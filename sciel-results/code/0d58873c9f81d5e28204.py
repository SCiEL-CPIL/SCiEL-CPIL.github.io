"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100
DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _objective(h):
    """Compute the value to minimize."""
    if not np.all(np.isfinite(h)):
        return np.inf
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _low_dim_eval(params):
    """Evaluate objective for low-dim parameterization."""
    # params: [a0_offset, a1, phi1, a2, phi2, a3, phi3]
    # h[i] = clamp(0.5 + params[0] + params[1]*cos(2*pi*1*i/N + params[2]) 
    #               + params[3]*cos(2*pi*2*i/N + params[4])
    #               + params[5]*cos(2*pi*3*i/N + params[6]), 0, 1)
    i = np.arange(N)
    h = 0.5 + params[0]
    h += params[1] * np.cos(2 * np.pi * 1 * i / N + params[2])
    h += params[3] * np.cos(2 * np.pi * 2 * i / N + params[4])
    h += params[5] * np.cos(2 * np.pi * 3 * i / N + params[6])
    h = np.clip(h, 0.0, 1.0)
    # Project to satisfy sum constraint
    diff = 50.0 - np.sum(h)
    h = h + diff / N
    h = np.clip(h, 0.0, 1.0)
    return _objective(h), h

def _nelder_mead(func, x0, maxiter=2000, tol=1e-8, time_limit=None, start_time=None):
    """Simple Nelder-Mead implementation with time guard."""
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        x = np.array(x0, dtype=float)
        x[i] += 0.1 if x[i] == 0 else 0.05 * x[i]
        simplex.append(x)
    
    fvals = [func(x) for x in simplex]
    best_iter = 0
    best_val = min(fvals)
    best_x = simplex[fvals.index(best_val)].copy()
    
    for it in range(maxiter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + (centroid - simplex[-1])
        fr = func(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + 2 * (centroid - simplex[-1])
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = func(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i])
        
        # Track best
        current_best = min(fvals)
        if current_best < best_val:
            best_val = current_best
            best_x = simplex[fvals.index(current_best)].copy()
    
    return best_x, best_val

def _full_dim_refine(h0, time_limit, start_time):
    """Refine in full dimension using gradient-free local search."""
    h = h0.copy()
    best_val = _objective(h)
    
    # Use coordinate descent with small perturbations
    np.random.seed(42)
    step_sizes = [0.01, 0.005, 0.001, 0.0005]
    
    for ss in step_sizes:
        improved = True
        while improved:
            if time.time() - start_time > time_limit:
                return h, best_val
            improved = False
            for i in range(N):
                if time.time() - start_time > time_limit:
                    return h, best_val
                # Try small perturbations
                for delta in [ss, -ss]:
                    h_new = h.copy()
                    h_new[i] = np.clip(h[i] + delta, 0.0, 1.0)
                    # Project to sum constraint
                    diff = 50.0 - np.sum(h_new)
                    # Distribute correction evenly, but keep within [0,1]
                    h_new = h_new + diff / N
                    h_new = np.clip(h_new, 0.0, 1.0)
                    val = _objective(h_new)
                    if val < best_val - 1e-12:
                        h = h_new
                        best_val = val
                        improved = True
                        break
    
    return h, best_val

def construct():
    start_time = time.time()
    time_limit = DEADLINE - start_time
    
    # Start with seed
    h_seed = np.full(N, 0.5)
    seed_val = _objective(h_seed)
    best_h = h_seed.copy()
    best_val = seed_val
    
    # Low-dimensional search with multiple random restarts
    np.random.seed(123)
    n_restarts = 8
    
    for r in range(n_restarts):
        if time.time() - start_time > time_limit - 30:
            break
        
        # Random initial params
        params = np.zeros(7)
        params[0] = 0.0  # offset
        params[1] = np.random.uniform(0.1, 0.4)  # amplitude 1st harmonic
        params[2] = np.random.uniform(0, 2 * np.pi)
        params[3] = np.random.uniform(0.05, 0.3)  # amplitude 2nd harmonic
        params[4] = np.random.uniform(0, 2 * np.pi)
        params[5] = np.random.uniform(0.02, 0.2)  # amplitude 3rd harmonic
        params[6] = np.random.uniform(0, 2 * np.pi)
        
        # Evaluate and refine with Nelder-Mead in low-dim space
        remaining = time_limit - (time.time() - start_time) - 15
        if remaining < 5:
            break
        
        best_params, best_ld_val = _nelder_mead(
            lambda p: _low_dim_eval(p)[0],
            params,
            maxiter=500,
            tol=1e-7,
            time_limit=remaining,
            start_time=start_time
        )
        
        _, h_ld = _low_dim_eval(best_params)
        
        # Refine in full dimension
        remaining = time_limit - (time.time() - start_time) - 5
        if remaining < 2:
            continue
        
        h_refined, val_refined = _full_dim_refine(h_ld, remaining, start_time)
        
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined.copy()
    
    # Final high-precision polish
    remaining = time_limit - (time.time() - start_time)
    if remaining > 3:
        h_final, val_final = _full_dim_refine(best_h, remaining, start_time)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
    
    return best_h
# EVOLVE-BLOCK-END
