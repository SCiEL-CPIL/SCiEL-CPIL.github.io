"""Rewritten program for third autocorrelation inequality optimization."""
import numpy as np
import time

N = 400
DEADLINE = 140.0  # safety margin before 150s kill


def _objective(h):
    """Compute the task value: |2*N*max(conv(h,h))/sum(h)^2|"""
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h, mode='full')
    max_val = np.max(conv)  # signed max, no abs
    return abs(2.0 * N * max_val / (s * s))


def _eval_with_time(h, t0, deadline):
    """Evaluate h, return (value, time_exceeded)."""
    if time.time() > deadline:
        return np.inf, True
    val = _objective(h)
    return val, False


def _nelder_mead(h0, f, max_iter=200, tol=1e-10, t0=None, deadline=None):
    """Simple Nelder-Mead for moderate dimension."""
    n = len(h0)
    # Build initial simplex
    simplex = [h0.copy()]
    for i in range(n):
        h = h0.copy()
        step = 0.01 * (abs(h0[i]) + 1.0)
        h[i] += step
        simplex.append(h)
    
    fvals = [f(x) for x in simplex]
    
    for it in range(max_iter):
        if t0 is not None and time.time() > deadline:
            break
        # Sort
        idx = np.argsort(fvals)
        simplex = [simplex[i] for i in idx]
        fvals = [fvals[i] for i in idx]
        
        if fvals[0] - fvals[-1] < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < fvals[0]:
            # Expansion
            gamma = 2.0
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = f(xe)
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
            rho = 0.5
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
    
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]


def _random_structures(rng):
    """Generate diverse random initial structures."""
    candidates = []
    
    # 1. Random signs with smooth magnitude
    signs = rng.choice([-1, 1], size=N)
    smooth_mag = np.exp(-0.5 * (np.linspace(-3, 3, N))**2) * rng.uniform(0.5, 1.0)
    candidates.append(signs * smooth_mag)
    
    # 2. Chirp-like: cos(quadratic phase)
    t = np.linspace(-1, 1, N)
    for freq in [1, 2, 5, 10, 20]:
        candidates.append(np.cos(freq * np.pi * t**2))
    
    # 3. Random with low-pass filtering
    raw = rng.standard_normal(N)
    # Simple moving average filter
    kernel = np.ones(5) / 5.0
    filtered = np.convolve(raw, kernel, mode='same')
    candidates.append(filtered)
    
    # 4. Alternating blocks
    block_size = rng.choice([10, 20, 50, 100])
    n_blocks = N // block_size
    block_vals = rng.uniform(-1, 1, size=n_blocks)
    h = np.repeat(block_vals, block_size)[:N]
    candidates.append(h)
    
    # 5. Sine wave with random phase
    for freq in [1, 3, 7, 15]:
        phase = rng.uniform(0, 2 * np.pi)
        candidates.append(np.sin(freq * 2 * np.pi * t + phase))
    
    # 6. Triangle wave
    tri = 2 * np.abs(2 * (t * 5 % 1) - 1) - 1
    candidates.append(tri)
    
    # 7. Random sparse: mostly zero with few non-zero
    h = np.zeros(N)
    nz_idx = rng.choice(N, size=rng.randint(50, 150), replace=False)
    h[nz_idx] = rng.uniform(-1, 1, size=len(nz_idx))
    candidates.append(h)
    
    # 8. Cosine window with random frequency
    for freq in [1, 2, 4, 8]:
        candidates.append(np.cos(2 * np.pi * freq * t) * np.exp(-0.5 * (t*2)**2))
    
    return candidates


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_val = _objective(best_h)  # 2.0
    
    def f(h):
        return _objective(h)
    
    # Phase 1: Evaluate random structures
    candidates = _random_structures(rng)
    
    for i, h in enumerate(candidates):
        if time.time() > deadline - 10:
            break
        val = _objective(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Also try some scaled versions
    for h in candidates[:5]:
        for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
            if time.time() > deadline - 10:
                break
            hs = h * scale
            val = _objective(hs)
            if val < best_val:
                best_val = val
                best_h = hs.copy()
    
    # Phase 2: Local optimization from best candidates
    if time.time() < deadline - 20:
        # Try Nelder-Mead from best_h
        h_opt, val_opt = _nelder_mead(best_h, f, max_iter=100, t0=t0, deadline=deadline)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Phase 3: More random restarts with local search
    if time.time() < deadline - 15:
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            # Generate new random candidate
            t = np.linspace(-1, 1, N)
            h = rng.standard_normal(N)
            # Apply random low-pass
            cutoff = rng.uniform(0.1, 2.0)
            h = np.convolve(h, np.sinc(cutoff * np.linspace(-1, 1, N)), mode='same')
            val = _objective(h)
            
            if val < best_val * 1.1:  # promising
                h_opt, val_opt = _nelder_mead(h, f, max_iter=50, t0=t0, deadline=deadline)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
    
    # Phase 4: Coordinate descent refinement
    if time.time() < deadline - 5:
        h = best_h.copy()
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(0, N, 2):  # step by 2 for speed
                if time.time() > deadline - 2:
                    break
                for delta in [-0.1, -0.01, 0.01, 0.1]:
                    h_new = h.copy()
                    h_new[i] += delta
                    val = _objective(h_new)
                    if val < best_val:
                        h = h_new
                        best_val = val
                        improved = True
                        break
    
    # Final check
    final_val = _objective(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h


if __name__ == "__main__":
    h = construct()
    print(f"Best value: {_objective(h)}")
