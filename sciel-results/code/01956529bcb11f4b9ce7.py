"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _enforce_sum(h):
    """Enforce sum(h) = N/2 and 0<=h<=1 with iterative projection."""
    h = h.copy()
    for _ in range(50):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-16:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    diff = N/2.0 - s
    if abs(diff) > 1e-16:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _solve_period_p(p):
    """
    Solve the period-p subproblem exactly.
    h[i] = pattern[i % p] for all i.
    Overlaps: pattern[j]*(1-pattern[(j+1)%p]) for j=0..p-1.
    Constraint: sum(pattern) = p/2, 0<=pattern<=1.
    Minimize max overlap.
    """
    if p == 1:
        # pattern = [0.5], overlap = 0.5*(1-0.5) = 0.25
        return 0.25, np.array([0.5])
    
    def objective(pattern):
        overlaps = np.array([pattern[j] * (1.0 - pattern[(j+1) % p]) for j in range(p)])
        return float(np.max(overlaps))
    
    # Use SLSQP on the small p-dimensional problem
    x0 = np.full(p, 0.5)
    bounds = [(0.0, 1.0)] * p
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - p/2.0}]
    
    best_score = float('inf')
    best_pattern = x0
    
    # Multi-start
    np.random.seed(42)
    for trial in range(20):
        if trial == 0:
            x_start = x0.copy()
        else:
            x_start = np.random.rand(p)
            x_start = (x_start - x_start.min()) / (x_start.max() - x_start.min() + 1e-10)
            x_start = np.clip(x_start, 0, 1)
            # Project to sum = p/2
            s = np.sum(x_start)
            x_start = x_start + (p/2.0 - s) / p
            x_start = np.clip(x_start, 0, 1)
        
        result = minimize(
            objective,
            x_start,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-15, 'disp': False}
        )
        if result.fun < best_score:
            best_score = result.fun
            best_pattern = result.x
    
    return best_score, best_pattern

def _build_from_period(p, pattern, N):
    """Build a length-N sequence from a period-p pattern."""
    h = np.zeros(N)
    for i in range(N):
        h[i] = pattern[i % p]
    return h

def _fourier_h(params, N):
    """Reconstruct h from Fourier parameters: [c0, a1, b1, a2, b2, ...]"""
    c0 = params[0]
    h = np.full(N, c0)
    k = 1
    while 2*k < len(params):
        a_k = params[2*k-1]
        b_k = params[2*k]
        h += a_k * np.cos(2*np.pi*k*np.arange(N)/N) + b_k * np.sin(2*np.pi*k*np.arange(N)/N)
        k += 1
    return h

def _fourier_objective(params):
    h = _fourier_h(params, N)
    if np.any(h < -0.1) or np.any(h > 1.1):
        return 1.0 + 10.0 * (np.sum(h[h<0]**2) + np.sum((h-1)[h>1]**2))
    h = np.clip(h, 0, 1)
    s = np.sum(h)
    if abs(s - N/2.0) > 0.5:
        return 1.0 + 10.0 * (s - N/2.0)**2
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Phase 1: Solve all period-p subproblems for divisors of 100
    divisors = [1, 2, 4, 5, 10, 20, 25, 50, 100]
    best_score = float('inf')
    best_h = None
    
    for p in divisors:
        if time.time() > deadline - 60.0:
            break
        try:
            score_p, pattern = _solve_period_p(p)
            h_p = _build_from_period(p, pattern, N)
            h_p = _enforce_sum(h_p)
            actual_score = _score(h_p)
            if actual_score < best_score:
                best_score = actual_score
                best_h = h_p.copy()
        except Exception:
            pass
    
    if best_h is None:
        # Fallback: 2-block
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        best_h = np.zeros(N)
        half = N // 2
        best_h[:half] = A
        best_h[half:] = B
        best_h = _enforce_sum(best_h)
        best_score = _score(best_h)
    
    h_final = best_h.copy()
    current_score = best_score
    
    def _try_update(h_candidate):
        nonlocal h_final, current_score
        h_c = _enforce_sum(h_candidate)
        if np.all(h_c >= -1e-12) and np.all(h_c <= 1.0 + 1e-12):
            h_c = np.clip(h_c, 0.0, 1.0)
            s = np.sum(h_c)
            if abs(s - N/2.0) < 1e-10:
                sc = _score(h_c)
                if sc < current_score - 1e-15:
                    h_final = h_c.copy()
                    current_score = sc
                    return True
        return False
    
    # Phase 2: Local SLSQP polish on the full problem
    try:
        if time.time() < deadline - 40.0:
            def objective_raw(h):
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return float(np.max(overlap))
            
            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
            bounds = [(0.0, 1.0)] * N
            
            result = minimize(
                objective_raw,
                h_final.copy(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 2000, 'ftol': 1e-15, 'disp': False}
            )
            _try_update(result.x)
    except Exception:
        pass
    
    # Phase 3: Global search via differential evolution on Fourier parameters
    try:
        if time.time() < deadline - 30.0:
            n_modes = 5
            n_params = 1 + 2 * n_modes  # c0 + a1..an + b1..bn
            
            # Bounds for Fourier parameters
            bounds_de = [(-0.5, 0.5)]  # c0 around 0.5
            for k in range(1, n_modes + 1):
                bounds_de.append((-0.4, 0.4))  # a_k
                bounds_de.append((-0.4, 0.4))  # b_k
            
            # Initialize c0 to 0.5
            x0_de = np.zeros(n_params)
            x0_de[0] = 0.5
            
            # Set initial Fourier coefficients from current best
            h_cur = h_final
            x0_de[0] = np.mean(h_cur)
            for k in range(1, n_modes + 1):
                x0_de[2*k-1] = 2.0/N * np.sum(h_cur * np.cos(2*np.pi*k*np.arange(N)/N))
                x0_de[2*k] = 2.0/N * np.sum(h_cur * np.sin(2*np.pi*k*np.arange(N)/N))
            
            result_de = differential_evolution(
                _fourier_objective,
                bounds=bounds_de,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-10,
                polish=True,
                init='latinhypercube',
                x0=x0_de
            )
            
            if result_de.fun < current_score:
                h_de = _fourier_h(result_de.x, N)
                _try_update(h_de)
                
    except Exception:
        pass
    
    # Phase 4: Try near-periodic structures with boundary adjustment
    try:
        if time.time() < deadline - 20.0:
            # For each period p, allow the last few elements to be free
            for p in [4, 5, 10, 20]:
                if time.time() > deadline - 15.0:
                    break
                score_p, pattern = _solve_period_p(p)
                # Build with free last p elements
                h_near = np.zeros(N)
                n_free = p
                n_fixed = N - n_free
                for i in range(n_fixed):
                    h_near[i] = pattern[i % p]
                # Last p elements are free, initialized to pattern
                for i in range(n_free):
                    h_near[n_fixed + i] = pattern[i]
                
                h_near = _enforce_sum(h_near)
                
                def obj_near(h):
                    shifted = np.zeros(N)
                    shifted[:-1] = h[1:]
                    overlap = h * (1.0 - shifted)
                    return float(np.max(overlap))
                
                # Only optimize the last n_free elements
                free_idx = list(range(n_fixed, N))
                
                def obj_free(params):
                    h_temp = h_near.copy()
                    for j, idx in enumerate(free_idx):
                        h_temp[idx] = params[j]
                    h_temp = np.clip(h_temp, 0, 1)
                    # Adjust sum
                    s = np.sum(h_temp)
                    diff = N/2.0 - s
                    # Distribute adjustment over free elements
                    h_temp[free_idx] += diff / n_free
                    h_temp = np.clip(h_temp, 0, 1)
                    return obj_near(h_temp)
                
                x0_free = h_near[free_idx].copy()
                bounds_free = [(0.0, 1.0)] * n_free
                
                result_near = minimize(
                    obj_free,
                    x0_free,
                    method='SLSQP',
                    bounds=bounds_free,
                    options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
                )
                
                h_opt = h_near.copy()
                for j, idx in enumerate(free_idx):
                    h_opt[idx] = result_near.x[j]
                _try_update(h_opt)
                
    except Exception:
        pass
    
    # Phase 5: Final high-precision polish
    try:
        if time.time() < deadline - 5.0:
            def objective_smooth(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                smax = _smooth_max(overlap, alpha=500.0)
                return t + 50000.0 * np.maximum(0.0, smax - t)**2
            
            x0_polish = np.concatenate([h_final, [current_score + 0.001]])
            bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]
            
            def constraint_sum(params):
                return np.sum(params[:N]) - N/2.0
            
            constraints_ext = [
                {'type': 'eq', 'fun': constraint_sum}
            ]
            
            result_final = minimize(
                objective_smooth,
                x0_polish,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 3000, 'ftol': 1e-16, 'disp': False}
            )
            
            if result_final.fun < 100.0:
                _try_update(result_final.x[:N])
                
    except Exception:
        pass
    
    # Ensure final result is valid
    h_final = _enforce_sum(h_final)
    return h_final
