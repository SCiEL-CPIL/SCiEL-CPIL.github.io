"""Erdos minimum overlap: minimize max(correlate(h, 1-h, full))/N*2."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 135.0  # safety margin below 150s kill
_START = time.time()


def remaining_time():
    return DEADLINE - (time.time() - _START)


def objective(h):
    """Compute the raw value (lower is better)."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def project(h):
    """Project h to satisfy: h in [0,1]^N, sum(h) ~ N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(100):
        s = h.sum()
        if abs(s - target) < 1e-12:
            break
        # Adjust by uniform shift, then re-clip
        h = h + (target - s) / N
        h = np.clip(h, 0.0, 1.0)
    return h


def construct():
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)  # = 1.0
    
    # Try several deterministic structured candidates
    candidates = []
    
    # 1. Cosine fundamental
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2 * np.pi * i / N)
    candidates.append(project(h_cos))
    
    # 2. Square wave (alternating blocks)
    h_sq = np.zeros(N)
    h_sq[::2] = 1.0
    h_sq[1::2] = 0.0
    candidates.append(project(h_sq))
    
    # 3. Block pattern: 50 ones, 50 zeros
    h_block = np.zeros(N)
    h_block[:N//2] = 1.0
    candidates.append(project(h_block))
    
    # 4. Random initializations
    rng = np.random.default_rng(42)
    for _ in range(10):
        h_rand = rng.uniform(0, 1, N)
        candidates.append(project(h_rand))
    
    # 5. Fourier-based: h_i = 0.5 + a*cos(2*pi*i/N) + b*sin(2*pi*i/N)
    for a in [0.1, 0.2, 0.3, 0.4]:
        for b in [0.0, 0.1, 0.2]:
            h_four = 0.5 + a * np.cos(2 * np.pi * i / N) + b * np.sin(2 * np.pi * i / N)
            h_four = project(h_four)
            candidates.append(h_four)
    
    # Evaluate all candidates
    for h_cand in candidates:
        if not np.all(np.isfinite(h_cand)):
            continue
        val = objective(h_cand)
        if val < best_val:
            best_val = val
            best_h = h_cand.copy()
    
    # Use scipy optimization to further refine best candidate
    if remaining_time() > 10:
        h_opt = best_h.copy()
        
        # Define objective for scipy (we want to minimize)
        def obj_scipy(x):
            h_p = project(x)
            return objective(h_p)
        
        # Try L-BFGS-B
        res = minimize(
            obj_scipy,
            h_opt,
            method='L-BFGS-B',
            bounds=[(0.0, 1.0)] * N,
            options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-12}
        )
        if remaining_time() > 5:
            h_res = project(res.x)
            val_res = objective(h_res)
            if val_res < best_val:
                best_val = val_res
                best_h = h_res.copy()
        
        # Try Nelder-Mead if still time
        if remaining_time() > 10:
            res_nm = minimize(
                obj_scipy,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if remaining_time() > 5:
                h_nm = project(res_nm.x)
                val_nm = objective(h_nm)
                if val_nm < best_val:
                    best_val = val_nm
                    best_h = h_nm.copy()
    
    # Final sanity check
    best_h = project(best_h)
    assert np.all(np.isfinite(best_h)), "Non-finite values in result"
    assert np.all(best_h >= -1e-10) and np.all(best_h <= 1.0 + 1e-10), "Out of bounds"
    assert abs(best_h.sum() - N / 2.0) < 1e-9 * N, "Sum constraint violated"
    
    return best_h


# EVOLVE-BLOCK-START
# (construct is already defined above)
# EVOLVE-BLOCK-END
