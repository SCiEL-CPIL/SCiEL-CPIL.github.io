"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _overlaps(h, N):
    """Compute overlap at each lag k=1..N-1: sum_i h[i]*(1-h[(i+k)%N])"""
    h = np.asarray(h)
    overlaps = np.zeros(N - 1)
    for k in range(1, N):
        shifted = np.roll(h, -k)
        overlaps[k - 1] = np.sum(h * (1.0 - shifted))
    return overlaps

def _max_overlap(h, N):
    o = _overlaps(h, N)
    return np.max(o)

def _smooth_max_overlap(h, N, beta=100.0):
    """Log-sum-exp smooth surrogate for max overlap."""
    o = _overlaps(h, N)
    m = np.max(o)
    if beta < 1e6:
        # log-sum-exp: m + log(sum(exp(beta*(o-m))))/beta
        s = np.sum(np.exp(beta * (o - m)))
        return m + np.log(s) / beta
    else:
        return m

def _project(h, N):
    """Project h onto [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    target = N / 2.0
    if abs(s - target) < 1e-12:
        return h
    # Adjust: scale or shift to meet sum constraint
    # Simple approach: add uniform offset then re-clip, iterate
    for _ in range(50):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    # Final correction on non-saturated entries
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        free = (h > 1e-8) & (h < 1.0 - 1e-8)
        if np.any(free):
            h[free] += diff / np.sum(free)
            h = np.clip(h, 0.0, 1.0)
    return h

def _local_search(h0, N, max_iter=2000, tol=1e-8):
    """Gradient-based local search with smooth max overlap."""
    h = h0.copy()
    h = _project(h, N)
    best_h = h.copy()
    best_val = _max_overlap(h, N)
    
    # Use scipy minimize with BFGS on smooth surrogate
    def obj(x):
        x = _project(x, N)
        return _smooth_max_overlap(x, N, beta=50.0)
    
    def grad(x):
        x = _project(x, N)
        # Numerical gradient for safety
        eps = 1e-5
        g = np.zeros(N)
        for i in range(N):
            xp = x.copy(); xp[i] += eps
            xp = _project(xp, N)
            xm = x.copy(); xm[i] -= eps
            xm = _project(xm, N)
            g[i] = (_smooth_max_overlap(xp, N, beta=50.0) - _smooth_max_overlap(xm, N, beta=50.0)) / (2 * eps)
        return g
    
    try:
        res = minimize(obj, h, method='L-BFGS-B', jac=grad,
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        h_new = _project(res.x, N)
        val = _max_overlap(h_new, N)
        if val < best_val:
            best_h = h_new
            best_val = val
    except Exception:
        pass
    
    # Coordinate descent polish
    for it in range(500):
        improved = False
        for i in range(N):
            old_h = h[i]
            # Try small perturbations
            for delta in [-0.01, -0.005, -0.001, 0.001, 0.005, 0.01]:
                h_new = h.copy()
                h_new[i] = np.clip(old_h + delta, 0.0, 1.0)
                h_new = _project(h_new, N)
                v = _max_overlap(h_new, N)
                if v < best_val - 1e-10:
                    best_val = v
                    best_h = h_new.copy()
                    improved = True
                    break
        if not improved:
            break
        h = best_h.copy()
    
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    best_h = np.full(N, 0.5)
    best_val = _max_overlap(best_h, N)
    
    # Diverse initializations
    inits = []
    # 1. Constant 0.5
    inits.append(np.full(N, 0.5))
    # 2. Square wave
    sw = np.zeros(N)
    sw[::2] = 1.0
    sw = _project(sw, N)
    inits.append(sw)
    # 3. Random
    rng = np.random.default_rng(42)
    for seed in range(10):
        r = rng.random(N)
        r = _project(r, N)
        inits.append(r)
    # 4. High frequency
    hf = 0.5 + 0.4 * np.sin(2 * np.pi * 5 * np.arange(N) / N)
    hf = _project(hf, N)
    inits.append(hf)
    
    for i, h0 in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        try:
            h, val = _local_search(h0, N)
            if val < best_val:
                best_val = val
                best_h = h.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        # More aggressive coordinate descent
        h = best_h.copy()
        for it in range(100):
            if time.time() - start_time > deadline - 3:
                break
            improved = False
            for i in range(N):
                if time.time() - start_time > deadline - 3:
                    break
                old_h = h[i]
                for delta in [-0.005, -0.002, -0.0005, 0.0005, 0.002, 0.005]:
                    h_new = h.copy()
                    h_new[i] = np.clip(old_h + delta, 0.0, 1.0)
                    h_new = _project(h_new, N)
                    v = _max_overlap(h_new, N)
                    if v < best_val - 1e-12:
                        best_val = v
                        best_h = h_new.copy()
                        h = h_new.copy()
                        improved = True
                        break
            if not improved:
                break
    
    return best_h
# EVOLVE-BLOCK-END
