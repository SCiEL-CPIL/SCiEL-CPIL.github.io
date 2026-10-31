"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _project_to_feasible(h):
    """Project h to satisfy sum=N/2 and 0<=h<=1."""
    target = N / 2.0
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if s < 1e-10:
        return np.full(N, 0.5)
    h = h * (target / s)
    h = np.clip(h, 0.0, 1.0)
    for _ in range(50):
        diff = target - h.sum()
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0
            if mask.any():
                h[mask] += diff / mask.sum()
            else:
                break
        else:
            mask = h > 0.0
            if mask.any():
                h[mask] += diff / mask.sum()
            else:
                break
        h = np.clip(h, 0.0, 1.0)
    return h

def _h_sine(params):
    """Generate h from sine parameters: [amp, offset, freq, phase]"""
    amp, offset, freq, phase = params
    x = np.arange(N)
    h = amp * np.sin(2 * np.pi * freq * x / N + phase) + offset
    return _project_to_feasible(h)

def _objective_sine(params):
    h = _h_sine(params)
    return _score(h)

def _h_ramp(params):
    """Generate h from ramp parameters: [start, end, width]
    h is 0 before start, 1 after end, linear in between.
    """
    start, end, width = params
    x = np.arange(N)
    h = np.zeros(N)
    # Simple step function with smooth transition
    # To avoid 1->0 drop, we might need a specific shape.
    # Let's try a simple step: 0 then 1.
    # But we know 1->0 at the end is bad.
    # So maybe 0 then 1, but the last element is forced to 0?
    # This is handled by projection? No, projection preserves shape roughly.
    
    # Let's try a "tent" or "triangle"
    # Or just a step.
    h[x < start] = 0.0
    h[x >= end] = 1.0
    mask = (x >= start) & (x < end)
    if mask.any():
        t = (x[mask] - start) / (end - start)
        h[mask] = t
    return _project_to_feasible(h)

def _objective_ramp(params):
    h = _h_ramp(params)
    return _score(h)

def _h_two_block(params):
    """Two block: [val1, val2, split]"""
    v1, v2, split = params
    split = int(np.clip(split, 1, N-1))
    h = np.zeros(N)
    h[:split] = v1
    h[split:] = v2
    return _project_to_feasible(h)

def _objective_two_block(params):
    h = _h_two_block(params)
    return _score(h)

def construct():
    deadline = time.time() + 140.0
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # 1. Optimize Sine family
    if time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        for _ in range(20):
            p0 = np.array([rng.uniform(0.2, 0.5), rng.uniform(0.3, 0.7), rng.uniform(0.5, 3.0), rng.uniform(0, 2*np.pi)])
            res = minimize(_objective_sine, p0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_h = _h_sine(res.x)
                
    # 2. Optimize Ramp family
    if time.time() < deadline - 30:
        for _ in range(20):
            p0 = np.array([rng.uniform(0, N), rng.uniform(0, N), rng.uniform(1, 10)])
            if p0[1] < p0[0]: p0[[0,1]] = p0[[1,0]]
            res = minimize(_objective_ramp, p0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_h = _h_ramp(res.x)
                
    # 3. Optimize Two-Block
    if time.time() < deadline - 30:
        for _ in range(20):
            p0 = np.array([rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(1, N-1)])
            res = minimize(_objective_two_block, p0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_h = _h_two_block(res.x)

    # 4. Local search refinement on best_h
    if time.time() < deadline - 10:
        rng = np.random.default_rng(123)
        h = best_h.copy()
        for _ in range(5000):
            if time.time() > deadline - 5: break
            idx = rng.integers(0, N)
            step = rng.uniform(-0.1, 0.1)
            h_trial = h.copy()
            h_trial[idx] = np.clip(h_trial[idx] + step, 0.0, 1.0)
            h_trial = _project_to_feasible(h_trial)
            sc = _score(h_trial)
            if sc < best_score - 1e-10:
                best_score = sc
                h = h_trial

    return h

if __name__ == "__main__":
    h = construct()
    print(_score(h))
