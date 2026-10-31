"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def _max_overlap_fast(h):
    """Compute max overlap using FFT-based circular cross-correlation."""
    N = len(h)
    one_minus_h = 1.0 - h
    H = np.fft.fft(h)
    G = np.fft.fft(one_minus_h)
    overlaps = np.real(np.fft.ifft(H * np.conj(G)))
    return np.max(overlaps)

def _soft_max_overlap(h, beta=50.0):
    """Smooth surrogate for max overlap using log-sum-exp."""
    N = len(h)
    one_minus_h = 1.0 - h
    H = np.fft.fft(h)
    G = np.fft.fft(one_minus_h)
    overlaps = np.real(np.fft.ifft(H * np.conj(G)))
    m = np.max(overlaps)
    # Avoid overflow in exp
    return m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta

def _score(h):
    """Score is the upper bound on C5, which is the max overlap."""
    return _max_overlap_fast(h)

def _fix_sum(h, N):
    """Ensure h sums to exactly N/2 and is in [0,1]."""
    h = np.clip(h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-14:
        if diff > 0:
            # Add mass to elements with most room
            order = np.argsort(1.0 - h)
            for idx in order:
                if diff <= 1e-14:
                    break
                add = min(diff, 1.0 - h[idx])
                h[idx] += add
                diff -= add
        else:
            # Subtract mass from elements with most mass
            order = np.argsort(-h)
            for idx in order:
                if -diff <= 1e-14:
                    break
                sub = min(-diff, h[idx])
                h[idx] -= sub
                diff += sub
    return h

def _try_alternating(N):
    """Try alternating high/low values."""
    a = 0.5
    h = np.zeros(N)
    h[0::2] = a
    h[1::2] = 1 - a
    return _fix_sum(h, N)

def _try_periodic_patterns(N):
    """Try various periodic patterns and return the best one."""
    best_h = None
    best_s = float('inf')
    
    # Pattern: blocks of high/low values
    for block_size in [2, 3, 4, 5]:
        n_blocks = N // block_size
        for a in [0.3, 0.4, 0.5, 0.6, 0.7]:
            h = np.zeros(N)
            for b in range(n_blocks):
                start = b * block_size
                if b % 2 == 0:
                    h[start:start+block_size] = a
                else:
                    h[start:start+block_size] = 1 - a
            h = _fix_sum(h, N)
            s = _score(h)
            if s < best_s:
                best_s = s
                best_h = h.copy()
    
    # Pattern: sinusoidal
    for freq in [1, 2, 3]:
        for amp in [0.3, 0.4, 0.5]:
            x = np.arange(N)
            h = 0.5 + amp * np.sin(2 * np.pi * freq * x / N)
            h = np.clip(h, 0, 1)
            h = _fix_sum(h, N)
            s = _score(h)
            if s < best_s:
                best_s = s
                best_h = h.copy()
    
    return best_h, best_s

def _optimize_smooth(h0, time_budget=20.0):
    """Optimize using L-BFGS-B with soft-max surrogate and sum penalty."""
    start = time.time()
    N = len(h0)
    beta = 50.0
    mu = 1000.0 # Penalty for sum constraint
    
    def _objective(x):
        if time.time() - start > time_budget:
            return 1e10
        h = 0.5 + x
        h = np.clip(h, 0, 1)
        sum_penalty = mu * (np.sum(h) - N/2.0)**2
        return _soft_max_overlap(h, beta=beta) + sum_penalty
    
    x0 = h0 - 0.5
    x0 = np.clip(x0, -0.5, 0.5)
    
    try:
        res = minimize(_objective, x0, method='L-BFGS-B',
                       bounds=[(-0.5, 0.5)] * N,
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        h_new = 0.5 + res.x
        h_new = np.clip(h_new, 0, 1)
        h_new = _fix_sum(h_new, N)
        return h_new, _score(h_new)
    except Exception:
        return h0, _score(h0)

def _local_search_fast(h, max_iter=5000, step_size=0.02, time_limit=15.0):
    """Fast local search using FFT-based scoring."""
    N = len(h)
    current = h.copy()
    current_score = _score(current)
    start = time.time()
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        if i == j:
            continue
        delta = random.uniform(-step_size, step_size)
        new_i = current[i] + delta
        new_j = current[j] - delta
        if new_i < 0 or new_i > 1 or new_j < 0 or new_j > 1:
            continue
        
        new_h = current.copy()
        new_h[i] = new_i
        new_h[j] = new_j
        
        new_score = _score(new_h)
        if new_score < current_score - 1e-12:
            current = new_h
            current_score = new_score
    
    return current, current_score

def _refine_with_scipy(h, time_budget=20.0):
    """Refine h using scipy minimize with soft-max surrogate."""
    start = time.time()
    N = len(h)
    
    best_h = h.copy()
    best_score = _score(h)
    
    for trial in range(4):
        if time.time() - start > time_budget - 3:
            break
        if trial == 0:
            x0 = h.copy()
        elif trial == 1:
            x0 = h + np.random.normal(0, 0.05, N)
            x0 = np.clip(x0, 0, 1)
        elif trial == 2:
            x0 = h + np.random.normal(0, 0.1, N)
            x0 = np.clip(x0, 0, 1)
        else:
            x0 = h + np.random.normal(0, 0.15, N)
            x0 = np.clip(x0, 0, 1)
        
        h_new, s_new = _optimize_smooth(x0, time_budget=5.0)
        if s_new < best_score:
            best_score = s_new
            best_h = h_new
    
    return best_h, best_score

def _final_polish(h, time_budget=10.0):
    """High-precision polish using coordinate descent."""
    N = len(h)
    current = h.copy()
    current_score = _score(current)
    start = time.time()
    
    improved = True
    while improved and time.time() - start < time_budget - 1:
        improved = False
        for i in range(N):
            if time.time() - start > time_budget - 1:
                break
            for j in range(N):
                if i == j:
                    continue
                if time.time() - start > time_budget - 1:
                    break
                d_lo = max(-current[i], current[j] - 1.0)
                d_hi = min(1.0 - current[i], current[j])
                if d_lo >= d_hi:
                    continue
                
                a, b = d_lo, d_hi
                for _ in range(15):
                    m1 = a + (b - a) / 3.0
                    m2 = b - (b - a) / 3.0
                    h1 = current.copy()
                    h1[i] += m1
                    h1[j] -= m1
                    s1 = _score(h1)
                    h2 = current.copy()
                    h2[i] += m2
                    h2[j] -= m2
                    s2 = _score(h2)
                    if s1 < s2:
                        b = m2
                    else:
                        a = m1
                
                best_delta = (a + b) / 2.0
                new_h = current.copy()
                new_h[i] += best_delta
                new_h[j] -= best_delta
                new_score = _score(new_h)
                if new_score < current_score - 1e-12:
                    current = new_h
                    current_score = new_score
                    improved = True
    
    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 130 # 20s safety margin from 150s limit
    
    random.seed(42)
    np.random.seed(42)
    
    # Start with multiple initial patterns
    h_alt = _try_alternating(N)
    s_alt = _score(h_alt)
    
    h_per, s_per = _try_periodic_patterns(N)
    
    if s_per < s_alt:
        best_h = h_per
        best_score = s_per
    else:
        best_h = h_alt
        best_score = s_alt
    
    # Local search from best
    if time.time() < deadline - 50:
        h1, s1 = _local_search_fast(best_h, max_iter=5000, step_size=0.02, time_limit=20.0)
        if s1 < best_score:
            best_score = s1
            best_h = h1
    
    # Refine with scipy using soft-max surrogate (multiple starts)
    if time.time() < deadline - 25:
        h_ref, s_ref = _refine_with_scipy(best_h, time_budget=20.0)
        if s_ref < best_score:
            best_score = s_ref
            best_h = h_ref
    
    # More local search after scipy refinement
    if time.time() < deadline - 15:
        h2, s2 = _local_search_fast(best_h, max_iter=3000, step_size=0.01, time_limit=10.0)
        if s2 < best_score:
            best_score = s2
            best_h = h2
    
    # Final high-precision polish with coordinate descent
    if time.time() < deadline - 5:
        h3, s3 = _final_polish(best_h, time_budget=8.0)
        if s3 < best_score:
            best_score = s3
            best_h = h3
    
    # Ensure exact sum and bounds
    best_h = _fix_sum(best_h, N)
    
    return best_h
# EVOLVE-BLOCK-END
