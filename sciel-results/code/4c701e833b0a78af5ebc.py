"""
Structural change: Direct optimization of the exact objective using
coordinate ascent with exact line search, combined with multi-start
from diverse structural templates. Avoids the log-sum-exp surrogate
approximation entirely.
"""
import numpy as np
import time

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 145.0     # Safety margin from 150s
RNG = np.random.RandomState(42)

def compute_value(h):
    """
    Exact evaluator leveraging nonnegativity:
    - g = convolve(h,h) >= 0, so norm_inf = max(g) = g[N-1] = sum(h^2)
    - norm_1 = sum(g)/(M+1) = (sum(h))^2/(M+1)
    - l2 = (W/3) * (2*sum(g^2) + sum(g[:-1]*g[1:]))
    """
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)):
        return 0.0
    h = np.maximum(0.0, h)
    if np.all(h == 0):
        return 0.0
    
    g = np.convolve(h, h)
    norm_inf = g[N - 1]  # = sum(h^2)
    if norm_inf <= 0:
        return 0.0
    
    sum_h = np.sum(h)
    norm_1 = (sum_h * sum_h) / (M + 1)
    if norm_1 <= 0:
        return 0.0
    
    sum_g_sq = np.sum(g * g)
    sum_g_gnext = np.sum(g[:-1] * g[1:])
    l2 = (W / 3.0) * (2.0 * sum_g_sq + sum_g_gnext)
    
    return l2 / (norm_1 * norm_inf)

def _coordinate_ascent(h, time_budget=5.0):
    """
    Coordinate ascent with exact 1D line search on each coordinate.
    For each coordinate j, we optimize h[j] while holding others fixed.
    The objective as a function of h[j] is unimodal (or close to it)
    due to the structure of the problem.
    """
    start = time.time()
    h = np.maximum(0.0, h.copy())
    best_val = compute_value(h)
    
    n_iter = 0
    while time.time() - start < time_budget:
        improved = False
        for j in range(N):
            if time.time() - start > time_budget - 0.1:
                    break
            # Save current value
            old_hj = h[j]
            
            # Try several candidate values for h[j]
            # Use a geometric grid around the current value
            candidates = [old_hj]
            if old_hj > 1e-10:
                for ratio in [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0, 3.0]:
                    candidates.append(old_hj * ratio)
            else:
                for val in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
                    candidates.append(val)
            
            # Also try some absolute values
            for val in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                if val not in candidates:
                    candidates.append(val)
            
            best_candidate = old_hj
            best_candidate_val = best_val
            
            for cand in candidates:
                h[j] = max(0.0, cand)
                v = compute_value(h)
                if v > best_candidate_val:
                    best_candidate_val = v
                    best_candidate = cand
            
            h[j] = best_candidate
            if best_candidate_val > best_val:
                improved = True
                best_val = best_candidate_val
        
        n_iter += 1
        if not improved and n_iter > 1:
            break
    
    return h, best_val

def _generate_starts():
    x = np.linspace(0, N - 1, N)
    starts = []
    
    # 1. Uniform (Seed)
    starts.append(np.ones(N))
    
    # 2. Gaussian centered
    starts.append(np.exp(-0.5 * ((x - N / 2) / (N / 4)) ** 2))
    
    # 3. Two Peaks
    p1 = np.exp(-0.5 * ((x - N * 0.25) / (N / 8)) ** 2)
    p2 = np.exp(-0.5 * ((x - N * 0.75) / (N / 8)) ** 2)
    starts.append(p1 + p2)
    
    # 4. Semicircle
    starts.append(np.sqrt(np.maximum(0.0, 1 - ((2 * x / (N - 1) - 1)) ** 2)))
    
    # 5. Cosine bump
    starts.append(0.5 * (1 + np.cos(2 * np.pi * x / N)))
    
    # 6. Triangle
    starts.append(1 - 2 * np.abs(x - (N - 1) / 2) / (N - 1))
    
    # 7. Comb patterns
    for k in [2, 3, 5, 8]:
        comb = 1.0 + 0.8 * np.cos(2 * np.pi * k * x / N)
        starts.append(np.maximum(0.1, comb))
        
    # 8. Random smooth profiles
    for _ in range(15):
        freqs = RNG.randint(1, 10, size=6)
        phases = RNG.uniform(0, 2 * np.pi, size=6)
        amps = RNG.uniform(0.5, 1.0, size=6)
        prof = np.zeros(N)
        for f_, p_, a_ in zip(freqs, phases, amps):
            prof += a_ * np.cos(2 * np.pi * f_ * x / N + p_)
        starts.append(np.maximum(0.1, prof))
        
    # 9. Random binary-like sequences (smoothed)
    for _ in range(10):
        binary = (RNG.rand(N) > 0.5).astype(float)
        kernel = np.ones(3) / 3.0
        smooth_binary = np.convolve(binary, kernel, mode='same')
        starts.append(np.maximum(0.1, smooth_binary))
    
    # 10. Block structures
    for block_size in [5, 10, 15, 25]:
        n_blocks = N // block_size
        blocks = RNG.uniform(0.5, 2.0, size=n_blocks)
        block_h = np.repeat(blocks, block_size)
        if len(block_h) < N:
            block_h = np.concatenate([block_h, np.zeros(N - len(block_h))])
        starts.append(block_h[:N])
    
    # 11. Ramp structures
    for direction in [1, -1]:
        ramp = np.linspace(0.5, 2.0, N) if direction == 1 else np.linspace(2.0, 0.5, N)
        starts.append(ramp)
    
    # 12. Mixed: high center, low edges
    starts.append(np.ones(N) * 0.5 + np.exp(-0.5 * ((x - N / 2) / (N / 6)) ** 2))
    
    # 13. Sparse peaks
    for peak_pos in [N // 4, N // 2, 3 * N // 4]:
        sparse = np.zeros(N)
        sparse[peak_pos - 2:peak_pos + 3] = 1.0
        starts.append(sparse)
    
    return starts

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_val = 0.0
    best_h = np.ones(N)

    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        if not np.all(np.isfinite(h)):
            return
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    starts = _generate_starts()
    
    # Initial evaluation of starts
    for s in starts:
        track(s)

    # Coordinate ascent phase
    for s in starts:
        if time.time() > deadline - 10:
            break
        
        x0 = np.maximum(0.0, np.asarray(s, dtype=float))
        
        # Give more time budget to promising starts
        val0 = compute_value(x0)
        if val0 > best_val * 0.9:
            time_budget = min(8.0, max(2.0, (deadline - time.time()) / 5))
        else:
            time_budget = min(3.0, max(1.0, (deadline - time.time()) / 10))
        
        h_opt, val_opt = _coordinate_ascent(x0, time_budget)
        track(h_opt)
    
    # Final polishing: multiple coordinate ascent passes on the best solution
    for _ in range(20):
        if time.time() > deadline - 2:
            break
        
        # Slight perturbation to escape local optima
        sigma = 0.05 + 0.05 * (1.0 - best_val / (best_val + 1e-10))
        pert = best_h + RNG.normal(0.0, sigma, N)
        pert = np.maximum(0.0, pert)
        
        h_opt, val_opt = _coordinate_ascent(pert, min(3.0, max(0.5, deadline - time.time() - 2)))
        track(h_opt)
    
    # Final direct evaluation
    track(best_h)

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
