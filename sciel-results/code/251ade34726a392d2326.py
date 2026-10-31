"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import numpy as np
import time
import math
import random
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _fix_sum(h, target=None):
    """Adjust h to have exact sum = target (default N/2) while keeping values in [0,1]."""
    if target is None:
        target = N / 2.0
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if abs(s - target) < 1e-14:
        return h
    # Iterative projection
    for _ in range(50):
        diff = target - h.sum()
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-12
        else:
            mask = h > 1e-12
        if not mask.any():
            break
        step = diff / mask.sum()
        h[mask] = np.clip(h[mask] + step, 0.0, 1.0)
    return h

def _make_h_from_blocks(block_values, block_lengths):
    """Create h array from repeating block pattern, then fix sum."""
    h = np.zeros(N)
    pos = 0
    for val, length in zip(block_values, block_lengths):
        for i in range(length):
            if pos + i < N:
                h[pos + i] = val
        pos += length
    # Fill remaining
    if pos < N:
        h[pos:] = block_values[0]
    return _fix_sum(h)

def _optimize_direct_limited(h0, max_time=30.0):
    """Local optimization using L-BFGS-B with projected gradient idea."""
    start_time = time.time()
    
    def objective_and_grad(params):
        h = np.clip(params, 0.0, 1.0)
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        # Use smooth max
        alpha = 200.0
        x_max = np.max(overlap)
        smax = x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (overlap - x_max))))
        # Penalty for sum constraint
        sum_penalty = 100.0 * (h.sum() - N/2.0)**2
        val = smax + sum_penalty
        
        # Gradient
        exps = np.exp(alpha * (overlap - x_max))
        weights = exps / np.sum(exps)
        
        grad = np.zeros(N)
        for i in range(N - 1):
            if weights[i] > 1e-15:
                grad[i] += weights[i] * (1.0 - h[i+1])
                grad[i+1] += weights[i] * (-h[i])
        if weights[N-1] > 1e-15:
            grad[N-1] += weights[N-1]
        
        # Sum penalty gradient
        grad += 200.0 * (h.sum() - N/2.0)
        
        return val, grad
    
    try:
        result = minimize(
            objective_and_grad,
            h0,
            method='L-BFGS-B',
            bounds=[(0.0, 1.0)] * N,
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-10, 'disp': False}
        )
        h_opt = _fix_sum(result.x)
        return h_opt, _score(h_opt)
    except Exception:
        return h0, _score(h0)
    
    return h0, _score(h0)

def _simulated_annealing_block(n_blocks, max_iter=5000, T0=0.1, T_end=0.001):
    """Simulated annealing over block structure."""
    best_h = None
    best_score = float('inf')
    
    # Initialize with random block structure
    block_lengths = [N // n_blocks] * n_blocks
    block_lengths[-1] += N - sum(block_lengths)
    block_values = np.random.uniform(0.2, 0.8, size=n_blocks)
    
    h = _make_h_from_blocks(block_values, block_lengths)
    current_score = _score(h)
    best_h = h.copy()
    best_score = current_score
    
    T = T0
    cooling = (T_end / T0) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        T *= cooling
        
        # Perturb: change one block value
        idx = random.randint(0, n_blocks - 1)
        old_val = block_values[idx].copy()
        block_values[idx] = np.clip(block_values[idx] + random.gauss(0, 0.05), 0.0, 1.0)
        
        h_new = _make_h_from_blocks(block_values, block_lengths)
        new_score = _score(h_new)
        
        if new_score < current_score or random.random() < math.exp(-(new_score - current_score) / max(T, 1e-10)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h_new.copy()
        else:
            block_values[idx] = old_val
        
        # Occasionally change block lengths
        if it % 100 == 0 and n_blocks > 1:
            i, j = random.sample(range(n_blocks), 2)
            delta = random.randint(-5, 5)
            if block_lengths[j] + delta > 2:
                block_lengths[i] += delta
                block_lengths[j] -= delta
    
    return best_h, best_score

def _local_search_polish(h, max_iter=2000):
    """Coordinate descent local search."""
    h = _fix_sum(h.copy())
    current_score = _score(h)
    
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for step in step_sizes:
        improved = True
        while improved:
            improved = False
            for i in range(N):
                for delta in [step, -step]:
                    h_test = h.copy()
                    h_test[i] = np.clip(h_test[i] + delta, 0.0, 1.0)
                    h_test = _fix_sum(h_test)
                    new_score = _score(h_test)
                    if new_score < current_score - 1e-12:
                        h = h_test
                        current_score = new_score
                        improved = True
                        break
        if not improved:
            break
    
    return h, current_score

def _generate_initial_guesses():
    """Generate diverse initial guesses."""
    guesses = []
    
    # 2-block
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h2 = np.zeros(N)
    half = N // 2
    h2[:half] = A
    h2[half:] = B
    guesses.append(_fix_sum(h2))
    
    # 3-block
    for n_blocks in [3, 4, 5]:
        for _ in range(5):
            block_values = np.random.uniform(0.2, 0.8, size=n_blocks)
            block_lengths = [N // n_blocks] * n_blocks
            block_lengths[-1] += N - sum(block_lengths)
            h = _make_h_from_blocks(block_values, block_lengths)
            guesses.append(h)
    
    # Sinusoidal
    for freq in [1, 2, 3]:
        for phase in [0, np.pi/2, np.pi]:
            t = np.linspace(0, 2*np.pi*freq, N, endpoint=False)
            h = 0.5 + 0.3 * np.sin(t + phase)
            guesses.append(_fix_sum(h))
    
    # Random
    for _ in range(10):
        h = np.random.uniform(0.0, 1.0, size=N)
        guesses.append(_fix_sum(h))
    
    return guesses

def construct():
    deadline = time.time() + 140.0
    
    # Generate initial guesses and pick best
    guesses = _generate_initial_guesses()
    h_best = None
    score_best = float('inf')
    for h in guesses:
        s = _score(h)
        if s < score_best:
            score_best = s
            h_best = h.copy()
    
    # Phase 1: Simulated annealing with different block counts
    for n_blocks in [2, 3, 4, 5, 6]:
        if time.time() > deadline - 40.0:
            break
        h_sa, score_sa = _simulated_annealing_block(n_blocks, max_iter=3000)
        if score_sa < score_best:
            score_best = score_sa
            h_best = h_sa.copy()
    
    # Phase 2: Local polish with L-BFGS-B
    if time.time() < deadline - 30.0:
        h_opt, score_opt = _optimize_direct_limited(h_best, max_time=25.0)
        if score_opt < score_best:
            score_best = score_opt
            h_best = h_opt.copy()
    
    # Phase 3: Coordinate descent polish
    if time.time() < deadline - 10.0:
        h_cd, score_cd = _local_search_polish(h_best, max_iter=2000)
        if score_cd < score_best:
            score_best = score_cd
            h_best = h_cd.copy()
    
    # Final: one more L-BFGS-B polish
    if time.time() < deadline - 5.0:
        h_final_opt, score_final = _optimize_direct_limited(h_best, max_time=3.0)
        if score_final < score_best:
            h_best = h_final_opt.copy()
            score_best = score_final
    
    # Ensure validity
    h_best = _fix_sum(h_best)
    h_best = np.clip(h_best, 0.0, 1.0)
    
    # Verify sum is close enough
    if abs(h_best.sum() - N/2.0) > 1e-6:
        h_best = _fix_sum(h_best)
    
    return h_best
