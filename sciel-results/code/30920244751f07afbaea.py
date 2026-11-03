import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = np.convolve(h, h, 'full')
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _generate_candidate(rng, strategy=0):
    """
    Generate a diverse candidate vector.
    """
    h = np.zeros(N)
    
    if strategy == 0:
        h = rng.standard_normal(N)
    elif strategy == 1:
        h = rng.choice([1.0, -1.0], N)
    elif strategy == 2:
        mask = rng.random(N) < 0.2
        h[mask] = rng.choice([1.0, -1.0], mask.sum())
    elif strategy == 3:
        num_blocks = 10
        block_size = N // num_blocks
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size if i < num_blocks - 1 else N
            sign = rng.choice([1.0, -1.0])
            h[start:end] = sign
    elif strategy == 4:
        noise = rng.standard_normal(N)
        H = np.fft.rfft(noise)
        cutoff = 10
        H[cutoff:] = 0
        h = np.fft.irfft(H, N)
        if np.sum(np.abs(h)) < 1e-6:
            h = rng.standard_normal(N)
    elif strategy == 5:
        x = np.arange(N)
        env = np.cos(np.pi * x / N) ** 2
        signs = rng.choice([1.0, -1.0], N)
        h = env * signs
    elif strategy == 6:
        # Cosine wave with random phase
        freq = rng.uniform(1, 5)
        phase = rng.uniform(0, 2*np.pi)
        h = np.cos(2*np.pi*freq*np.arange(N)/N + phase)
    elif strategy == 7:
        # Two-tone signal
        f1 = rng.uniform(1, 3)
        f2 = rng.uniform(3, 7)
        p1 = rng.uniform(0, 2*np.pi)
        p2 = rng.uniform(0, 2*np.pi)
        h = np.cos(2*np.pi*f1*np.arange(N)/N + p1) + np.cos(2*np.pi*f2*np.arange(N)/N + p2)
    else:
        h = rng.standard_normal(N)
        
    return h

def _simulated_annealing(h, deadline, rng, t_start_temp=10.0, t_end_temp=0.01, alpha=0.999):
    """
    Simulated annealing with block-wise and individual perturbations.
    """
    h = h.copy()
    best_h = h.copy()
    best_score = _score(h)
    current_score = best_score
    temp = t_start_temp
    
    while time.time() < deadline and temp > t_end_temp:
        temp *= alpha
        
        # Choose perturbation type
        if rng.random() < 0.7:
            # Block-wise flip
            block_size = rng.integers(5, 50)
            start = rng.integers(0, N - block_size + 1)
            h_new = h.copy()
            h_new[start:start+block_size] *= -1
        elif rng.random() < 0.5:
            # Individual flip
            idx = rng.integers(0, N)
            h_new = h.copy()
            h_new[idx] *= -1
        else:
            # Gaussian perturbation
            h_new = h + 0.1 * rng.standard_normal(N)
        
        # Normalize to keep sum constant
        s_new = np.sum(h_new)
        if abs(s_new) > 1e-9:
            h_new = h_new / s_new * np.sum(h)
        
        new_score = _score(h_new)
        
        # Acceptance criterion
        if new_score < current_score:
            h = h_new
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h_new.copy()
        else:
            # Accept with probability exp(-delta/temp)
            delta = new_score - current_score
            if delta < 100 * temp:  # Avoid overflow
                prob = np.exp(-delta / temp)
                if rng.random() < prob:
                    h = h_new
                    current_score = new_score
    
    return best_h, best_score

def _local_search_flip(h, deadline, rng):
    """
    Greedy local search with sign flipping.
    """
    h = h.copy()
    best_score = _score(h)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for j in range(N):
            if time.time() > deadline:
                break
            orig = h[j]
            h[j] = -orig
            new_val = _score(h)
            if new_val < best_score:
                best_score = new_val
                improved = True
            else:
                h[j] = orig
        
        # If no improvement, try random flips
        if not improved:
            for _ in range(20):
                if time.time() > deadline:
                    break
                j = rng.integers(0, N)
                orig = h[j]
                h[j] = -orig
                new_val = _score(h)
                if new_val < best_score:
                    best_score = new_val
                    improved = True
                else:
                    h[j] = orig
    
    return h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin for 150s limit
    
    best_score = np.inf
    best_h = np.ones(N)
    
    rng = np.random.default_rng(42)
    
    strategies = [0, 1, 2, 3, 4, 5, 6, 7]
    
    # 1. Initial Phase: Generate and screen candidates
    num_initial_candidates = 80
    
    candidates = []
    for i in range(num_initial_candidates):
        if time.time() > deadline - 60:
            break
        strat = strategies[i % len(strategies)]
        h = _generate_candidate(rng, strat)
        s = np.sum(h)
        if abs(s) < 1e-9:
            continue
        h = h / s
        val = _score(h)
        candidates.append((val, h))
        
    # Sort by initial score
    candidates.sort(key=lambda x: x[0])
    
    # Refine the top K candidates using simulated annealing
    K = 15
    for i, (init_val, h_init) in enumerate(candidates[:K]):
        if time.time() > deadline - 30:
            break
        
        local_deadline = min(deadline - 10.0, time.time() + 4.0)
        
        h_opt, score_opt = _simulated_annealing(h_init, local_deadline, rng)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt
    
    # 2. Final Phase: Intensive polishing with local search
    if time.time() < deadline - 10.0 and best_score < np.inf:
        final_deadline = deadline - 2.0
        h_polished, score_polished = _local_search_flip(best_h, final_deadline, rng)
        
        if score_polished < best_score:
            best_score = score_polished
            best_h = h_polished
    
    # 3. One more SA pass if time permits
    if time.time() < deadline - 5.0 and best_score < np.inf:
        sa_deadline = deadline
        h_sa, score_sa = _simulated_annealing(best_h, sa_deadline, rng, t_start_temp=5.0, t_end_temp=0.005)
        
        if score_sa < best_score:
            best_score = score_sa
            best_h = h_sa
    
    # Ensure valid return
    if not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
