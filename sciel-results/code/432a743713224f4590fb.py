import numpy as np
import time
import random

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    # Target: Minimize max(|P(z)|) / sqrt(24)
    # Optimal target raw value is ~1.2583758
    
    n_coeffs = 24
    fft_size = 4096  # Dense sampling
    
    def score(coeffs):
        # coeffs is a list of +1/-1
        # Use FFT to evaluate on unit circle
        # P(z) = sum c_k z^k
        # fft gives sum c_k w^{-jk} where w = exp(i 2pi / N)
        # This is P(w^{-j}). Magnitude is same as P(w^j).
        f = np.fft.fft(coeffs, n=fft_size)
        max_val = np.max(np.abs(f))
        return max_val / np.sqrt(n_coeffs)
    
    def hill_climb(initial_coeffs):
        current = initial_coeffs.copy()
        current_score = score(current)
        improved = True
        last_improvement_time = time.time()
        
        while improved and (time.time() - start_time < time_limit):
            improved = False
            indices = list(range(n_coeffs))
            random.shuffle(indices)
            
            for i in indices:
                # Flip coefficient i
                old_val = current[i]
                current[i] = -old_val
                new_score = score(current)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    last_improvement_time = time.time()
                else:
                    current[i] = old_val
                
                # Check time limit inside loop
                if time.time() - start_time > time_limit:
                    return current, current_score
        
        return current, current_score

    best_coeffs = None
    best_score = float('inf')
    
    # Initial candidates
    # 1. All ones
    # 2. Golay-Rudin-Shapiro truncated (from seed)
    # 3. Random
    
    gr_p, gr_q = [1], [1]
    while len(gr_p) < n_coeffs:
        gr_p, gr_q = gr_p + gr_q, gr_p + [-c for c in gr_q]
    gr_seed = gr_p[:n_coeffs]
    
    candidates = [
        [1] * n_coeffs,
        gr_seed,
        [random.choice([1, -1]) for _ in range(n_coeffs)]
    ]
    
    # Run hill climbing on initial candidates
    for cand in candidates:
        if time.time() - start_time > time_limit:
            break
        coeffs, sc = hill_climb(cand)
        if sc < best_score:
            best_score = sc
            best_coeffs = coeffs
    
    # Random restarts
    while time.time() - start_time < time_limit:
        random_seed = [random.choice([1, -1]) for _ in range(n_coeffs)]
        coeffs, sc = hill_climb(random_seed)
        if sc < best_score:
            best_score = sc
            best_coeffs = coeffs

    if best_coeffs is None:
        return gr_seed
        
    return best_coeffs
