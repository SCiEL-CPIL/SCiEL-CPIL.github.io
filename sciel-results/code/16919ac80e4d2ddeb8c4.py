import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    N = 4096
    thetas = 2 * np.pi * np.arange(N) / N
    z = np.exp(1j * thetas)
    
    # Precompute powers for all k up to 23
    # powers[k, j] = z_j^k
    powers = np.ones((24, N), dtype=complex)
    for k in range(1, 24):
        powers[k] = powers[k-1] * z
        
    def fast_eval(coeffs):
        # coeffs is a numpy array of shape (24,)
        # We want max_j |sum_k coeffs[k] * powers[k, j]|
        # This is equivalent to max_j |powers.T @ coeffs|
        vals = powers.T @ coeffs
        return np.max(np.abs(vals))
        
    # Initial candidate: Golay-Rudin-Shapiro truncated to 24
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:24], dtype=np.float64)
    current_coeffs = best_coeffs.copy()
    best_score = fast_eval(best_coeffs)
    current_score = best_score
    
    # Simulated Annealing Parameters
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Simulated Annealing
    while time.time() < deadline - 10.0:  # Leave 10s for final polish
        if T < T_min:
            break
            
        # Pick a random bit to flip
        i = rng.integers(0, 24)
        
        # Calculate change in score
        # new_score = fast_eval(new_coeffs)
        # To speed up, we can approximate or just do it. 
        # Since N=4096 and 24 coeffs, fast_eval is very fast (~10us).
        # 100k iterations * 10us = 1s. So we can do millions.
        
        new_coeffs = current_coeffs.copy()
        new_coeffs[i] *= -1
        new_score = fast_eval(new_coeffs)
        
        delta = new_score - current_score
        
        if delta < 0:
            # Accept
            current_coeffs = new_coeffs
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current_coeffs.copy()
        else:
            # Accept with probability exp(-delta / T)
            prob = np.exp(-delta / T)
            if rng.random() < prob:
                current_coeffs = new_coeffs
                current_score = new_score
        
        T *= cooling_rate
        
    # Phase 2: Local Search (Hill Climbing) from best_coeffs
    # This polishes the result to the local minimum.
    improved = True
    while improved:
        improved = False
        for i in range(24):
            new_coeffs = best_coeffs.copy()
            new_coeffs[i] *= -1
            new_score = fast_eval(new_coeffs)
            if new_score < best_score:
                best_coeffs = new_coeffs
                best_score = new_score
                improved = True
                
    # If time remains, do a few more random restarts with local search
    while time.time() < deadline:
        # Random start
        current_coeffs = rng.choice([-1, 1], size=24).astype(np.float64)
        current_score = fast_eval(current_coeffs)
        
        # Local search
        improved = True
        while improved:
            improved = False
            for i in range(24):
                new_coeffs = current_coeffs.copy()
                new_coeffs[i] *= -1
                new_score = fast_eval(new_coeffs)
                if new_score < current_score:
                    current_coeffs = new_coeffs
                    current_score = new_score
                    improved = True
                    
        if current_score < best_score:
            best_score = current_score
            best_coeffs = current_coeffs.copy()
            
        # Check if we hit the known optimum
        if abs(best_score / np.sqrt(24) - 1.2583758) < 1e-6:
            break

    return [int(c) for c in best_coeffs]
