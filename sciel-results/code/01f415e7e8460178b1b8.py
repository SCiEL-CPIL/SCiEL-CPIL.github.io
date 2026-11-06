"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute all arithmetic progressions with length >= 2
    # For each AP, store the indices (0-based)
    aps = []
    for d in range(1, N // 2 + 1):
        for start in range(N - d):
            # AP: start, start+d, start+2d, ...
            # Length = floor((N-1 - start) / d) + 1
            # Only include if length >= 2
            if start + d < N:
                indices = list(range(start, N, d))
                if len(indices) >= 2:
                    aps.append(indices)
    
    num_aps = len(aps)
    
    # Convert APs to list of numpy arrays for faster indexing
    aps_np = [np.array(ap, dtype=np.int32) for ap in aps]
    
    def compute_max_abs_sum(signs):
        """Compute maximum absolute sum over all APs."""
        max_abs = 1  # Singletons always have abs sum 1
        for ap in aps_np:
            s = np.sum(signs[ap])
            abs_s = abs(s)
            if abs_s > max_abs:
                max_abs = abs_s
        return max_abs
    
    # Simulated Annealing
    rng = random.Random(42)
    
    # Initial solution: random signs
    signs = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
    current_cost = compute_max_abs_sum(signs)
    best_signs = signs.copy()
    best_cost = current_cost
    
    T = 10.0  # Initial temperature
    cooling_rate = 0.999
    
    while time.time() < deadline:
        if T < 0.01:
            break
            
        # Pick a random index to flip
        idx = rng.randrange(N)
        old_sign = signs[idx]
        signs[idx] = -old_sign
        
        new_cost = compute_max_abs_sum(signs)
        
        if new_cost <= current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_signs = signs.copy()
        else:
            # Accept with probability exp(-(new_cost - current_cost) / T)
            delta = new_cost - current_cost
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_cost = new_cost
                else:
                    signs[idx] = old_sign
        
        T *= cooling_rate
    
    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
