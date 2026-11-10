"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _precompute_aps(n):
    """Precompute all arithmetic progression index lists for [0, n)."""
    aps = []
    for d in range(1, n + 1):
        for s in range(0, n):
            # AP: s, s+d, s+2d, ... < n
            idx = list(range(s, n, d))
            if len(idx) >= 1:
                aps.append(idx)
    return aps

def _max_abs_sum_with_early_exit(signs, aps, threshold):
    """Compute max abs AP sum, but return early if we exceed threshold."""
    best = 1  # Singletons guarantee at least 1
    for ap in aps:
        total = 0
        for i in ap:
            total += signs[i]
            if abs(total) > threshold:
                return abs(total)  # Early exit
        if abs(total) > best:
            best = abs(total)
            if best > threshold:
                return best
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 256
    rng = random.Random(12345)
    
    # Precompute all APs once
    aps = _precompute_aps(n)
    
    # Sort APs by length descending - longer APs more likely to have large sums
    # Actually, let's keep them in a useful order. Let's sort by length desc to find large sums faster.
    aps_sorted = sorted(aps, key=len, reverse=True)
    
    def evaluate(signs):
        return _max_abs_sum_with_early_exit(signs, aps_sorted, 10**9)
    
    def evaluate_limited(signs, limit):
        return _max_abs_sum_with_early_exit(signs, aps_sorted, limit)
    
    # Multi-start simulated annealing
    best_signs = None
    best_score = 10**9
    
    while time.time() < deadline:
        # Initialize randomly
        signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
        current_score = evaluate(signs)
        
        if current_score < best_score:
            best_score = current_score
            best_signs = signs.copy()
        
        # Simulated annealing
        T = 2.0
        T_min = 0.05
        cooling = 0.9995
        local_best = signs.copy()
        local_best_score = current_score
        
        stall_count = 0
        max_stall = 2000
        
        while time.time() < deadline:
            # Pick random position to flip
            i = rng.randint(0, n - 1)
            signs[i] *= -1
            
            # Evaluate with early exit using current score as threshold
            new_score = evaluate_limited(signs, current_score)
            delta = new_score - current_score
            
            if delta <= 0:
                current_score = new_score
                stall_count = 0
                if new_score < local_best_score:
                    local_best_score = new_score
                    local_best = signs.copy()
                if new_score < best_score:
                    best_score = new_score
                    best_signs = signs.copy()
            else:
                # Accept with probability exp(-delta / T)
                if T > T_min:
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        current_score = new_score
                    else:
                        # Reject: revert
                        signs[i] *= -1
                else:
                    signs[i] *= -1
                stall_count += 1
            
            T *= cooling
            
            if T < T_min:
                # Reheat and possibly restart
                T = 2.0
                if stall_count > max_stall:
                    break
                if rng.random() < 0.3:
                    signs[:] = local_best
                    current_score = local_best_score
                    stall_count = 0
        
        # Check if we should continue with another restart
        if time.time() >= deadline - 5.0:
            break
    
    # Return as list of Python integers
    return best_signs.tolist()
# EVOLVE-BLOCK-END
