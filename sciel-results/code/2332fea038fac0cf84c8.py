import time
import random

def construct():
    n = 256
    
    # Precompute AP index lists
    ap_indices = []
    for d in range(1, n + 1):
        for a in range(1, n - d + 1):
            indices = []
            idx = a - 1
            while idx < n:
                indices.append(idx)
                idx += d
            ap_indices.append(indices)
            
    def get_max_discrepancy_limited(signs, limit):
        max_abs = 0
        for indices in ap_indices:
            s = 0
            for idx in indices:
                s += signs[idx]
                if s > limit:
                    return s
                if -s > limit:
                    return -s
                if s > max_abs:
                    max_abs = s
                elif -s > max_abs:
                    max_abs = -s
        return max_abs

    def get_full_discrepancy(signs):
        return get_max_discrepancy_limited(signs, n)

    # Initialize with Thue-Morse
    signs = []
    for i in range(1, n + 1):
        s2 = bin(i).count('1')
        signs.append(1 if s2 % 2 == 0 else -1)
        
    current_max = get_full_discrepancy(signs)
    best_signs = signs[:]
    best_max = current_max
    
    start_time = time.time()
    time_limit = 55.0
    
    while (time.time() - start_time) < time_limit:
        # Local Search
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            indices = list(range(n))
            random.shuffle(indices)
            for i in indices:
                # Flip
                signs[i] *= -1
                # Check if new max is strictly less than current_max
                # Limit is current_max - 1
                new_max = get_max_discrepancy_limited(signs, current_max - 1)
                if new_max < current_max:
                    current_max = new_max
                    improved = True
                    if current_max < best_max:
                        best_max = current_max
                        best_signs = signs[:]
                else:
                    # Revert
                    signs[i] *= -1
        
        if (time.time() - start_time) >= time_limit:
            break
            
        # Perturbation
        K = random.randint(5, 15)
        flip_indices = random.sample(range(n), K)
        for i in flip_indices:
            signs[i] *= -1
            
        current_max = get_full_discrepancy(signs)
        
    return best_signs
