import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    target_low = 128
    target_high = 256
    num_targets = target_high - target_low + 1 # 129
    
    def score_numpy(A_arr):
        k = len(A_arr)
        if k == 0:
            return 100000
        # Compute all pair sums (i <= j)
        sums = np.add.outer(A_arr, A_arr)
        upper_sums = sums[np.triu_indices(k, k=0)]
        
        # Filter to target range
        valid_mask = (upper_sums >= target_low) & (upper_sums <= target_high)
        shifted_sums = upper_sums[valid_mask] - target_low
        
        if len(shifted_sums) == 0:
            mults = np.zeros(num_targets, dtype=int)
        else:
            mults = np.bincount(shifted_sums, minlength=num_targets)
            
        max_mult = np.max(mults)
        uncovered = np.sum(mults == 0)
        
        if uncovered > 0:
            # Heavy penalty for not covering
            return 10000 + max_mult + uncovered * 100
        return max_mult

    best_A = None
    best_score = float('inf')
    
    # Generate diverse initializations
    inits = []
    sizes = [16, 17, 18, 20, 22, 25, 30]
    
    for k in sizes:
        # 1. Uniform in [64, 128]
        for _ in range(3):
            try:
                A_init = np.array(random.sample(range(64, 129), k))
                inits.append(A_init)
            except ValueError:
                pass
                
        # 2. Evenly spaced in [64, 128]
        if k > 1:
            step = (128 - 64) / (k - 1)
            A_init = np.array([int(64 + i * step) for i in range(k)])
            inits.append(A_init)
            
        # 3. Random in [0, 128]
        for _ in range(3):
            try:
                A_init = np.array(random.sample(range(0, 129), k))
                inits.append(A_init)
            except ValueError:
                pass
                
        # 4. Clustered: half in [0, 64], half in [64, 128]
        k1 = k // 2
        k2 = k - k1
        if k1 > 0 and k2 > 0:
             try:
                A1 = random.sample(range(0, 65), k1)
                A2 = random.sample(range(64, 129), k2)
                A_init = np.array(A1 + A2)
                inits.append(A_init)
             except ValueError:
                pass

    # Shuffle inits to randomize order
    random.shuffle(inits)

    # Local Search
    for A_start in inits:
        if time.time() > deadline - 10.0:
            break
            
        A = A_start.copy()
        current_score = score_numpy(A)
        
        if current_score < best_score:
            best_score = current_score
            best_A = A.copy()
            
        no_improve_count = 0
        last_improve_time = time.time()
        
        while time.time() < deadline - 5.0:
            # Random Swap
            idx = random.randint(0, len(A) - 1)
            a = A[idx]
            b = random.randint(0, 128)
            
            if b == a:
                continue
                
            A_new = A.copy()
            A_new[idx] = b
            
            new_score = score_numpy(A_new)
            
            if new_score < current_score:
                A = A_new
                current_score = new_score
                no_improve_count = 0
                last_improve_time = time.time()
                
                if current_score < best_score:
                    best_score = current_score
                    best_A = A.copy()
            else:
                no_improve_count += 1
                
            # If stuck, break early to try next init
            if no_improve_count > 500:
                if time.time() - last_improve_time > 0.5:
                    break

    if best_A is None:
        # Fallback to seed
        return list(range(64, 129))
        
    return list(best_A)

