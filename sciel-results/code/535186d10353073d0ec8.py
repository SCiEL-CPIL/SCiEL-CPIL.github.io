"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time
import random

def construct():
    n = 128
    target_low = 128
    target_high = 256
    size = n + 1  # 129
    
    def objective(x):
        # x is a numpy array of shape (129,)
        # Threshold to binary
        b = (x > 0.5).astype(np.int8)
        
        # Convolution
        c = np.convolve(b, b)
        
        # Extract relevant sums
        # c[s] is the count for sum s
        # We need sums 128 to 256
        c_relevant = c[128:257]
        
        # Compute d_s: 1 if s is even and b[s//2] == 1, else 0
        s_vals = np.arange(128, 257)
        is_even = (s_vals % 2 == 0)
        mid_indices = s_vals // 2
        d_vals = np.zeros_like(c_relevant, dtype=np.int8)
        d_vals[is_even] = b[mid_indices[is_even]]
        
        # Count unordered pairs
        counts = (c_relevant + d_vals) / 2.0
        
        # Check coverage
        uncovered = np.sum(counts == 0)
        
        if uncovered > 0:
            return 1000.0 + uncovered
            
        max_count = np.max(counts)
        return max_count

    start_time = time.time()
    
    # Initial guess: A = [64, 128]
    init_b = np.zeros(129, dtype=np.int8)
    init_b[64:129] = 1
    
    try:
        res = differential_evolution(
            objective, 
            bounds=[(0, 1)] * 129, 
            popsize=20, 
            maxiter=100, 
            tol=1e-4, 
            seed=42,
            polish=False,
            init='latinhypercube'
        )
    except Exception:
        res = None

    best_x = res.x if res is not None else init_b.astype(float)
    
    # Refine with local search
    current_A = set(np.where(best_x > 0.5)[0].tolist())
    
    def get_max_count_and_uncovered(A_set):
        max_count = 0
        uncovered = 0
        for s in range(target_low, target_high + 1):
            count = 0
            a_min = max(0, s - 128)
            a_max = min(128, s // 2)
            for a in range(a_min, a_max + 1):
                if a in A_set and (s - a) in A_set:
                    count += 1
            if count == 0:
                uncovered += 1
            else:
                if count > max_count:
                    max_count = count
        return max_count, uncovered

    current_max, current_uncovered = get_max_count_and_uncovered(current_A)
    
    if current_uncovered > 0:
        current_A = set(range(64, 129))
        current_max, current_uncovered = get_max_count_and_uncovered(current_A)
        
    best_max = current_max
    best_A = set(current_A)
    
    time_limit = 55
    
    for _ in range(2000):
        if time.time() - start_time > time_limit:
            break
            
        # Try removing an element
        if len(current_A) > 1:
            elem = random.choice(list(current_A))
            new_A = current_A - {elem}
            new_max, new_uncovered = get_max_count_and_uncovered(new_A)
            if new_uncovered == 0 and new_max <= current_max:
                current_A = new_A
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_A = set(current_A)
                    
        # Try swapping
        candidates = [i for i in range(129) if i not in current_A]
        if candidates:
            elem = random.choice(list(current_A))
            add_elem = random.choice(candidates)
            new_A = current_A - {elem} | {add_elem}
            new_max, new_uncovered = get_max_count_and_uncovered(new_A)
            if new_uncovered == 0 and new_max < current_max:
                current_A = new_A
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_A = set(current_A)

    return list(best_A)
# EVOLVE-BLOCK-END
