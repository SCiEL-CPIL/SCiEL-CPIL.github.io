import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 20
    max_val = 10000
    
    # Precompute all pairs (i, j) with i <= j for n=20
    # There are n*(n+1)/2 = 210 pairs
    pairs_i = np.array([i for i in range(n) for j in range(i, n)], dtype=np.int32)
    pairs_j = np.array([j for i in range(n) for j in range(i, n)], dtype=np.int32)
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using NumPy vectorization."""
        A_arr = np.array(A, dtype=np.int64)
        sums = A_arr[pairs_i] + A_arr[pairs_j]
        prods = A_arr[pairs_i] * A_arr[pairs_j]
        sumset_size = len(np.unique(sums))
        prodset_size = len(np.unique(prods))
        return max(sumset_size, prodset_size)
    
    # Initialize with a mix of strategies to give SA a good starting point
    # Strategy 1: Geometric progression (good for products)
    # Strategy 2: Arithmetic progression (good for sums, but bad for products)
    # Strategy 3: Smooth numbers (good for both)
    
    def get_smooth_pool():
        pool = set()
        for a in range(0, 14):
            v2 = 1 << a
            if v2 > max_val: break
            for b in range(0, 9):
                v3 = v2 * (3 ** b)
                if v3 > max_val: break
                for c in range(0, 6):
                    v5 = v3 * (5 ** c)
                    if v5 > max_val: break
                    for d in range(0, 5):
                        v7 = v5 * (7 ** d)
                        if v7 > max_val: break
                        pool.add(v7)
        return sorted(pool)
    
    smooth_pool = get_smooth_pool()
    
    # Initial candidates
    # 1. Smallest smooth numbers
    init_smooth = smooth_pool[:n]
    
    # 2. Geometric-like spread
    init_geo = [int(max_val ** (i / (n - 1))) for i in range(n)]
    # Ensure distinctness and <= max_val
    init_geo = sorted(set(init_geo))
    while len(init_geo) < n:
        init_geo.append(init_geo[-1] + 1)
    init_geo = init_geo[:n]
    
    # 3. Random smooth
    init_rand_smooth = random.sample(smooth_pool, n)
    
    candidates = [init_smooth, init_geo, init_rand_smooth]
    
    best_set = None
    best_score = float('inf')
    
    for cand in candidates:
        score = evaluate(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]
            
    current_set = best_set[:]
    current_score = best_score
    
    # Simulated Annealing Parameters
    T = 50.0
    T_min = 0.1
    cooling = 0.9995
    
    # Track best
    best_set_sa = current_set[:]
    best_score_sa = current_score
    
    # Precompute indices for fast access? No, we swap values directly.
    
    # To speed up SA, we can evaluate the change in score more efficiently?
    # Full re-evaluation is ~2000 ops. 100k evals = 200M ops. ~2s.
    # We have 80s. So we can do ~3-4 million evals.
    # Let's aim for ~200k-500k evals to be safe and allow for overhead.
    
    max_evals = 300000
    eval_count = 0
    
    while T > T_min and time.time() < deadline - 5 and eval_count < max_evals:
        # Pick a random element to change
        i = random.randint(0, n - 1)
        old_val = current_set[i]
        
        # Generate a new value
        # Bias towards values close to current to ensure small steps?
        # Or random uniform?
        # Let's try a mix: 50% local perturbation, 50% global random
        if random.random() < 0.5:
            # Local perturbation: +/- delta
            delta = random.randint(1, 50)
            sign = random.choice([-1, 1])
            new_val = old_val + sign * delta
            if new_val < 1: new_val = 1
            if new_val > max_val: new_val = max_val
        else:
            new_val = random.randint(1, max_val)
            
        if new_val == old_val:
            continue
            
        # Check distinctness
        # If new_val is already in current_set (at index j != i), we have a collision.
        # In SA, we usually reject invalid states or handle them.
        # Let's just reject if it causes a duplicate.
        if new_val in current_set:
            continue
            
        # Apply change
        current_set[i] = new_val
        
        # Evaluate
        new_score = evaluate(current_set)
        eval_count += 1
        
        delta_score = new_score - current_score
        
        if delta_score < 0 or random.random() < math.exp(-delta_score / T):
            # Accept
            current_score = new_score
            if new_score < best_score_sa:
                best_score_sa = new_score
                best_set_sa = current_set[:]
        else:
            # Reject
            current_set[i] = old_val
            
        T *= cooling
        
    return sorted(best_set_sa)
