import numpy as np
import time
import random
import itertools

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def compute_ratio(h):
        """Compute ||h||_2^2 / (||h||_1 * ||h||_inf * ||h*h||_inf)"""
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        l_inf = np.max(h)
        if l1 < 1e-15 or l_inf < 1e-15:
            return 0.0
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv < 1e-15:
            return 0.0
        return (l2_sq) / (l1 * l_inf * max_conv)

    def compute_ratio_fast(h):
        """Faster version using direct computation for binary vectors"""
        ones = np.where(h > 1e-10)[0]
        if len(ones) == 0:
            return 0.0
        # For binary: ratio = 1 / max_conv
        # max_conv = max number of pairs (i,j) in ones such that i+j = k
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-15:
            return 0.0
        return 1.0 / max_conv

    # Key insight: For binary vectors, ratio = 1/max_autoconv_peak
    # So we want to MINIMIZE the maximum autoconvolution peak.
    # This is equivalent to finding a Sidon set (B2 set) where all pairwise sums are distinct.
    # Maximum Sidon set in [0, N-1] has size ~sqrt(N).
    # For N=50, we can have Sidon sets of size ~7-8.
    # But we can also use non-binary heights to improve the ratio.
    
    best_ratio = 0.0
    best_h = None
    
    # Strategy 1: Find good binary (Sidon-like) sets
    # Strategy 2: Refine with non-binary heights
    
    # Generate candidate binary sets
    candidates = []
    
    # Known good Sidon sets in [0, 49]
    # These are sets where all pairwise sums i+j (i<=j) are distinct
    sidon_sets = [
        [0, 1, 4, 10, 18, 23, 30],
        [0, 1, 4, 10, 18, 27, 33],
        [0, 2, 5, 11, 17, 26, 34],
        [0, 1, 5, 11, 17, 23, 31],
        [0, 2, 3, 10, 18, 25, 32],
        [0, 1, 3, 9, 17, 25, 32],
        [0, 2, 7, 12, 19, 26, 33],
        [0, 1, 4, 9, 16, 25, 34],
    ]
    
    for s in sidon_sets:
        h = np.zeros(N)
        for idx in s:
            if idx < N:
                h[idx] = 1.0
        candidates.append(h)
    
    # Generate random sparse sets
    random.seed(42)
    for size in range(5, 12):
        for _ in range(500):
            s = set(random.sample(range(N), size))
            h = np.zeros(N)
            for idx in s:
                h[idx] = 1.0
            candidates.append(h)
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline:
            break
        r = compute_ratio(h)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.zeros(N)
        best_h[0] = 1.0
        best_ratio = compute_ratio(best_h)
    
    # Local search on binary vectors
    current_h = best_h.copy()
    current_ratio = best_ratio
    T = 1.0
    T_min = 1e-4
    cooling = 0.9995
    max_iterations = 50000
    
    for i in range(max_iterations):
        if time.time() > deadline:
            break
        
        move_type = random.random()
        if move_type < 0.35:
            # Add
            zeros = np.where(current_h < 1e-10)[0]
            if len(zeros) == 0:
                continue
            idx = random.choice(zeros)
            new_h = current_h.copy()
            new_h[idx] = 1.0
        elif move_type < 0.65:
            # Remove
            ones = np.where(current_h > 1e-10)[0]
            if len(ones) <= 2:
                continue
            idx = random.choice(ones)
            new_h = current_h.copy()
            new_h[idx] = 0.0
        else:
            # Swap
            ones = np.where(current_h > 1e-10)[0]
            zeros = np.where(current_h < 1e-10)[0]
            if len(ones) == 0 or len(zeros) == 0:
                continue
            idx_out = random.choice(ones)
            idx_in = random.choice(zeros)
            new_h = current_h.copy()
            new_h[idx_out] = 0.0
            new_h[idx_in] = 1.0
        
        new_ratio = compute_ratio(new_h)
        
        if new_ratio > current_ratio:
            current_h = new_h
            current_ratio = new_ratio
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_h = current_h.copy()
        else:
            delta = current_ratio - new_ratio
            if delta < 1e-15:
                current_h = new_h
                current_ratio = new_ratio
            else:
                if random.random() < np.exp(-delta / T):
                    current_h = new_h
                    current_ratio = new_ratio
        
        T = max(T * cooling, T_min)
        
        if i % 1000 == 0 and i > 0:
            if random.random() < 0.15:
                current_h = best_h.copy()
                current_ratio = best_ratio
                T = 0.3
    
    # Now try to improve with non-binary heights
    # The key insight: for fixed support S, the optimal heights can be found
    # by optimizing over h_i > 0 for i in S
    
    if best_h is not None:
        # Get the support
        support = np.where(best_h > 1e-10)[0]
        
        if len(support) > 0:
            # Start with the current best and do coordinate descent/ascent
            h_opt = best_h.copy()
            
            # Try to optimize heights on the support
            # The ratio is scale invariant in a specific way:
            # If we scale all heights by c, ratio scales by 1/c^2
            # So we should normalize: set max(h) = 1
            # Then optimize the relative heights
            
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                improved = False
                for idx in support:
                    # Try different values for h[idx]
                    current_val = h_opt[idx]
                    best_val = current_val
                    best_r = compute_ratio(h_opt)
                    
                    # Try a grid of values
                    for val in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                        if abs(val - current_val) < 1e-6:
                            continue
                        h_test = h_opt.copy()
                        h_test[idx] = val
                        r_test = compute_ratio(h_test)
                        if r_test > best_r + 1e-10:
                            best_r = r_test
                            best_val = val
                    
                    if best_val != current_val:
                        h_opt[idx] = best_val
                        improved = True
                
                if not improved:
                    break
            
            final_ratio = compute_ratio(h_opt)
            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_h = h_opt
    
    # Final attempt: try random non-binary vectors
    for _ in range(200):
        if time.time() > deadline:
            break
        
        # Random vector with random support and random heights
        size = random.randint(5, 10)
        support = set(random.sample(range(N), size))
        h = np.zeros(N)
        for idx in support:
            h[idx] = random.uniform(0.3, 1.0)
        
        # Normalize so max = 1
        h_max = np.max(h)
        if h_max > 1e-10:
            h /= h_max
        
        r = compute_ratio(h)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.zeros(N)
        best_h[0] = 1.0
    
    return list(best_h)
