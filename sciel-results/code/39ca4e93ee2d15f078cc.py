import numpy as np
import random
import time

N = 400

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_x = None
    best_peak = float('inf')
    
    # Range of M to explore
    M_values = [15, 20, 25, 30, 35]
    
    for M in M_values:
        # Number of restarts
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
                
            support = generate_support_randomized(M, N)
            if support is None:
                continue
                
            signs, peak = optimize_signs(support, M)
            
            if peak < best_peak:
                best_peak = peak
                best_x = create_vector(support, signs, M)
                
    if best_x is None:
        # Fallback
        M = 20
        idx = np.random.choice(N, M, replace=False)
        x = np.zeros(N)
        x[idx] = 1.0 / np.sqrt(M)
        return x
        
    return best_x

def generate_support_randomized(M, N):
    support = [0]
    diff_counts = {}
    
    for _ in range(M - 1):
        best_candidate = None
        best_max_diff = float('inf')
        
        candidates = np.random.choice(N, size=min(N, 200), replace=False)
        
        for cand in candidates:
            if cand in support:
                continue
                
            max_new = 0
            for s in support:
                d1 = cand - s
                d2 = s - cand
                
                c1 = diff_counts.get(d1, 0) + 1
                c2 = diff_counts.get(d2, 0) + 1
                
                if c1 > max_new: max_new = c1
                if c2 > max_new: max_new = c2
            
            current_max = max(diff_counts.values()) if diff_counts else 0
            result_max = max(current_max, max_new)
            
            if result_max < best_max_diff:
                best_max_diff = result_max
                best_candidate = cand
                if best_max_diff == 1:
                    break
                    
        if best_candidate is None:
            return None
            
        support.append(best_candidate)
        
        for s in support[:-1]:
            d1 = best_candidate - s
            d2 = s - best_candidate
            diff_counts[d1] = diff_counts.get(d1, 0) + 1
            diff_counts[d2] = diff_counts.get(d2, 0) + 1
            
    return support

def calculate_peak(support, signs, M):
    sums = {}
    for i in range(M):
        for j in range(M):
            k = support[i] + support[j]
            val = signs[i] * signs[j]
            if k in sums:
                sums[k] += val
            else:
                sums[k] = val
                
    if not sums:
        return 0.0
        
    max_val = 0.0
    for v in sums.values():
        abs_v = abs(v)
        if abs_v > max_val:
            max_val = abs_v
            
    return max_val / M

def optimize_signs(support, M):
    signs = np.ones(M)
    current_peak = calculate_peak(support, signs, M)
    
    improved = True
    while improved:
        improved = False
        for i in range(M):
            signs[i] *= -1
            new_peak = calculate_peak(support, signs, M)
            if new_peak < current_peak - 1e-12:
                current_peak = new_peak
                improved = True
            else:
                signs[i] *= -1
                
    return signs, current_peak

def create_vector(support, signs, M):
    x = np.zeros(N)
    norm = 1.0 / np.sqrt(M)
    for i, idx in enumerate(support):
        x[idx] = signs[i] * norm
    return x
