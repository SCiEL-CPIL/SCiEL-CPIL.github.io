import itertools
import time
from math import log
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def fast_score_set(U):
    """Faster scoring using numpy for larger sets."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all pair sums (i <= j)
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    sum_count = len(np.unique(sums))
    
    # Compute all pair differences (i < j), including negatives and 0
    i_idx, j_idx = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx] - U_arr[i_idx]
    # Include 0 and negatives
    all_diffs = np.concatenate([diffs, -diffs, np.array([0])])
    diff_count = len(np.unique(all_diffs))
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_score = 1.0
    best_set = [0, 1]
    
    # Phase 1: Exhaustive search for small n (3-5) to find strong seeds
    max_val_limit_small = 50
    max_n_limit_small = 5
    
    for n in range(3, max_n_limit_small + 1):
        if time.time() > deadline:
            break
        range_vals = range(1, max_val_limit_small + 1)
        
        for comb in itertools.combinations(range_vals, n - 1):
            U = [0] + list(comb)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
        
        # Check deadline more frequently for larger n
        if n >= 4 and time.time() > deadline:
            break
    
    # Phase 2: Local search from the best seed
    # Try to improve by perturbing elements
    max_val_limit_large = 200
    max_iterations = 5000
    
    current_set = best_set[:]
    current_score = best_score
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        n = len(current_set)
        
        # Try different perturbation strategies
        strategy = iteration % 3
        
        if strategy == 0:
            # Replace one element with a random new one
            if n < 2:
                continue
            idx = (iteration + 1) % n
            if idx == 0:
                # Keep 0 fixed, modify another
                idx = 1 + (iteration % (n - 1))
            new_val = 1 + (iteration * 7 + idx * 13) % max_val_limit_large
            if new_val not in current_set:
                candidate = current_set[:]
                candidate[idx] = new_val
                candidate = sorted(set(candidate))
                if 0 not in candidate:
                    continue
                s = fast_score_set(candidate)
                if s > current_score:
                    current_score = s
                    current_set = candidate
                    if s > best_score:
                        best_score = s
                        best_set = candidate[:]
        
        elif strategy == 1:
            # Add a new element
            if n < 15:  # Don't make sets too large
                new_val = 1 + (iteration * 11 + 5) % max_val_limit_large
                if new_val not in current_set:
                    candidate = sorted(set(current_set + [new_val]))
                    if 0 in candidate:
                        s = fast_score_set(candidate)
                        if s > current_score:
                            current_score = s
                            current_set = candidate
                            if s > best_score:
                                best_score = s
                                best_set = candidate[:]
        
        else:
            # Remove an element (if n > 3)
            if n > 3:
                idx = 1 + (iteration % (n - 1))  # Don't remove 0
                candidate = sorted(set(current_set[:idx] + current_set[idx+1:]))
                if 0 in candidate and len(candidate) >= 2:
                    s = fast_score_set(candidate)
                    if s > current_score:
                        current_score = s
                        current_set = candidate
                        if s > best_score:
                            best_score = s
                            best_set = candidate[:]
    
    # Phase 3: Try structured sets and geometric progressions
    structured = [
        [0, 1, 3, 7],
        [0, 1, 3, 8],
        [0, 1, 3, 9],
        [0, 1, 4, 9],
        [0, 1, 4, 16],
        [0, 1, 2, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 8, 16],
        [0, 1, 4, 9, 16],
        [0, 1, 2, 4, 8],
        [0, 1, 2, 4, 9],
        [0, 1, 3, 6, 10],
        [0, 1, 3, 9, 27],
        [0, 1, 4, 16, 64],
        [0, 1, 2, 6, 18],
        [0, 1, 3, 12, 48],
    ]
    for s_set in structured:
        if time.time() > deadline:
            break
        s = fast_score_set(s_set)
        if s > best_score:
            best_score = s
            best_set = s_set[:]
    
    # Phase 4: Geometric-like sets with varying ratios
    for base in [2, 3, 4, 5]:
        if time.time() > deadline:
            break
        for length in range(3, 8):
            if time.time() > deadline:
                break
            U = [0]
            val = 1
            for i in range(length):
                U.append(val)
                val *= base
            if max(U) <= 500:
                s = fast_score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
    
    # Phase 5: Fine-tune the best set with small perturbations
    if best_score > 1.0:
        current_set = best_set[:]
        current_score = best_score
        fine_tune_iterations = 2000
        
        for iteration in range(fine_tune_iterations):
            if time.time() > deadline:
                break
            
            n = len(current_set)
            if n < 2:
                break
            
            # Try small changes: replace an element with nearby value
            idx = 1 + (iteration % (n - 1))  # Don't change 0
            old_val = current_set[idx]
            
            # Try values near old_val
            for delta in [-2, -1, 1, 2]:
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                if new_val in current_set:
                    continue
                
                candidate = current_set[:]
                candidate[idx] = new_val
                candidate = sorted(set(candidate))
                if 0 not in candidate or len(candidate) != n:
                    continue
                
                s = fast_score_set(candidate)
                if s > current_score:
                    current_score = s
                    current_set = candidate
                    if s > best_score:
                        best_score = s
                        best_set = candidate[:]
                    break  # Move to next iteration after first improvement
    
    return best_set

if __name__ == "__main__":
    start = time.time()
    U = construct()
    elapsed = time.time() - start
    print("Result:", U)
    print("Score:", score_set(U))
    print("Time: %.2f s" % elapsed)
