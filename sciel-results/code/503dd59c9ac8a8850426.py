import numpy as np
import itertools
import random

def score_set(U):
    U = list(set(U))
    if 0 not in U:
        return -1
    U.sort()
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Calculate sumset
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    
    sum_size = len(sumset)
    diff_size = len(diffset)
    
    if sum_size == 0:
        return -1
        
    ratio = diff_size / sum_size
    if ratio <= 0:
        return -1
        
    log_ratio = np.log2(ratio)
    log_max = np.log2(2 * max_u + 1)
    
    if log_max == 0:
        return -1
        
    return 1 + log_ratio / log_max

def construct():
    best_score = -1
    best_set = [0, 1, 2, 4]
    
    # 1. Analytical construction: Powers of 2
    # U = {0, 1, 2, 4, ..., 2^(k-1)}
    # This maximizes the difference set size relative to sum set size for a given max element.
    for k in range(2, 12):
        U = [0] + [2**i for i in range(1, k)]
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U[:]

    # 2. Local search around the best power-of-2 set
    # Try replacing some elements with nearby integers to see if we can improve the ratio
    base_set = best_set[:]
    base_score = best_score
    
    # We will perform a few rounds of local optimization
    # For each element (except 0), try changing it to a nearby value
    # We need to be careful not to create duplicates or remove 0
    
    current_set = base_set[:]
    current_score = base_score
    
    improved = True
    max_iterations = 100
    iter_count = 0
    
    while improved and iter_count < max_iterations:
        improved = False
        iter_count += 1
        # Try modifying each element in the set (except 0)
        for i in range(1, len(current_set)):
            original_val = current_set[i]
            # Try values in a small window around original_val
            # Also try some random perturbations
            candidates_to_try = set()
            
            # Nearby values
            for delta in range(-3, 4):
                new_val = original_val + delta
                if new_val > 0:
                    candidates_to_try.add(new_val)
            
            # Random perturbations
            for _ in range(5):
                new_val = random.randint(1, 2 * original_val + 5)
                candidates_to_try.add(new_val)
            
            for new_val in candidates_to_try:
                if new_val == original_val:
                    continue
                # Check for duplicates
                if new_val in current_set and new_val != original_val:
                    continue
                
                # Create new set
                new_set = current_set[:]
                new_set[i] = new_val
                new_set.sort()
                
                s = score_set(new_set)
                if s > current_score + 1e-9:
                    current_set = new_set
                    current_score = s
                    improved = True
                    # Update best global
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current_set[:]
                    break # Restart loop with the new set for this index
    
    # 3. Broader random search with larger sets, biased towards sparse structures
    # The score depends on log2(|U-U|/|U+U|) / log2(2*max(U)+1).
    # Sparse sets have large |U-U|/|U+U| ratios.
    # Powers of 2 are very sparse.
    # Let's try geometric progressions with other bases.
    
    for base in [1.5, 2, 3, 4, 5, 10]:
        for max_k in range(3, 10):
            U = [0]
            val = 1
            for i in range(max_k):
                U.append(int(round(val)))
                val *= base
            # Remove duplicates and ensure 0 is present
            U = sorted(list(set(U)))
            if 0 not in U:
                U.append(0)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]

    # 4. Specific known good sets or heuristic search
    # Try sets of the form {0, a, 2a, 4a, ...} (scaled powers of 2)
    for a in range(1, 10):
        for k in range(2, 8):
            U = [0] + [a * (2**i) for i in range(1, k)]
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]

    return best_set

# Self-test
if __name__ == "__main__":
    U = construct()
    print("Constructed Set:", U)
    print("Score:", score_set(U))
