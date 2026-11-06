import itertools
import random
import time

def is_sidon(subset):
    """Check if a subset has all unordered pair sums unique."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    """Compute the size of the largest Sidon subset of S."""
    n = len(S)
    max_size = 0
    # Iterate through all subsets using bitmask
    # 2^12 = 4096 subsets
    for mask in range(1, 1 << n):
        # Count bits
        size = bin(mask).count('1')
        if size <= max_size:
            continue
        # Extract subset
        subset = [S[i] for i in range(n) if (mask >> i) & 1]
        if is_sidon(subset):
            max_size = size
    return max_size

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin
    
    # Candidate sets
    candidates = []
    
    # 1. Contiguous low
    candidates.append(list(range(12)))
    # 2. Contiguous high
    candidates.append(list(range(117, 129)))
    # 3. Arithmetic progression
    candidates.append(list(range(0, 128, 11)))
    # 4. Seed
    candidates.append([0, 22, 1, 21, 2, 20, 3, 19, 4, 18, 5, 17])
    # 5. Pairs summing to 64
    pairs = [(i, 64-i) for i in range(6)]
    cand5 = []
    for p in pairs:
        cand5.extend(p)
    candidates.append(cand5)
    
    # 6. Random sets
    random.seed(42)
    for _ in range(50):
        cand = random.sample(range(129), 12)
        candidates.append(cand)
        
    best_set = None
    best_score = float('inf')
    
    for cand in candidates:
        if time.time() > deadline:
            break
        score = max_sidon_size(cand)
        if score < best_score:
            best_score = score
            best_set = cand
            
    # If time allows, do a local search around the best set?
    # Given 12 elements, local search might not be necessary if random search found a good one.
    # But we can try swapping elements in the best set to see if we can reduce the score further.
    
    if best_set is None:
        return list(range(12))
        
    # Local refinement: try replacing one element with another to see if score drops
    # This is expensive, so only do it if we have time.
    if time.time() < deadline - 5:
        improved = True
        current_best = list(best_set)
        current_score = best_score
        
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                for val in range(129):
                    if val in current_best:
                        continue
                    # Swap
                    old_val = current_best[i]
                    current_best[i] = val
                    new_set = current_best[:]
                    new_score = max_sidon_size(new_set)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        best_set = new_set
                        best_score = new_score
                    else:
                        current_best[i] = old_val
                        # If score is same, maybe keep? No, we want min.
                        # If score is same, swapping might lead to better later?
                        # For now, only accept strict improvement.
                        
    return best_set
