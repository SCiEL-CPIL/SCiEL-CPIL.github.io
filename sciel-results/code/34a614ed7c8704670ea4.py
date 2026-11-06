import time
import random

def is_sidon(subset):
    """Check if a subset has all unordered pair sums unique (including i=j)."""
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
    """Compute the size of the largest Sidon subset of S exactly."""
    n = len(S)
    max_size = 0
    # Iterate through all subsets using bitmask
    # 2^12 = 4096 subsets
    for mask in range(1, 1 << n):
        size = bin(mask).count('1')
        if size <= max_size:
            continue
        subset = [S[i] for i in range(n) if (mask >> i) & 1]
        if is_sidon(subset):
            max_size = size
    return max_size

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin before 90s kill
    
    def elapsed():
        return time.time() - start_time
    
    # We want to minimize the largest Sidon subset size of our 12-element set.
    # The minimum possible is 1 (no two elements can form a valid pair with unique sums).
    # But for 12 elements, it's likely we can get down to 4 or 5.
    # Actually, let's think: a Sidon set of size k has C(k,2)+k = k(k+1)/2 + k = k(k+3)/2 sums.
    # For k=5: 5*8/2 = 20 sums.
    # For k=4: 4*7/2 = 14 sums.
    
    # Strategy: 
    # 1. Start with a random set.
    # 2. Use local search: swap one element with another to reduce max_sidon_size.
    # 3. Since max_sidon_size is expensive (4096 subsets * 78 pair sums), we need to be smart.
    
    # Better strategy:
    # The max_sidon_size is determined by the "densest" Sidon subset.
    # We want to make it hard to find large Sidon subsets.
    # One approach: make many pair sums collide.
    
    # Let's use a simulated annealing or random restart local search.
    
    def evaluate(S):
        return max_sidon_size(S)
    
    # Initial candidate: random
    random.seed(12345)
    
    best_set = random.sample(range(129), 12)
    best_score = evaluate(best_set)
    
    # Local search with swaps
    improved = True
    while improved and elapsed() < deadline:
        improved = False
        for i in range(12):
            if elapsed() > deadline:
                break
            for val in range(129):
                if val in best_set:
                    continue
                old_val = best_set[i]
                best_set[i] = val
                new_score = evaluate(best_set)
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    best_set[i] = old_val
    
    # If we still have time, try random restarts
    while elapsed() < deadline - 10:
        # Random perturbation
        current = list(best_set)
        # Swap 2-3 random elements
        for _ in range(3):
            i = random.randint(0, 11)
            val = random.randint(0, 128)
            while val in current:
                val = random.randint(0, 128)
            current[i] = val
        
        current_score = evaluate(current)
        if current_score < best_score:
            best_set = current
            best_score = current_score
        
        # Local search from this new starting point
        improved = True
        while improved and elapsed() < deadline - 5:
            improved = False
            for i in range(12):
                if elapsed() > deadline - 5:
                    break
                for val in range(129):
                    if val in best_set:
                        continue
                    old_val = best_set[i]
                    best_set[i] = val
                    new_score = evaluate(best_set)
                    if new_score < best_score:
                        best_score = new_score
                        improved = True
                    else:
                        best_set[i] = old_val
    
    return best_set
