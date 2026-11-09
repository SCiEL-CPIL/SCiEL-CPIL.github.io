import random
import time
import itertools

def is_sidon_subset(subset):
    """
    Check if a subset is Sidon.
    Sidon: all unordered pair sums x+y (x<=y) are unique.
    """
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(ambient_set):
    """
    Find the size of the largest Sidon subset of ambient_set.
    Uses exhaustive search for small sets (12 elements).
    2^12 = 4096 subsets. Very fast.
    """
    max_size = 0
    # Check subsets in descending order of size
    # But it's faster to just iterate all and keep max.
    # Or prune: if a set of size k is not Sidon, its supersets are not.
    # But we want the MAX Sidon subset.
    # So we check all subsets.
    
    # Optimization: Sort ambient set? Not necessary for subset generation.
    # Use bitmasks or combinations.
    n = len(ambient_set)
    # We know max possible is n.
    # Check sizes from n down to 1.
    for k in range(n, 0, -1):
        found = False
        for comb in itertools.combinations(ambient_set, k):
            if is_sidon_subset(list(comb)):
                return k
        # If no subset of size k is Sidon, try k-1
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    best_set = list(range(12))
    best_val = max_sidon_size(best_set)
    
    # If we already have 3, we can't do better (min is 2, which is impossible for n=12)
    if best_val == 3:
        return best_set
        
    # Search strategy:
    # 1. Try small ranges [0, R]
    # 2. Random sampling within small ranges
    # 3. Local search
    
    # Since n=12, max_sidon_size is fast (~ms).
    # We can try many candidates.
    
    candidates = []
    
    # Try consecutive integers in various ranges
    for start in range(0, 10):
        for length in range(12, 20):
            cand = list(range(start, start + length))
            if len(cand) == 12:
                candidates.append(cand)
                
    # Random sets in [0, 30]
    for _ in range(10000):
        if time.time() > deadline:
            break
        cand = random.sample(range(0, 31), 12)
        candidates.append(cand)
        
    # Random sets in [0, 128] but biased to small numbers
    for _ in range(10000):
        if time.time() > deadline:
            break
        # Use exponential distribution to favor small numbers
        cand = []
        while len(cand) < 12:
            x = int(random.expovariate(1/5.0)) # Mean 5
            if x > 128:
                x = random.randint(0, 128)
            if x not in cand:
                cand.append(x)
        candidates.append(cand)

    # Evaluate candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        val = max_sidon_size(cand)
        if val < best_val:
            best_val = val
            best_set = cand
            if best_val == 3:
                return best_set
                
    # Local search on best_set
    if time.time() < deadline - 5.0:
        current = list(best_set)
        current_val = best_val
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try swapping one element with a random other in [0, 128]
            for _ in range(1000):
                if time.time() > deadline - 5.0:
                    break
                idx = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val in current and current.index(new_val) != idx:
                    continue
                old_val = current[idx]
                current[idx] = new_val
                val = max_sidon_size(current)
                if val < current_val:
                    current_val = val
                    best_set = list(current)
                    best_val = val
                    improved = True
                    if best_val == 3:
                        return best_set
                else:
                    current[idx] = old_val
                    
    return best_set
