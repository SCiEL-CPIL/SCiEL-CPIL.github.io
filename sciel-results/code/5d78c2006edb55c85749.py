def largest_sidon_subset(S):
    """Find largest subset of S with all pair sums (i<=j) unique."""
    S = sorted(S)
    n = len(S)
    from itertools import combinations
    
    for k in range(n, 1, -1):
        for subset in combinations(S, k):
            sums = set()
            valid = True
            for i in range(len(subset)):
                for j in range(i, len(subset)):
                    s = subset[i] + subset[j]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid:
                return k
    return 2  # any two distinct elements form a Sidon set

def construct():
    import random
    from itertools import combinations
    
    best_set = list(range(12))
    best_size = largest_sidon_subset(best_set)
    
    # Try various structured sets
    candidates = []
    
    # Arithmetic progressions
    for start in range(0, 120):
        for diff in range(1, 12):
            S = [start + i*diff for i in range(12)]
            if max(S) <= 128:
                candidates.append(S)
    
    # Random sets
    random.seed(42)
    for _ in range(500):
        S = random.sample(range(129), 12)
        candidates.append(S)
    
    # Try to find a set with small largest Sidon subset
    for S in candidates:
        size = largest_sidon_subset(S)
        if size < best_size:
            best_size = size
            best_set = S
            if best_size == 2:
                break
    
    return best_set
