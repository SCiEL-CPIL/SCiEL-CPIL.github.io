def is_sidon(subset):
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    S = sorted(S)
    n = len(S)
    from itertools import combinations
    for k in range(n, 2, -1):
        for subset in combinations(S, k):
            if is_sidon(subset):
                return k
    for subset in combinations(S, 3):
        if is_sidon(subset):
            return 3
    return 2

def construct():
    import random
    from itertools import combinations
    
    best_set = list(range(12))
    best_size = max_sidon_size(best_set)
    
    random.seed(42)
    
    # Try to find a set with max sidon size 3
    # Random search
    for _ in range(5000):
        S = random.sample(range(129), 12)
        size = max_sidon_size(S)
        if size < best_size:
            best_size = size
            best_set = S
            if best_size == 3:
                # We can't get below 3 easily, but let's check if 2 is possible
                # Actually, 2 is possible if all triples are APs, which is impossible for 12 elems.
                # So 3 is the target.
                pass
    
    # Local search around best_set if best_size > 3
    # Or just return best_set
    
    return best_set
