def construct():
    import random
    from math import gcd
    from itertools import combinations

    def is_valid_triple(a, b, c):
        return not (gcd(a, b) == gcd(a, c) == gcd(b, c))

    def can_add(x, current_set):
        # Check if adding x to current_set creates a forbidden triple
        # A forbidden triple (a, b, x) exists if gcd(a,b) == gcd(a,x) == gcd(b,x)
        # We only need to check pairs (a, b) in current_set
        for a, b in combinations(current_set, 2):
            if gcd(a, b) == gcd(a, x) == gcd(b, x):
                return False
        return True

    best_set = set()
    best_size = 0
    
    # Number of restarts
    # 60 seconds is plenty for 1000-5000 restarts of size 128 greedy
    for _ in range(2000):
        current_set = set()
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, current_set):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
    # Try to improve best_set with local search (adding any remaining valid numbers)
    # The greedy might have missed some if the order was bad, but a second pass over all numbers
    # in sorted order or random order on the current best_set might help.
    improved = True
    while improved:
        improved = False
        remaining = set(range(1, 129)) - best_set
        for x in list(remaining):
            if can_add(x, best_set):
                best_set.add(x)
                remaining.discard(x)
                improved = True
                
    return list(best_set)
