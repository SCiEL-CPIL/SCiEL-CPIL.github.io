def construct():
    import random
    from math import gcd
    from itertools import combinations

    n = 128
    
    # 1. Precompute all bad triples
    # A triple (a,b,c) is bad if gcd(a,b)==gcd(a,c)==gcd(b,c)
    bad_triples = []
    # Map from element to list of bad triples it's in
    element_triples = {i: [] for i in range(1, n + 1)}
    
    # Precompute gcds
    gcds = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            gcds[i][j] = gcd(i, j)
            gcds[j][i] = gcds[i][j]

    for a in range(1, n+1):
        for b in range(a+1, n+1):
            g_ab = gcds[a][b]
            for c in range(b+1, n+1):
                if gcds[a][c] == g_ab and gcds[b][c] == g_ab:
                    bad_triples.append((a, b, c))
                    element_triples[a].append((a, b, c))
                    element_triples[b].append((a, b, c))
                    element_triples[c].append((a, b, c))
    
    # Helper to check if adding x to current set creates a bad triple
    # We can just check all bad triples involving x
    def is_compatible(x, current_set):
        for triple in element_triples[x]:
            # Check if the other two elements are in current_set
            # triple is (a,b,c) where one is x
            # Get the other two
            others = [t for t in triple if t != x]
            if others[0] in current_set and others[1] in current_set:
                return False
        return True

    best_set = set()
    
    # Run greedy multiple times with different orderings
    for run in range(50):
        random.seed(run)
        
        # Heuristic: Sort by degree (number of bad triples) ascending
        # Elements with fewer constraints are "safer" to pick early?
        # Or should we pick high degree to eliminate conflicts?
        # For Max Independent Set, picking low degree first is a common greedy heuristic.
        
        candidates = list(range(1, n + 1))
        
        # Randomize the sort slightly to explore different solutions
        degrees = [(len(element_triples[i]), random.random(), i) for i in candidates]
        degrees.sort()
        sorted_candidates = [d[2] for d in degrees]
        
        current_set = set()
        
        for x in sorted_candidates:
            if is_compatible(x, current_set):
                current_set.add(x)
        
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            
    # Local Search: Try to improve best_set
    # Try swapping an element in best_set with one outside to see if we can add more?
    # Since we want max size, a simple local search is:
    # Try to add an element not in set. If it conflicts, try to remove one of the conflicting elements 
    # if that allows the new element to be added AND we don't lose size (impossible with 1-for-1) 
    # BUT maybe removing 1 allows adding 2?
    
    # Let's do a simple "repair" loop:
    # For each element not in best_set, check conflicts.
    # If conflicts are few, maybe we can remove a subset of conflicts and add this element?
    # This is complex. Let's just do a few more greedy runs with different heuristics.
    
    # Heuristic 2: Sort by value (ascending)
    for run in range(50):
        random.seed(100 + run)
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        # Sort by value
        candidates.sort()
        
        current_set = set()
        for x in candidates:
            if is_compatible(x, current_set):
                current_set.add(x)
        
        if len(current_set) > len(best_set):
            best_set = current_set.copy()

    # Heuristic 3: Sort by number of prime factors (ascending)
    def count_prime_factors(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count

    pf_counts = {i: count_prime_factors(i) for i in range(1, n+1)}
    for run in range(50):
        random.seed(200 + run)
        candidates = list(range(1, n + 1))
        # Sort by prime factor count, then random
        candidates.sort(key=lambda x: (pf_counts[x], random.random()))
        
        current_set = set()
        for x in candidates:
            if is_compatible(x, current_set):
                current_set.add(x)
        
        if len(current_set) > len(best_set):
            best_set = current_set.copy()

    return list(best_set)
