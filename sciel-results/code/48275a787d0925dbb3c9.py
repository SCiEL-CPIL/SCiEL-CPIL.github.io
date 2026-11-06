def construct():
    import time
    from math import gcd
    from itertools import combinations
    
    deadline = time.time() + 75  # safety margin before 90s kill
    
    n = 128
    # Precompute gcd table
    gcd_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def has_bad_triple_with_new(x, selected):
        """Check if adding x to selected creates a bad triple."""
        # For each pair (a, b) in selected, check if gcd(x,a) == gcd(x,b) == gcd(a,b)
        for i in range(len(selected)):
            a = selected[i]
            gxa = gcd_table[x][a]
            for j in range(i+1, len(selected)):
                b = selected[j]
                gxb = gcd_table[x][b]
                if gxa == gxb and gxa == gcd_table[a][b]:
                    return True
        return False
    
    def has_bad_triple(selected):
        """Check if selected contains any bad triple."""
        for i in range(len(selected)):
            a = selected[i]
            for j in range(i+1, len(selected)):
                b = selected[j]
                gab = gcd_table[a][b]
                for k in range(j+1, len(selected)):
                    c = selected[k]
                    if gcd_table[a][c] == gab and gcd_table[b][c] == gab:
                        return True
        return False
    
    # Greedy construction with multiple orderings
    best_solution = []
    
    # Try different greedy orderings
    orderings = [
        list(range(1, n+1)),
        list(range(n, 0, -1)),
        # Sort by number of divisors (fewer divisors = more "prime-like")
        sorted(range(1, n+1), key=lambda x: sum(1 for d in range(1, x+1) if x % d == 0)),
        # Sort by largest prime factor
        sorted(range(1, n+1), key=lambda x: max((d for d in range(1, x+1) if x % d == 0), default=1)),
    ]
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        selected = []
        for x in ordering:
            if not has_bad_triple_with_new(x, selected):
                selected.append(x)
        if len(selected) > len(best_solution):
            best_solution = selected[:]
    
    # Local search: try to add elements, then swap
    if time.time() > deadline:
        return best_solution
    
    # Try to add any missing element
    for attempt in range(200):
        if time.time() > deadline:
            break
        improved = False
        for x in range(1, n+1):
            if x not in best_solution:
                if not has_bad_triple_with_new(x, best_solution):
                    best_solution.append(x)
                    improved = True
                    break
        if not improved:
            break
    
    # Local search: swap out one element to allow adding two
    if time.time() > deadline:
        return best_solution
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        improved = False
        for i, remove_x in enumerate(best_solution):
            if time.time() > deadline:
                break
            candidate = best_solution[:i] + best_solution[i+1:]
            # Try to add up to 2 elements
            added = []
            for x in range(1, n+1):
                if x not in candidate and x not in added:
                    if not has_bad_triple_with_new(x, candidate + added):
                        added.append(x)
                        if len(added) >= 2:
                            break
            if len(added) > 1:
                best_solution = candidate + added
                improved = True
                break
        if not improved:
            break
    
    # Final local search: remove one, add best possible
    if time.time() > deadline:
        return best_solution
    
    for iteration in range(300):
        if time.time() > deadline:
            break
        improved = False
        for i, remove_x in enumerate(best_solution):
            if time.time() > deadline:
                break
            candidate = best_solution[:i] + best_solution[i+1:]
            best_add = None
            for x in range(1, n+1):
                if x not in candidate:
                    if not has_bad_triple_with_new(x, candidate):
                        if best_add is None or x > best_add:
                            best_add = x
            if best_add is not None and best_add != remove_x:
                # Check if this actually helps (it won't change size, but may enable future additions)
                # Skip for now, focus on size improvement
                pass
        if not improved:
            break
    
    # Ensure nonempty
    if not best_solution:
        best_solution = [1]
    
    return best_solution
