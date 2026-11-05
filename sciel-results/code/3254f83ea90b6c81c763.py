def construct():
    import time
    import random
    from itertools import combinations
    from math import gcd
    
    deadline = time.time() + 55  # safety margin
    
    # Precompute gcd table
    n = 128
    gcd_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    
    # Precompute all bad triples
    bad_triples = set()
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            gab = gcd_table[a][b]
            for c in range(b+1, n+1):
                if gcd_table[a][c] == gab and gcd_table[b][c] == gab:
                    bad_triples.add((a, b, c))
    
    # Convert to adjacency: for each element, which triples it's in
    # For greedy: track which elements are "blocked" if we add x
    # Use bitmasks for speed
    # element -> list of triples containing it
    elem_triples = [[] for _ in range(n+1)]
    for (a, b, c) in bad_triples:
        elem_triples[a].append((a, b, c))
        elem_triples[b].append((a, b, c))
        elem_triples[c].append((a, b, c))
    
    def is_valid_triple_free(selected_set):
        """Check if selected_set has no bad triple."""
        sel = set(selected_set)
        for (a, b, c) in bad_triples:
            if a in sel and b in sel and c in sel:
                return False
        return True
    
    def greedy_randomized(order):
        """Greedy: add elements in given order if no bad triple is formed."""
        selected = set()
        # For speed, track which triples are "active" (have 2 elements selected)
        # When we add x, check if any triple with x and 2 already-selected elements exists
        triple_count = {}  # triple -> count of selected elements
        for x in order:
            # Check if adding x creates any bad triple
            ok = True
            for (a, b, c) in elem_triples[x]:
                other1, other2 = (a, b) if x == c else ((a, c) if x == b else (b, c))
                if other1 in selected and other2 in selected:
                    ok = False
                    break
            if ok:
                selected.add(x)
        return selected
    
    best = set()
    best_size = 0
    
    # Strategy 1: Various greedy orderings
    elements = list(range(1, n+1))
    
    # Try different orderings
    for trial in range(200):
        if time.time() > deadline:
            break
        
        if trial < 10:
            # Deterministic orderings
            if trial == 0:
                order = elements[:]  # natural order
            elif trial == 1:
                order = elements[::-1]  # reverse
            elif trial == 2:
                # Sort by number of bad triples (fewest first)
                order = sorted(elements, key=lambda x: len(elem_triples[x]))
            elif trial == 3:
                order = sorted(elements, key=lambda x: -len(elem_triples[x]))
            elif trial == 4:
                # Powers of 2 first, then others
                order = sorted(elements, key=lambda x: (x & -x) != x, then=x) if False else sorted(elements, key=lambda x: (0 if x & -x == x else 1, x))
            elif trial == 5:
                # Odd numbers first
                order = sorted(elements, key=lambda x: (x % 2 == 0, x))
            elif trial == 6:
                # Even numbers first
                order = sorted(elements, key=lambda x: (x % 2 != 0, x))
            elif trial == 7:
                # Sort by smallest prime factor
                def spf(x):
                    if x % 2 == 0: return 2
                    if x % 3 == 0: return 3
                    if x % 5 == 0: return 5
                    if x % 7 == 0: return 7
                    if x % 11 == 0: return 11
                    if x % 13 == 0: return 13
                    if x % 17 == 0: return 17
                    if x % 19 == 0: return 19
                    if x % 23 == 0: return 23
                    if x % 29 == 0: return 29
                    if x % 31 == 0: return 31
                    if x % 37 == 0: return 37
                    if x % 41 == 0: return 41
                    if x % 43 == 0: return 43
                    if x % 47 == 0: return 47
                    if x % 53 == 0: return 53
                    if x % 59 == 0: return 59
                    if x % 61 == 0: return 61
                    if x % 67 == 0: return 67
                    if x % 71 == 0: return 71
                    if x % 73 == 0: return 73
                    if x % 79 == 0: return 79
                    if x % 83 == 0: return 83
                    if x % 89 == 0: return 89
                    if x % 97 == 0: return 97
                    return x
                order = sorted(elements, key=lambda x: spf(x))
            elif trial == 8:
                # Sort by number of distinct prime factors
                def num_pf(x):
                    count = 0
                    d = 2
                    while d * d <= x:
                        if x % d == 0:
                            count += 1
                            while x % d == 0:
                                x //= d
                        d += 1
                    if x > 1:
                        count += 1
                    return count
                order = sorted(elements, key=lambda x: (num_pf(x), x))
            else:
                order = sorted(elements, key=lambda x: (num_pf(x), -len(elem_triples[x]), x))
        else:
            # Random orderings
            order = elements[:]
            random.shuffle(order)
        
        result = greedy_randomized(order)
        if len(result) > best_size:
            best_size = len(result)
            best = set(result)
    
    # Local search: try to swap elements to improve
    if time.time() < deadline:
        for iteration in range(50):
            if time.time() > deadline:
                break
            improved = False
            # Try adding any element not in best
            for x in range(1, n+1):
                if x not in best:
                    # Check if adding x is valid
                    ok = True
                    for (a, b, c) in elem_triples[x]:
                        other1, other2 = (a, b) if x == c else ((a, c) if x == b else (b, c))
                        if other1 in best and other2 in best:
                            ok = False
                            break
                    if ok:
                        best.add(x)
                        improved = True
            # Try removing one and adding two
            if not improved and len(best) >= 2:
                best_list = list(best)
                random.shuffle(best_list)
                for i in range(min(20, len(best_list))):
                    if time.time() > deadline:
                        break
                    to_remove = best_list[i]
                    temp_best = best - {to_remove}
                    # Try to add two elements
                    added = 0
                    for x in range(1, n+1):
                        if x not in temp_best and added < 2:
                            ok = True
                            for (a, b, c) in elem_triples[x]:
                                other1, other2 = (a, b) if x == c else ((a, c) if x == b else (b, c))
                                if other1 in temp_best and other2 in temp_best:
                                    ok = False
                                    break
                            if ok:
                                temp_best.add(x)
                                added += 1
                    if len(temp_best) > len(best):
                        best = temp_best
                        best_size = len(best)
                        improved = True
    
    return sorted(best)
