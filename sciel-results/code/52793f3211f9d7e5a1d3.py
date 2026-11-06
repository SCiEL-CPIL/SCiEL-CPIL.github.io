def construct():
    import time
    from math import gcd
    from itertools import combinations
    
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return DEADLINE - elapsed()
    
    # Precompute gcd table for [1,128]
    n = 128
    gcd_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def is_valid_triple(a, b, c):
        """Check if triple (a,b,c) violates the condition (all three pairwise gcds equal)"""
        gab = gcd_table[a][b]
        gac = gcd_table[a][c]
        gbc = gcd_table[b][c]
        return gab == gac == gbc
    
    def can_add(selected_set, x):
        """Check if x can be added to selected_set without creating a bad triple"""
        sel_list = list(selected_set)
        for i in range(len(sel_list)):
            for j in range(i+1, len(sel_list)):
                a = sel_list[i]
                b = sel_list[j]
                if is_valid_triple(a, b, x):
                    return False
        return True
    
    def count_bad_triples(selected):
        """Count number of bad triples in selected"""
        bad = 0
        sel = list(selected)
        for i in range(len(sel)):
            for j in range(i+1, len(sel)):
                for k in range(j+1, len(sel)):
                    if is_valid_triple(sel[i], sel[j], sel[k]):
                        bad += 1
        return bad
    
    # Greedy construction: try to build a large set
    # Strategy: prioritize numbers with "rare" gcd patterns
    # Start with numbers that are hard to form triples with
    
    # Initial greedy: add numbers in an order that maximizes flexibility
    # Numbers with many small prime factors create more gcd=1 or gcd=2 situations
    
    # Let's try: include all numbers, then remove those causing conflicts
    # Or: greedy add in smart order
    
    # Smart greedy: process numbers, try to add if possible
    # Order: try numbers that are "diverse" in their gcd relationships
    
    def greedy_construct():
        selected = set()
        # Try adding in order of increasing "conflict potential"
        # Numbers that are prime or have unique factors are safer
        # Actually, let's just try all numbers 1-128 and greedily add
        for x in range(1, n+1):
            if elapsed() > DEADLINE:
                break
            if can_add(selected, x):
                selected.add(x)
        return selected
    
    # Local search: try to swap elements to improve
    def local_search(selected):
        sel_list = list(selected)
        improved = True
        while improved and time_left() > 5.0:
            improved = False
            # Try removing one element and adding another
            for i in range(len(sel_list)):
                if time_left() < 3.0:
                    break
                for j in range(len(sel_list)):
                    if i == j:
                        continue
                    # Try swapping sel_list[i] with sel_list[j]
                    # Actually, try removing sel_list[i] and adding some other number
                    removed = sel_list[i]
                    new_selected = selected - {removed}
                    # Try adding numbers not in new_selected
                    for x in range(1, n+1):
                        if x in new_selected:
                            continue
                        if can_add(new_selected, x):
                            # This swap increases or maintains size
                            # But we want to try to ADD, so this is only useful if we can add without removing
                            pass
                    # Try: remove one, add one different
                    for x in range(1, n+1):
                        if x in selected:
                            continue
                        if x == removed:
                            continue
                        new_selected = selected - {removed}
                        if can_add(new_selected, x):
                            # Check if this actually helps (same size, but maybe enables future additions)
                            # For now, just accept if size is same or larger
                            selected = new_selected | {x}
                            sel_list = list(selected)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return selected
    
    # Better local search: hill climbing with random restarts
    import random
    
    def random_restart():
        random.seed()
        # Start with a random subset
        size = random.randint(20, 50)
        candidates = list(range(1, n+1))
        random.shuffle(candidates)
        selected = set()
        for x in candidates:
            if len(selected) >= size:
                break
            if can_add(selected, x):
                selected.add(x)
        return selected
    
    def hill_climb(selected):
        sel_list = list(selected)
        best = selected
        best_size = len(selected)
        
        while time_left() > 2.0:
            # Try all single swaps
            found_improvement = False
            for i in range(len(sel_list)):
                if time_left() < 1.0:
                    break
                removed = sel_list[i]
                base = best - {removed}
                for x in range(1, n+1):
                    if x in best or x == removed:
                        continue
                    if can_add(base, x):
                        new_set = base | {x}
                        if len(new_set) > best_size:
                            best = new_set
                            best_size = len(new_set)
                            sel_list = list(best)
                            found_improvement = True
                            break
                if found_improvement:
                    break
            if not found_improvement:
                # Try adding without removing
                for x in range(1, n+1):
                    if x in best:
                        continue
                    if can_add(best, x):
                        best = best | {x}
                        best_size += 1
                        sel_list = list(best)
                        found_improvement = True
                        break
                if not found_improvement:
                    break
        return best
    
    # Start with greedy
    best = greedy_construct()
    
    # Try multiple local searches
    for _ in range(10):
        if time_left() < 10.0:
            break
        candidate = random_restart()
        candidate = hill_climb(candidate)
        if len(candidate) > len(best):
            best = candidate
    
    # Final attempt to add more
    sel_list = list(best)
    for x in range(1, n+1):
        if time_left() < 2.0:
            break
        if x not in best and can_add(best, x):
            best.add(x)
    
    return list(best)
