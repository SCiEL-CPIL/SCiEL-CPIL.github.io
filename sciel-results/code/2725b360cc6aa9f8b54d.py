def construct():
    import time
    from math import gcd
    import random
    
    deadline = time.time() + 75
    
    # Precompute gcd table
    gcd_table = [[0] * 129 for _ in range(129)]
    for a in range(1, 129):
        for b in range(a, 129):
            g = gcd(a, b)
            gcd_table[a][b] = g
            gcd_table[b][a] = g
    
    def get_gcd(a, b):
        return gcd_table[a][b]
    
    def num_distinct_pf(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    def total_pf(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    def num_divisors(n):
        count = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
            i += 1
        return count
    
    all_numbers = list(range(1, 129))
    
    def greedy(order):
        selected = []
        for x in order:
            can_add = True
            for i in range(len(selected)):
                a = selected[i]
                ga = get_gcd(a, x)
                for j in range(i + 1, len(selected)):
                    b = selected[j]
                    gb = get_gcd(b, x)
                    gab = get_gcd(a, b)
                    if ga == gb == gab:
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                selected.append(x)
        return selected
    
    # Multiple orderings
    best = []
    
    # Ordering 1: by decreasing distinct prime factors
    order1 = sorted(all_numbers, key=lambda x: (-num_distinct_pf(x), -total_pf(x), -num_divisors(x), x))
    best = greedy(order1)
    
    # Ordering 2: by decreasing total prime factors
    order2 = sorted(all_numbers, key=lambda x: (-total_pf(x), -num_distinct_pf(x), x))
    r2 = greedy(order2)
    if len(r2) > len(best):
        best = r2
    
    # Ordering 3: by decreasing number of divisors
    order3 = sorted(all_numbers, key=lambda x: (-num_divisors(x), x))
    r3 = greedy(order3)
    if len(r3) > len(best):
        best = r3
    
    # Random orderings
    random.seed(42)
    for _ in range(20):
        if time.time() > deadline:
            break
        order = all_numbers[:]
        random.shuffle(order)
        r = greedy(order)
        if len(r) > len(best):
            best = r
    
    # Local search on best
    selected = best
    selected_set = set(selected)
    unselected = [x for x in all_numbers if x not in selected_set]
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try adding each unselected number with removal of 1 selected number
        for x in unselected:
            if time.time() > deadline - 5:
                break
            for y in selected:
                # Check if replacing y with x gives valid set of same size
                # Or if we can add x and remove y to get valid set
                new_sel = [s for s in selected if s != y]
                new_sel.append(x)
                # Quick validity check
                valid = True
                n = len(new_sel)
                for i in range(n):
                    if not valid:
                        break
                    a = new_sel[i]
                    for j in range(i + 1, n):
                        if not valid:
                            break
                        b = new_sel[j]
                        for k in range(j + 1, n):
                            c = new_sel[k]
                            if get_gcd(a, b) == get_gcd(a, c) == get_gcd(b, c):
                                valid = False
                                break
                if valid:
                    # Same size, but might enable further improvements
                    # Only accept if it helps (for now, skip same-size swaps)
                    pass
            # Try adding x without removing (shouldn't be possible if greedy was optimal)
            # Try removing 2 numbers to add x (net -1, not helpful)
        
        # Try removing one number and see if we can add two others
        for y in selected:
            if time.time() > deadline - 5:
                break
            temp_sel = [s for s in selected if s != y]
            temp_set = set(temp_sel)
            # Try to add two numbers from unselected
            for i in range(len(unselected)):
                x1 = unselected[i]
                # Check if x1 can be added to temp_sel
                can_add_x1 = True
                for a in temp_sel:
                    ga = get_gcd(a, x1)
                    for b in temp_sel:
                        if a == b:
                            continue
                        gb = get_gcd(b, x1)
                        gab = get_gcd(a, b)
                        if ga == gb == gab:
                            can_add_x1 = False
                            break
                    if not can_add_x1:
                        break
                if not can_add_x1:
                    continue
                new_sel = temp_sel + [x1]
                # Now try to add x2
                for j in range(len(unselected)):
                    if i == j:
                        continue
                    x2 = unselected[j]
                    can_add_x2 = True
                    for a in new_sel:
                        ga = get_gcd(a, x2)
                        for b in new_sel:
                            if a == b:
                                continue
                            gb = get_gcd(b, x2)
                            gab = get_gcd(a, b)
                            if ga == gb == gab:
                                can_add_x2 = False
                                break
                        if not can_add_x2:
                            break
                    if can_add_x2:
                        final_sel = new_sel + [x2]
                        selected = final_sel
                        selected_set = set(selected)
                        unselected = [u for u in all_numbers if u not in selected_set]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return selected
