import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill
    
    N = 256
    # Perfect squares up to 512
    squares = set()
    s = 1
    while s * s <= 512:
        squares.add(s * s)
        s += 1
    
    # Diagonal constraint: 2*a must not be a perfect square
    # 2*a = perfect square means a = 2*k^2 for some k
    diagonal_forbidden = set()
    k = 1
    while 2 * k * k <= N:
        diagonal_forbidden.add(2 * k * k)
        k += 1
    
    # Build conflict graph: edge between a and b if a+b is a perfect square (a < b)
    # Also mark diagonal conflicts
    # conflict[a] = set of b > a that conflict with a
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        if a in diagonal_forbidden:
            continue  # can't include a at all
        for b in range(a + 1, N + 1):
            if b in diagonal_forbidden:
                continue
            if a + b in squares:
                conflict[a].add(b)
                conflict[b].add(a)
    
    # Valid numbers (not diagonally forbidden)
    valid_nums = [i for i in range(1, N + 1) if i not in diagonal_forbidden]
    
    # Greedy: sort by degree (fewest conflicts first) and add if no conflict with current set
    # Try multiple orderings for diversity
    
    def greedy_construct(order):
        chosen = set()
        for v in order:
            if v in diagonal_forbidden:
                continue
            # Check if v conflicts with any chosen
            ok = True
            for c in chosen:
                if c in conflict[v]:
                    ok = False
                    break
            if ok:
                chosen.add(v)
        return chosen
    
    def local_search(chosen, time_limit=5.0):
        chosen = set(chosen)
        # Precompute conflicts for fast lookup
        chosen_list = list(chosen)
        
        while time.time() - start_time < DEADLINE - 5:
            improved = False
            # Try removing one element and adding one or more
            # Single swap: remove x, add y
            for x in chosen_list:
                # Try adding each non-chosen valid number
                for y in valid_nums:
                    if y in chosen:
                        continue
                    # Check if y conflicts with (chosen - {x})
                    ok = True
                    for c in chosen:
                        if c == x:
                            continue
                        if c in conflict[y]:
                            ok = False
                            break
                    if ok and y not in conflict[x]:  # y doesn't conflict with x either (redundant but safe)
                        # Actually we just need y not to conflict with chosen-{x}
                        new_size = len(chosen)  # remove x, add y, same size
                        # Try double: remove x, add y1, y2
                        # For now single swap
                        if new_size > len(chosen):
                            pass
                        # Same size swap - only useful if it enables further improvements
                        # Let's try: remove x, add y (same size), then try to add more
                        new_chosen = (chosen - {x}) | {y}
                        # Try to greedily add more
                        can_add = True
                        added_any = False
                        for z in valid_nums:
                            if z in new_chosen:
                                continue
                            ok2 = True
                            for c in new_chosen:
                                if c in conflict[z]:
                                    ok2 = False
                                    break
                            if ok2:
                                new_chosen.add(z)
                                added_any = True
                        if len(new_chosen) > len(chosen):
                            chosen = new_chosen
                            chosen_list = list(chosen)
                            improved = True
                            break
                if improved:
                    break
                if time.time() - start_time > DEADLINE - 5:
                    break
            if not improved:
                # Try removal of a problematic element to allow additions
                # Or try a different strategy: remove x, see if we can add 2+ elements
                found = False
                for x in chosen_list:
                    without_x = chosen - {x}
                    # Greedily add as many as possible
                    new_chosen = set(without_x)
                    for z in valid_nums:
                        if z in new_chosen:
                            continue
                        ok2 = True
                        for c in new_chosen:
                            if c in conflict[z]:
                                ok2 = False
                                break
                        if ok2:
                            new_chosen.add(z)
                    if len(new_chosen) > len(chosen):
                        chosen = new_chosen
                        chosen_list = list(chosen)
                        found = True
                        break
                if not found:
                    break
        
        return chosen
    
    # Multiple greedy starts with different orderings
    best_set = set()
    best_size = 0
    
    # Ordering 1: by degree (ascending)
    degrees = {v: len(conflict[v]) for v in valid_nums}
    order1 = sorted(valid_nums, key=lambda v: degrees[v])
    
    # Ordering 2: by value
    order2 = sorted(valid_nums)
    
    # Ordering 3: by value descending
    order3 = sorted(valid_nums, reverse=True)
    
    # Ordering 4: by degree descending
    order4 = sorted(valid_nums, key=lambda v: -degrees[v])
    
    # Ordering 5: random-ish (use a fixed seed pattern)
    order5 = sorted(valid_nums, key=lambda v: (v * 7 + 13) % 37)
    
    for order in [order1, order2, order3, order4, order5]:
        if time.time() - start_time > DEADLINE - 10:
            break
        gset = greedy_construct(order)
        if len(gset) > best_size:
            best_size = len(gset)
            best_set = set(gset)
        
        # Local search on this greedy result
        if time.time() - start_time < DEADLINE - 15:
            lset = local_search(gset)
            if len(lset) > best_size:
                best_size = len(lset)
                best_set = set(lset)
    
    # Additional local search iterations on the best set
    if time.time() - start_time < DEADLINE - 10 and len(best_set) > 0:
        # Try more aggressive local search
        current = set(best_set)
        for iteration in range(20):
            if time.time() - start_time > DEADLINE - 5:
                break
            improved = False
            current_list = list(current)
            for x in current_list:
                if time.time() - start_time > DEADLINE - 5:
                    break
                without_x = current - {x}
                new_chosen = set(without_x)
                # Greedily add
                for z in valid_nums:
                    if z in new_chosen:
                        continue
                    ok = True
                    for c in new_chosen:
                        if c in conflict[z]:
                            ok = False
                            break
                    if ok:
                        new_chosen.add(z)
                if len(new_chosen) > len(current):
                    current = new_chosen
                    improved = True
                    break
            if not improved:
                # Try pairwise swaps
                for i, x in enumerate(current_list):
                    if time.time() - start_time > DEADLINE - 5:
                        break
                    for y in valid_nums:
                        if y in current:
                            continue
                        new_chosen = (current - {x}) | {y}
                        # Try to add more
                        for z in valid_nums:
                            if z in new_chosen:
                                continue
                            ok = True
                            for c in new_chosen:
                                if c in conflict[z]:
                                    ok = False
                                    break
                            if ok:
                                new_chosen.add(z)
                        if len(new_chosen) > len(current):
                            current = new_chosen
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                break
            if len(current) > len(best_set):
                best_set = set(current)
                best_size = len(current)
    
    # Final verification
    result = sorted(best_set)
    # Verify all pairs
    valid = True
    for i in range(len(result)):
        a = result[i]
        if a in diagonal_forbidden:
            valid = False
            break
        for j in range(i, len(result)):
            b = result[j]
            if a + b in squares:
                valid = False
                break
        if not valid:
            break
    
    if not valid or len(result) == 0:
        # Fallback: simple valid set
        result = [1, 3, 5, 7, 10, 12, 14, 15]
        # Verify fallback
        final_valid = True
        for i in range(len(result)):
            a = result[i]
            for j in range(i, len(result)):
                b = result[j]
                if a + b in squares:
                    final_valid = False
                    break
            if not final_valid:
                break
        if not final_valid:
            result = [1]
    
    return result
