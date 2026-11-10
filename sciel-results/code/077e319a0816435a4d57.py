def construct():
    import time
    start = time.time()
    deadline = start + 80.0  # 80s safety margin
    
    N = 512
    from math import gcd
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Build conflict: conflict[i][j] = True if lcm(i+1, j+1) > 512
    # Use bitset for efficiency
    n = N
    # conflict_mask[i] is a set of indices j where lcm(i+1, j+1) > 512
    conflict = [set() for _ in range(n)]
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            if a * b // gcd(a, b) > N:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Greedy: start with all numbers, remove conflicting ones
    # Better: greedy independent set
    # Order numbers by degree (ascending) for greedy
    degrees = [len(conflict[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: (degrees[x], x))
    
    selected = set()
    for idx in order:
        if time.time() > deadline:
            break
        # Check if idx conflicts with any selected
        can_add = True
        for s in selected:
            if s in conflict[idx]:
                can_add = False
                break
        if can_add:
            selected.add(idx)
    
    # Local improvement: try to add removed elements, swap
    # Try 1-swap: for each unselected u, see if we can add it by removing one selected
    improved = True
    while improved and time.time() < deadline:
        improved = False
        unselected = [i for i in range(n) if i not in selected]
        for u in unselected:
            if time.time() > deadline:
                break
            # Find selected elements that conflict with u
            conflicts_with_u = [s for s in selected if s in conflict[u]]
            if not conflicts_with_u:
                selected.add(u)
                improved = True
            else:
                # Try removing one conflict to add u
                # Only beneficial if we remove fewer than we gain (net +1)
                # Since we're adding 1 and removing 1, no net gain
                # But maybe removing 1 lets us add more later - skip for now
                pass
    
    # Try 2-swap: remove 2 selected, add 1 unselected (net -1, bad) 
    # Or remove 1 selected, add 2 unselected (net +1)
    # Let's try: for each pair of unselected (u1, u2) that don't conflict,
    # check if there's a selected s that conflicts with both, and removing s lets us add both
    if time.time() < deadline - 5:
        unselected = [i for i in range(n) if i not in selected]
        # Group unselected by which selected they conflict with
        # For each unselected u, find minimal set of selected to remove
        # Simpler: try all pairs of unselected that are compatible
        for i in range(len(unselected)):
            if time.time() > deadline:
                break
            u1 = unselected[i]
            for j in range(i + 1, len(unselected)):
                if time.time() > deadline:
                    break
                u2 = unselected[j]
                # Check u1 and u2 don't conflict
                if u2 in conflict[u1]:
                    continue
                # Find selected that conflict with u1 or u2
                blockers1 = [s for s in selected if s in conflict[u1]]
                blockers2 = [s for s in selected if s in conflict[u2]]
                blockers = set(blockers1) | set(blockers2)
                # If we remove all blockers, we can add u1 and u2
                # Net gain = 2 - len(blockers)
                if len(blockers) <= 1:
                    for s in blockers:
                        selected.remove(s)
                    selected.add(u1)
                    selected.add(u2)
                    improved = True
                    break
            if improved:
                break
    
    result = sorted([i + 1 for i in selected])
    if not result:
        result = [1]
    return result
