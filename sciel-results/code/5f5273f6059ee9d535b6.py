def construct():
    N = 256

    # Precompute all 3-term arithmetic progressions (non-constant) within [1, N]
    # An AP is (a, a+d, a+2d) with a >= 1, d >= 1, a+2d <= N
    # We represent each AP by its max element and the other two.
    # Actually, we need all APs. Let's list them as tuples (a, b, c) with a < b < c and b-a == c-b.
    # a + d = b, b + d = c => c = a + 2d.
    # So a in 1..N, d in 1..(N-a)//2.
    
    all_aps = []
    # element -> list of AP indices containing it
    elem_to_ap = [[] for _ in range(N + 1)]
    
    ap_idx = 0
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 2 + 1):
            c = a + 2 * d
            b = a + d
            ap_idx += 1
            all_aps.append((a, b, c))
            elem_to_ap[a].append(ap_idx)
            elem_to_ap[b].append(ap_idx)
            elem_to_ap[c].append(ap_idx)
    
    total_aps = ap_idx

    # Start with all elements in S
    S = set(range(1, N + 1))
    
    # active_aps: set of AP indices that are still "active" (all 3 elements in S)
    # Initially all APs are active.
    # We maintain active_count[x] = number of active APs containing x.
    
    active_aps = set(range(1, total_aps + 1))
    active_count = [0] * (N + 1)
    
    # Initialize active_count
    for ap_i in range(1, total_aps + 1):
        a, b, c = all_aps[ap_i - 1]
        active_count[a] += 1
        active_count[b] += 1
        active_count[c] += 1
    
    # Greedy deletion:
    # While there are active APs:
    #   Find an active AP whose max element has the highest active_count.
    #   Ties: pick the one with the smallest max element (to preserve smaller numbers? 
    #         Actually, we want to remove the element with smallest reciprocal weight, i.e., largest value.
    #         But the constraint is we remove the max of the AP.
    #         Wait, the "Smart Greedy Hitting Set" idea was:
    #         Pick the AP whose max element covers the most other active APs.
    #         Then remove that max element.
    #         This removes that element from S, and breaks all active APs containing it.
    
    # Let's refine:
    # We want to minimize the sum of reciprocals of removed elements.
    # Equivalent to maximizing the sum of reciprocals of remaining elements.
    # Greedy: At each step, pick an element to remove.
    # Which element? The one that breaks the most active APs, but weighted by its reciprocal cost?
    # Or simply: pick the AP, then remove its max element.
    # Which AP to pick? The one where the max element has the highest "impact" (active_count).
    
    # Let's try: 
    # 1. Find the element x in S that has the highest active_count[x].
    # 2. If active_count[x] == 0, we are done (no active APs).
    # 3. Remove x from S.
    # 4. For each active AP containing x, remove it from active_aps and decrement active_count for its other two elements.
    
    # Why this?
    # If we remove x, we break all active APs containing x.
    # The "cost" is 1/x. The "benefit" is breaking active_count[x] APs.
    # We want to maximize benefit per cost? Or just maximize benefit?
    # If we just maximize active_count[x], we might remove a large x (small cost) that breaks many APs.
    # This seems good.
    
    # However, we must ensure that we only consider elements in S.
    # And we must stop when no active APs remain.
    
    while active_aps:
        # Find x in S with max active_count[x]
        # If tie, pick the largest x (smallest reciprocal weight, cheapest to remove)
        # Or pick the smallest x?
        # Let's pick the largest x among those with max active_count.
        # Because we want to preserve small numbers (high reciprocal weight).
        
        best_x = None
        best_count = -1
        for x in range(1, N + 1):
            if x in S:
                if active_count[x] > best_count:
                    best_count = active_count[x]
                    best_x = x
                elif active_count[x] == best_count:
                    # Tie-breaker: prefer larger x (cheaper to remove)
                    if best_x is None or x > best_x:
                        best_x = x
        
        if best_x is None or best_count == 0:
            # This should not happen if active_aps is non-empty,
            # because every active AP has 3 elements in S, so their active_count >= 1.
            break
        
        # Remove best_x from S
        S.remove(best_x)
        
        # Update active_aps and active_count
        # For each active AP containing best_x:
        for ap_i in elem_to_ap[best_x]:
            if ap_i in active_aps:
                # Remove this AP
                active_aps.remove(ap_i)
                a, b, c = all_aps[ap_i - 1]
                # Decrement active_count for the other two elements
                for y in (a, b, c):
                    if y != best_x:
                        active_count[y] -= 1
        
        # Note: active_count[best_x] is no longer relevant since best_x is not in S.
        # But we don't need to reset it.
    
    return list(S)
