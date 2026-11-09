def _local_search_fast(subset):
    S = set(subset)
    reachable = {0}
    # Rebuild reachable sums for initial S
    for x in S:
        new_sums = {s + x for s in reachable if s + x <= 200}
        if 200 in new_sums:
            # Should not happen if S is valid, but safety check
            pass 
        reachable |= new_sums
        
    changed = True
    while changed:
        changed = False
        for x in range(1, 101):
            if x in S:
                continue
            # Check if adding x creates sum 200
            # This happens if 200-x is a subset sum of S
            if (200 - x) in reachable:
                continue
            # Safe to add
            S.add(x)
            new_sums = {s + x for s in reachable if s + x <= 200}
            reachable |= new_sums
            changed = True
    return list(S)
