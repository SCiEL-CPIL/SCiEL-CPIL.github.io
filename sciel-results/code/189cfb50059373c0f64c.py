import time

def construct():
    deadline = time.time() + 55.0
    N = 100
    T = 200
    universe = list(range(1, N + 1))

    # Greedy descending seed (same as original but we will polish)
    def greedy_desc():
        selected, reachable = [], {0}
        for x in range(N, 0, -1):
            new = {s + x for s in reachable if s + x <= T}
            if T not in new:
                selected.append(x)
                reachable |= new
        return selected

    best = greedy_desc()
    best_len = len(best)

    def is_valid(s):
        # subset sum check for T
        # Use bitset via Python ints for speed: bit i set iff sum i reachable
        bits = 1  # bit 0
        for x in s:
            bits |= (bits << x)
        return ((bits >> T) & 1) == 0

    # Verify seed
    if not is_valid(best):
        # Fallback to a trivially valid set
        best = list(range(1, 70))
        best_len = len(best)

    def try_improve(s):
        nonlocal best, best_len
        s = list(s)
        cur_len = len(s)
        # Build bitset
        bits = 1
        for x in s:
            bits |= (bits << x)
        # Precompute for each element whether removing it still avoids T? Not needed for insertion.
        # Candidate insertions: elements not in s, sorted descending
        sset = set(s)
        candidates = [x for x in range(N, 0, -1) if x not in sset]
        # Try to insert one at a time greedily
        changed = True
        while changed:
            changed = False
            for x in candidates:
                if time.time() > deadline:
                    return
                # Check if adding x creates T
                # new bits: bits | (bits << x). Check if bit T is set in (bits << x) & ~bits? 
                # Actually we need: does there exist s in S such that s + x = T? 
                # i.e., is bit (T-x) set in bits?
                if x <= T and ((bits >> (T - x)) & 1):
                    continue
                # Safe to add
                s.append(x)
                sset.add(x)
                bits |= (bits << x)
                candidates.remove(x)
                cur_len += 1
                changed = True
                if cur_len > best_len:
                    best = list(s)
                    best_len = cur_len
            # After one full pass, if no change, stop (greedy local optimum for insertion)
            # But we might have added something that blocks nothing else, so loop again is fine but 
            # the 'for' loop already iterates all remaining candidates. If any was added, 
            # it might unblock others? No, adding elements only creates more subset sums, 
            # making it harder to add more. So a single pass of trying to add is sufficient 
            # if we check in descending order? Actually, adding a large number might make 
            # a smaller number invalid if they sum to T. But if we check descending, 
            # and we add a large one, it might block a smaller one that we haven't tried yet.
            # But if we don't add the large one, the smaller one might be addable.
            # The current loop tries ALL candidates in the current list. 
            # If we add x, we remove it from candidates. 
            # This is a greedy heuristic.

    try_improve(best)

    # Local search: try swapping one element for another to increase size?
    # Swapping keeps size same. We want to increase size.
    # So we need to find a way to add an element by removing a "blocking" element and adding two others?
    # Or simply the greedy insertion above might have missed a combo.
    
    # Let's try a different greedy order: Ascending
    def greedy_asc():
        selected, reachable = [], {0}
        for x in range(1, N + 1):
            new = {s + x for s in reachable if s + x <= T}
            if T not in new:
                selected.append(x)
                reachable |= new
        return selected

    s2 = greedy_asc()
    if len(s2) > best_len and is_valid(s2):
        best = s2
        best_len = len(s2)
    try_improve(best)

    # Try "skip small, take large"
    def greedy_mixed():
        # Start with large, then fill small
        selected, reachable = [], {0}
        # First pass: large
        for x in range(N, 0, -1):
            new = {s + x for s in reachable if s + x <= T}
            if T not in new:
                selected.append(x)
                reachable |= new
        # Second pass: try adding small ones that weren't added
        sset = set(selected)
        for x in range(1, N + 1):
            if x in sset:
                continue
            new = {s + x for s in reachable if s + x <= T}
            if T not in new:
                selected.append(x)
                reachable |= new
                sset.add(x)
        return selected

    s3 = greedy_mixed()
    if len(s3) > best_len and is_valid(s3):
        best = s3
        best_len = len(s3)
    try_improve(best)

    return best
