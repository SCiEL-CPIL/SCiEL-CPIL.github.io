def construct():
    N = 128
    deadline = time.time() + 50  # safety margin

    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            gcd_table[a][b] = gcd(a, b)
            gcd_table[b][a] = gcd_table[a][b]

    # Precompute conflict pairs
    conflict_pair = [dict() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_table[a][b]
            cset = set()
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if gcd_table[a][c] == gab and gcd_table[b][c] == gab:
                    cset.add(c)
            if cset:
                conflict_pair[a][b] = cset
                conflict_pair[b][a] = cset

    def can_add(selected, x):
        # selected is a set
        for a in selected:
            if a >= x:
                continue
            if a in conflict_pair:
                for b in conflict_pair[a]:
                    if b in selected and b < x:
                        if x in conflict_pair[a][b]:
                            return False
        return True

    def greedy_build(order):
        selected = set()
        for x in order:
            if can_add(selected, x):
                selected.add(x)
        return selected

    def local_search(selected, time_limit=5.0):
        selected = set(selected)
        start = time.time()
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            # Try adding
            candidates = [x for x in range(1, N + 1) if x not in selected]
            random.shuffle(candidates)
            for x in candidates:
                if can_add(selected, x):
                    selected.add(x)
                    improved = True
                    break
            
            if not improved:
                # Try 1-swap
                sel_list = list(selected)
                random.shuffle(sel_list)
                for x in sel_list:
                    remaining = selected - {x}
                    for y in range(1, N + 1):
                        if y not in remaining:
                            if can_add(remaining, y):
                                # Check if we can add more to new_sel
                                new_sel = remaining | {y}
                                # Try to extend
                                ext_candidates = [z for z in range(1, N + 1) if z not in new_sel]
                                random.shuffle(ext_candidates)
                                for z in ext_candidates:
                                    if can_add(new_sel, z):
                                        selected = new_sel | {z}
                                        improved = True
                                        break
                                if improved:
                                    break
                    if improved:
                        break
        return selected

    best = set()
    
    # Strategy 1: Greedy by number
    result = greedy_build(list(range(1, N + 1)))
    result = local_search(result, time_limit=5.0)
    if len(result) > len(best):
        best = set(result)

    # Strategy 2: Greedy by largest
    result = greedy_build(list(range(N, 0, -1)))
    result = local_search(result, time_limit=5.0)
    if len(result) > len(best):
        best = set(result)

    # Strategy 3: Random restarts
    while time.time() < deadline:
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        result = greedy_build(random_order)
        time_left = deadline - time.time()
        if time_left < 2: break
        ts = min(2.0, max(0.5, time_left * 0.3))
        result = local_search(result, time_limit=ts)
        if len(result) > len(best):
            best = set(result)

    return list(best) if best else [1]
