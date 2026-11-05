def construct():
    import time
    start_time = time.time()
    DEADLINE = 58  # Safety margin before 60s limit

    N = 180

    # Precompute all forbidden triples: (a, b, c) with a < b < c and a*(b+c) == b*c
    # For each element, store its conflict pairs
    conflict_pairs = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            diff = b - a
            if diff == 0:
                continue
            num = a * b
            if num % diff != 0:
                continue
            c = num // diff
            if c > b and c <= N:
                conflict_pairs[a].append((b, c))
                conflict_pairs[b].append((a, c))
                conflict_pairs[c].append((a, b))

    def greedy_add(in_set, result, order):
        """Add elements in given order if possible."""
        for x in order:
            ok = True
            for (u, v) in conflict_pairs[x]:
                if in_set[u] and in_set[v]:
                    ok = False
                    break
            if ok:
                in_set[x] = True
                result.add(x)

    def try_add_all(in_set, result, candidates):
        """Try to add all possible candidates repeatedly until no more can be added."""
        changed = True
        while changed:
            changed = False
            for y in candidates:
                if in_set[y]:
                    continue
                ok = True
                for (u, v) in conflict_pairs[y]:
                    if in_set[u] and in_set[v]:
                        ok = False
                        break
                if ok:
                    in_set[y] = True
                    result.add(y)
                    changed = True

    # Multiple greedy strategies
    best_size = 0
    best_result = None
    best_in_set = None

    # Strategy 1: Start with high range, add lower
    in_set = [False] * (N + 1)
    result = set()
    for i in range(91, N + 1):
        in_set[i] = True
        result.add(i)
    greedy_add(in_set, result, range(90, 0, -1))
    try_add_all(in_set, result, range(1, N + 1))
    if len(result) > best_size:
        best_size = len(result)
        best_result = set(result)
        best_in_set = in_set[:]

    # Strategy 2: Start with high range, add lower in ascending order
    in_set = [False] * (N + 1)
    result = set()
    for i in range(91, N + 1):
        in_set[i] = True
        result.add(i)
    greedy_add(in_set, result, range(1, 91))
    try_add_all(in_set, result, range(1, N + 1))
    if len(result) > best_size:
        best_size = len(result)
        best_result = set(result)
        best_in_set = in_set[:]

    # Strategy 3: Pure greedy from high to low
    in_set = [False] * (N + 1)
    result = set()
    greedy_add(in_set, result, range(N, 0, -1))
    try_add_all(in_set, result, range(1, N + 1))
    if len(result) > best_size:
        best_size = len(result)
        best_result = set(result)
        best_in_set = in_set[:]

    # Strategy 4: Pure greedy from low to high
    in_set = [False] * (N + 1)
    result = set()
    greedy_add(in_set, result, range(1, N + 1))
    try_add_all(in_set, result, range(1, N + 1))
    if len(result) > best_size:
        best_size = len(result)
        best_result = set(result)
        best_in_set = in_set[:]

    # Strategy 5: Start with multiples of some structure
    # Elements that are unlikely to be in triples: large numbers
    in_set = [False] * (N + 1)
    result = set()
    for i in range(121, N + 1):
        in_set[i] = True
        result.add(i)
    greedy_add(in_set, result, range(120, 0, -1))
    try_add_all(in_set, result, range(1, N + 1))
    if len(result) > best_size:
        best_size = len(result)
        best_result = set(result)
        best_in_set = in_set[:]

    # Use the best result for local search
    in_set = best_in_set[:]
    result = set(best_result)

    # Local search: try to increase size by removing 1 and adding 2+
    improved = True
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        for x in sorted(result):
            if time.time() - start_time > DEADLINE:
                break
            
            # Remove x
            in_set[x] = False
            result.discard(x)
            
            # Try to add as many as possible
            added = []
            for y in range(1, N + 1):
                if in_set[y]:
                    continue
                ok = True
                for (u, v) in conflict_pairs[y]:
                    if in_set[u] and in_set[v]:
                        ok = False
                        break
                if ok:
                    in_set[y] = True
                    result.add(y)
                    added.append(y)
            
            if len(added) > 1:
                # Net gain, keep it
                improved = True
            elif len(added) == 1:
                # Swap, might help later but no immediate gain
                # Keep it and mark improved to continue searching
                improved = True
            else:
                # No improvement, restore x
                in_set[x] = True
                result.add(x)

    # Final polish: try to add any remaining
    for y in range(1, N + 1):
        if not in_set[y]:
            ok = True
            for (u, v) in conflict_pairs[y]:
                if in_set[u] and in_set[v]:
                    ok = False
                    break
            if ok:
                in_set[y] = True
                result.add(y)

    return list(result)
