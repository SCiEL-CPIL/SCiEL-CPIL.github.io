<<<<<<< SEARCH
def construct():
    t0 = time.time()
    DL = t0 + 70.0

    # ---- Proven 58-edge construction (40 rim + 17 fan chords + 1 bridge) ----
    rim = set()
    for i in range(N):
        a, b = i, (i + 1) % N
        rim.add((min(a, b), max(a, b)))

    fan_chords = set()
    for d in range(2, 19):
        fan_chords.add((0, d))

    bridge = (2, 21)
    e58 = frozenset(rim | fan_chords | {bridge})

    # Verify the known-good construction
    cycs58 = _build_fan_cycles(set(e58))
    if cycs58 is not None:
        best_edges = sorted(e58)
        best_cycs = cycs58
    else:
        # Should never happen, but fallback
        best_edges = sorted(rim | fan_chords | {bridge})
        best_cycs = []
        for k in range(3, N + 1):
            best_cycs.append(list(range(k)))
=======
def construct():
    t0 = time.time()
    DL = t0 + 70.0

    # ---- Proven 58-edge construction (40 rim + 17 fan chords + 1 bridge) ----
    rim = set()
    for i in range(N):
        a, b = i, (i + 1) % N
        rim.add((min(a, b), max(a, b)))

    fan_chords = set()
    for d in range(2, 19):
        fan_chords.add((0, d))

    bridge = (2, 21)
    e58 = frozenset(rim | fan_chords | {bridge})

    # Verify the known-good construction
    cycs58 = _build_fan_cycles(set(e58))
    if cycs58 is not None:
        best_edges = sorted(e58)
        best_cycs = cycs58
    else:
        best_edges = sorted(rim | fan_chords | {bridge})
        best_cycs = []
        for k in range(3, N + 1):
            best_cycs.append(list(range(k)))

    # ---- Greedy edge removal: try to remove non-rim edges one by one ----
    def _find_cycle_in_graph(k, edge_set, deadline, start=0):
        """Find a k-cycle using DFS with aggressive pruning."""
        adj = {}
        for u, v in edge_set:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        # For small k (<=25), use DFS
        if k > 25:
            return None  # Too slow for DFS
        
        def dfs(path, visited):
            if time.time() > deadline:
                return None
            if len(path) == k:
                if path[0] in adj.get(path[-1], set()):
                    return list(path)
                return None
            # Pruning: not enough vertices
            if len(path) + (N - len(visited)) < k:
                return None
            # Pruning: degree check - last vertex must have unvisited neighbors
            last = path[-1]
            unvisited_neighbors = [nb for nb in adj.get(last, set()) if nb not in visited]
            if not unvisited_neighbors:
                return None
            for nb in unvisited_neighbors:
                visited.add(nb)
                path.append(nb)
                result = dfs(path, visited)
                if result is not None:
                    return result
                path.pop()
                visited.remove(nb)
            return None
        
        for s in range(start, min(start + 3, N)):
            result = dfs([s], {s})
            if result is not None:
                return result
        return None

    def _try_all_cycles(edge_set, deadline):
        """Try to find all 38 cycles in the given edge set."""
        cset = set(edge_set)
        cycs = []
        for k in range(3, N + 1):
            if time.time() > deadline:
                return None
            if k == N:
                c = list(range(N))
            elif k <= 19:
                c = list(range(k))
            else:
                # Try standard pattern first
                if k == 20:
                    c = list(range(2, 22))
                elif k == 21:
                    c = [0, 2] + list(range(21, 40))
                elif k == 22:
                    c = [2, 1, 0] + list(range(39, 20, -1))
                else:
                    dp = 41 - k
                    c = [0] + list(range(39, dp - 1, -1))
                
                if not _verify_cyc(c, cset):
                    # Try to find alternative
                    c = _find_cycle_in_graph(k, cset, deadline)
                
                if c is None or not _verify_cyc(c, cset):
                    return None
            cycs.append(c)
        return cycs

    # Try removing each non-rim edge
    non_rim = sorted(fan_chords | {bridge})
    for edge_to_remove in non_rim:
        if time.time() > DL - 10:
            break
        cand = set(e58)
        cand.discard(edge_to_remove)
        cycs = _try_all_cycles(cand, DL - 10)
        if cycs is not None:
            if len(cand) < len(best_edges):
                best_edges = sorted(cand)
                best_cycs = cycs
                e58 = frozenset(cand)  # Update base for next iteration
>>>>>>> REPLACE
