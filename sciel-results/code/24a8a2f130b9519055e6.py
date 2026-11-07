import random, time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(a, b):
    return (a, b) if a < b else (b, a)

def construct():
    t0 = time.time()
    deadline = t0 + 55.0
    n = P['n']
    full_edges = set(_edge_key(u, v) for u, v in P['edges'])
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    class TimeExceeded(Exception):
        pass

    def find_hamiltonian_cycle(adj_cur, start, deadline, rng):
        # randomized DFS with backtracking, returns list of vertices or None
        n2 = len(adj_cur)
        visited = [False] * n2
        path = [start]
        visited[start] = True
        step_count = [0]

        def dfs():
            step_count[0] += 1
            if step_count[0] % 5000 == 0 and time.time() > deadline:
                raise TimeExceeded()
            if len(path) == n2:
                return start in adj_cur[path[-1]]
            u = path[-1]
            nbrs = list(adj_cur[u])
            # Warnsdorff: prefer neighbors with fewer unvisited neighbors
            def warnsdorff_score(w):
                if visited[w]:
                    return 100
                return sum(1 for x in adj_cur[w] if not visited[x])
            nbrs.sort(key=lambda w: (warnsdorff_score(w), rng.random()))
            for w in nbrs:
                if not visited[w]:
                    visited[w] = True
                    path.append(w)
                    if dfs():
                        return True
                    path.pop()
                    visited[w] = False
            return False

        try:
            if dfs():
                return list(path)
        except TimeExceeded:
            return None
        return None

    def verify_cycles(cycles):
        used = set()
        for cyc in cycles:
            if len(cyc) < 3:
                return False
            if len(set(cyc)) != len(cyc):
                return False
            for i in range(len(cyc)):
                a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                k = _edge_key(a, b)
                if k in used:
                    return False
                used.add(k)
        return True

    def try_coloring_search(adj_cur, deadline, rng):
        """
        Try to find a 3-edge-coloring where each color class is a Hamiltonian cycle.
        We assign colors 0,1,2 to edges such that at each vertex, the three incident
        edges get distinct colors, and each color forms a single cycle covering all vertices.
        
        Approach: backtracking search over edges, assigning colors.
        At each vertex, once two edges are colored, the third is forced.
        We check connectivity at the end.
        """
        n2 = len(adj_cur)
        # Build edge list
        edge_list = []
        for u in range(n2):
            for v in sorted(adj_cur[u]):
                if u < v:
                    edge_list.append((u, v))
        
        # For each vertex, list of incident edge indices
        vert_edges = [[] for _ in range(n2)]
        for ei, (u, v) in enumerate(edge_list):
            vert_edges[u].append(ei)
            vert_edges[v].append(ei)
        
        # Color assignment: -1 means unassigned
        color = [-1] * len(edge_list)
        
        # For each vertex, track which colors are used
        vert_color_used = [set() for _ in range(n2)]
        
        # Find a good starting edge
        # We'll do backtracking with constraint propagation
        
        def is_consistent():
            for v in range(n2):
                if len(vert_color_used[v]) > 3:
                    return False
            return True
        
        def propagate(v):
            """Given vertex v, if two edges are colored, force the third."""
            eids = vert_edges[v]
            if len(eids) != 3:
                return True
            colors_at_v = []
            forced_eid = None
            forced_color = None
            for eid in eids:
                if color[eid] != -1:
                    colors_at_v.append(color[eid])
                else:
                    forced_eid = eid
            if len(colors_at_v) == 2:
                used = set(colors_at_v)
                for c in range(3):
                    if c not in used:
                        if forced_eid is not None and color[forced_eid] == -1:
                            color[forced_eid] = c
                            vert_color_used[v].add(c)
                            u, w = edge_list[forced_eid]
                            other = w if u == v else u
                            vert_color_used[other].add(c)
                            return propagate(other)
                        elif forced_eid is None:
                            return True
                        else:
                            return False
            elif len(colors_at_v) == 3:
                return True
            return True
        
        def check_cycles():
            """Check if each color class forms a single Hamiltonian cycle."""
            for c in range(3):
                # Build adjacency for color c
                adj_c = [set() for _ in range(n2)]
                for ei, (u, v) in enumerate(edge_list):
                    if color[ei] == c:
                        adj_c[u].add(v)
                        adj_c[v].add(u)
                # Check each vertex has degree 2
                for v in range(n2):
                    if len(adj_c[v]) != 2:
                        return False
                # Check connectivity: do a DFS from vertex 0
                visited = [False] * n2
                stack = [0]
                visited[0] = True
                count = 1
                while stack:
                    u = stack.pop()
                    for w in adj_c[u]:
                        if not visited[w]:
                            visited[w] = True
                            count += 1
                            stack.append(w)
                if count != n2:
                    return False
            return True
        
        def backtrack(ei):
            if time.time() > deadline:
                raise TimeExceeded()
            if ei == len(edge_list):
                return check_cycles()
            
            u, v = edge_list[ei]
            # Try each color
            for c in range(3):
                if c in vert_color_used[u] or c in vert_color_used[v]:
                    continue
                # Assign color c to edge ei
                color[ei] = c
                vert_color_used[u].add(c)
                vert_color_used[v].add(c)
                
                # Propagate
                ok = propagate(u) and propagate(v)
                if ok and backtrack(ei + 1):
                    return True
                
                # Undo
                color[ei] = -1
                vert_color_used[u].discard(c)
                vert_color_used[v].discard(c)
                # Undo propagation effects - this is tricky, we need to be careful
                # For now, let's just rely on the fact that we're doing a fresh search
                # Actually, propagation may have changed other colors, so we need to undo those too
                # This is getting complex. Let's use a different approach.
            
            return False
        
        # The propagation undo is complex. Let's use a simpler approach:
        # Just do random assignment and check at the end, with some heuristics.
        
        # Alternative: try random 3-colorings with local search
        for _ in range(100):
            if time.time() > deadline:
                raise TimeExceeded()
            # Random initial coloring
            for ei in range(len(edge_list)):
                color[ei] = rng.randrange(3)
            
            # Local search: try to fix degree constraints
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for v in range(n2):
                    eids = vert_edges[v]
                    colors = [color[e] for e in eids]
                    if len(set(colors)) != 3:
                        # Find an edge to recolor
                        from collections import Counter
                        cnt = Counter(colors)
                        # Find the color with highest count, recolor one of those edges
                        for e in eids:
                            if cnt[color[e]] >= 2:
                                old_c = color[e]
                                u, w = edge_list[e]
                                other = w if u == v else u
                                # Find a color not used at v or other
                                new_c = None
                                for c in range(3):
                                    if c != old_c and c not in [color[e2] for e2 in vert_edges[v] if e2 != e] and c not in [color[e2] for e2 in vert_edges[other] if e2 != e]:
                                        new_c = c
                                        break
                                if new_c is not None:
                                    color[e] = new_c
                                    improved = True
                                    break
            
            if check_cycles():
                # Extract cycles
                cycles = []
                for c in range(3):
                    adj_c = [set() for _ in range(n2)]
                    for ei, (u, v) in enumerate(edge_list):
                        if color[ei] == c:
                            adj_c[u].add(v)
                            adj_c[v].add(u)
                    # Extract cycle
                    cyc = [0]
                    prev = -1
                    curr = 0
                    while True:
                        nxts = [w for w in adj_c[curr] if w != prev]
                        if not nxts:
                            break
                        nxt = nxts[0]
                        if nxt == 0:
                            break
                        cyc.append(nxt)
                        prev = curr
                        curr = nxt
                    if len(cyc) == n2:
                        cycles.append(cyc)
                if len(cycles) == 3:
                    return cycles
        return None

    best = None
    rng = random.Random(12345)
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        # work on a copy of adjacency
        adj_cur = [set(s) for s in adj]
        cycles = []
        ok = True

        # First, try the coloring approach
        if attempt % 3 == 1:
            col_result = try_coloring_search(adj_cur, deadline, rng)
            if col_result is not None:
                return [[int(v) for v in cyc] for cyc in col_result]

        for c in range(3):
            # pick a start vertex
            if c == 0:
                start = rng.randrange(n)
            else:
                # pick vertex with highest remaining degree
                starts = list(range(n))
                starts.sort(key=lambda v: len(adj_cur[v]))
                # pick from top half
                top_half = starts[len(starts)//2:]
                start = top_half[rng.randrange(len(top_half))]

            cyc = find_hamiltonian_cycle(adj_cur, start, deadline, rng)
            if cyc is None:
                ok = False
                break
            cycles.append(cyc)
            # remove edges of this cycle
            for i in range(len(cyc)):
                a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                k = _edge_key(a, b)
                if k not in full_edges:
                    ok = False
                    break
                adj_cur[a].discard(b)
                adj_cur[b].discard(a)
            if not ok:
                break

        if ok and len(cycles) == 3 and verify_cycles(cycles):
            res = [[int(v) for v in cyc] for cyc in cycles]
            return res

        # track best partial result
        if len(cycles) >= 2 and verify_cycles(cycles):
            if best is None or len(cycles) > len(best):
                best = [[int(v) for v in cyc] for cyc in cycles]
            elif len(cycles) == len(best) and rng.random() < 0.1:
                # occasionally replace for diversity
                best = [[int(v) for v in cyc] for cyc in cycles]

        # if we have 1 cycle, track it too
        if len(cycles) == 1 and verify_cycles(cycles):
            if best is None or len(cycles) > len(best):
                best = [[int(v) for v in cyc] for cyc in cycles]

        # reset rng for diversity
        rng = random.Random(attempt * 7919 + 12345)

    if best is not None:
        return best

    # fallback: single Hamiltonian cycle
    adj_cur = [set(s) for s in adj]
    rng_fb = random.Random(42)
    cyc = find_hamiltonian_cycle(adj_cur, 0, deadline, rng_fb)
    if cyc is not None:
        return [[int(v) for v in cyc]]
    return [[int(v) for v in list(range(n))]]
