"""Strong edge colourings (induced matching partition) via branch-and-bound."""

def construct():
    import time
    import sys
    sys.setrecursionlimit(1000000)

    start_time = time.time()
    DEADLINE = 140.0

    def elapsed():
        return time.time() - start_time

    def instance(n, d, seed):
        import random
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        edge_index = {e: i for i, e in enumerate(edges)}

        # Build conflict graph: two edges conflict if they share a vertex or are joined by an edge
        conflict = [set() for _ in range(m)]
        for i in range(m):
            u, v = edges[i]
            for j in range(i + 1, m):
                w, x = edges[j]
                if u == w or u == x or v == w or v == x:
                    conflict[i].add(j)
                    conflict[j].add(i)
                else:
                    if (u, w) in edge_index or (u, x) in edge_index or (v, w) in edge_index or (v, x) in edge_index:
                        conflict[i].add(j)
                        conflict[j].add(i)

        conflict = [sorted(s) for s in conflict]
        deg_conflict = [len(c) for c in conflict]

        # Order vertices by degree descending (Welsh-Powell style)
        order = sorted(range(m), key=lambda i: -deg_conflict[i])
        pos_in_order = [0] * m
        for idx, v in enumerate(order):
            pos_in_order[v] = idx

        # Greedy initialization
        color = [-1] * m
        for i in order:
            taken = set()
            for j in conflict[i]:
                if color[j] != -1:
                    taken.add(color[j])
            c = 0
            while c in taken:
                c += 1
            color[i] = c
        best = max(color) + 1 if m > 0 else 0

        if elapsed() > DEADLINE:
            return [color[i] for i in range(m)]

        # B&B to find optimal or near-optimal coloring
        # We'll try K from best-1 down, using B&B for each K
        # For each K, we check if a valid K-coloring exists

        # Precompute earlier neighbors for B&B
        earlier_neighbors = [set() for _ in range(m)]
        for i in range(m):
            for j in conflict[i]:
                if pos_in_order[j] < pos_in_order[i]:
                    earlier_neighbors[i].add(j)
        earlier_neighbors = [sorted(s) for s in earlier_neighbors]

        # B&B function: try to color all vertices with K colors
        # Returns True if found, and fills color array
        def bb_search(K, color_arr):
            # color_arr is a list of length m, -1 means uncolored
            # We process in order[0], order[1], ...
            # Use recursion with backtracking
            # For speed, we use a stack-based approach or recursion with good pruning
            
            # For each vertex, the available colors are those not used by earlier neighbors
            # We'll use a recursive function
            
            def backtrack(pos):
                if pos == m:
                    return True
                if elapsed() > DEADLINE:
                    return False
                v = order[pos]
                # Find available colors
                taken = set()
                for j in earlier_neighbors[v]:
                    if color_arr[j] != -1:
                        taken.add(color_arr[j])
                # Try colors in a smart order: prefer colors that are already used (to minimize new colors)
                # But since we're fixing K, we just try 0..K-1
                for c in range(K):
                    if c not in taken:
                        color_arr[v] = c
                        if backtrack(pos + 1):
                            return True
                        color_arr[v] = -1
                return False
            
            return backtrack(0)

        # Try to improve: for K from best-1 down to 1
        # But this could be slow. Let's limit the search.
        # We'll try at most a few K values.
        
        # Actually, for these small graphs, let's just do a full B&B for K = best-1
        # If it finds a solution, great. Otherwise, we keep best.
        
        # To make B&B faster, use a better heuristic:
        # Order vertices by degree descending (already done)
        # For color assignment, try colors in order of how many vertices they can still be used for
        
        # Let's try K = best - 1
        if best > 1:
            test_color = [-1] * m
            if bb_search(best - 1, test_color):
                best = best - 1
                color = test_color[:]
                # Try to go even lower
                while best > 1:
                    if elapsed() > DEADLINE:
                        break
                    test_color2 = [-1] * m
                    if bb_search(best - 1, test_color2):
                        best = best - 1
                        color = test_color2[:]
                    else:
                        break

        return [color[i] for i in range(m)]

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        if elapsed() > DEADLINE:
            edges = instance(n, d, seed)
            out.append(list(range(len(edges))))
        else:
            out.append(solve_instance(n, d, seed))

    return out
