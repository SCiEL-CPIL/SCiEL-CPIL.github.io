"""Strong edge colouring via backtracking with forward checking and bitmasks."""
# EVOLVE-BLOCK-START
def construct():
    import random
    from collections import defaultdict

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        # Try many times to build a bounded-degree graph
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e in edges:
                continue
            if degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def strong_edge_coloring(edges, n, max_deg):
        if not edges:
            return []
        
        m = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Precompute conflict masks
        # conflict_mask[i] is an integer where bit j is set if edge i conflicts with edge j
        conflict_mask = [0] * m
        for i in range(m):
            u, v = edges[i]
            for j in range(i + 1, m):
                x, y = edges[j]
                # Check if they share a vertex
                if x == u or x == v or y == u or y == v:
                    conflict_mask[i] |= (1 << j)
                    conflict_mask[j] |= (1 << i)
                    continue
                # Check if joined by an edge
                if x in adj[u] or x in adj[v] or y in adj[u] or y in adj[v]:
                    conflict_mask[i] |= (1 << j)
                    conflict_mask[j] |= (1 << i)
        
        # Order edges by descending degree in conflict graph
        # degree[i] = number of 1s in conflict_mask[i]
        degs = [bin(conflict_mask[i]).count('1') for i in range(m)]
        order = sorted(range(m), key=lambda i: -degs[i])
        
        # Greedy solution to get an upper bound
        color_greedy = [-1] * m
        used_masks = [] # list of masks for each color used so far in greedy
        for i in order:
            placed = False
            for c_idx, mask in enumerate(used_masks):
                if (conflict_mask[i] & mask) == 0:
                    color_greedy[i] = c_idx
                    used_masks[c_idx] = mask | (1 << i)
                    placed = True
                    break
            if not placed:
                color_greedy[i] = len(used_masks)
                used_masks.append(1 << i)
        
        upper_bound = len(used_masks)
        
        # Lower bound: max degree in conflict graph + 1
        lower_bound = max(degs) + 1 if degs else 1
        
        # Backtracking solver
        # We try to find a coloring with K colors
        def solve_with_k(K):
            # color_masks[c] is a bitmask of edges assigned color c
            color_masks = [0] * K
            assignment = [-1] * m
            nodes_visited = [0]
            max_nodes = 500000 # Limit to prevent timeout
            
            def backtrack(idx):
                if nodes_visited[0] > max_nodes:
                    return False
                nodes_visited[0] += 1
                
                if idx == m:
                    return True
                
                e = order[idx]
                # Try each color
                for c in range(K):
                    if (conflict_mask[e] & color_masks[c]) == 0:
                        # Assign color c to edge e
                        assignment[e] = c
                        color_masks[c] |= (1 << e)
                        
                        if backtrack(idx + 1):
                            return True
                        
                        # Undo
                        color_masks[c] ^= (1 << e)
                        assignment[e] = -1
                return False
            
            return backtrack(0), assignment

        # Try K from lower_bound to upper_bound
        # We want the minimum K, so we start from lower_bound
        # However, if lower_bound is too low, it might take long to fail.
        # We can start from lower_bound and go up.
        # If it fails for a long time, we might switch to a heuristic, but let's try first.
        
        final_assignment = None
        for K in range(lower_bound, upper_bound + 1):
            # Optimization: If K is small, backtracking might be slow.
            # But m is small.
            found, assignment = solve_with_k(K)
            if found:
                final_assignment = assignment
                break
        
        if final_assignment is None:
            # Fallback to greedy if backtracking fails or is too slow (shouldn't happen for small K)
            final_assignment = color_greedy
        
        return [final_assignment[edge_to_idx[e]] for e in edges]

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        if not edges:
            out.append([])
            continue
        colors = strong_edge_coloring(edges, n, d)
        out.append(colors)
    return out
# EVOLVE-BLOCK-END
