"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(edges, n, d):
        if not edges:
            return []
        
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
            
        M = len(edges)
        edge_set = set(edges)
        
        # Build Conflict Graph
        # Vertices 0..M-1 correspond to edges in `edges`
        conflict = [set() for _ in range(M)]
        for i in range(M):
            u, v = edges[i]
            for j in range(i + 1, M):
                x, y = edges[j]
                # Check if adjacent (share vertex)
                if u == x or u == y or v == x or v == y:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    continue
                # Check if joined (connected by an edge in G)
                # Edges are (min, max) tuples
                if (min(u,x), max(u,x)) in edge_set:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    continue
                if (min(u,y), max(u,y)) in edge_set:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    continue
                if (min(v,x), max(v,x)) in edge_set:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    continue
                if (min(v,y), max(v,y)) in edge_set:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    continue

        # Greedy Upper Bound
        def greedy_color():
            col = [-1] * M
            # Order by degree in conflict graph (descending)
            order = sorted(range(M), key=lambda x: len(conflict[x]), reverse=True)
            for v in order:
                used = set()
                for u in conflict[v]:
                    if col[u] != -1:
                        used.add(col[u])
                for c in range(M):
                    if c not in used:
                        col[v] = c
                        break
            return col

        greedy_sol = greedy_color()
        ub = max(greedy_sol) + 1
        lb = max(len(s) for s in conflict) + 1
        
        # Backtracking Solver with Dynamic Ordering and Symmetry Breaking
        coloring = [-1] * M
        
        def backtrack_dynamic(K):
            def get_next_vertex():
                min_avail = K + 1
                next_v = -1
                for v in range(M):
                    if coloring[v] == -1:
                        used = set()
                        for u in conflict[v]:
                            if coloring[u] != -1:
                                used.add(coloring[u])
                        avail_count = K - len(used)
                        if avail_count < min_avail:
                            min_avail = avail_count
                            next_v = v
                            if avail_count == 0:
                                return -1 # Fail signal
                return next_v

            def recurse():
                # Check if all colored
                if all(c != -1 for c in coloring):
                    return True
                
                v = get_next_vertex()
                if v == -1:
                    return False # Fail
                
                used = set()
                for u in conflict[v]:
                    if coloring[u] != -1:
                        used.add(coloring[u])
                
                # Symmetry breaking: 
                # The set of colors used so far must be {0, 1, ..., k-1}
                # If we introduce a new color, it must be k.
                colors_used_so_far = set(c for c in coloring if c != -1)
                next_new_color = len(colors_used_so_far)
                
                for c in range(K):
                    if c not in used:
                        # Enforce symmetry breaking
                        if c not in colors_used_so_far and c != next_new_color:
                            continue
                        
                        coloring[v] = c
                        if recurse():
                            return True
                        coloring[v] = -1
                return False
            
            return recurse()

        # Try K from lb to ub
        for K in range(lb, ub + 1):
            coloring = [-1] * M
            if backtrack_dynamic(K):
                return coloring
        
        # Fallback to greedy if something went wrong
        return greedy_sol

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        res = solve_instance(edges, n, d)
        out.append(res)
    return out
# EVOLVE-BLOCK-END
