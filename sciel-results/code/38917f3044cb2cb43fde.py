import time
import sys

def construct():
    start_time = time.time()
    deadline = 150.0
    
    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        adj = [[False] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = True
                    adj[v][u] = True
        return adj

    def bron_kerbosch(adj, n):
        # Returns list of maximal cliques (list of vertex indices)
        cliques = []
        R = set()
        P = set(range(n))
        X = set()
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(list(R))
                return
            
            # Pivoting: choose u in P union X with max degree in P
            # Simple pivot: just pick first one if any
            if P:
                u = next(iter(P))
            elif X:
                u = next(iter(X))
            else:
                return
                
            # P \ N(u)
            P_minus_Nu = [v for v in P if not adj[u][v]]
            
            for v in P_minus_Nu:
                bk(R | {v}, P & set(adj[v]), X & set(adj[v]))
                P.remove(v)
                X.add(v)
        
        bk(R, P, X)
        return cliques

    def solve_hitting_set(cliques, n):
        if not cliques:
            return []
        
        # Convert cliques to bitmasks of vertices
        clique_masks = []
        for c in cliques:
            mask = 0
            for v in c:
                mask |= (1 << v)
            clique_masks.append(mask)
        
        # Precompute for each vertex, which cliques it covers
        # vertex_clique_mask[v] is a bitmask of cliques (indices) that contain v
        vertex_clique_mask = [0] * n
        for i, cm in enumerate(clique_masks):
            for v in range(n):
                if (cm >> v) & 1:
                    vertex_clique_mask[v] |= (1 << i)
        
        total_cliques = len(clique_masks)
        all_cliques_mask = (1 << total_cliques) - 1
        
        best_size = n # Upper bound
        best_set = []
        
        # Lower bound function
        def lower_bound(uncovered_mask):
            if uncovered_mask == 0:
                return 0
            # Max coverage of any single vertex among uncovered cliques
            max_cov = 0
            # We can just use 1, or calculate max. 
            # For speed, let's just use 1. Or precompute max possible.
            # Actually, a better LB is: ceil(popcount(uncovered) / max_cov_any_vertex)
            # But max_cov depends on which vertices are still "available". 
            # In this branching scheme, we don't "remove" vertices from availability, 
            # we just pick them. Once picked, they are in the set.
            # The branching is on "which vertex to pick to cover this specific clique".
            # So all vertices are always candidates? 
            # No, if we pick a vertex, we don't pick it again.
            # But we don't "forbid" other vertices.
            # So the LB is simply ceil(popcount(uncovered) / max_possible_coverage).
            # max_possible_coverage is the max number of cliques any vertex can cover.
            return 1 # Conservative LB to be safe and fast. 
            # We can improve this later if needed.
        
        # Recursive solver
        # uncovered_mask: bitmask of cliques not yet covered
        # current_size: number of vertices picked
        # picked_mask: bitmask of vertices already picked (to avoid re-picking? No, picking same vertex twice is useless)
        # Actually, we don't need picked_mask for correctness, just for efficiency if we want to ensure we don't pick same vertex.
        # But since we branch on vertices in a specific clique, and remove covered cliques, 
        # we might branch on a vertex already picked? 
        # If a vertex is already picked, its cliques are already removed from uncovered.
        # So if a clique is in uncovered, none of its vertices are picked yet.
        # So we don't need to check if vertex is already picked.
        
        def dfs(uncovered_mask, current_size):
            nonlocal best_size, best_set
            if uncovered_mask == 0:
                if current_size < best_size:
                    best_size = current_size
                    # Reconstruct set? We can pass the set or store path.
                    # For now, just track size. We need the actual set to return.
                    # We can store the path in a list.
                return
            
            if current_size >= best_size:
                return
            
            # Heuristic: Pick a clique to branch on
            # Find an uncovered clique with minimum number of vertices?
            # Or just the first one.
            # Let's find the first set bit in uncovered_mask
            c_idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
            c_mask = clique_masks[c_idx]
            
            # Get vertices in this clique
            vertices = []
            for v in range(n):
                if (c_mask >> v) & 1:
                    vertices.append(v)
            
            # Order vertices by coverage (descending) to find good solutions faster
            # coverage = number of uncovered cliques this vertex hits
            def get_coverage(v):
                # Count bits in (vertex_clique_mask[v] & uncovered_mask)
                m = vertex_clique_mask[v] & uncovered_mask
                return bin(m).count('1')
            
            vertices.sort(key=get_coverage, reverse=True)
            
            for v in vertices:
                new_uncovered = uncovered_mask & ~vertex_clique_mask[v]
                dfs(new_uncovered, current_size + 1)
                
                # Time check
                if time.time() > deadline - 5:
                    return

        # Initial call
        dfs(all_cliques_mask, 0)
        
        # The above DFS only tracks size. We need to return the actual set.
        # Let's modify to track the set.
        # To avoid copying lists, we can use a stack or global array.
        # Given small size, copying is fine.
        
        best_set = []
        
        def dfs2(uncovered_mask, current_size, current_set):
            nonlocal best_size, best_set
            if uncovered_mask == 0:
                if current_size < best_size:
                    best_size = current_size
                    best_set = current_set.copy()
                return
            
            if current_size >= best_size:
                return
            
            c_idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
            c_mask = clique_masks[c_idx]
            
            vertices = []
            for v in range(n):
                if (c_mask >> v) & 1:
                    vertices.append(v)
            
            def get_coverage(v):
                m = vertex_clique_mask[v] & uncovered_mask
                return bin(m).count('1')
            
            vertices.sort(key=get_coverage, reverse=True)
            
            for v in vertices:
                new_uncovered = uncovered_mask & ~vertex_clique_mask[v]
                dfs2(new_uncovered, current_size + 1, current_set + [v])
                
                if time.time() > deadline - 5:
                    return

        dfs2(all_cliques_mask, 0, [])
        return best_set

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 10:
            # Fallback to seed solution if running out of time
            adj = random_graph(n, p, seed)
            touched = set()
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u][v]:
                        touched.add(u)
                        touched.add(v)
            results.append(sorted(touched))
            continue
            
        adj = random_graph(n, p, seed)
        cliques = bron_kerbosch(adj, n)
        # Filter cliques of size >= 2
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            results.append([])
        else:
            solution = solve_hitting_set(cliques, n)
            results.append(sorted(solution))
            
    return results
