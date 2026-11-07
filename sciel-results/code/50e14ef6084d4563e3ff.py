def construct():
    from random import Random
    import sys
    sys.setrecursionlimit(10000)

    def instance(n, k, p, seed):
        rng = Random(seed)
        cross = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross.append((u, v))
        return cross

    def solve_instance(n, k, p, seed):
        cross = instance(n, k, p, seed)
        
        # Edge IDs
        # C-edges: (u, w) with u < w < k
        # X-edges: (u, v) with u < k <= v
        
        c_edges = []
        c_edge_to_id = {}
        for u in range(k):
            for w in range(u + 1, k):
                c_edge_to_id[(u, w)] = len(c_edges)
                c_edges.append((u, w))
        
        x_edges = []
        x_edge_to_id = {}
        for (u, v) in cross:
            # u is in [0, k), v is in [k, n)
            x_edge_to_id[(u, v)] = len(x_edges)
            x_edges.append((u, v))
            
        num_c = len(c_edges)
        num_x = len(x_edges)
        total_edges = num_c + num_x
        
        if total_edges == 0:
            return []
            
        # Baseline: 1 clique for C, 1 for each cross edge
        baseline = 1 + num_x
        
        # Precompute neighbors in C for each v in I
        # N[v] is a set of u in [0, k) such that (u, v) is an edge
        N = [set() for _ in range(n)]
        for (u, v) in cross:
            N[v].add(u)
            
        # Map edge ID to (type, u, v)
        # type 0: C-edge, type 1: X-edge
        
        # We will use a bitmask for uncovered edges.
        # Since total_edges can be ~1700, Python int is fine.
        
        # To speed up, we can precompute for each edge the list of cliques that cover it?
        # No, generate on the fly.
        
        # Helper to get clique edges bitmask
        # Clique is defined by a set of vertices.
        
        def get_clique_edges(vertices):
            # vertices is a list of ints
            mask = 0
            vs = sorted(vertices)
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    u, w = vs[i], vs[j]
                    if u < k and w < k:
                        # C-edge
                        if u > w: u, w = w, u
                        eid = c_edge_to_id.get((u, w), -1)
                        if eid != -1:
                            mask |= (1 << eid)
                    elif u < k and w >= k:
                        # X-edge
                        eid = x_edge_to_id.get((u, w), -1)
                        if eid != -1:
                            mask |= (1 << (num_c + eid))
                    elif w < k and u >= k:
                        # X-edge (u is v, w is u)
                        eid = x_edge_to_id.get((w, u), -1)
                        if eid != -1:
                            mask |= (1 << (num_c + eid))
                    else:
                        # Both in I, no edge
                        pass
            return mask

        # Precompute mask for the big clique C
        mask_C = 0
        for eid in range(num_c):
            mask_C |= (1 << eid)
            
        # B&B State
        # uncovered: int bitmask
        # best: int
        
        best = baseline
        
        def dfs(uncovered, count):
            nonlocal best
            if uncovered == 0:
                if count < best:
                    best = count
                return
                
            # Lower bound
            # If C is fully uncovered, we can use C.
            if (uncovered & mask_C) == mask_C:
                # Using C covers all C-edges.
                # Remaining X-edges must be covered individually.
                rem_x = bin(uncovered >> num_c).count("1")
                lb = count + 1 + rem_x
                if lb >= best:
                    return
                # Branch: Use C
                new_uncovered = uncovered & ~mask_C
                dfs(new_uncovered, count + 1)
                
            # Pick an uncovered edge
            # Find the lowest set bit
            lsb = uncovered & -uncovered
            eid = lsb.bit_length() - 1
            
            if eid < num_c:
                # It's a C-edge
                u, w = c_edges[eid]
                # Candidates:
                # 1. Subsets of C containing u, w
                # 2. S U {v} containing u, w
                
                # Try C if not already handled (handled above if fully uncovered)
                # If not fully uncovered, we can't use C.
                
                # Generate cliques containing (u, w)
                # 1. Pure C subsets
                # We can try adding vertices to {u, w}
                # To limit, try maximal extensions?
                
                # 2. Mixed cliques
                # Find v in I such that u, w in N[v]
                for v in range(k, n):
                    if u in N[v] and w in N[v]:
                        # Clique S U {v} where S contains u, w
                        # S is a subset of N[v] containing u, w
                        # We can try S = N[v] (maximal)
                        # Or S = {u, w}
                        # Let's try maximal first for pruning
                        S_max = list(N[v])
                        mask = get_clique_edges(S_max + [v])
                        if (mask & uncovered) == mask:
                            dfs(uncovered & ~mask, count + 1)
                        
                        # Also try smaller sets?
                        # If maximal doesn't work, maybe smaller does.
                        # But branching on all subsets is expensive.
                        # We can try {u, w, v}
                        mask_small = get_clique_edges([u, w, v])
                        if (mask_small & uncovered) == mask_small:
                            dfs(uncovered & ~mask_small, count + 1)
                            
            else:
                # It's an X-edge
                x_eid = eid - num_c
                u, v = x_edges[x_eid]
                # Candidates:
                # 1. {u, v}
                mask_edge = (1 << eid)
                dfs(uncovered & ~mask_edge, count + 1)
                
                # 2. S U {v} containing u
                # S subset of N[v] containing u
                # Try maximal
                S_max = list(N[v])
                mask = get_clique_edges(S_max + [v])
                if (mask & uncovered) == mask:
                    dfs(uncovered & ~mask, count + 1)
                    
                # Try {u, v} is already done.
                # Try intermediate?
                # Maybe just maximal and edge is enough for small instances?
                
        dfs((1 << total_edges) - 1, 0)
        
        # Reconstruct solution? 
        # The task requires returning the list of cliques.
        # The B&B above only finds the count.
        # We need to store the path.
        
        # Let's modify to store the best path.
        best_cliques = None
        best_val = baseline
        
        def dfs2(uncovered, count, path):
            nonlocal best_cliques, best_val
            if uncovered == 0:
                if count < best_val:
                    best_val = count
                    best_cliques = list(path)
                return
                
            if (uncovered & mask_C) == mask_C:
                rem_x = bin(uncovered >> num_c).count("1")
                if count + 1 + rem_x >= best_val:
                    return
                # Use C
                new_path = path + [list(range(k))]
                dfs2(uncovered & ~mask_C, count + 1, new_path)
                
            lsb = uncovered & -uncovered
            eid = lsb.bit_length() - 1
            
            if eid < num_c:
                u, w = c_edges[eid]
                for v in range(k, n):
                    if u in N[v] and w in N[v]:
                        S_max = list(N[v])
                        mask = get_clique_edges(S_max + [v])
                        if (mask & uncovered) == mask:
                            dfs2(uncovered & ~mask, count + 1, path + [S_max + [v]])
                        
                        mask_small = get_clique_edges([u, w, v])
                        if (mask_small & uncovered) == mask_small:
                            dfs2(uncovered & ~mask_small, count + 1, path + [[u, w, v]])
            else:
                x_eid = eid - num_c
                u, v = x_edges[x_eid]
                mask_edge = (1 << eid)
                dfs2(uncovered & ~mask_edge, count + 1, path + [[u, v]])
                
                S_max = list(N[v])
                mask = get_clique_edges(S_max + [v])
                if (mask & uncovered) == mask:
                    dfs2(uncovered & ~mask, count + 1, path + [S_max + [v]])
                    
        dfs2((1 << total_edges) - 1, 0, [])
        
        if best_cliques is None:
            # Fallback to baseline
            cliques = [list(range(k))]
            for (u, v) in cross:
                cliques.append([u, v])
            return cliques
        
        return best_cliques

    result = []
    for (n, k, p, seed) in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        cliques = solve_instance(n, k, p, seed)
        # Ensure types are int and list
        clean_cliques = []
        for c in cliques:
            clean_cliques.append([int(x) for x in c])
        result.append(clean_cliques)
    return result
