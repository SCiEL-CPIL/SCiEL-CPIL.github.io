def construct():
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import coo_matrix
    from random import Random

    def instance_edges(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def get_maximal_cliques(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = []
        def bronk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            if P:
                u = max(P | X, key=lambda u: len(P & adj[u]))
                candidates = P - adj[u]
            else:
                candidates = P
                
            for v in list(candidates):
                bronk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
                
        bronk(set(), set(range(n)), set())
        return cliques

    def minimize_solution(selected, vert_to_cliques, n, num_cliques):
        # Convert selected to set for fast lookup
        sel = set(selected)
        # Check validity and minimality
        # We want to remove vertices if possible
        changed = True
        while changed:
            changed = False
            for v in list(sel):
                # Check if removing v leaves all cliques covered
                # A clique is covered if it contains any vertex in sel
                # If we remove v, cliques that ONLY contained v (among sel) become uncovered
                # We need to check if there is any clique C such that C ∩ sel == {v}
                # If such a clique exists, we cannot remove v.
                # Otherwise, we can remove v.
                can_remove = True
                # Optimization: Pre-compute for each vertex v, the set of cliques where v is the ONLY selected vertex?
                # This is expensive to update dynamically.
                # Simple check:
                for c_idx in vert_to_cliques[v]:
                    C = None # We need the clique content or just check intersection
                    # We don't have clique content in vert_to_cliques.
                    # We need to pass cliques or a way to check.
                    pass
                # Let's restructure: 
                # We need to know for each clique, how many selected vertices it has.
                # Maintain a count array `clique_counts`.
                # Initialize `clique_counts` for current `sel`.
                # For each v in sel, for each c in vert_to_cliques[v], count[c]++.
                # To test removing v:
                #   For each c in vert_to_cliques[v]:
                #      if count[c] == 1:
                #          can_remove = False
                #          break
                #   if can_remove:
                #      sel.remove(v)
                #      for c in vert_to_cliques[v]: count[c]--
                #      changed = True
                #      break # restart loop as counts changed
                pass
        # The logic above is a bit messy to implement in a small snippet without passing extra state.
        # Let's implement a simpler version inside the function.
        pass

    def solve_set_cover(cliques, n):
        if not cliques:
            return []
        
        # 1. Build vertex-to-clique mapping
        vert_to_cliques = [[] for _ in range(n)]
        for i, C in enumerate(cliques):
            for v in C:
                vert_to_cliques[v].append(i)
        
        # 2. Prune Dominated Vertices
        # A vertex u is dominated by v if vert_to_cliques[u] is a subset of vert_to_cliques[v]
        # We keep v, remove u.
        # If they are identical, keep the one with smaller index? Or just one.
        active = [True] * n
        for u in range(n):
            if not active[u]:
                continue
            for v in range(n):
                if u == v or not active[v]:
                    continue
                # Check if vert_to_cliques[u] is subset of vert_to_cliques[v]
                # Use sets for faster check
                if set(vert_to_cliques[u]).issubset(set(vert_to_cliques[v])):
                    active[u] = False
                    break
        
        active_vertices = [i for i in range(n) if active[i]]
        
        # If all vertices are dominated? (Should not happen, at least one maximal vertex remains)
        if not active_vertices:
            active_vertices = [0]
            
        # 3. Build MILP for active vertices
        # Variables: x_0 ... x_{m-1} corresponding to active_vertices
        m = len(active_vertices)
        c = np.ones(m)
        
        # Constraints: For each clique, sum of x_v for v in clique (and active) >= 1
        # Note: Every clique must contain at least one active vertex.
        # If a clique C contains only dominated vertices, let u in C. u dominated by v => v in C.
        # So C contains v (active).
        
        row_indices = []
        col_indices = []
        values = []
        
        for c_idx, C in enumerate(cliques):
            # Find active vertices in this clique
            active_in_C = [v for v in C if active[v]]
            if not active_in_C:
                # This shouldn't happen, but if it does, the problem is infeasible with active vars?
                # Or we missed a dominance relation.
                # Fallback: add all vertices?
                pass
            for v in active_in_C:
                # Map v to index in active_vertices
                # We need a fast mapping.
                # Since n is small, linear search or dict is fine.
                # Let's use a dict for vertex -> active_index
                pass
        
        # Precompute mapping
        v_to_idx = {v: i for i, v in enumerate(active_vertices)}
        
        for c_idx, C in enumerate(cliques):
            for v in C:
                if active[v]:
                    row_indices.append(c_idx)
                    col_indices.append(v_to_idx[v])
                    values.append(1)
                    
        A = coo_matrix((values, (row_indices, col_indices)), shape=(len(cliques), m)).tocsr()
        
        lb = np.ones(len(cliques))
        ub = np.full(len(cliques), np.inf)
        
        constraints = LinearConstraint(A, lb, ub)
        bounds = Bounds(0, 1)
        integrality = np.ones(m)
        
        try:
            res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status == 0:
                sol_active = [i for i in range(m) if res.x[i] > 0.5]
                sol = [active_vertices[i] for i in sol_active]
                # Minimality check
                # Implement the count-based minimization
                sel = set(sol)
                counts = np.zeros(len(cliques), dtype=int)
                for v in sel:
                    for c_idx in vert_to_cliques[v]:
                        counts[c_idx] += 1
                        
                changed = True
                while changed:
                    changed = False
                    for v in list(sel):
                        can_remove = True
                        for c_idx in vert_to_cliques[v]:
                            if counts[c_idx] == 1:
                                can_remove = False
                                break
                        if can_remove:
                            sel.remove(v)
                            for c_idx in vert_to_cliques[v]:
                                counts[c_idx] -= 1
                            changed = True
                            break
                return sorted(list(sel))
            else:
                pass
        except Exception:
            pass
            
        # Greedy Fallback (on active vertices? No, on all, but prefer active?)
        # Just use standard greedy on all vertices, then minimize.
        uncovered = set(range(len(cliques)))
        selected = set()
        
        # Use active vertices first? Or just all.
        # Greedy on all vertices is safer.
        while uncovered:
            best_v = -1
            best_count = 0
            for v in range(n):
                count = 0
                for c_idx in vert_to_cliques[v]:
                    if c_idx in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
                    
            if best_v == -1:
                break
                
            selected.add(best_v)
            for c_idx in vert_to_cliques[best_v]:
                uncovered.discard(c_idx)
                
        # Minimality check on greedy result
        sel = selected
        counts = np.zeros(len(cliques), dtype=int)
        for v in sel:
            for c_idx in vert_to_cliques[v]:
                counts[c_idx] += 1
                
        changed = True
        while changed:
            changed = False
            for v in list(sel):
                can_remove = True
                for c_idx in vert_to_cliques[v]:
                    if counts[c_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    sel.remove(v)
                    for c_idx in vert_to_cliques[v]:
                        counts[c_idx] -= 1
                    changed = True
                    break
        return sorted(list(sel))

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance_edges(n, p, seed)
        cliques = get_maximal_cliques(n, edges)
        sol = solve_set_cover(cliques, n)
        out.append(sol)
        
    return out
