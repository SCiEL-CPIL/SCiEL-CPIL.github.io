import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s kill

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch(adj, P, X, C, maximal_cliques):
        if not P and not X:
            if len(C) >= 2:
                maximal_cliques.append(frozenset(C))
            return
        
        P_union_X = P | X
        if P_union_X:
            # Pivot selection: choose vertex u in P union X with max degree in P
            pivot = None
            max_inter = -1
            for v in P_union_X:
                inter = len(P & adj[v])
                if inter > max_inter:
                    max_inter = inter
                    pivot = v
            
            # Branch on P \ N(pivot)
            candidates = P - adj[pivot] if pivot is not None else P
            
            for v in list(candidates):
                new_P = P & adj[v]
                new_X = X & adj[v]
                new_C = C | {v}
                bron_kerbosch(adj, new_P, new_X, new_C, maximal_cliques)
                P.remove(v)
                X.add(v)

    def find_min_transversal(n, edges):
        if not edges:
            return []
        
        adj = build_adj(n, edges)
        maximal_cliques = []
        # Bron-Kerbosch
        bron_kerbosch(adj, set(range(n)), set(), set(), maximal_cliques)
        
        if not maximal_cliques:
            return []

        n_cliques = len(maximal_cliques)
        
        # Precompute: for each vertex, bitmask of cliques it hits
        # Since n_cliques can be up to ~100-200 for n=40, we can use Python ints as bitmasks
        vertex_clique_mask = [0] * n
        for i, c in enumerate(maximal_cliques):
            mask_bit = 1 << i
            for v in c:
                vertex_clique_mask[v] |= mask_bit
        
        # Greedy initialization
        # We want to pick vertices that cover the most uncovered cliques.
        # unhit_mask: bitmask of cliques not yet hit.
        unhit_mask = (1 << n_cliques) - 1
        transversal = set()
        
        # To improve greedy, we can weight cliques. 
        # A clique is "harder" if it has fewer vertices.
        # Let's calculate a weight for each clique: 1 / len(clique) or similar?
        # Or just use frequency.
        
        while unhit_mask:
            if time.time() > deadline:
                break
            
            best_v = -1
            best_score = -1
            
            # For each vertex not in transversal, calculate how many unhit cliques it covers
            # And potentially a secondary score: sum of weights of those cliques
            for v in range(n):
                if v in transversal:
                    continue
                
                # Mask of unhit cliques that v can cover
                covers = unhit_mask & vertex_clique_mask[v]
                
                if covers == 0:
                    continue
                
                # Primary score: number of cliques covered
                # Popcount of covers
                # Python 3.8+ has int.bit_count()
                count = covers.bit_count()
                
                # Secondary score: Prefer covering cliques that are "small" (fewer options)
                # We can precompute clique weights based on size
                # weight[i] = 1 / len(maximal_cliques[i])
                # Sum of weights of covered unhit cliques
                # This is slower, so let's stick to count first, or use a simple tie breaker
                # Tie breaker: prefer vertex with higher degree? Or lower index?
                # Let's try: score = count * 1000 + len(adj[v])
                
                score = count * 1000 + len(adj[v])
                
                if score > best_score:
                    best_score = score
                    best_v = v
            
            if best_v == -1:
                # Should not happen if unhit_mask is not 0 and graph is connected enough
                # But if we get stuck, pick any vertex from an unhit clique
                # Find first unhit clique
                m = unhit_mask
                while m:
                    lowbit = m & (-m)
                    idx = lowbit.bit_length() - 1
                    m ^= lowbit
                    best_v = next(iter(maximal_cliques[idx]))
                    break
            
            transversal.add(best_v)
            unhit_mask &= ~vertex_clique_mask[best_v]
        
        # Greedy removal (Minimization)
        changed = True
        while changed:
            changed = False
            for v in list(transversal):
                if time.time() > deadline:
                    return sorted(transversal)
                # Check if v is redundant
                # v is redundant if every clique containing v also contains another vertex in transversal
                redundant = True
                # Get all cliques containing v
                v_cliques_mask = vertex_clique_mask[v]
                
                # Iterate through cliques in v_cliques_mask
                m = v_cliques_mask
                while m:
                    lowbit = m & (-m)
                    idx = lowbit.bit_length() - 1
                    m ^= lowbit
                    c = maximal_cliques[idx]
                    # Check if any other vertex in c is in transversal
                    # We can check if (transversal_mask & c_mask) != 0, excluding v
                    # But we don't have transversal_mask as bitmask for quick check against clique mask?
                    # Let's just iterate vertices in c
                    found_other = False
                    for u in c:
                        if u != v and u in transversal:
                            found_other = True
                            break
                    if not found_other:
                        redundant = False
                        break
                
                if redundant:
                    transversal.remove(v)
                    changed = True
        
        # Branch and Bound
        best_size = len(transversal)
        best_solution = set(transversal)
        
        # Sort cliques by size (descending) for better pruning?
        # Actually, for B&B, we usually pick the "most constrained" variable/clique.
        # Let's keep original order but use bitmask for speed.
        
        # Precompute clique masks for vertices? No, we have vertex_clique_mask.
        # We need to check if a vertex is in a clique.
        # Let's create a list of vertices for each clique for quick iteration in B&B branching.
        clique_vertices = [list(c) for c in maximal_cliques]
        
        # B&B Search
        # State: unhit_mask, current_set (as bitmask of vertices), current_size
        # Vertex bitmask: 1 << v
        
        def search(unhit_mask, current_mask, current_size, remaining_budget):
            nonlocal best_size, best_solution
            
            if current_size >= best_size:
                return
            
            if not unhit_mask:
                if current_size < best_size:
                    best_size = current_size
                    # Reconstruct set from mask
                    sol = set()
                    m = current_mask
                    while m:
                        lowbit = m & (-m)
                        v = lowbit.bit_length() - 1
                        m ^= lowbit
                        sol.add(v)
                    best_solution = sol
                return
            
            if time.time() > remaining_budget:
                return
            
            # Lower bound:
            # If we have disjoint cliques, we need at least 1 vertex for each.
            # Greedy disjoint clique cover:
            # Pick a clique, remove it and all overlapping cliques, repeat.
            # This is O(C^2) per node, might be too slow.
            # Simple LB: 1 if unhit_mask != 0.
            # Better LB: Max number of disjoint cliques in unhit_mask.
            # Let's try a cheap LB: 
            # If there is a clique of size 1 (none here), LB=1.
            # Let's just use 1.
            
            if current_size + 1 >= best_size:
                return
            
            # Choose a clique to branch on
            # Heuristic: Pick the unhit clique with the fewest candidate vertices
            # Candidate vertices: vertices in the clique that are NOT in current_set
            
            best_clique_idx = -1
            best_candidates = None
            min_cand_count = float('inf')
            
            m = unhit_mask
            while m:
                lowbit = m & (-m)
                idx = lowbit.bit_length() - 1
                m ^= lowbit
                
                c_verts = clique_vertices[idx]
                # Count candidates
                cand_count = 0
                # Check which vertices in c_verts are not in current_mask
                # We can do: for v in c_verts: if not (current_mask & (1 << v)): cand_count++
                # Or: candidates_mask = clique_mask[idx] & ~current_mask
                # We don't have clique_mask as bitmask of vertices precomputed.
                # Let's compute it on the fly or precompute it.
                
                # Let's precompute clique_vertex_mask: bitmask of vertices in each clique
                # This is better.
                
            # Precomputation for clique_vertex_mask
            # Do this outside search
            
        # Precompute clique_vertex_mask
        clique_vertex_mask = [0] * n_cliques
        for i, c in enumerate(maximal_cliques):
            mask = 0
            for v in c:
                mask |= (1 << v)
            clique_vertex_mask[i] = mask
            
        def search(unhit_mask, current_mask, current_size, remaining_budget):
            nonlocal best_size, best_solution
            
            if current_size >= best_size:
                return
            
            if not unhit_mask:
                if current_size < best_size:
                    best_size = current_size
                    sol = set()
                    m = current_mask
                    while m:
                        lowbit = m & (-m)
                        v = lowbit.bit_length() - 1
                        m ^= lowbit
                        sol.add(v)
                    best_solution = sol
                return
            
            if time.time() > remaining_budget:
                return
            
            if current_size + 1 >= best_size:
                return
            
            # Choose clique to branch on
            best_clique_idx = -1
            best_candidates_mask = 0
            min_cand_count = float('inf')
            
            m = unhit_mask
            while m:
                lowbit = m & (-m)
                idx = lowbit.bit_length() - 1
                m ^= lowbit
                
                # Candidates are vertices in clique idx not in current_mask
                cand_mask = clique_vertex_mask[idx] & ~current_mask
                cand_count = cand_mask.bit_count()
                
                if cand_count < min_cand_count:
                    min_cand_count = cand_count
                    best_clique_idx = idx
                    best_candidates_mask = cand_mask
                    if min_cand_count <= 1:
                        break
            
            if min_cand_count == 0:
                return # Should not happen if unhit_mask is valid
            
            # Sort candidates by "impact"
            # Impact = number of unhit cliques covered by this vertex
            # We can estimate this using vertex_clique_mask[v] & unhit_mask
            
            candidates = []
            m = best_candidates_mask
            while m:
                lowbit = m & (-m)
                v = lowbit.bit_length() - 1
                m ^= lowbit
                # Score: how many unhit cliques does v cover?
                covers = vertex_clique_mask[v] & unhit_mask
                score = covers.bit_count()
                candidates.append((score, v))
            
            # Sort by score descending
            candidates.sort(key=lambda x: -x[0])
            
            for _, v in candidates:
                if time.time() > remaining_budget:
                    return
                
                new_mask = current_mask | (1 << v)
                new_unhit = unhit_mask & ~vertex_clique_mask[v]
                
                search(new_unhit, new_mask, current_size + 1, remaining_budget)
        
        # Run B&B
        bb_start = time.time()
        bb_budget = min(deadline - bb_start, 20) # Limit B&B time
        if bb_budget > 0:
            # Start search from empty set? Or from current transversal?
            # Starting from empty set allows finding a potentially smaller solution than greedy.
            # But greedy is usually close.
            # Let's start from empty set to be safe, but use best_size from greedy as initial bound.
            search((1 << n_cliques) - 1, 0, 0, bb_start + bb_budget)
            
            if best_size < len(transversal):
                transversal = best_solution
            else:
                # If B&B didn't improve, try local search on current transversal
                # 1-exchange: remove v, add u if it covers all cliques v covered that weren't covered by others
                # This is complex. Let's stick to the greedy removal which is already done.
                pass

        return sorted(transversal)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance(n, p, seed)
        sol = find_min_transversal(n, edges)
        results.append(sol)
        if time.time() > deadline:
             # If we ran out of time, ensure we return what we have
             # But we need to return 3 lists. 
             # If we didn't finish all 3, we should return best effort or empty?
             # Better to return valid sets for completed ones and try fast heuristic for others?
             # Actually, 120s is plenty for N=40. The loop is very fast.
             pass

    return results
