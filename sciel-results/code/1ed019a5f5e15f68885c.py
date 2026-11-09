import time
import random
import numpy as np

def _build_polarity_graph_pg2_5():
    """
    Build the polarity graph of PG(2,5).
    Points: 31 points in PG(2,5).
    A point is adjacent to all points on its polar line (excluding itself).
    This graph is C4-free.
    """
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                p = [x, y, z]
                # Normalize: first non-zero element to 1
                for i in range(3):
                    if p[i] != 0:
                        inv = pow(p[i], 3, 5)
                        p = [(c * inv) % 5 for c in p]
                        break
                if p not in points:
                    points.append(p)
    
    n = len(points)
    assert n == 31, f"Expected 31 points, got {n}"
    
    edges = []
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            p, q = points[i], points[j]
            dot = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
            if dot == 0:
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
    
    return edges, adj, n

def _to_bitmask(adj, n):
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        masks[i] = mask
    return masks

def _max_independent_set_size_bitmask(adj_mask, n, time_limit=1.0):
    start_time = time.time()
    best = [0]
    
    all_mask = (1 << n) - 1
    # comp_mask[i] = vertices that are NOT adjacent to i (and not i itself)
    # We want to find a set of vertices with no edges between them.
    # In the complement graph, we want a clique.
    # Or in G, we want an independent set.
    # Branch and bound on G.
    
    comp_mask = [0] * n
    for i in range(n):
        # Non-neighbors of i within all_mask, excluding i
        comp_mask[i] = all_mask ^ adj_mask[i] ^ (1 << i)

    def _search(candidates, current_size):
        if time.time() - start_time > time_limit:
            return
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
            
        # Bound: current_size + number of remaining candidates
        if current_size + candidates.bit_count() <= best[0]:
            return
            
        # Heuristic: pick vertex with max degree in the induced subgraph of candidates in the COMPLEMENT?
        # No, pick vertex with max degree in G?
        # If we pick v, we either include it (remove its neighbors from candidates) or exclude it.
        # To prune well, we should pick a vertex that is "central" in the candidate set.
        # In G, a vertex with high degree in candidates has many neighbors, so including it removes many candidates.
        # This is good for the "include" branch, but bad for the "exclude" branch?
        # Actually, standard Max Independent Set heuristic: pick vertex with max degree in the candidate subgraph of G.
        
        max_deg = -1
        v = -1
        temp_cands = candidates
        while temp_cands:
            lsb = temp_cands & -temp_cands
            idx = lsb.bit_length() - 1
            # Degree of idx in candidates (within G)
            deg = (adj_mask[idx] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = idx
            temp_cands ^= lsb
            
        if v == -1:
            return

        # Branch 1: Include v
        # New candidates: candidates \ (N(v) U {v})
        new_cands_inc = candidates & ~(adj_mask[v] | (1 << v))
        _search(new_cands_inc, current_size + 1)
        
        # Branch 2: Exclude v
        new_cands_exc = candidates & ~(1 << v)
        _search(new_cands_exc, current_size)

    _search(all_mask, 0)
    return best[0]

def _find_compatible_sets(adj, n, max_size=6):
    """
    Find sets S of vertices that have pairwise disjoint neighborhoods in G.
    """
    # Incompatibility: u and v are incompatible if N(u) intersects N(v)
    # We need S such that for all u, w in S, N(u) \cap N(w) = empty.
    # This is equivalent to finding a clique in the "disjointness" graph?
    # Or an independent set in the "intersection" graph.
    
    incomp = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & adj[j]:
                incomp[i] |= (1 << j)
                incomp[j] |= (1 << i)
                
    sets_found = []
    
    def backtrack(start, current_set, current_mask):
        if len(current_set) > max_size:
            return
        if len(current_set) > 0:
            sets_found.append(list(current_set))
            
        # Candidates must be > start and not incompatible with any in current_set
        # current_mask is the bitmask of vertices in current_set
        # We need v such that (incomp[v] & current_mask) == 0
        candidates = []
        for v in range(start, n):
            if current_mask & incomp[v]:
                continue
            candidates.append(v)
            
        for v in candidates:
            current_set.append(v)
            current_mask |= (1 << v)
            backtrack(v + 1, current_set, current_mask)
            current_set.pop()
            current_mask ^= (1 << v)
            
    backtrack(0, [], 0)
    return sets_found

def _alpha_subgraph_bitmask(adj_mask, n, exclude_mask, time_limit=0.3):
    """
    Compute max independent set size in G[V \\ exclude_mask].
    """
    all_mask = (1 << n) - 1
    candidates = all_mask & ~exclude_mask
    if candidates == 0:
        return 0
    
    start_time = time.time()
    best = [0]
    
    def _search(cands, current_size):
        if time.time() - start_time > time_limit:
            return
        if not cands:
            if current_size > best[0]:
                best[0] = current_size
            return
            
        if current_size + cands.bit_count() <= best[0]:
            return
            
        max_deg = -1
        v = -1
        temp = cands
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            deg = (adj_mask[idx] & cands).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = idx
            temp ^= lsb
            
        if v == -1:
            return

        # Include v
        new_cands_inc = cands & ~(adj_mask[v] | (1 << v))
        _search(new_cands_inc, current_size + 1)
        
        # Exclude v
        new_cands_exc = cands & ~(1 << v)
        _search(new_cands_exc, current_size)

    _search(candidates, 0)
    return best[0]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Build Base Graph
    edges, adj, n = _build_polarity_graph_pg2_5()
    
    best_n = n
    best_edges = list(edges)
    best_adj = [set(s) for s in adj]
    
    # Check base alpha
    adj_mask = _to_bitmask(adj, n)
    alpha = _max_independent_set_size_bitmask(adj_mask, n, time_limit=2.0)
    if alpha >= 11:
        return {'n': n, 'edges': edges}

    # 2. Extension Loop
    while time.time() < deadline - 15:
        found_extension = False
        
        # Precompute adjacency masks for current graph
        adj_mask = _to_bitmask(adj, n)
        
        # Generate candidate compatible sets
        # Try larger sets first as they reduce alpha more
        candidates = _find_compatible_sets(adj, n, max_size=6)
        
        # Sort by size descending
        candidates.sort(key=lambda s: -len(s))
        
        # Limit candidates
        max_candidates_per_round = 1000
        candidates = candidates[:max_candidates_per_round]
        
        # Try each candidate
        for S in candidates:
            if time.time() > deadline - 10:
                break
                
            s_size = len(S)
            if s_size < 2 and n > 33:
                continue
                
            exclude_mask = 0
            for v in S:
                exclude_mask |= (1 << v)
                
            # Quick check: alpha of G[V\S] must be <= 9
            # If alpha(G[V\S]) >= 10, then adding v (connected to S) might create an independent set of size 11?
            # No, v is connected to S, so v is NOT in any independent set containing vertices from S.
            # Any independent set in G' is either:
            # 1. An independent set in G[V\S] (size <= alpha(G[V\S]))
            # 2. {v} U I, where I is an independent set in G[V\S] that has no neighbors of v?
            #    v is connected to S. So I must be disjoint from S.
            #    So I is an independent set in G[V\S].
            #    So size is 1 + alpha(G[V\S]).
            # We need max(alpha(G), 1 + alpha(G[V\S])) <= 10.
            # We know alpha(G) <= 10.
            # So we need 1 + alpha(G[V\S]) <= 10 => alpha(G[V\S]) <= 9.
            
            alpha_u = _alpha_subgraph_bitmask(adj_mask, n, exclude_mask, time_limit=0.2)
            
            if alpha_u <= 9:
                # Valid extension
                new_v = n
                new_n = n + 1
                new_adj = [set(s) for s in adj]
                new_adj.append(set(S))
                for u in S:
                    new_adj[u].add(new_v)
                
                new_edges = list(edges)
                for u in S:
                    new_edges.append([u, new_v])
                
                # Update state
                edges = new_edges
                adj = new_adj
                n = new_n
                best_n = n
                best_edges = new_edges
                best_adj = new_adj
                
                found_extension = True
                break
        
        if not found_extension:
            # Try to "repair" or try different compatible sets?
            # For now, break if no extension found.
            break

    # Final Validation
    if best_n > 1:
        c4_free = True
        for i in range(best_n):
            for j in range(i+1, best_n):
                if len(best_adj[i] & best_adj[j]) > 1:
                    c4_free = False
                    break
            if not c4_free:
                break
        if not c4_free:
             return {'n': 31, 'edges': _build_polarity_graph_pg2_5()[0]}
        
        adj_mask_final = _to_bitmask(best_adj, best_n)
        alpha_final = _max_independent_set_size_bitmask(adj_mask_final, best_n, time_limit=5.0)
        if alpha_final >= 11:
             base_edges, _, _ = _build_polarity_graph_pg2_5()
             return {'n': 31, 'edges': base_edges}

    return {'n': best_n, 'edges': best_edges}
