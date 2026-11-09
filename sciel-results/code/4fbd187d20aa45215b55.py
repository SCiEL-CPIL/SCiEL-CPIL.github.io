import time
import random
import numpy as np

def _build_polarity_graph_pg2_5():
    """
    Build the polarity graph of PG(2,5).
    Points: 31 points in PG(2,5).
    A point is adjacent to all points on its polar line.
    This graph is C4-free and has independence number 8.
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
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = all_mask ^ adj_mask[i] ^ (1 << i)

    # Use a simple branch and bound with bitmasks
    def _search(candidates, current_size):
        if time.time() - start_time > time_limit:
            return
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
            
        # Bound
        if current_size + candidates.bit_count() <= best[0]:
            return
            
        # Pick vertex with max degree in candidates (complement graph degree)
        # This helps prune
        max_deg = -1
        v = -1
        temp_cands = candidates
        while temp_cands:
            lsb = temp_cands & -temp_cands
            idx = lsb.bit_length() - 1
            deg = (comp_mask[idx] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = idx
            temp_cands ^= lsb
            
        if v == -1:
            return

        # Include v
        new_cands = candidates & comp_mask[v]
        _search(new_cands, current_size + 1)
        
        # Exclude v
        new_cands2 = candidates & ~(1 << v)
        _search(new_cands2, current_size)

    _search(all_mask, 0)
    return best[0]

def _greedy_independent_set_size(adj_mask, n):
    """
    Fast greedy lower bound for independence number.
    """
    remaining = (1 << n) - 1
    size = 0
    while remaining:
        # Pick vertex with min degree in remaining subgraph
        min_deg = n
        v = -1
        temp = remaining
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            deg = (adj_mask[idx] & remaining).bit_count()
            if deg < min_deg:
                min_deg = deg
                v = idx
            temp ^= lsb
        
        if v == -1:
            break
            
        size += 1
        # Remove v and its neighbors
        remaining &= ~(adj_mask[v] | (1 << v))
        
    return size

def _find_compatible_sets(adj, n, max_size=5):
    """
    Find sets S of vertices that have pairwise disjoint neighborhoods in G.
    """
    # Build incompatibility graph: edge if neighborhoods intersect
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
    Compute max independent set size in G[V \\ exclude_mask] using bitmask.
    adj_mask[i] is the full adjacency mask for vertex i in the original graph.
    exclude_mask: bitmask of vertices to exclude.
    """
    all_mask = (1 << n) - 1
    candidates = all_mask & ~exclude_mask
    if candidates == 0:
        return 0
    
    # Build complement masks restricted to candidates
    comp_mask = [0] * n
    for i in range(n):
        # Non-neighbors of i within candidates, excluding i itself
        comp_mask[i] = candidates & ~(adj_mask[i] | (1 << i))
    
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
            
        # Pick vertex with max degree in complement within cands
        max_deg = -1
        v = -1
        temp = cands
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            deg = (comp_mask[idx] & cands).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = idx
            temp ^= lsb
            
        if v == -1:
            return

        # Include v
        new_cands = cands & comp_mask[v]
        _search(new_cands, current_size + 1)
        
        # Exclude v
        new_cands2 = cands & ~(1 << v)
        _search(new_cands2, current_size)

    _search(candidates, 0)
    return best[0]

def _alpha_upper_bound_greedy(adj_mask, n, exclude_mask):
    """
    Quick greedy upper bound: repeatedly pick vertex with max degree in remaining,
    count how many we can pick. This gives a lower bound on alpha, not upper.
    For upper bound, use: alpha <= n - max_matching_lower_bound... complex.
    Simpler: use greedy to find a large independent set as lower bound.
    """
    remaining = ((1 << n) - 1) & ~exclude_mask
    size = 0
    while remaining:
        # Pick vertex with min degree in remaining subgraph (greedy for max independent set)
        min_deg = n
        v = -1
        temp = remaining
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            deg = (adj_mask[idx] & remaining).bit_count()
            if deg < min_deg:
                min_deg = deg
                v = idx
            temp ^= lsb
        
        if v == -1:
            break
            
        size += 1
        # Remove v and its neighbors
        remaining &= ~(adj_mask[v] | (1 << v))
        
    return size

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
        
        # Generate candidate compatible sets (pairwise disjoint neighborhoods)
        # Limit max_size to 4 to keep candidate count manageable
        candidates = _find_compatible_sets(adj, n, max_size=4)
        
        # Sort by size descending (larger sets reduce alpha more effectively)
        candidates.sort(key=lambda s: -len(s))
        
        # Limit number of candidates tried per round to stay within time
        max_candidates_per_round = 800
        candidates = candidates[:max_candidates_per_round]
        
        # Try each candidate
        for S in candidates:
            if time.time() > deadline - 10:
                break
                
            # Quick size check: if |S| is too small, alpha(G-V\S) is likely still >= 10
            s_size = len(S)
            if s_size < 2 and n > 32:
                continue
                
            # Calculate alpha of G[V \ S] using bitmask
            exclude_mask = 0
            for v in S:
                exclude_mask |= (1 << v)
                
            # Quick lower bound check: if greedy finds independent set of size 10 in G[V\S],
            # then alpha >= 10, so we can't extend
            greedy_lb = _alpha_upper_bound_greedy(adj_mask, n, exclude_mask)
            if greedy_lb >= 10:
                continue
                
            alpha_u = _alpha_subgraph_bitmask(adj_mask, n, exclude_mask, time_limit=0.25)
            
            # Check if adding v is valid
            # New alpha = max(alpha(G), alpha_u + 1)
            # We know alpha(G) <= 10
            # So we just need alpha_u + 1 <= 10 => alpha_u <= 9
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
                break # Restart with new graph
        
        if not found_extension:
            # No extension found in this round
            break

    # Final Validation
    if best_n > 1:
        # Quick C4 check
        c4_free = True
        for i in range(best_n):
            for j in range(i+1, best_n):
                if len(best_adj[i] & best_adj[j]) > 1:
                    c4_free = False
                    break
            if not c4_free:
                break
        if not c4_free:
             # Fallback
             return {'n': 31, 'edges': _build_polarity_graph_pg2_5()[0]}
        
        adj_mask_final = _to_bitmask(best_adj, best_n)
        alpha_final = _max_independent_set_size_bitmask(adj_mask_final, best_n, time_limit=5.0)
        if alpha_final >= 11:
             # Fallback to base
             base_edges, _, _ = _build_polarity_graph_pg2_5()
             return {'n': 31, 'edges': base_edges}

    return {'n': best_n, 'edges': best_edges}
