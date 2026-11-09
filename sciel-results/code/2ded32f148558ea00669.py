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
    """
    Find the size of the maximum independent set using branch and bound.
    Uses bitmask operations for speed.
    """
    start_time = time.time()
    best = [0]
    
    all_mask = (1 << n) - 1
    # comp_mask[i] = vertices not adjacent to i, excluding i itself
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = all_mask ^ adj_mask[i] ^ (1 << i)
    
    # Precompute degree in complement for ordering
    comp_deg = [0] * n
    for i in range(n):
        comp_deg[i] = comp_mask[i].bit_count()
    
    # Sort vertices by complement degree (ascending for better pruning)
    order = sorted(range(n), key=lambda x: comp_deg[x])
    order_idx = {v: i for i, v in enumerate(order)}
    
    def _search(candidates, current_size, time_left):
        if time.time() - start_time > time_limit:
            return
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
            
        # Bound: can't exceed current_size + number of candidates
        if current_size + candidates.bit_count() <= best[0]:
            return
            
        # Pick vertex with minimum complement degree among candidates (for better pruning)
        # This is a heuristic for maximum independent set
        min_deg = n + 1
        v = -1
        temp_cands = candidates
        while temp_cands:
            lsb = temp_cands & -temp_cands
            idx = lsb.bit_length() - 1
            deg = (comp_mask[idx] & candidates).bit_count()
            if deg < min_deg:
                min_deg = deg
                v = idx
            temp_cands ^= lsb
            
        if v == -1:
            return

        # Include v: add v to independent set, remove v and its neighbors from candidates
        new_cands = candidates & comp_mask[v]
        _search(new_cands, current_size + 1, time_left)
        
        # Exclude v
        new_cands2 = candidates & ~(1 << v)
        _search(new_cands2, current_size, time_left)

    _search(all_mask, 0, 0)
    return best[0]

def _find_max_compatible_set(adj, n, max_size=6):
    """
    Find a large set of vertices with pairwise disjoint neighborhoods.
    These are independent sets in the incompatibility graph.
    Uses greedy + local search.
    """
    # Build incompatibility: u ~ v if N(u) ∩ N(v) ≠ ∅
    incomp = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & adj[j]:
                incomp[i].add(j)
                incomp[j].add(i)
    
    # Greedy: pick vertex with fewest incompatibilities first
    remaining = set(range(n))
    current_set = []
    
    while remaining and len(current_set) < max_size:
        # Pick vertex with minimum incompatibility degree in remaining
        best_v = None
        best_deg = n + 1
        for v in remaining:
            deg = len(incomp[v] & remaining)
            if deg < best_deg:
                best_deg = deg
                best_v = v
        
        if best_v is None:
            break
        
        current_set.append(best_v)
        remaining.discard(best_v)
        # Remove all incompatible vertices from remaining
        for u in incomp[best_v]:
            remaining.discard(u)
    
    return current_set

def _alpha_estimate(adj, n, time_limit=0.3):
    """
    Quick estimate of independence number using greedy + limited search.
    Returns an upper bound estimate.
    """
    if n <= 1:
        return n
    
    # Greedy lower bound
    remaining = set(range(n))
    greedy_set = []
    while remaining:
        # Pick vertex with minimum degree in remaining
        best_v = min(remaining, key=lambda v: len(adj[v] & remaining))
        greedy_set.append(best_v)
        remaining.discard(best_v)
        for u in adj[best_v]:
            remaining.discard(u)
    
    greedy_size = len(greedy_set)
    
    # Quick branch and bound with tight time limit
    adj_mask = _to_bitmask(adj, n)
    alpha = _max_independent_set_size_bitmask(adj_mask, n, time_limit=time_limit)
    
    return max(greedy_size, alpha)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
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
    # Strategy: try to add vertices connected to compatible sets
    # A compatible set S means all vertices in S have disjoint neighborhoods
    # Adding a new vertex v connected to all of S is valid if S is compatible
    
    extension_count = 0
    while time.time() < deadline - 15:
        found_extension = False
        
        # Find a good compatible set to connect the new vertex to
        # We want S such that alpha(G[V\S]) is small
        # Heuristic: find large compatible sets, then evaluate their impact
        
        # Generate candidate compatible sets using greedy
        S = _find_max_compatible_set(adj, n, max_size=5)
        
        if not S:
            break
        
        # Try this set
        U = set(range(n)) - set(S)
        u_list = sorted(list(U))
        n_u = len(u_list)
        
        if n_u == 0:
            break
        
        # Build subgraph on U
        sub_adj = [set() for _ in range(n_u)]
        u_map = {u: i for i, u in enumerate(u_list)}
        for i in range(n_u):
            for j in range(i+1, n_u):
                u, v = u_list[i], u_list[j]
                if v in adj[u]:
                    sub_adj[i].add(j)
                    sub_adj[j].add(i)
        
        sub_adj_mask = _to_bitmask(sub_adj, n_u)
        
        # Check if alpha(G[V\S]) <= 9 (so that alpha(G+new_v) <= 10)
        alpha_u = _max_independent_set_size_bitmask(sub_adj_mask, n_u, time_limit=1.0)
        
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
            
            extension_count += 1
            found_extension = True
        
        if not found_extension:
            # Try smaller sets
            for max_s in [4, 3, 2, 1]:
                if time.time() > deadline - 10:
                    break
                
                S = _find_max_compatible_set(adj, n, max_size=max_s)
                if not S:
                    continue
                
                U = set(range(n)) - set(S)
                u_list = sorted(list(U))
                n_u = len(u_list)
                
                if n_u == 0:
                    continue
                
                sub_adj = [set() for _ in range(n_u)]
                u_map = {u: i for i, u in enumerate(u_list)}
                for i in range(n_u):
                    for j in range(i+1, n_u):
                        u, v = u_list[i], u_list[j]
                        if v in adj[u]:
                            sub_adj[i].add(j)
                            sub_adj[j].add(i)
                
                sub_adj_mask = _to_bitmask(sub_adj, n_u)
                alpha_u = _max_independent_set_size_bitmask(sub_adj_mask, n_u, time_limit=0.5)
                
                if alpha_u <= 9:
                    new_v = n
                    new_n = n + 1
                    new_adj = [set(s) for s in adj]
                    new_adj.append(set(S))
                    for u in S:
                        new_adj[u].add(new_v)
                    
                    new_edges = list(edges)
                    for u in S:
                        new_edges.append([u, new_v])
                    
                    edges = new_edges
                    adj = new_adj
                    n = new_n
                    best_n = n
                    best_edges = new_edges
                    best_adj = new_adj
                    
                    extension_count += 1
                    found_extension = True
                    break
            
            if not found_extension:
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
            base_edges, _, _ = _build_polarity_graph_pg2_5()
            return {'n': 31, 'edges': base_edges}
        
        adj_mask_final = _to_bitmask(best_adj, best_n)
        alpha_final = _max_independent_set_size_bitmask(adj_mask_final, best_n, time_limit=5.0)
        if alpha_final >= 11:
            # Fallback to base
            base_edges, _, _ = _build_polarity_graph_pg2_5()
            return {'n': 31, 'edges': base_edges}
    
    return {'n': best_n, 'edges': best_edges}
