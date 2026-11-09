import time
import random

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
    
    def _search(candidates, current_size):
        nonlocal best
        if time.time() - start_time > time_limit:
            return
        
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
            
        # Bound: can't exceed current_size + number of candidates
        if current_size + candidates.bit_count() <= best[0]:
            return
            
        # Pick vertex with maximum degree in the original graph restricted to candidates
        max_deg_orig = -1
        v = -1
        temp_cands = candidates
        while temp_cands:
            lsb = temp_cands & -temp_cands
            idx = lsb.bit_length() - 1
            deg = (adj_mask[idx] & candidates).bit_count()
            if deg > max_deg_orig:
                max_deg_orig = deg
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

def _generate_compatible_set_candidates(adj, n, max_size=5, num_candidates=10, time_limit=0.5):
    """
    Generate several candidate compatible sets.
    A compatible set is a set of vertices with pairwise disjoint neighborhoods.
    """
    # Build incompatibility: u ~ v if N(u) ∩ N(v) ≠ ∅
    incomp_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and (adj[i] & adj[j]):
                mask |= (1 << j)
        incomp_mask[i] = mask
    
    candidates = []
    start_time = time.time()
    
    # 1. Greedy based on min incompatibility degree
    remaining_mask = (1 << n) - 1
    current_set = []
    while remaining_mask and len(current_set) < max_size:
        best_v = -1
        best_deg = n + 1
        temp = remaining_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            deg = (incomp_mask[idx] & remaining_mask).bit_count()
            if deg < best_deg:
                best_deg = deg
                best_v = idx
            temp ^= lsb
        if best_v == -1:
            break
        current_set.append(best_v)
        remaining_mask &= ~(1 << best_v)
        remaining_mask &= ~incomp_mask[best_v]
    if current_set:
        candidates.append(current_set)
    
    # 2. Randomized Greedy
    for _ in range(num_candidates):
        if time.time() - start_time > time_limit:
            break
        remaining_mask = (1 << n) - 1
        current_set = []
        while remaining_mask and len(current_set) < max_size:
            temp = remaining_mask
            count = temp.bit_count()
            if count == 0: break
            r = random.randint(0, count - 1)
            t = temp
            for k in range(r):
                t &= t - 1
            lsb = t & -t
            v = lsb.bit_length() - 1
            
            current_set.append(v)
            remaining_mask &= ~(1 << v)
            remaining_mask &= ~incomp_mask[v]
        if current_set and current_set not in candidates:
            candidates.append(current_set)
            
    return candidates

def _add_vertex(adj, n, S, edges):
    """Add a new vertex connected to set S. Returns new adj, n, edges."""
    new_v = n
    new_n = n + 1
    new_adj = [set(s) for s in adj]
    new_adj.append(set(S))
    for u in S:
        new_adj[u].add(new_v)
    
    new_edges = list(edges)
    for u in S:
        new_edges.append([u, new_v])
    
    return new_adj, new_n, new_edges

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
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
    extension_count = 0
    while time.time() < deadline - 15:
        found_extension = False
        
        # Generate candidate compatible sets
        candidates = _generate_compatible_set_candidates(adj, n, max_size=5, num_candidates=20, time_limit=0.5)
        
        if not candidates:
            # Try smaller sizes
            candidates = _generate_compatible_set_candidates(adj, n, max_size=3, num_candidates=10, time_limit=0.2)
            if not candidates:
                break

        # Sort candidates by size (larger is better for extension? 
        # Actually, larger S means smaller V\S, so alpha(V\S) is more likely to be small.
        # So try larger sets first.)
        candidates.sort(key=lambda x: len(x), reverse=True)
        
        for S in candidates:
            if time.time() > deadline - 10:
                break
                
            U = set(range(n)) - set(S)
            u_list = sorted(list(U))
            n_u = len(u_list)
            
            if n_u == 0:
                continue
            
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
            
            # Check if alpha(G[V\S]) <= 9
            alpha_u = _max_independent_set_size_bitmask(sub_adj_mask, n_u, time_limit=0.5)
            
            if alpha_u <= 9:
                # Valid extension
                new_adj, new_n, new_edges = _add_vertex(adj, n, S, edges)
                
                # Update state
                adj = new_adj
                n = new_n
                edges = new_edges
                best_n = n
                best_edges = list(edges)
                best_adj = [set(s) for s in adj]
                
                extension_count += 1
                found_extension = True
                break # Try next extension step
        
        if not found_extension:
            # If no extension found with max_size=5, try smaller sets explicitly
            for max_s in [4, 3, 2, 1]:
                if time.time() > deadline - 10:
                    break
                
                candidates = _generate_compatible_set_candidates(adj, n, max_size=max_s, num_candidates=10, time_limit=0.2)
                for S in candidates:
                    U = set(range(n)) - set(S)
                    u_list = sorted(list(U))
                    n_u = len(u_list)
                    if n_u == 0: continue
                    
                    sub_adj = [set() for _ in range(n_u)]
                    u_map = {u: i for i, u in enumerate(u_list)}
                    for i in range(n_u):
                        for j in range(i+1, n_u):
                            u, v = u_list[i], u_list[j]
                            if v in adj[u]:
                                sub_adj[i].add(j)
                                sub_adj[j].add(i)
                    
                    sub_adj_mask = _to_bitmask(sub_adj, n_u)
                    alpha_u = _max_independent_set_size_bitmask(sub_adj_mask, n_u, time_limit=0.2)
                    
                    if alpha_u <= 9:
                        new_adj, new_n, new_edges = _add_vertex(adj, n, S, edges)
                        
                        adj = new_adj
                        n = new_n
                        edges = new_edges
                        best_n = n
                        best_edges = list(edges)
                        best_adj = [set(s) for s in adj]
                        
                        extension_count += 1
                        found_extension = True
                        break
                if found_extension:
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
