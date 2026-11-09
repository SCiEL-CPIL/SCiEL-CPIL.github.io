"""
C4-free graph construction without independent 11-sets.
Uses polarity graph PG(2,5) as a base (n=31, alpha=8) and attempts to extend
to n=39 by adding vertices with carefully chosen neighborhoods that preserve
C4-freeness and keep alpha < 11.

Strategy:
1. Build the polarity graph of PG(2,5) (31 vertices, C4-free, alpha=8).
2. Try to add vertices 31..n-1 one at a time. For each new vertex, choose a
   subset S of existing vertices to connect to, such that:
     - S is an "independent set" in the incompatibility graph (no two vertices
       in S share a common neighbor in the current graph).
   This ensures C4-freeness.
3. Use local search to add more edges where possible.
4. Verify alpha < 11 using exact bitmask branch-and-bound.
"""
import time
import random
import numpy as np

def _build_polarity_graph_q5():
    """
    Build the polarity graph of PG(2,5).
    Points: 31 points in PG(2,5).
    A point is adjacent to all points on its polar line.
    """
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                p = [x, y, z]
                # Normalize: first non-zero element to 1
                for i in range(3):
                    if p[i] != 0:
                        inv = pow(p[i], q - 2, q)  # Fermat's little theorem
                        p = [(c * inv) % q for c in p]
                        break
                if p not in points:
                    points.append(p)
    
    n = len(points)
    assert n == 31, f"Expected 31 points, got {n}"
    
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            p, qq = points[i], points[j]
            dot = (p[0] * qq[0] + p[1] * qq[1] + p[2] * qq[2]) % q
            if dot == 0:
                adj[i][j] = True
                adj[j][i] = True
    
    return adj, n

def _to_bitmask(adj, n):
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i][j]:
                mask |= (1 << j)
        masks[i] = mask
    return masks

def _max_independent_set_size_bitmask(adj_mask, n, time_limit=2.0):
    """
    Exact maximum independent set size using bitmask branch-and-bound.
    adj_mask[i] is the bitmask of neighbors of vertex i.
    """
    start_time = time.time()
    best = [0]
    all_mask = (1 << n) - 1
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = all_mask ^ adj_mask[i] ^ (1 << i)
    
    def _search(candidates, current_size):
        if time.time() - start_time > time_limit:
            return
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
        if current_size + candidates.bit_count() <= best[0]:
            return
        
        # Pick vertex with max degree in complement within candidates
        max_deg = -1
        v = -1
        temp = candidates
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            deg = (comp_mask[idx] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = idx
            temp ^= lsb
        
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
        remaining &= ~(adj_mask[v] | (1 << v))
    return size

def _check_c4_free(adj, n):
    """
    Check if graph is C4-free: no two vertices have more than one common neighbor.
    """
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                common = np.sum(adj[u] & adj[v])
                if common > 1:
                    return False
    return True

def _try_add_vertex(adj, n_current, new_idx, n_total):
    """
    Try to add a new vertex at index new_idx by connecting it to a subset S
    of existing vertices such that no two vertices in S share a common neighbor.
    Returns the updated adj matrix if successful, else None.
    """
    # Build incompatibility: u and v are incompatible if they share a common neighbor
    # We need S such that for all u,v in S, N(u) ∩ N(v) = ∅
    
    # Greedy: shuffle existing vertices, pick greedily
    candidates = list(range(n_current))
    random.shuffle(candidates)
    
    chosen = []
    # For efficiency, maintain a set of "blocked" vertices
    # When we pick u, we block all v that share a neighbor with u
    
    # Precompute for each vertex its neighborhood
    neighbors = [np.where(adj[i])[0] for i in range(n_current)]
    
    blocked = [False] * n_current
    
    for u in candidates:
        if blocked[u]:
            continue
        # Check u is not blocked
        chosen.append(u)
        # Block all v that share a neighbor with u
        for x in neighbors[u]:
            if x >= n_current:
                continue
            for y in neighbors[x]:
                if y != u and y < n_current:
                    blocked[y] = True
    
    if len(chosen) == 0:
        return None
    
    # Create new adj
    new_adj = np.zeros((n_total, n_total), dtype=bool)
    new_adj[:n_current, :n_current] = adj[:n_current, :n_current]
    
    for u in chosen:
        new_adj[new_idx, u] = True
        new_adj[u, new_idx] = True
    
    return new_adj

def _local_search_add_edges(adj, n):
    """
    Try to add edges that don't create C4s.
    Adding edge (u,v) is safe if there's no edge between N(u) and N(v).
    """
    changed = True
    iterations = 0
    while changed and iterations < 50:
        changed = False
        iterations += 1
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u][v]:
                    nu = np.where(adj[u])[0]
                    nv = np.where(adj[v])[0]
                    if len(nu) == 0 or len(nv) == 0:
                        adj[u][v] = True
                        adj[v][u] = True
                        changed = True
                    else:
                        # Check if any edge between N(u) and N(v)
                        conflict = False
                        for x in nu:
                            if np.any(adj[x, nv]):
                                conflict = True
                                break
                        if not conflict:
                            adj[u][v] = True
                            adj[v][u] = True
                            changed = True
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Build base polarity graph
    base_adj, base_n = _build_polarity_graph_q5()
    
    best_n = base_n
    best_edges = []
    for u in range(base_n):
        for v in range(u + 1, base_n):
            if base_adj[u, v]:
                best_edges.append([u, v])
    
    # Try to extend to larger n
    for n_target in [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]:
        if time.time() > deadline - 15:
            break
        
        best_adj_for_n = None
        best_edge_count = -1
        
        for attempt in range(15):
            if time.time() > deadline - 10:
                break
            
            adj = np.zeros((n_target, n_target), dtype=bool)
            adj[:base_n, :base_n] = base_adj
            
            # Add vertices one by one
            success = True
            for new_idx in range(base_n, n_target):
                if time.time() > deadline - 10:
                    success = False
                    break
                
                # Try multiple random orderings
                found = False
                for trial in range(5):
                    new_adj = _try_add_vertex(adj, new_idx, new_idx, n_target)
                    if new_adj is not None:
                        adj = new_adj
                        found = True
                        break
                
                if not found:
                    success = False
                    break
            
            if not success:
                continue
            
            # Local search to add more edges
            adj = _local_search_add_edges(adj, n_target)
            
            # Verify C4-free
            if not _check_c4_free(adj, n_target):
                continue
            
            # Check independence number
            adj_mask = _to_bitmask(adj, n_target)
            greedy_alpha = _greedy_independent_set_size(adj_mask, n_target)
            
            if greedy_alpha >= 11:
                continue
            
            # Exact check
            exact_alpha = _max_independent_set_size_bitmask(adj_mask, n_target, time_limit=3.0)
            if exact_alpha >= 11:
                continue
            
            # Valid graph
            edge_count = np.sum(adj) // 2
            if edge_count > best_edge_count:
                best_edge_count = edge_count
                best_adj_for_n = adj.copy()
            
            if time.time() > deadline - 10:
                break
        
        if best_adj_for_n is not None:
            edges = []
            for u in range(n_target):
                for v in range(u + 1, n_target):
                    if best_adj_for_n[u, v]:
                        edges.append([u, v])
            
            if n_target > best_n:
                best_n = n_target
                best_edges = edges
    
    return {'n': best_n, 'edges': best_edges}
