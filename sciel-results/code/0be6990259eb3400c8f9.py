"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _construct_polarity_graph(q):
    """
    Construct the polarity graph of PG(2, q) using the standard orthogonal polarity.
    Vertices are 1D subspaces (points) of F_q^3.
    Two points are adjacent iff their representatives are orthogonal under the form x1*y1 + x2*y2 - x3*y3 = 0.
    This graph is C4-free.
    """
    reps = []
    seen = set()
    
    for z in range(q):
        for y in range(q):
            for x in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, -1, q)
                    norm = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, -1, q)
                    norm = (0, 1, (z * inv_y) % q)
                else:
                    norm = (0, 0, 1)
                
                if norm not in seen:
                    seen.add(norm)
                    reps.append(norm)
    
    n = len(reps)
    if n != q*q + q + 1:
        raise ValueError(f"Expected {q*q+q+1} vertices, got {n}")
        
    adj = [set() for _ in range(n)]
    edges = []
    
    # Form: u[0]*v[0] + u[1]*v[1] - u[2]*v[2] == 0
    for i in range(n):
        for j in range(i+1, n):
            u = reps[i]
            v = reps[j]
            val = (u[0]*v[0] + u[1]*v[1] - u[2]*v[2]) % q
            if val == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
                
    return n, edges, adj

def _check_no_c4(n, edges):
    """Check that no two vertices have two common neighbours."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        for j in range(i+1, n):
            common = adj[i] & adj[j]
            if len(common) >= 2:
                return False
    return True

def _find_independent_set(n, adj, target_size, deadline=None):
    """
    Find an independent set of size 'target_size'.
    Returns the set of vertices if found, None otherwise.
    """
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        masks[i] = mask
    
    all_mask = (1 << n) - 1
    path = []
    
    def branch(remaining_mask, current_set):
        if deadline and time.time() > deadline:
            return None
        if len(current_set) >= target_size:
            return current_set[:]
        if remaining_mask == 0:
            return None
        
        # Upper bound
        if len(current_set) + remaining_mask.bit_count() < target_size:
            return None
            
        # Pick vertex with max degree in remaining
        temp = remaining_mask
        v_idx = -1
        max_deg = -1
        while temp:
            lsb = temp & -temp
            v = (lsb.bit_length() - 1)
            neighbors_in_rem = masks[v] & remaining_mask
            deg = neighbors_in_rem.bit_count()
            if deg > max_deg:
                max_deg = deg
                v_idx = v
            temp -= lsb
            
        if v_idx == -1:
            return None
            
        v_bit = 1 << v_idx
        
        # Include v
        new_remaining = remaining_mask & ~(masks[v_idx] | v_bit)
        res = branch(new_remaining, current_set + [v_idx])
        if res:
            return res
            
        # Exclude v
        new_remaining = remaining_mask & ~v_bit
        res = branch(new_remaining, current_set)
        if res:
            return res
            
        return None

    return branch(all_mask, [])

def _verify_graph(n, edges, target_alpha=11, deadline=None):
    """Verify graph has no C4 and alpha < target_alpha."""
    if not _check_no_c4(n, edges):
        return False, None
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    is_set = _find_independent_set(n, adj, target_alpha, deadline)
    if is_set:
        return False, is_set
        
    return True, None

def construct():
    """
    Construct a C4-free graph with no independent 11-set.
    Strategy:
    1. Construct PG(2, 7) (n=57).
    2. Take the first 43 vertices.
    3. Check for independent set of 11.
    4. If found, prune vertices until valid.
    5. Return the best valid graph (max n).
    """
    start_time = time.time()
    deadline = start_time + 120 # Safety margin
    
    # Try PG(2, 7)
    try:
        n_full, edges_full, adj_full = _construct_polarity_graph(7)
        # n_full = 57
        
        # We want n <= 43. Let's take the first 43 vertices.
        target_n = 43
        if n_full < target_n:
            target_n = n_full
            
        # Subset of vertices: 0 to target_n-1
        active_vertices = list(range(target_n))
        
        # Build subgraph
        current_n = target_n
        current_edges = []
        current_adj = [set() for _ in range(current_n)]
        
        for u, v in edges_full:
            if u < current_n and v < current_n:
                current_edges.append([u, v])
                current_adj[u].add(v)
                current_adj[v].add(u)
                
        # Check validity
        is_valid, is_set = _verify_graph(current_n, current_edges, target_alpha=11, deadline=deadline)
        
        if not is_valid:
            # Prune
            # We have an independent set of 11.
            # Remove vertices from this set to break it.
            # We can remove up to 43 - 39 = 4 vertices and still be better than seed (31).
            # Actually, we want to maximize n.
            # Let's try to remove 1 vertex at a time.
            
            while not is_valid and current_n > 35: # Safety bound
                if time.time() > deadline - 5:
                    break
                
                # Pick a vertex from the found independent set to remove
                # We need to re-find the set if we modify the graph
                # For efficiency, we can just remove a vertex from the previous set
                # and re-verify.
                
                # Remove the last vertex of the independent set
                v_to_remove = is_set[-1]
                
                # Rebuild graph without v_to_remove
                # Map old indices to new indices
                # This is O(n^2). n is small (43).
                
                new_n = current_n - 1
                new_edges = []
                new_adj = [set() for _ in range(new_n)]
                
                # Create mapping
                old_to_new = {}
                new_idx = 0
                for i in range(current_n):
                    if i != v_to_remove:
                        old_to_new[i] = new_idx
                        new_idx += 1
                        
                for u, v in current_edges:
                    if u == v_to_remove or v == v_to_remove:
                        continue
                    nu = old_to_new[u]
                    nv = old_to_new[v]
                    new_edges.append([nu, nv])
                    new_adj[nu].add(nv)
                    new_adj[nv].add(nu)
                    
                current_n = new_n
                current_edges = new_edges
                current_adj = new_adj
                
                # Re-verify
                is_valid, is_set = _verify_graph(current_n, current_edges, target_alpha=11, deadline=deadline)
                
        if is_valid:
            return {'n': current_n, 'edges': current_edges}
            
    except Exception as e:
        pass

    # Fallback: PG(2, 5) n=31
    try:
        n5, edges5, adj5 = _construct_polarity_graph(5)
        is_valid, _ = _verify_graph(n5, edges5, target_alpha=11, deadline=deadline)
        if is_valid:
            return {'n': n5, 'edges': edges5}
    except Exception as e:
        pass

    # Fallback: 10 disjoint triangles (n=30, alpha=10)
    edges_30 = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges_30 += [[a,b], [a,c], [b,c]]
        
    return {'n': 30, 'edges': edges_30}
# EVOLVE-BLOCK-END
