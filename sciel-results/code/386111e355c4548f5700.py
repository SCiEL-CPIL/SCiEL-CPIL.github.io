import time
import random
import itertools

# EVOLVE-BLOCK-START

def _has_c4_bitmask(n, adj):
    """Check if graph has a C4 using bitmasks. Returns True if C4 found."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if common & (common - 1):  # at least two bits set
                return True
    return False

def _max_independent_set_size_bitmask(n, adj, target=11):
    """
    Check if there exists an independent set of size >= target.
    Returns the size of the maximum independent set found (capped at target).
    Uses branch and bound with better pruning.
    """
    # Precompute degree for ordering
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    
    # Sort vertices by degree descending for better branching
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    # Map old vertex index to new position in order
    pos_in_order = [0] * n
    for i, v in enumerate(order):
        pos_in_order[v] = i
    
    # Rebuild adj in new order
    new_adj = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and (adj[i] >> j) & 1:
                new_adj[i] |= (1 << pos_in_order[j])
    
    # Now use branch and bound on reordered graph
    def max_is(vertices_mask, remaining, current_size):
        if remaining <= 0:
            return current_size
        if not vertices_mask:
            return current_size
        
        # Upper bound
        count = bin(vertices_mask).count('1')
        if count + current_size < target:
            return current_size
        if count < remaining:
            return current_size + count
        
        # Pick vertex with highest degree in remaining subgraph
        # Use first set bit for simplicity, but try a few heuristics
        lsb = vertices_mask & -vertices_mask
        v = lsb.bit_length() - 1
        
        # Option 1: include v
        new_mask = vertices_mask & ~(1 << v) & ~new_adj[v]
        include_val = max_is(new_mask, remaining - 1, current_size + 1)
        
        if include_val >= target:
            return include_val
            
        # Option 2: exclude v
        exclude_val = max_is(vertices_mask ^ (1 << v), remaining, current_size)
        
        return max(include_val, exclude_val)
    
    all_vertices = (1 << n) - 1
    return max_is(all_vertices, target, 0)

def _verify_graph(n, edges):
    """Verify the graph is C4-free and has no independent set of size 11."""
    if n < 1:
        return False
    adj = [0] * n
    for u, v in edges:
        if u == v:
            return False
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if adj[u] & (1 << v):
            return False # Duplicate edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    if _has_c4_bitmask(n, adj):
        return False
    
    if _max_independent_set_size_bitmask(n, adj, 11) >= 11:
        return False
    
    return True

def _build_polarity_graph_q5():
    """
    Construct the polarity graph of PG(2,5).
    n = 31.
    """
    q = 5
    # Points of PG(2, q)
    points = []
    # Affine points: (1, x, y)
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    # Points at infinity: (0, 1, x)
    for x in range(q):
        points.append((0, 1, x))
    # Point at infinity: (0, 0, 1)
    points.append((0, 0, 1))
    
    n = len(points)
    adj = [0] * n
    edges = []
    
    # Polarity: u . v = 0 mod q
    # Using standard dot product
    for i in range(n):
        u = points[i]
        for j in range(i + 1, n):
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append([i, j])
                
    return n, adj, edges

def _find_independent_set(adj, n, size):
    """Find an independent set of given size in the graph."""
    # Greedy approach with backtracking
    def backtrack(available_mask, chosen, needed):
        if needed == 0:
            return chosen[:]
        if not available_mask:
            return None
        if bin(available_mask).count('1') < needed:
            return None
        
        lsb = available_mask & -available_mask
        v = lsb.bit_length() - 1
        
        # Try including v
        new_mask = available_mask & ~(1 << v) & ~adj[v]
        result = backtrack(new_mask, chosen + [v], needed - 1)
        if result is not None:
            return result
        
        # Try excluding v
        return backtrack(available_mask ^ (1 << v), chosen, needed)
    
    all_vertices = (1 << n) - 1
    return backtrack(all_vertices, [], size)

def _construct_39_vertex_graph():
    """
    Construct a 39-vertex C4-free graph with alpha < 11.
    Strategy: Start with PG(2,5) polarity graph (31 vertices),
    find an independent set of size 8, add 8 new vertices each connected
    to all vertices in the independent set, and make the 8 new vertices a clique.
    """
    deadline = time.time() + 110
    
    n_base, adj_base, edges_base = _build_polarity_graph_q5()
    
    # Find an independent set of size 8 in the base graph
    independent_set = _find_independent_set(adj_base, n_base, 8)
    
    if independent_set is None:
        # Fallback to base graph
        return {'n': n_base, 'edges': edges_base}
    
    # Create new graph with 39 vertices
    n_new = n_base + 8
    adj = adj_base.copy()
    edges = edges_base.copy()
    
    # Add 8 new vertices (indices 31-38)
    for i in range(8):
        adj.append(0)
    
    # Connect each new vertex to all vertices in the independent set
    for i in range(8):
        new_v = n_base + i
        for v in independent_set:
            adj[new_v] |= (1 << v)
            adj[v] |= (1 << new_v)
            edges.append([new_v, v])
    
    # Make the 8 new vertices a clique
    for i in range(8):
        for j in range(i + 1, 8):
            u, v = n_base + i, n_base + j
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
    
    # Verify the construction
    if _verify_graph(n_new, edges):
        return {'n': n_new, 'edges': edges}
    
    # If verification fails, fall back to base
    return {'n': n_base, 'edges': edges_base}

def _construct_large_graph():
    """
    Try to construct a graph with n > 31.
    First tries the deterministic 39-vertex construction,
    then falls back to random extensions.
    """
    deadline = time.time() + 110
    
    # Try the deterministic construction first
    result = _construct_39_vertex_graph()
    
    if result['n'] >= 39:
        return result
    
    # If that didn't work, try random extensions from base
    n_base, adj_base, edges_base = _build_polarity_graph_q5()
    
    best_result = {'n': n_base, 'edges': edges_base}
    
    for target_n in range(39, n_base, -1):
        if time.time() > deadline:
            break
            
        found = False
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            random.seed(attempt * 1000 + target_n)
            
            adj = adj_base.copy()
            edges = edges_base.copy()
            n = n_base
            
            success = True
            for _ in range(target_n - n_base):
                new_v = n
                n += 1
                adj.append(0)
                
                candidates = list(range(new_v))
                random.shuffle(candidates)
                
                chosen = []
                available = set(range(new_v))
                
                while available and len(chosen) < 5:
                    s = random.choice(list(available))
                    chosen.append(s)
                    available.remove(s)
                    
                    ns = adj[s]
                    forb_mask = 0
                    temp = ns
                    while temp:
                        lsb = temp & -temp
                        u = lsb.bit_length() - 1
                        forb_mask |= adj[u]
                        temp ^= lsb
                    
                    new_available = set()
                    for v in available:
                        if not (forb_mask & (1 << v)):
                            new_available.add(v)
                    available = new_available
                    
                for s in chosen:
                    adj[new_v] |= (1 << s)
                    adj[s] |= (1 << new_v)
                    edges.append([new_v, s])
                    
            if not _has_c4_bitmask(n, adj):
                if _max_independent_set_size_bitmask(n, adj, 11) < 11:
                    best_result = {'n': n, 'edges': edges.copy()}
                    found = True
                    break
                    
        if found:
            break
            
    return best_result

def construct():
    """Construct a C4-free graph with no independent set of size 11, maximizing n."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Try the deterministic 39-vertex construction
    result = _construct_39_vertex_graph()
    
    if time.time() > deadline:
        # Return best known valid graph
        n_base, adj_base, edges_base = _build_polarity_graph_q5()
        return {'n': n_base, 'edges': edges_base}
    
    if _verify_graph(result['n'], result['edges']):
        return result
    
    # Try random extensions
    result = _construct_large_graph()
    
    if time.time() > deadline:
        n_base, adj_base, edges_base = _build_polarity_graph_q5()
        return {'n': n_base, 'edges': edges_base}
    
    if _verify_graph(result['n'], result['edges']):
        return result
    
    # Fallback: polarity graph of PG(2,5)
    n_base, adj_base, edges_base = _build_polarity_graph_q5()
    return {'n': n_base, 'edges': edges_base}

# EVOLVE-BLOCK-END
