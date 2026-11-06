import time
import random

def _has_c4_bitmask(n, adj):
    """Check if graph has a C4 using bitmasks. Returns True if C4 found."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if common & (common - 1):  # at least two bits set
                return True
    return False

def _independent_set_size_bitmask(n, adj, target=11, time_limit=5.0):
    """Check if there exists an independent set of size >= target using branching with bitmasks."""
    start_time = time.time()
    
    # Memoization is not effective due to large state space.
    # We use DFS with pruning.
    
    # Pre-compute degrees? 
    # degrees = [adj[i].bit_count() for i in range(n)]
    
    def max_is(vertices_mask, remaining):
        if remaining <= 0:
            return 0
        if not vertices_mask:
            return 0
        
        # Time check
        if time.time() - start_time > time_limit:
            return 0 # Assume small? Or raise? 
            # Better to return a low value to indicate "not found" quickly, 
            # but this might be incorrect. 
            # However, for construction, we just need to know if it's < 11.
            # If we time out, we might incorrectly think alpha < 11.
            # To be safe, if time runs out, we should probably assume it's >= 11 
            # and reject the graph, or use a different verification.
            # Let's return -1 to signal timeout.
            return -1

        # Upper bound: count bits in vertices_mask
        count = vertices_mask.bit_count()
        if count < remaining:
            return count
        
        # Heuristic: pick vertex with max degree in induced subgraph
        # This prunes the 'include' branch more effectively.
        v = -1
        max_deg = -1
        temp = vertices_mask
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            deg = (adj[u] & vertices_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = u
            temp ^= lsb
        
        if v == -1:
            return 0
            
        lsb_v = 1 << v
        
        # Option 1: include v
        new_mask = vertices_mask & ~(lsb_v) & ~adj[v]
        include_val = 1 + max_is(new_mask, remaining - 1)
        if include_val == -1: return -1
        if include_val >= target:
            return include_val # Found a large set
        
        # Option 2: exclude v
        exclude_val = max_is(vertices_mask ^ lsb_v, remaining)
        if exclude_val == -1: return -1
        
        return max(include_val, exclude_val)
    
    all_vertices = (1 << n) - 1
    result = max_is(all_vertices, target)
    return result

def _verify_graph(n, edges):
    """Verify the graph is C4-free and has no independent set of size 11."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    if _has_c4_bitmask(n, adj):
        return False
    
    # Check alpha
    alpha = _independent_set_size_bitmask(n, adj, 11, time_limit=10.0)
    if alpha == -1:
        # Timeout, assume invalid to be safe? Or try again?
        # Let's assume invalid.
        return False
    if alpha >= 11:
        return False
    
    return True

def _try_add_edge(adj, n, u, v):
    """Check if adding edge (u,v) creates a C4. Returns True if safe to add."""
    common = adj[u] & adj[v]
    return not (common & (common - 1))  # safe if fewer than 2 common neighbors

def _construct_graph(n, seed):
    """
    Construct a maximal C4-free graph on n vertices using random greedy.
    Returns edges list if valid (alpha < 11), else None.
    """
    random.seed(seed)
    adj = [0] * n
    edges = []
    
    # Generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)
    
    # Add edges greedily
    for u, v in all_edges:
        if _try_add_edge(adj, n, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
            
    # Check independence number
    alpha = _independent_set_size_bitmask(n, adj, 11, time_limit=5.0)
    if alpha != -1 and alpha < 11:
        return edges
    return None

def construct():
    """Construct a C4-free graph with no independent set of size 11, maximizing n."""
    start_time = time.time()
    deadline = start_time + 110  # Leave margin
    
    # Try to construct graphs with decreasing n
    for n in range(43, 29, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple seeds for each n
        attempts = 0
        while time.time() < deadline - 20: # Stop 20s before deadline
            attempts += 1
            seed = attempts * 12345 + n
            edges = _construct_graph(n, seed)
            if edges:
                # Verify just in case
                if _verify_graph(n, edges):
                    return {'n': n, 'edges': edges}
                found = True
                break
        
        if found:
            break
            
    # Fallback: 10 disjoint triangles (n=30)
    edges = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}
