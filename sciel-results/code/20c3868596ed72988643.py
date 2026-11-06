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

def _has_independent_set(adj, n, target):
    """
    Check if there exists an independent set of size >= target.
    Uses branch and bound with bitmasks.
    Returns True if an independent set of size 'target' exists.
    """
    # We want to find if there is an IS of size 'target'.
    # This is equivalent to finding a clique of size 'target' in the complement.
    # But we can just do IS on the original graph.
    
    # Optimization: If n < target, return False
    if n < target:
        return False
    
    # adj is a list of bitmasks
    
    def exists_is(vertices_mask, k):
        if k == 0:
            return True
        if not vertices_mask:
            return False
        
        # Upper bound: if number of vertices left < k, impossible
        if vertices_mask.bit_count() < k:
            return False
        
        # Heuristic: Pick a vertex. 
        # Using the vertex with the highest degree in the induced subgraph 
        # often prunes the "include" branch more effectively (removes more vertices).
        # However, calculating degree for all vertices in the mask is O(n^2).
        # For n=43, this is fast.
        
        # Let's try a simpler heuristic first: lowest bit.
        # If that's too slow, we can switch to max degree.
        # Given the tight deadline, let's stick to a robust but simple heuristic.
        # Actually, "lowest bit" is very fast.
        
        lsb = vertices_mask & -vertices_mask
        v = lsb.bit_length() - 1
        
        # Option 1: Include v
        # Remove v and its neighbors
        new_mask_include = vertices_mask & ~(1 << v) & ~adj[v]
        if exists_is(new_mask_include, k - 1):
            return True
            
        # Option 2: Exclude v
        new_mask_exclude = vertices_mask ^ (1 << v)
        if exists_is(new_mask_exclude, k):
            return True
            
        return False

    all_vertices = (1 << n) - 1
    return exists_is(all_vertices, target)

def _build_polarity_graph_q5():
    """
    Construct the polarity graph of PG(2,5).
    n = 31.
    """
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for x in range(q):
        points.append((0, 1, x))
    points.append((0, 0, 1))
    
    n = len(points)
    adj = [0] * n
    edges = []
    
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

def _construct_graph():
    deadline = time.time() + 110
    
    # Start with base graph
    n_base, adj_base, edges_base = _build_polarity_graph_q5()
    
    # Verify base
    if _has_c4_bitmask(n_base, adj_base) or _has_independent_set(adj_base, n_base, 11):
        # Fallback if base is invalid (should not happen)
        return {'n': 31, 'edges': edges_base}
        
    best_n = n_base
    best_edges = edges_base.copy()
    
    # Try to extend to 39, 38, ...
    for target_n in range(39, n_base, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple random attempts
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            random.seed(attempt * 12345 + target_n)
            
            adj = adj_base.copy()
            edges = edges_base.copy()
            n = n_base
            
            success = True
            for _ in range(target_n - n_base):
                new_v = n
                n += 1
                adj.append(0)
                
                # We need to select a neighbor set S for new_v.
                # S must be an independent set in the "conflict graph" H.
                # H has vertices 0..n-1. u~w in H if adj[u] & adj[w] != 0.
                # We can generate random independent sets in H.
                
                # To make it efficient, we can just try to add a vertex with a random subset of neighbors
                # and check C4 constraint locally.
                
                # Better: Generate a random subset of existing vertices, check if it's valid.
                # Validity: For all u, w in S, adj[u] & adj[w] == 0.
                
                # Let's try to pick neighbors greedily from a shuffled list.
                candidates = list(range(new_v))
                random.shuffle(candidates)
                
                chosen = []
                # We want chosen to be "large" to constrain the graph, 
                # but not too large to violate C4-free property.
                # In PG(2,5), degrees are around 5-6.
                # Let's aim for degree 4-8 for new vertices.
                
                for s in candidates:
                    # Check if s is compatible with all chosen
                    compatible = True
                    for s2 in chosen:
                        if adj[s] & adj[s2] != 0:
                            compatible = False
                            break
                    if compatible:
                        chosen.append(s)
                        if len(chosen) >= 8: # Limit degree
                            break
                        # Optional: Stop if we've picked enough
                        if len(chosen) >= 4 and random.random() < 0.5:
                            break
                
                # Add edges
                for s in chosen:
                    adj[new_v] |= (1 << s)
                    adj[s] |= (1 << new_v)
                    edges.append([new_v, s])
                
                # Quick check: Did we create a C4?
                # We can check this incrementally, but _has_c4_bitmask is O(n^2).
                # For n=39, it's fast.
                if _has_c4_bitmask(n, adj):
                    success = False
                    break
                    
                # Check independence number periodically?
                # It's expensive. Let's check at the end of the extension loop.
                # Or, we can check if alpha < 11.
                # If alpha >= 11, this path is dead.
                # But checking alpha every step is slow.
                # Let's check every 2 steps or at the end.
                # Given the small n, checking at the end of the full extension is safer for correctness,
                # but might waste time on invalid paths.
                # Let's check alpha every time we add a vertex, but only if n is small enough?
                # No, let's check alpha only at the end of the target_n loop.
                
            if not success:
                continue
                
            # Final verification
            if _has_c4_bitmask(n, adj) or _has_independent_set(adj, n, 11):
                continue
                
            # Valid graph found
            best_n = n
            best_edges = edges.copy()
            found = True
            break # Found a valid graph for this target_n, try larger? No, we iterate down.
            # Wait, the loop is `for target_n in range(39, n_base, -1)`.
            # If we find a valid graph for 39, we break out of the inner loop.
            # Then we should break out of the outer loop too, as 39 is the max we target.
            
        if found:
            break
            
    return {'n': best_n, 'edges': best_edges}

def construct():
    start_time = time.time()
    result = _construct_graph()
    return result
