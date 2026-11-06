import random
import time

def _make_bitmask_adj(edge_list, n=16):
    """Convert edge list to list of integer bitmasks representing adjacency."""
    adj = [0] * n
    for u, v in edge_list:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_triangle_free(adj):
    """Verify that adjacency bitmasks define a simple triangle-free graph."""
    n = len(adj)
    for u in range(n):
        mask = adj[u]
        while mask:
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            mask ^= lb
            # Check if u and v share any common neighbor
            if adj[v] & adj[u]:
                return False
    return True

def _is_3_colorable(adj):
    """
    Check if the graph is 3-colorable using backtracking with MRV heuristic.
    Returns True if 3-colorable, False otherwise.
    """
    n = len(adj)
    color = [0] * n
    
    def backtrack():
        # Find uncolored vertex with fewest available colors (MRV)
        best_v = -1
        min_avail = 8
        best_avail_mask = 0
        
        for v in range(n):
            if color[v] == 0:
                # Determine which colors are unavailable due to colored neighbors
                unavail = 0
                mask = adj[v]
                while mask:
                    lb = mask & (-mask)
                    u = lb.bit_length() - 1
                    mask ^= lb
                    if color[u] != 0:
                        unavail |= (1 << (color[u] - 1))
                avail = 7 ^ unavail  # 3 colors = bits 0,1,2
                count = bin(avail).count('1')
                if count == 0:
                    return False
                if count < min_avail:
                    min_avail = count
                    best_v = v
                    best_avail_mask = avail
                    if count == 1:
                        break
        
        if best_v == -1:
            return True  # All vertices colored
        
        # Try each available color for best_v
        avail = best_avail_mask
        while avail:
            lb = avail & (-avail)
            c = lb.bit_length()  # color is 1-indexed: bit 0 -> color 1, bit 1 -> color 2, bit 2 -> color 3
            avail ^= lb
            color[best_v] = c
            if backtrack():
                return True
            color[best_v] = 0
        return False
    
    return backtrack()

def _greedy_add_edges(adj, edge_set, missing_edges, rng, deadline):
    """
    Greedily add edges from missing_edges in random order while maintaining triangle-freeness.
    Returns updated adj and edge_set.
    """
    for u, v in missing_edges:
        if time.time() > deadline:
            break
        # Check if adding (u,v) creates a triangle
        common = adj[u] & adj[v]
        if common == 0:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_set.add((u, v))
            # Remove from missing_edges
            try:
                missing_edges.remove((u, v))
            except ValueError:
                pass
    return adj, edge_set, missing_edges

def _swap_search(adj, edge_set, missing_edges, rng, deadline, max_swaps=500):
    """
    Local search: try swapping edges to increase edge count while maintaining
    triangle-freeness and non-3-colorability.
    """
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        
        # Pick a random missing edge
        if not missing_edges:
            break
        idx = rng.randint(0, len(missing_edges) - 1)
        u, v = missing_edges[idx]
        
        # Find common neighbors of u and v
        common = adj[u] & adj[v]
        if common == 0:
            # Can add directly
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_set.add((u, v))
            missing_edges.pop(idx)
            continue
        
        # Try removing an edge incident to a common neighbor to unblock (u,v)
        common_list = []
        m = common
        while m:
            lb = m & (-m)
            w = lb.bit_length() - 1
            m ^= lb
            common_list.append(w)
        
        rng.shuffle(common_list)
        
        for w in common_list:
            if time.time() > deadline:
                break
            
            # Try removing (u,w) or (w,v)
            for edge_to_remove in [(u, w), (w, v)]:
                ru, rv = min(edge_to_remove), max(edge_to_remove)
                
                if (ru, rv) not in edge_set:
                    continue
                
                # Save state
                adj_backup = [a for a in adj]
                edge_set_backup = set(edge_set)
                
                # Remove edge
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                edge_set.discard((ru, rv))
                
                # Check if (u,v) can now be added
                new_common = adj[u] & adj[v]
                if new_common == 0:
                    # Add (u,v)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    
                    # Try to add more edges greedily
                    for mu, mv in missing_edges[:30]:
                        if (mu, mv) == (u, v):
                            continue
                        c2 = adj[mu] & adj[mv]
                        if c2 == 0:
                            adj[mu] |= (1 << mv)
                            adj[mv] |= (1 << mu)
                            edge_set.add((mu, mv))
                            try:
                                missing_edges.remove((mu, mv))
                            except ValueError:
                                pass
                    
                    # Check if graph is still non-3-colorable
                    if not _is_3_colorable(adj):
                        # Good swap
                        # Update missing_edges
                        missing_edges = [(a, b) for (a, b) in all_edges if (a, b) not in edge_set]
                        break
                    else:
                        # Revert
                        adj = adj_backup
                        edge_set = edge_set_backup
                
                # Revert if we didn't break
                if adj != adj_backup:
                    adj = adj_backup
                    edge_set = edge_set_backup
        
        # Update missing_edges periodically
        if _ % 50 == 0:
            missing_edges = [(a, b) for (a, b) in all_edges if (a, b) not in edge_set]
    
    return adj, edge_set, missing_edges

def _build_grotzsch_edges():
    """Return the edge list of the Grötzsch graph (Mycielski of C5) on vertices 0..10."""
    # Cycle on 0..4
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    # Mycielski: vertices 5..9 are clones of 0..4
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]]
    # Vertex 10 connected to all clones 5..9
    edges += [[10, 5 + u] for u in range(5)]
    return edges

def construct():
    """
    Construct a triangle-free graph on 16 vertices that is not 3-colorable,
    maximizing the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Start with Grötzsch graph on vertices 0..10
    grotzsch = _build_grotzsch_edges()
    
    best_adj = None
    best_edge_set = None
    best_count = 0
    
    # Multiple restarts with different random seeds
    seed = 0
    while time.time() < deadline - 5.0:
        rng = random.Random(seed)
        
        # Initialize with Grötzsch graph
        adj = _make_bitmask_adj(grotzsch, n)
        edge_set = set()
        for u, v in grotzsch:
            edge_set.add((min(u, v), max(u, v)))
        
        # Get missing edges
        missing_edges = [(u, v) for (u, v) in all_edges if (u, v) not in edge_set]
        rng.shuffle(missing_edges)
        
        # Phase 1: Greedy completion
        adj, edge_set, missing_edges = _greedy_add_edges(adj, edge_set, missing_edges, rng, deadline)
        
        # Phase 2: Local search with swaps
        if time.time() < deadline - 3.0:
            adj, edge_set, missing_edges = _swap_search(adj, edge_set, missing_edges, rng, deadline)
        
        # Validate
        if _is_triangle_free(adj) and not _is_3_colorable(adj):
            count = len(edge_set)
            if count > best_count:
                best_count = count
                best_adj = [a for a in adj]
                best_edge_set = set(edge_set)
        
        seed += 1
    
    # If no valid graph found, fallback to Grötzsch
    if best_adj is None:
        best_adj = _make_bitmask_adj(grotzsch, n)
        best_edge_set = set()
        for u, v in grotzsch:
            best_edge_set.add((min(u, v), max(u, v)))
    
    # Convert to edge list
    result_edges = []
    for u, v in sorted(best_edge_set):
        result_edges.append([u, v])
    
    return result_edges
