import random
import time

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
            # If u and v are adjacent, their common neighbors must be empty
            # adj[u] & adj[v] should not contain any bit other than possibly u and v?
            # Actually, if (u,v) is an edge, and w is a neighbor of both, then u-v-w is a triangle.
            # So adj[u] & adj[v] must be 0?
            # Wait. adj[u] contains v. adj[v] contains u.
            # So adj[u] & adj[v] will contain u and v? No.
            # adj[u] has bit v. adj[v] has bit u.
            # (1<<v) is in adj[u]. Is (1<<v) in adj[v]? No, unless self-loop.
            # So adj[u] & adj[v] should be 0 for triangle-free?
            # Let's check.
            # If w is common neighbor: w in adj[u] and w in adj[v].
            # Then bit w is in adj[u] & adj[v].
            # So if adj[u] & adj[v] != 0, there is a triangle.
            # BUT, does adj[u] contain u? No.
            # Does adj[v] contain v? No.
            # So if (u,v) is an edge, adj[u] has bit v, adj[v] has bit u.
            # adj[u] & adj[v] does NOT contain u or v.
            # So adj[u] & adj[v] must be 0.
            if adj[u] & adj[v]:
                return False
    return True

def _is_3_colorable(adj):
    """Check if the graph is 3-colorable using backtracking."""
    n = len(adj)
    color = [0] * n
    
    def backtrack():
        # MRV: pick uncolored vertex with fewest available colors
        best_v = -1
        min_avail = 8
        for v in range(n):
            if color[v] == 0:
                unavail = 0
                m = adj[v]
                while m:
                    lb = m & (-m)
                    u = lb.bit_length() - 1
                    m ^= lb
                    if color[u] != 0:
                        unavail |= (1 << (color[u]-1))
                avail = 7 ^ unavail
                count = bin(avail).count('1')
                if count == 0:
                    return False
                if count < min_avail:
                    min_avail = count
                    best_v = v
                    if count == 1:
                        break
        
        if best_v == -1:
            return True # All colored
            
        unavail = 0
        m = adj[best_v]
        while m:
            lb = m & (-m)
            u = lb.bit_length() - 1
            m ^= lb
            if color[u] != 0:
                unavail |= (1 << (color[u]-1))
        
        avail = 7 ^ unavail
        
        c = 1
        while c <= 3:
            if avail & (1 << (c-1)):
                color[best_v] = c
                if backtrack():
                    return True
                color[best_v] = 0
            c += 1
        return False

    return backtrack()

def _get_missing_edges(adj, n=16):
    """Return list of edges that can be added without creating a triangle."""
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] & (1 << v)):
                # Check if adding (u,v) creates a triangle
                # This is equivalent to checking if adj[u] & adj[v] == 0
                # Wait, adj[u] & adj[v] checks common neighbors.
                # If there is a common neighbor w, then u-w-v is a path of length 2.
                # Adding u-v makes it a triangle.
                # So we need adj[u] & adj[v] == 0.
                if adj[u] & adj[v] == 0:
                    candidates.append((u, v))
    return candidates

def construct():
    """
    Construct a triangle-free graph on 16 vertices that is not 3-colorable,
    maximizing the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 16
    best_edges = []
    best_count = 0
    
    grotzsch_edges = _build_grotzsch_edges()
    # Pad with isolated vertices 11..15
    initial_edges = list(grotzsch_edges)
    
    # We will use a local search
    # State: adj bitmask
    # We maintain the edge list for reconstruction
    
    def evaluate(adj):
        """Returns (is_not_3_colorable, edge_count)"""
        count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
        is_3c = _is_3_colorable(adj)
        return (not is_3c), count

    # Initial state
    adj = _make_bitmask_adj(initial_edges, n)
    is_not_3c, count = evaluate(adj)
    if is_not_3c:
        best_count = count
        # Reconstruct edge list from adj for best
        best_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    best_edges.append([u, v])
    
    rng = random.Random(42)
    
    # Local Search
    # We try to add edges. If it becomes 3-colorable, we revert.
    # If we get stuck, we remove an edge to explore other regions.
    
    stuck_counter = 0
    max_stuck = 50
    
    while time.time() < deadline:
        # Get candidates for addition
        candidates = _get_missing_edges(adj, n)
        
        if not candidates:
            # No edges can be added
            # Try to remove an edge to unlock new possibilities?
            # Or just break if we can't improve
            if stuck_counter > max_stuck:
                # Try a different random restart?
                # For now, just break or try removing
                # Let's try removing a random edge to see if we can add different ones
                all_edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u] & (1 << v):
                            all_edges.append((u, v))
                if all_edges:
                    u, v = rng.choice(all_edges)
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    stuck_counter = 0
                else:
                    break
            else:
                stuck_counter += 1
                continue
        
        # Pick a random candidate
        u, v = rng.choice(candidates)
        
        # Tentatively add
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Check 3-colorability
        is_3c = _is_3_colorable(adj)
        
        if not is_3c:
            # Valid and non-3-colorable
            count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
            if count > best_count:
                best_count = count
                best_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            best_edges.append([i, j])
            stuck_counter = 0
        else:
            # Revert
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            stuck_counter += 1
            
            if stuck_counter > max_stuck:
                # Perturb: remove a random edge
                all_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            all_edges.append((i, j))
                if all_edges:
                    i, j = rng.choice(all_edges)
                    adj[i] ^= (1 << j)
                    adj[j] ^= (1 << i)
                    stuck_counter = 0
                else:
                    break

    # Final validation
    if best_count == 0:
        # Fallback to Grötzsch
        best_edges = list(grotzsch_edges)
        
    adj_final = _make_bitmask_adj(best_edges, n)
    if not _is_triangle_free(adj_final) or _is_3_colorable(adj_final):
        # If validation fails, return Grötzsch
        best_edges = list(grotzsch_edges)
        
    return best_edges
