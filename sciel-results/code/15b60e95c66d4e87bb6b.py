"""Construct a dense triangle-free 4-chromatic graph on 16 vertices."""
import time
import random

# EVOLVE-BLOCK-START

def _hamming_weight(x):
    """Return the number of 1-bits in x."""
    w = 0
    while x:
        x &= x - 1
        w += 1
    return w

def _clebsch_edges():
    """
    Return edges of the Clebsch graph (folded 5-cube) on 16 vertices 0..15.
    Vertices are 5-bit strings with MSB=0, i.e. integers 0..15.
    Two vertices are adjacent iff Hamming distance is 1 or 4.
    This graph is triangle-free, 4-chromatic, and has 40 edges.
    """
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            d = _hamming_weight(i ^ j)
            if d == 1 or d == 4:
                edges.append([i, j])
    return edges

def _build_adjacency(edges, n=16):
    """Build a bitset adjacency matrix from an edge list."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_triangle_free(adj, n=16):
    """Check if graph with bitset adjacency is triangle-free."""
    for u in range(n):
        a = adj[u]
        for v in range(n):
            if not (a >> v) & 1 or v <= u:
                continue
            # Check if u and v share a common neighbor
            if adj[u] & adj[v]:
                return False
    return True

def _is_3_colorable(adj, n=16):
    """
    Check 3-colorability via backtracking with MRV heuristic.
    Returns True if the graph is 3-colorable, False otherwise.
    """
    color = [-1] * n
    # Precompute neighbor lists
    neighbors = [[] for _ in range(n)]
    for u in range(n):
        a = adj[u]
        while a:
            b = a & (-a)
            v = b.bit_length() - 1
            a ^= b
            if v > u:
                neighbors[u].append(v)
                neighbors[v].append(u)
    
    # Count degrees
    deg = [len(neighbors[i]) for i in range(n)]
    
    def backtrack(uncolored, domains):
        """
        Backtracking search.
        uncolored: frozenset or tuple of uncolored vertices
        domains: dict mapping vertex -> set of available colors
        """
        if not uncolored:
            return True
        
        # MRV: pick uncolored vertex with smallest domain
        best_v = -1
        best_size = 4
        for v in uncolored:
            ds = domains[v]
            s = len(ds)
            if s == 0:
                return False
            if s < best_size:
                best_size = s
                best_v = v
                if s == 1:
                    break
        
        if best_v == -1:
            return True
        
        # Try each color in the domain
        for c in sorted(domains[best_v]):
            # Check consistency: no neighbor has color c
            conflict = False
            for nb in neighbors[best_v]:
                if color[nb] == c:
                    conflict = True
                    break
            if conflict:
                continue
            
            color[best_v] = c
            # Update domains of neighbors
            new_domains = dict(domains)
            new_uncolored = set(uncolored)
            new_uncolored.remove(best_v)
            feasible = True
            for nb in neighbors[best_v]:
                if nb in new_uncolored:
                    new_domains[nb] = new_domains[nb] - {c}
                    if not new_domains[nb]:
                        feasible = False
                        break
            
            if feasible:
                if backtrack(frozenset(new_uncolored), new_domains):
                    return True
            
            color[best_v] = -1
        
        return False
    
    # Initial domains: all 3 colors for all vertices
    initial_domains = {i: {0, 1, 2} for i in range(n)}
    return backtrack(frozenset(range(n)), initial_domains)

def _local_search(adj, edges, deadline, n=16):
    """
    Local search that tries to remove one edge and add multiple edges.
    """
    best_edges = list(edges)
    best_count = len(edges)
    
    # Iterate through all edges to remove
    for i in range(len(edges)):
        if time.time() > deadline:
            break
            
        u, v = edges[i]
        
        # Snapshot
        adj_snap = list(adj)
        edges_snap = list(edges)
        
        # Remove edge (u, v)
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        # Find and remove from list
        for j, e in enumerate(edges):
            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                edges.pop(j)
                break
                
        # Find all safe edges
        safe_edges = []
        for a in range(n):
            for b in range(a+1, n):
                if (adj[a] >> b) & 1:
                    continue
                if adj[a] & adj[b]:
                    continue
                safe_edges.append((a, b))
                
        # Greedily add safe edges
        for se in safe_edges:
            a, b = se
            # Check if still safe (adj might have changed)
            if (adj[a] >> b) & 1:
                continue
            if adj[a] & adj[b]:
                continue
                
            # Add
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
            edges.append([a, b])
            
            # Check 3-colorability
            if not _is_3_colorable(adj, n):
                # Keep
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = list(edges)
            else:
                # Revert
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                edges.pop()
                
        # Restore state
        adj = adj_snap
        edges = edges_snap
        
    return best_edges

def construct():
    """
    Construct a triangle-free graph on vertices 0..15 that is not 3-colourable.
    Maximize the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Start with Clebsch graph
    edges = _clebsch_edges()
    adj = _build_adjacency(edges)
    
    # Verify it's valid
    assert _is_triangle_free(adj), "Clebsch graph should be triangle-free"
    assert not _is_3_colorable(adj), "Clebsch graph should not be 3-colorable"
    
    # Run local search
    best_edges = _local_search(adj, edges, deadline)
    
    # Return the best found
    return best_edges

# EVOLVE-BLOCK-END
