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

def _add_edge_safely(adj, edges, u, v, n=16):
    """
    Try to add edge (u,v) while maintaining triangle-freeness.
    Returns True if edge was added, False otherwise.
    """
    if (adj[u] >> v) & 1:
        return False  # Already exists
    # Check triangle: no common neighbor
    if adj[u] & adj[v]:
        return False
    # Add the edge
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    edges.append([u, v])
    return True

def _remove_edge(adj, edges, u, v, n=16):
    """Remove edge (u,v) from adjacency and edge list."""
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)
    # Remove from edges list
    for i, e in enumerate(edges):
        if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
            edges.pop(i)
            break

def _random_search(adj, edges, deadline, rng, n=16):
    """
    Randomized search: try to add edges and occasionally remove them
    to explore the space of triangle-free graphs, looking for a 4-chromatic
    graph with more edges than the current best.
    """
    best_edge_count = len(edges)
    best_edges_snapshot = list(edges)
    best_adj_snapshot = list(adj)
    
    # Precompute all non-edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline:
        # Randomly pick a non-edge and try to add it
        # Focus on adding edges to increase density
        attempt = rng.random()
        
        if attempt < 0.7:
            # Try adding a random non-edge
            # Pick a random pair and check if it's a non-edge
            for _ in range(50):  # Try up to 50 random pairs
                u = rng.randrange(n)
                v = rng.randrange(n)
                if u == v:
                    continue
                if (adj[u] >> v) & 1:
                    continue
                # Try to add
                if _add_edge_safely(adj, edges, u, v, n):
                    if len(edges) > best_edge_count:
                        # Check if 3-colorable
                        if not _is_3_colorable(adj, n):
                            best_edge_count = len(edges)
                            best_edges_snapshot = list(edges)
                            best_adj_snapshot = list(adj)
                    break
        elif attempt < 0.85:
            # Try a systematic add: go through vertices in order
            u = rng.randrange(n)
            for v in range(n):
                if v == u or (adj[u] >> v) & 1:
                    continue
                if _add_edge_safely(adj, edges, u, v, n):
                    if len(edges) > best_edge_count:
                        if not _is_3_colorable(adj, n):
                            best_edge_count = len(edges)
                            best_edges_snapshot = list(edges)
                            best_adj_snapshot = list(adj)
                    break
        else:
            # Occasionally remove an edge to explore (simulated annealing style)
            if edges:
                idx = rng.randrange(len(edges))
                u, v = edges[idx]
                _remove_edge(adj, edges, u, v, n)
    
    return best_edges_snapshot

def construct():
    """
    Construct a triangle-free graph on vertices 0..15 that is not 3-colourable.
    Maximize the number of edges. Uses the Clebsch graph (40 edges) as baseline
    and runs a randomized search to attempt to find a denser 4-chromatic
    triangle-free graph.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Start with Clebsch graph
    edges = _clebsch_edges()
    adj = _build_adjacency(edges)
    
    # Verify it's valid
    assert _is_triangle_free(adj), "Clebsch graph should be triangle-free"
    assert not _is_3_colorable(adj), "Clebsch graph should not be 3-colorable"
    
    base_count = len(edges)
    
    # If we don't have enough time, just return the Clebsch graph
    if time.time() > deadline - 5:
        return edges
    
    # Run randomized search
    rng = random.Random(42)
    best_edges = _random_search(adj, edges, deadline, rng)
    
    # Return the best found
    return best_edges

# EVOLVE-BLOCK-END
