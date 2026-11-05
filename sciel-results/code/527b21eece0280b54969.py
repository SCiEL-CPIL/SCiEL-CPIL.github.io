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

def _greedy_maximal_triangle_free(all_pairs, rng, n=16):
    """
    Greedy construction: shuffle all pairs, add each if it doesn't create a triangle.
    Returns the resulting edge list and adjacency.
    """
    adj = [0] * n
    edges = []
    
    for u, v in all_pairs:
        # Check if adding (u,v) creates a triangle
        if adj[u] & adj[v]:
            continue  # Would create a triangle
        # Add the edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edges.append([u, v])
    
    return edges, adj

def construct():
    """
    Construct a triangle-free graph on vertices 0..15 that is not 3-colourable.
    Maximize the number of edges. Uses greedy maximal triangle-free construction
    with multiple random orderings, then checks 3-colorability.
    """
    start_time = time.time()
    deadline = start_time + 120  # Safety margin for evaluator's 150s limit
    
    # Start with Clebsch graph as baseline
    base_edges = _clebsch_edges()
    base_adj = _build_adjacency(base_edges)
    best_edges = list(base_edges)
    best_count = len(best_edges)
    
    # Precompute all possible pairs
    all_pairs = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    
    # Try many random orderings
    rng = random.Random(42)
    
    while time.time() < deadline - 5:
        # Shuffle a copy of all_pairs
        shuffled = all_pairs[:]
        rng.shuffle(shuffled)
        
        # Greedy construction
        edges, adj = _greedy_maximal_triangle_free(shuffled, rng, n=16)
        
        if len(edges) > best_count:
            # Check if this graph is 3-colorable
            if not _is_3_colorable(adj, n=16):
                best_edges = edges
                best_count = len(edges)
                # If we found a very dense graph, we can stop early
                if best_count >= 50:
                    break
    
    return best_edges

# EVOLVE-BLOCK-END
