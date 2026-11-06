"""
Triangle-free 4-chromatic graph on 16 vertices: maximize edges.
Strategy: Start from a strong seed (Grötzsch-based), then use randomized
local search with simulated-annealing-like acceptance to escape local optima.
Representation: bitmasks for adjacency and candidate edges for speed.
"""
# EVOLVE-BLOCK-START
import random
import time

def _build_grotzsch_edges():
    """Return edge list of Grötzsch graph on vertices 0..10."""
    c5 = [(0,1),(1,2),(2,3),(3,4),(4,0)]
    edges = [list(e) for e in c5]
    for u,v in c5:
        edges.append([u, 5+v])
        edges.append([v, 5+u])
    for i in range(5):
        edges.append([10, 5+i])
    return edges

def _edges_to_adj(edge_list, n=16):
    adj = [0]*n
    for u,v in edge_list:
        adj[u] |= (1<<v)
        adj[v] |= (1<<u)
    return adj

def _is_triangle_free(adj, n=16):
    for u in range(n):
        mask = adj[u]
        while mask:
            lb = mask & (-mask)
            v = lb.bit_length()-1
            mask ^= lb
            # Check if any common neighbor of u and v exists
            if adj[u] & adj[v] & ~(1<<u) & ~(1<<v):
                return False
    return True

def _is_3_colorable(adj, n=16):
    """
    Exact backtracking 3-colorability check.
    Returns True if 3-colorable, False otherwise.
    Uses bitmasks and MRV heuristic.
    """
    # available colors per vertex: bitmask of colors {0,1,2}
    # We'll do recursive backtracking with constraint propagation.
    # For speed on 16 vertices, this is feasible if we prune well.
    
    # Precompute neighbor masks
    # colors[v] = -1 means uncolored, else 0,1,2
    colors = [-1]*n
    
    # Domain for each vertex: initially all 3 colors
    # We'll use a simple recursive search with forward checking.
    
    def can_color(v, c):
        """Check if vertex v can be colored with color c given current colors."""
        mask = adj[v]
        while mask:
            lb = mask & (-mask)
            u = lb.bit_length()-1
            mask ^= lb
            if colors[u] == c:
                return False
        return True
    
    def backtrack(unassigned):
        if not unassigned:
            return True
        # MRV: pick unassigned vertex with fewest available colors
        best_v = -1
        best_count = 10
        best_avail = 0
        for v in unassigned:
            avail = 0
            for c in range(3):
                if can_color(v, c):
                    avail |= (1<<c)
            cnt = bin(avail).count('1')
            if cnt == 0:
                return False  # dead end
            if cnt < best_count:
                best_count = cnt
                best_v = v
                best_avail = avail
                if cnt == 1:
                    break
        
        # Try colors for best_v
        avail = best_avail
        while avail:
            lb = avail & (-avail)
            c = lb.bit_length()-1
            avail ^= lb
            colors[best_v] = c
            if backtrack(unassigned - {best_v}):
                return True
            colors[best_v] = -1
        return False
    
    # Initial unassigned set
    unassigned = frozenset(range(n))
    return backtrack(unassigned)

def _local_search(seed_edges, deadline, seed):
    """
    Start from seed_edges, perform randomized local search to maximize edges
    while maintaining triangle-free and not-3-colorable.
    """
    n = 16
    rng = random.Random(seed)
    
    # Start with a greedy completion of the seed
    adj = _edges_to_adj(seed_edges, n)
    edge_set = set()
    for u,v in seed_edges:
        edge_set.add((min(u,v), max(u,v)))
    
    # Greedy add: try all missing edges in random order
    all_pairs = [(u,v) for u in range(n) for v in range(u+1, n)]
    remaining = [(u,v) for (u,v) in all_pairs if (u,v) not in edge_set]
    rng.shuffle(remaining)
    
    for u,v in remaining:
        if time.time() > deadline:
            break
        # Check triangle-free
        if adj[u] & adj[v] == 0:
            adj[u] |= (1<<v)
            adj[v] |= (1<<u)
            edge_set.add((u,v))
    
    # Now we have a maximal triangle-free graph (with respect to this order).
    # Verify it's not 3-colorable. If it is, we need to remove some edges to break 3-colorability?
    # Actually, removing edges makes it MORE likely to be 3-colorable.
    # So if the greedy completion IS 3-colorable, we have a problem.
    # We need a graph that is NOT 3-colorable.
    # The Grötzsch core ensures 4-chromaticity only if we don't add edges that allow 3-coloring.
    # Adding edges can sometimes make a graph 3-colorable? No, adding edges makes it HARDER to color.
    # Wait: adding edges adds constraints. So if a graph is 3-colorable, adding edges might make it not 3-colorable.
    # If a graph is NOT 3-colorable, adding edges keeps it NOT 3-colorable.
    # So starting from Grötzsch (not 3-colorable) and adding edges preserves non-3-colorability.
    # So the greedy completion should be fine.
    
    # Let's verify just in case.
    if _is_3_colorable(adj, n):
        # This shouldn't happen if we started from Grötzsch, but just in case
        # Fall back to just Grötzsch
        return seed_edges, len(seed_edges)
    
    # Now try to improve by swapping edges?
    # Actually, the greedy step already maximizes edges for a given random order.
    # Different random orders yield different maximal triangle-free graphs.
    # We want the one with the most edges.
    # But we also need to ensure non-3-colorability (which is preserved).
    
    # Let's do a few more random restarts within the time limit.
    best_count = len(edge_set)
    best_adj = adj[:]
    best_edge_set = set(edge_set)
    
    # Try a few more random orders
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        adj2 = _edges_to_adj(seed_edges, n)
        edge_set2 = set(edge_set)
        for u,v in seed_edges:
            edge_set2.add((min(u,v), max(u,v)))
        
        remaining2 = [(u,v) for (u,v) in all_pairs if (u,v) not in edge_set2]
        rng.shuffle(remaining2)
        
        for u,v in remaining2:
            if time.time() > deadline:
                break
            if adj2[u] & adj2[v] == 0:
                adj2[u] |= (1<<v)
                adj2[v] |= (1<<u)
                edge_set2.add((u,v))
        
        if len(edge_set2) > best_count:
            if not _is_3_colorable(adj2, n):
                best_count = len(edge_set2)
                best_adj = adj2[:]
                best_edge_set = set(edge_set2)
    
    # Convert best_adj to edge list
    result_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1<<v):
                result_edges.append([u,v])
    
    return result_edges, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Base seed: Grötzsch graph
    seed_edges = _build_grotzsch_edges()
    best_edges = seed_edges[:]
    best_count = len(seed_edges)
    
    # Try multiple random seeds
    seed = 42
    while time.time() < deadline - 10:
        edges, count = _local_search(seed_edges, deadline, seed)
        if count > best_count:
            best_count = count
            best_edges = edges
        seed += 1
    
    # Final validation
    adj = _edges_to_adj(best_edges, 16)
    if not _is_triangle_free(adj, 16) or _is_3_colorable(adj, 16):
        # Fallback
        return seed_edges
    
    return best_edges
# EVOLVE-BLOCK-END
