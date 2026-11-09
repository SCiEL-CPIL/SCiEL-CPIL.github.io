"""C4-free graph with high minimum degree (R(C4,K1,39) lower bound)."""
import time
import random

# EVOLVE-BLOCK-START
def _can_add_edge(adj, n, u, v):
    """
    Check if adding edge (u, v) creates a C4.
    A C4 is formed if there exist x, y such that:
    u-x, x-v, v-y, y-u are edges.
    This implies x is a neighbor of v (in current graph) and y is a neighbor of u (in current graph).
    And (x, y) is an edge in the current graph.
    Also x != y (to avoid triangle u-x-v-y-u where x=y? No, if x=y, it's triangle u-x-v. Triangles are OK).
    So we check if there is an edge between N(u) and N(v) connecting distinct vertices.
    """
    # Get neighbors of u and v
    nu = adj[u]
    nv = adj[v]
    
    # For each y in N(u), check if it is connected to any x in N(v) where x != y
    # We can optimize by checking neighbors of u against neighbors of v
    # If there's an edge between any y in N(u) and x in N(v) with x != y, then adding (u,v) creates a C4
    
    # Iterate through neighbors of u
    temp = nu
    while temp:
        y_bit = temp & (-temp)
        y = y_bit.bit_length() - 1
        temp ^= y_bit
        
        # Check if y is connected to any x in N(v) where x != y
        # The neighbors of v that are also neighbors of y (excluding y itself)
        common = nv & adj[y]
        # Remove y from common if it's there (we need x != y)
        common &= ~(1 << y)
        if common:
            return False
    
    return True

def _greedy_build(n, rng, min_degree_target):
    """
    Build a C4-free graph on n vertices with minimum degree at least min_degree_target.
    Uses greedy edge addition, trying to add edges in a random order.
    """
    adj = [0] * n
    deg = [0] * n
    
    # Create list of all possible edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    # Shuffle edges randomly
    rng.shuffle(edges)
    
    # Try to add each edge if it doesn't create a C4
    for u, v in edges:
        if _can_add_edge(adj, n, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
    
    # Check minimum degree
    md = min(deg)
    if md < min_degree_target:
        return None
    
    # Convert to edge list format
    edge_list = []
    for i in range(n):
        ai = adj[i]
        for j in range(i + 1, n):
            if (ai >> j) & 1:
                edge_list.append([i, j])
    
    return {'n': n, 'edges': edge_list}

def _try_build(n, rng, seed_offset=0):
    """Try to build a C4-free graph on n vertices with min degree >= n-39."""
    min_degree_target = n - 39
    if min_degree_target < 1:
        min_degree_target = 1
    
    result = _greedy_build(n, rng, min_degree_target)
    return result

def construct():
    deadline = time.time() + 135.0  # safety margin under 150s
    best = None  # (n, mindeg, result)
    
    # Try to maximize n from 47 down; for each n try many seeds
    seeds = [12345, 999, 42, 20240501, 7, 123, 555, 31337, 88, 271828,
             314159, 271, 4242, 13579, 24680, 11, 22, 33, 44, 55,
             66, 77, 888, 9999, 123456, 654321, 777, 8888, 99999, 100000]
    
    for n in range(47, 38, -1):
        found = False
        for s in seeds:
            if time.time() > deadline:
                break
            rng = random.Random(s * 1000 + n * 7)
            res = _try_build(n, rng)
            if res is not None:
                # Compute min degree from edges
                deg = [0] * n
                for e in res['edges']:
                    deg[e[0]] += 1
                    deg[e[1]] += 1
                md = min(deg)
                
                if best is None or n > best[0] or (n == best[0] and md > best[1]):
                    best = (n, md, res)
                found = True
                # Don't break immediately - try more seeds to potentially find higher min degree
                # But if we have a good result, we can break after a few more tries
                if md >= n - 39 + 2:  # If min degree is well above target, stop
                    break
        if found:
            break
    
    if best is not None:
        return best[2]
    
    # fallback: empty graph on 39
    return {'n': 39, 'edges': []}
# EVOLVE-BLOCK-END
