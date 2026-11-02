import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Determine minimum d for LUW
    # Girth of LUW CD(d, k) is at least 2d + 1
    min_d = (g - 1) // 2
    if (g - 1) % 2 != 0:
        min_d += 1
        
    # For k=5, g=9, min_d = 4. N = 2 * 5^4 = 1250.
    
    # Try LUW
    n, edges = _luw_construct_optimized(k, min_d)
    
    # Verify
    if _verify_girth(n, edges, g):
        return n, edges
        
    # Fallback: larger d
    n, edges = _luw_construct_optimized(k, min_d + 1)
    return n, edges

def _luw_construct_optimized(k, d):
    """
    Optimized LUW construction using integer indices.
    Vertices are encoded as (side, x_1, ..., x_d) where side in {0,1} and x_i in {0,...,k-1}.
    Index = side * k^d + x_1 * k^(d-1) + ... + x_d
    """
    q = k
    qd = q ** d
    n_vertices = 2 * qd
    
    # Precompute rule
    rule = {}
    rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
    
    # Precompute powers of q for fast encoding/decoding
    q_powers = [1] * (d + 1)
    for i in range(1, d + 1):
        q_powers[i] = q_powers[i-1] * q
    
    def encode(side, x_tuple):
        """Encode (side, x_tuple) to integer index."""
        val = 0
        for i in range(d):
            val += x_tuple[i] * q_powers[d - 1 - i]
        return side * qd + val
    
    def decode(idx):
        """Decode integer index to (side, x_tuple)."""
        side = idx // qd
        val = idx % qd
        x = [0] * d
        for i in range(d - 1, -1, -1):
            x[i] = val % q
            val //= q
        return side, tuple(x)
    
    edges = []
    # Iterate over all vertices in side 0
    for i in range(qd):
        side, x = decode(i)
        # Generate neighbors
        for first in range(q):
            # y[1] = first, y[0] is unused
            y = [0] * (d + 1)  # indices 1..d used
            y[1] = first
            # y[0] is unused in LUW definition
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # y[c] = x[c] + y[a] * x[b] (mod q)
                    # x[c] in original is x[c-1] in 0-indexed
                    y[c] = (x[c-1] + y[a] * x[b-1]) % q
                else:
                    # y[c] = x[c] - x[a] * y[b] (mod q)
                    y[c] = (x[c-1] - x[a-1] * y[b]) % q
            # Neighbor is (1-side, (y[1], ..., y[d]))
            neighbor_tuple = tuple(y[1:d+1])
            neighbor_idx = encode(1 - side, neighbor_tuple)
            edges.append((i, neighbor_idx))
    
    return n_vertices, edges

def _verify_girth(n, edges, g):
    """Verify that the graph has girth at least g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree and simplicity
    for u in range(n):
        if len(adj[u]) != len(set(adj[u])):
            return False
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

# Keep the original functions
def _luw_construct(k, d):
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    rule = {}
    rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def _construct_5_9(deadline):
    pass

def _random_5_regular(n, rng):
    pass

def _get_adjacency(edges, n):
    pass

def _find_short_cycle(adj, n, max_len):
    pass

def _improve_girth(edges, n, target_girth, deadline, rng):
    pass

def _estimate_girth(adj, n, upper_bound):
    pass

def _has_cycle_of_length(adj, n, length):
    pass
