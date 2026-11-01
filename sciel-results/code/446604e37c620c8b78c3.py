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
    q = k
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
            
    n_vertices = 2 * (q ** d)
    # Map (side, x_tuple) to index
    # side 0: 0 to q^d - 1
    # side 1: q^d to 2*q^d - 1
    
    def get_index(side, x):
        # x is a tuple of length d
        val = 0
        for i in range(d):
            val = val * q + x[i]
        if side == 0:
            return val
        else:
            return q ** d + val
            
    def get_side_and_x(idx):
        if idx < q ** d:
            side = 0
            val = idx
        else:
            side = 1
            val = idx - q ** d
            
        x = [0] * d
        for i in range(d - 1, -1, -1):
            x[i] = val % q
            val //= q
        return side, tuple(x)

    edges = []
    # Iterate over all vertices in side 0
    for i in range(q ** d):
        side, x = get_side_and_x(i)
        # Generate neighbors
        for first in range(q):
            y = [0] * (d + 1) # indices 1..d
            y[1] = first
            # y[0] is unused in LUW definition?
            # The LUW definition usually has x_0, ..., x_d.
            # The seed code used:
            # y = [None, first] + [0] * (d - 1)
            # So y has length d+1?
            # Let's check the seed code logic carefully.
            
            # Seed:
            # y = [None, first] + [0] * (d - 1)
            # This creates a list of length d.
            # Indices: 0 is None, 1 is first, 2..d are 0.
            # Wait, the loop is for c in range(2, d+1).
            # So y[c] is accessed.
            # If d=4, y has length 4. Indices 0,1,2,3.
            # c=2,3,4.
            # y[4] would be out of bounds if length is 4.
            # Seed code: y = [None, first] + [0] * (d - 1)
            # If d=4, length is 1 + 1 + 3 = 5?
            # [None, first] is length 2.
            # [0]*(d-1) is length 3.
            # Total length 5.
            # Indices 0,1,2,3,4.
            # c=2,3,4.
            # y[2], y[3], y[4] are valid.
            
            # Let's re-implement with list of length d+1 (indices 1..d)
            # Actually, the seed uses 1-based indexing for x and y in the formula.
            # x = (None, x1, ..., xd)?
            # Seed: start = (0, (None,) + (0,) * d)
            # So x has length d+1.
            # x[0] is None. x[1]..x[d] are values.
            
            # In my optimized version:
            # x is a tuple of length d (indices 0..d-1).
            # I need to map to 1-based for the rule.
            
            # Let's stick to the seed's logic but with integer IDs.
            pass

    # This is getting complex. Let's just use the seed's tuple logic but ensure it's fast enough.
    # The seed's logic is O(N * k * d).
    # For N=1250, k=5, d=4, this is 1250 * 5 * 4 = 25000 operations. Very fast.
    # The verification is the bottleneck.
    
    return _luw_construct(k, d) # Fallback to seed implementation if optimized is too complex to write correctly in short time

def _verify_girth(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
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
