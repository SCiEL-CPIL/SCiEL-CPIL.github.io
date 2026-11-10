import time
import random
import sys

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Strategy:
    1. If k=3 and g=16, attempt to construct a 900-vertex graph via tensor product 
       of two 30-vertex cubic graphs of girth 8.
    2. Fallback to Lazebnik-Ustimenko-Woldar (LUW) construction.
    """
    if k != 3 or g != 16:
        # Fallback for other parameters: use LUW or simple construction
        return _luw_construct(k, g)

    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    # Step 1: Find a 30-vertex cubic graph of girth 8
    graph_30 = _find_small_cubic_girth_8(30, 8, deadline)
    
    if graph_30 is not None:
        # Step 2: Tensor product
        n1 = 30
        edges1 = graph_30
        # The tensor product of G1 and G2 has vertices (u, v)
        # Edges ((u, v), (u', v')) if (u, u') in E1 and (v, v') in E2
        # Since G1 is bipartite (girth 8 is even), the product is 2-connected?
        # The product of two connected bipartite graphs is connected?
        # Actually, the tensor product of two connected graphs is connected iff at least one is bipartite.
        # Both are bipartite, so it is connected.
        
        n = n1 * n1
        edges = []
        # Build adjacency for G1
        adj1 = [[] for _ in range(n1)]
        for u, v in edges1:
            adj1[u].append(v)
            adj1[v].append(u)
            
        for u in range(n1):
            for v in range(n1):
                node_id = u * n1 + v
                for u_prime in adj1[u]:
                    for v_prime in adj1[v]:
                        node_id2 = u_prime * n1 + v_prime
                        if node_id < node_id2:
                            edges.append((node_id, node_id2))
        
        # Verify girth (optional, but good for safety)
        # If verification fails, fallback
        if _verify_girth(n, edges, g) and _verify_degree(n, edges, k):
            return n, edges

    # Fallback: LUW construction
    return _luw_construct(k, g)

def _luw_construct(k, g):
    # Original seed logic
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _find_small_cubic_girth_8(n, target_girth, deadline):
    # Randomized local search for a cubic graph of girth 8 on 30 vertices
    # We use a multigraph representation initially, then simplify?
    # Better: Generate random 3-regular graphs and check.
    
    import random
    random.seed(42)
    
    for _ in range(10000):
        if time.time() > deadline:
            return None
            
        # Generate a random 3-regular graph
        edges = _generate_random_3reg(n)
        if edges is None:
            continue
            
        # Check girth
        if _check_girth(n, edges, target_girth):
            return edges
            
        # Local search: try to swap edges to break short cycles
        # This is complex to implement robustly in short time.
        # Instead, we just try many random graphs.
        # 30 vertices, 45 edges. Random 3-reg generation is fast.
        
    return None

def _generate_random_3reg(n):
    if n % 2 != 0:
        return None
    # Configuration model
    half_edges = [i for i in range(n) for _ in range(3)]
    random.shuffle(half_edges)
    edges = set()
    valid = True
    for i in range(0, len(half_edges), 2):
        u, v = half_edges[i], half_edges[i+1]
        if u == v:
            valid = False
            break
        if (min(u,v), max(u,v)) in edges:
            valid = False
            break
        edges.add((min(u,v), max(u,v)))
    if not valid:
        return None
    # Check degree
    deg = [0]*n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != 3 for d in deg):
        return None
    return list(edges)

def _check_girth(n, edges, target_girth):
    # BFS from each node to find shortest cycle
    # For n=30, this is fast.
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        # BFS
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < target_girth:
                        return False
    return True

def _verify_degree(n, edges, k):
    deg = [0]*n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return all(d == k for d in deg)

def _verify_girth(n, edges, target_girth):
    # Reuse _check_girth
    return _check_girth(n, edges, target_girth)
