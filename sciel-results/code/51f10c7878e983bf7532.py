import time
import math
from collections import deque

def construct(k, g):
    deadline = time.time() + 115
    
    if k == 3 and g == 16:
        # Try Klein construction with prime powers q ≡ 1 (mod 4)
        # q=257 gives n=514, but we need to verify girth
        # Try multiple candidates and pick the smallest valid one
        candidates = [257, 521, 1021]
        
        best_result = None
        for q in candidates:
            if time.time() > deadline - 5:
                break
            if not _is_prime_power(q) or q % 4 != 1:
                continue
            
            n, edges = _klein_graph(q)
            if n is None or n == 0:
                continue
            
            # Fast girth check
            if _check_girth_fast(n, edges, g):
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
        
        if best_result is not None:
            return best_result
        
        # Fallback to LUW if no small graph found
        return _luw_fallback(k, g)
    
    # Generic fallback for other k, g
    return _luw_fallback(k, g)

def _is_prime_power(n):
    if n < 2:
        return False
    # Check if n is a prime power
    for p in range(2, int(math.sqrt(n)) + 1):
        if n % p == 0:
            m = n
            while m % p == 0:
                m //= p
            if m == 1:
                return True
    return True  # n itself is prime

def _klein_graph(q):
    # Klein construction: vertices are (u, v) where u,v in GF(q)
    # For simplicity, use the standard Klein graph on 2q vertices
    # Edges: (u, v) connected to (u+1, v), (u, v+1), (u, v+a) for non-residue a
    # But the classic Klein graph has a specific structure
    
    # Actually, let me use a different approach:
    # The (3,16)-cage has 936 vertices. Let me try to construct it directly.
    # For now, return None and use fallback.
    
    # Try a simpler algebraic construction
    # Vertices: pairs (i, j) where i in range(q), j in range(3)
    # This gives 3q vertices
    n = 3 * q
    edges = []
    
    # Find a non-quadratic residue
    a = None
    for x in range(1, q):
        if pow(x, (q - 1) // 2, q) == q - 1:
            a = x
            break
    
    if a is None:
        return None, None
    
    # Construct edges for a 3-regular graph
    # This is a simplified version; the actual Klein graph is more complex
    for i in range(q):
        for j in range(3):
            u = i * 3 + j
            # Three neighbors
            # 1. (i+1, j)
            v1 = ((i + 1) % q) * 3 + j
            edges.append((u, v1))
            
            # 2. (i, j+1) with some transformation
            j2 = (j + 1) % 3
            i2 = (i + j) % q
            v2 = i2 * 3 + j2
            edges.append((u, v2))
            
            # 3. (i*a, (j+2)%3)
            i3 = (i * a) % q
            j3 = (j + 2) % 3
            v3 = i3 * 3 + j3
            edges.append((u, v3))
    
    # Remove duplicates and self-loops
    edge_set = set()
    unique_edges = []
    for u, v in edges:
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in edge_set:
            edge_set.add(key)
            unique_edges.append(key)
    
    return n, unique_edges

def _check_girth_fast(n, edges, g):
    """Check if graph has girth at least g using BFS from each vertex."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return False
    
    # For girth >= 16, we need no cycles of length < 16
    # Use BFS from each vertex, but limit depth to g//2 - 1 = 7
    # If we find a vertex at distance d from start where d + dist_to_start < g, we have a cycle
    
    # Optimization: Only check from a subset of vertices if n is large
    # For n <= 1000, check all vertices
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2:
                continue
                
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True

def _luw_fallback(k, g):
    # The original LUW construction
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
