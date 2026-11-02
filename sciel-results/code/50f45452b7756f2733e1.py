import time
import math
from collections import deque

def construct(k, g):
    deadline = time.time() + 115
    
    if k == 3 and g == 16:
        # Try circulant graph search first (faster to check)
        result = _search_circulant(k, g, deadline)
        if result is not None:
            return result
        
        # Try to find a small graph using Klein construction
        candidates = [257, 521, 1021, 2053, 4109, 8213]
        
        for q in candidates:
            if time.time() > deadline:
                break
            if not _is_prime(q) or q % 4 != 1:
                continue
            
            n, edges = _klein_graph(q)
            if n is None:
                continue
            
            # Check girth
            if _check_girth(n, edges, g):
                return n, edges
        
        # Fallback to LUW if no small graph found
        return _luw_fallback(k, g)
    
    # Generic fallback for other k, g
    return _luw_fallback(k, g)

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def _klein_graph(q):
    # Vertices 0..q-1 are U, q..2q-1 are V
    n = 2 * q
    edges = []
    
    # Find a non-quadratic residue 'a'
    a = None
    for x in range(1, q):
        if pow(x, (q - 1) // 2, q) == q - 1:
            a = x
            break
    
    if a is None:
        return None, None
        
    # Add edges from U to V
    for u in range(q):
        u2 = (u * u) % q
        v1 = u2
        v2 = (u2 + a) % q
        v3 = (u2 - a) % q
        
        edges.append((u, q + v1))
        edges.append((u, q + v2))
        edges.append((u, q + v3))
    
    # Add edges from V to U (to make it 3-regular, we need edges in both directions)
    # Actually, in a bipartite graph, each vertex has degree 3
    # V vertices: each v should have exactly 3 neighbors in U
    # For v = u^2, neighbors are u, -u, and... let's compute properly
    # This construction is tricky; let's use a different approach
    # For now, return only the U-V edges and let the caller verify
    
    return n, edges

def _check_girth(n, edges, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree first
    for i in range(n):
        if len(adj[i]) != 3:
            return False
    
    # BFS from each vertex to find shortest cycle
    # Use non-backtracking walks for accurate girth check
    for start in range(n):
        # dist[v][parent] = distance from start to v where parent is the previous vertex
        # For efficiency, we'll do BFS up to depth g//2
        dist = {}
        dist[(start, -1)] = 0
        queue = deque([(start, -1)])
        
        while queue:
            u, parent = queue.popleft()
            d = dist[(u, parent)]
            if d >= g // 2:
                continue
                
            for v in adj[u]:
                if v == parent:
                    continue
                key = (v, u)
                if key not in dist:
                    dist[key] = d + 1
                    queue.append((v, u))
                else:
                    # Found a cycle
                    cycle_len = d + 1 + dist[key]
                    if cycle_len < g:
                        return False
                        
    return True

def _search_circulant(k, g, deadline):
    # Search for a circulant graph C(n; S) where S is a set of k symmetric generators
    # For k=3, S = {a, n-a, b} where b = n/2 (if n is even)
    # We need to find n and S such that girth >= g
    
    # Moore bound for k=3, g=16: 2 * (3^7 - 1)/2 = 3^7 - 1 = 2186
    # But we know smaller graphs exist
    # Try n values from a reasonable range
    
    min_n = 500  # Start from a reasonable lower bound
    max_n = 1500  # Upper bound for search
    
    best_result = None
    
    for n in range(min_n, max_n + 1):
        if time.time() > deadline:
            break
        
        # For k=3, we need 3 generators
        # Option 1: {a, n-a, n/2} if n is even
        if n % 2 == 0:
            half = n // 2
            # Try different values of a
            for a in range(1, n // 2):
                if a == half:
                    continue
                S = {a, n - a, half}
                # Check if this gives a simple graph (no loops, no repeated edges)
                # a != 0, n-a != 0, half != 0 (already ensured)
                # a != n-a (i.e., a != n/2, already checked)
                # a != half, n-a != half
                
                if _check_circulant_girth(n, S, g):
                    # Build edges
                    edges = []
                    for i in range(n):
                        for s in S:
                            j = (i + s) % n
                            if i < j:
                                edges.append((i, j))
                            elif i == j:
                                # Loop, invalid
                                break
                        else:
                            if best_result is None or n < best_result[0]:
                                best_result = (n, edges)
                            break
                        break
        
        # Option 2: {a, b, c} with a+b+c = 0 mod n (for odd n)
        # This is more complex, skip for now
    
    return best_result

def _check_circulant_girth(n, S, g):
    # Check if circulant graph C(n; S) has girth >= g
    # A cycle of length L exists if there are s1, ..., sL in S (with signs) such that
    # s1 + s2 + ... + sL = 0 mod n, and the partial sums are all distinct (non-zero)
    
    # For efficiency, we check for short cycles up to length g-1
    # Use BFS in the Cayley graph
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for i in range(n):
        for s in S:
            j1 = (i + s) % n
            j2 = (i - s) % n
            if j1 != i:
                adj[i].append(j1)
            if j2 != i and j2 != j1:
                adj[i].append(j2)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return False
    
    # BFS from each vertex to find shortest cycle
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2:
                continue
                
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                else:
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
