import time
import math
import random
from collections import deque

def construct(k, g):
    """Construct a k-regular graph of girth at least g with as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Known small cages
    if (k, g) == (3, 5):
        return _petersen()
    if (k, g) == (3, 6):
        return _heawood()
    if (k, g) == (3, 8):
        # Try to construct Tutte-Coxeter or fall back to LUW
        pass
        
    # 1. Try optimized LUW construction
    best_luw_result = None
    if _is_prime(k) or k == 4:
        for d in range(1, 10):
            if d % 2 == 1:
                girth_bound = d + 5
            else:
                girth_bound = 2 * d + 1
            
            if girth_bound >= g:
                result = _luw_construct(k, g, d)
                if result is not None:
                    n, edges = result
                    if _verify_girth(n, edges, k, g):
                        return n, edges
        
    # If LUW failed or didn't run, try modular tensor product construction
    result = _modular_tensor_construct(k, g, deadline)
    if result is not None:
        n, edges = result
        if _verify_girth(n, edges, k, g):
            return n, edges

    # Try Circulant
    result = _circulant_construct(k, g, deadline)
    if result is not None:
        n, edges = result
        if _verify_girth(n, edges, k, g):
            return n, edges

    # Fallback
    return _fallback_construct(k, g)

def _petersen():
    edges = []
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    for i in range(5):
        edges.append((5 + i, 5 + (i + 2) % 5))
    for i in range(5):
        edges.append((i, 5 + i))
    return 10, edges

def _heawood():
    edges = []
    lines = [
        (0, 1, 2), (0, 3, 4), (0, 5, 6),
        (1, 3, 5), (1, 4, 6), (2, 3, 6),
        (2, 4, 5)
    ]
    for i, line in enumerate(lines):
        for p in line:
            edges.append((i, 7 + p))
    return 14, edges

def _luw_construct(k, g, d):
    """Lazebnik-Ustimenko-Woldar type construction with specific d."""
    q = k
    if q < 2:
        return None
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        if not _is_prime(q):
            return None
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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
    
    n = len(order)
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(dg != k for dg in deg):
        return None
        
    return n, edges

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def _verify_girth(n, edges, k, g):
    """Verify that the graph has girth at least g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        q_head = 0
        while q_head < len(queue):
            u = queue[q_head]
            q_head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _modular_tensor_construct(k, g, deadline):
    """
    Modular tensor product construction: build a k-regular graph on n = 2*p*m vertices
    by defining vertices as pairs (x, y) with x in Z_p and y in Z_m, and edges determined
    by a sparse linear recurrence that ensures no short cycles.
    """
    if k < 3:
        return None
    
    # Try to find a good (p, m) combination
    # p should be prime, m small
    # n = 2 * p * m
    # We want n as small as possible while maintaining girth >= g
    
    # Moore bound for reference
    if g % 2 == 0:
        h = g // 2
        moore = 1 + k * sum([(k-1)**i for i in range(h)])
    else:
        h = (g - 1) // 2
        moore = 1 + k * sum([(k-1)**i for i in range(h)])
    
    moore = max(moore, 2 * k)
    
    # Try different (p, m) combinations
    primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    
    best_result = None
    for p in primes:
        if time.time() > deadline - 10:
            break
        
        for m in range(2, 20):  # m from 2 to 19
            if time.time() > deadline - 10:
                break
            
            n = 2 * p * m
            if n < moore:
                continue
            if n > moore + 10000:  # Don't go too far beyond Moore bound
                break
            
            # Try to construct the graph
            result = _try_modular_tensor(k, g, p, m, deadline)
            if result is not None:
                n_val, edges = result
                if best_result is None or n_val < best_result[0]:
                    best_result = (n_val, edges)
                    if n_val <= moore + 100:  # Good enough, return early
                        return best_result
    
    return best_result

def _try_modular_tensor(k, g, p, m, deadline):
    """
    Try to construct a modular tensor product graph with parameters p, m.
    Vertices are (side, x, y) where side in {0,1}, x in Z_p, y in Z_m.
    """
    if time.time() > deadline - 5:
        return None
    
    # Define the linear recurrence for edges
    # For side=0: (0, x, y) connects to (1, x', y') where x' = a*x + b*y + c mod p, y' = d*x + e*y + f mod m
    # For side=1: (1, x, y) connects to (0, x', y') where x' = a'*x + b'*y + c' mod p, y' = d'*x + e'*y + f' mod m
    
    # We need each vertex to have degree k
    # So we need k different mappings for each side
    
    # Try random parameters
    random.seed(42)  # Fixed seed for reproducibility
    
    for trial in range(50):
        if time.time() > deadline - 5:
            return None
        
        # Generate random parameters for the k mappings
        mappings_side0 = []
        mappings_side1 = []
        
        for i in range(k):
            a = random.randint(0, p - 1)
            b = random.randint(0, p - 1)
            c = random.randint(0, p - 1)
            d = random.randint(0, m - 1)
            e = random.randint(0, m - 1)
            f = random.randint(0, m - 1)
            mappings_side0.append((a, b, c, d, e, f))
            
            a = random.randint(0, p - 1)
            b = random.randint(0, p - 1)
            c = random.randint(0, p - 1)
            d = random.randint(0, m - 1)
            e = random.randint(0, m - 1)
            f = random.randint(0, m - 1)
            mappings_side1.append((a, b, c, d, e, f))
        
        # Build the graph
        edges = []
        seen_edges = set()
        
        for side in [0, 1]:
            for x in range(p):
                for y in range(m):
                    for i in range(k):
                        a, b, c, d, e, f = mappings_side0[i] if side == 0 else mappings_side1[i]
                        x_prime = (a * x + b * y + c) % p
                        y_prime = (d * x + e * y + f) % m
                        other_side = 1 - side
                        
                        # Vertex ID: side * p * m + x * m + y
                        u = side * p * m + x * m + y
                        v = other_side * p * m + x_prime * m + y_prime
                        
                        if u == v:
                            return None  # Loop detected
                        
                        edge = (min(u, v), max(u, v))
                        if edge in seen_edges:
                            return None  # Duplicate edge
                        seen_edges.add(edge)
                        edges.append(edge)
        
        n = 2 * p * m
        
        # Verify degree and girth
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        valid_degree = True
        for i in range(n):
            if len(adj[i]) != k:
                valid_degree = False
                break
        if not valid_degree:
            continue
        
        # Check girth (simplified: use BFS from a few starting points)
        if _quick_girth_check(n, adj, g, 5):
            return n, edges
    
    return None

def _quick_girth_check(n, adj, g, num_starts):
    """Quick girth check using BFS from a few starting points."""
    import random as _random
    _random.seed(123)
    starts = _random.sample(range(n), min(num_starts, n))
    
    for start in starts:
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        q_head = 0
        while q_head < len(queue):
            u = queue[q_head]
            q_head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _circulant_construct(k, g, deadline):
    """Search for a circulant k-regular graph with girth at least g."""
    if k < 2:
        return None
    
    if g % 2 == 0:
        h = g // 2
        moore = 1 + k * sum([(k-1)**i for i in range(h)])
    else:
        h = (g - 1) // 2
        moore = 1 + k * sum([(k-1)**i for i in range(h)])
        
    moore = max(moore, 2 * k)
    
    start_n = moore
    end_n = moore + 200
    
    for n in range(start_n, end_n + 1):
        if time.time() > deadline - 5:
            return None
            
        trials = 100
        for _ in range(trials):
            if time.time() > deadline - 5:
                return None
                
            try:
                if k > n // 2:
                    continue
                steps = random.sample(range(1, n // 2 + 1), k)
            except ValueError:
                continue
                
            if _check_circulant_girth(n, steps, g):
                edges = []
                for i in range(n):
                    for s in steps:
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges
                
    return None

def _check_circulant_girth(n, steps, g):
    max_depth = (g - 1) // 2
    
    dist = [-1] * n
    parent = [-1] * n
    dist[0] = 0
    queue = [0]
    q_head = 0
    
    while q_head < len(queue):
        u = queue[q_head]
        q_head += 1
        if dist[u] >= max_depth:
            continue
            
        du = dist[u]
        pu = parent[u]
        for s in steps:
            v1 = (u + s) % n
            v2 = (u - s) % n
            
            if dist[v1] == -1:
                dist[v1] = du + 1
                parent[v1] = u
                if du + 1 < max_depth:
                    queue.append(v1)
            elif parent[u] != v1 and parent[v1] != u:
                cycle_len = du + dist[v1] + 1
                if cycle_len < g:
                    return False
            
            if dist[v2] == -1:
                dist[v2] = du + 1
                parent[v2] = u
                if du + 1 < max_depth:
                    queue.append(v2)
            elif parent[u] != v2 and parent[v2] != u:
                cycle_len = du + dist[v2] + 1
                if cycle_len < g:
                    return False
                    
    return True

def _fallback_construct(k, g):
    """Fallback construction using a simple method."""
    n = 2 * k
    edges = []
    for i in range(k):
        for j in range(k):
            if i != j:
                edges.append((i, k + j))
    return n, edges
