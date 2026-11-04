"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Try multiple d values from small to large, find the smallest d
    # that produces a valid graph with girth >= g
    best_n = None
    best_edges = None
    
    # The LUW graph CD(d,q) has girth at least d+5 (odd d) or d+4 (even d)
    # But the actual girth may be higher. Try smaller d values first.
    # For k=3, g=16: d=7 gives bound 12, but actual girth might be >= 16
    
    max_d = 30
    for candidate_d in range(3, max_d + 1):
        if time.time() > deadline - 30:
            break
        
        if candidate_d % 2 == 1:
            girth_bound = candidate_d + 5
        else:
            girth_bound = candidate_d + 4
        
        # If the guaranteed bound already exceeds g, we can use this d
        # If not, we still try it but must verify
        n, edges = _generate_luw_graph(k, g, candidate_d)
        
        if n is None or n == 0:
            continue
        
        # If the guaranteed girth bound is sufficient, accept without full verification
        # (the construction guarantees it)
        if girth_bound >= g:
            # Still verify for safety, but this should always pass
            if _verify_girth_fast(n, edges, k, g):
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
            break  # This is the smallest d with guaranteed girth, no need to try larger
        else:
            # Girth bound is insufficient, must verify actual girth
            if _verify_girth_fast(n, edges, k, g):
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
    
    if best_n is None:
        # Fallback: use the guaranteed d
        d = None
        for candidate_d in range(3, max_d + 1):
            if candidate_d % 2 == 1:
                girth_bound = candidate_d + 5
            else:
                girth_bound = candidate_d + 4
            if girth_bound >= g:
                d = candidate_d
                break
        if d is None:
            d = max(3, g - 5)
        best_n, best_edges = _generate_luw_graph(k, g, d)
    
    return best_n, best_edges

def _generate_luw_graph(k, g, d=None):
    """Generate a k-regular graph with girth at least g using LUW construction.
    
    The Lazebnik-Ustimenko-Woldar graph CD(d, q) is a q-regular bipartite graph
    with girth at least d+5 (if d odd) or d+4 (if d even).
    """
    q = k
    
    if d is None:
        # Choose d to ensure girth >= g
        d = None
        for candidate_d in range(3, 30):
            if candidate_d % 2 == 1:
                girth_bound = candidate_d + 5
            else:
                girth_bound = candidate_d + 4
            if girth_bound >= g:
                d = candidate_d
                break
        if d is None:
            d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Define field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Define the rule for each coordinate
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]
    
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
    
    # BFS from the zero vertex to get its connected component
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Only add edge once (from side 0 to side 1)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges

def _moore_bound(k, g):
    """Compute the Moore bound for k-regular graphs of girth g."""
    if g % 2 == 0:
        # Even girth: n >= k + k*(k-1)^(g/2 - 1)
        return k + k * (k - 1) ** (g // 2 - 1)
    else:
        # Odd girth: n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        total = 1
        term = 1
        for i in range((g - 3) // 2 + 1):
            total += k * term
            term *= (k - 1)
        return total

def _verify_girth_fast(n, edges, k, target_girth):
    """Verify that the graph has girth at least target_girth using BFS.
    Returns True if girth >= target_girth, False otherwise."""
    if n == 0:
        return True
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degrees
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Use BFS from each vertex to find shortest cycle
    # In a k-regular graph, we only need to check up to target_girth/2 depth
    half = target_girth // 2
    
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= half:
                continue  # No shorter cycle possible from here
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a back edge, check if it forms a short cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < target_girth:
                        return False
    
    return True
# EVOLVE-BLOCK-END
