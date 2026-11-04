"""Construct small 3-regular graphs of large girth using random BFS with backtracking."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth >= g on as few vertices as possible.
    Uses random BFS with backtracking (girth-avoiding greedy growth).
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    if k <= 0 or g < 4:
        return 0, []
    
    # For the specific case k=3, g=16, use targeted search
    # Moore bound: 1 + k * (k-1)^(g/2 - 1) for even girth
    # For k=3, g=16: 1 + 3*(2^7) = 1 + 384 = 385 (but this is for trees, actual bound is higher)
    # The actual Moore bound for (3,16) is 2*sum_{i=0}^{7} 2^i = 2*(2^8-1) = 510
    # Best known: 936 vertices
    
    best_n = None
    best_edges = None
    
    def try_random_bfs(max_depth=None):
        """
        Random BFS: grow a tree level by level, at each vertex pick random unused neighbors.
        When all vertices at a level have been assigned, close the graph by connecting
        remaining stubs in a way that avoids short cycles.
        """
        nonlocal best_n, best_edges
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            return None
        
        # Strategy: build a regular graph via random permutation-based construction
        # For k-regular graphs, we can use the configuration model with girth checking
        
        # For k=3, g=16: try to find a graph with ~936 or fewer vertices
        # Use a structured approach: start with a base graph and try to find valid ones
        
        # Method: Random regular graph generation + girth verification
        # Generate random 3-regular graphs with n vertices, check girth >= 16
        
        # The Moore bound for (3,16) gives minimum 512 vertices
        # Best known is 936. Let's try random search around that range.
        
        target_n = 936  # Start from best known
        
        # Try a few random 3-regular graphs
        for attempt in range(200):
            elapsed = time.time() - start_time
            if elapsed > deadline - 10:
                break
            
            n = target_n + random.randint(-20, 20)
            if n < 512 or n % 2 != 0:  # 3-regular needs even number of vertices
                continue
            
            # Generate random 3-regular graph using configuration model
            edges = _generate_random_regular_graph(n, k)
            if edges is None:
                continue
            
            # Check girth
            girth = _compute_girth(n, edges)
            if girth >= g:
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                return n, edges
        
        return None
    
    def _generate_random_regular_graph(n, degree):
        """Generate a random n-vertex degree-regular graph using configuration model."""
        if n * degree % 2 != 0:
            return None
        
        # Configuration model: each vertex has 'degree' stubs
        stubs = []
        for v in range(n):
            for _ in range(degree):
                stubs.append(v)
        
        random.shuffle(stubs)
        
        # Pair up stubs
        edges = set()
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None  # Loop, invalid
            edge = (min(u, v), max(u, v))
            if edge in edges:
                return None  # Multi-edge, invalid
            edges.add(edge)
        
        return list(edges)
    
    def _compute_girth(n, edges):
        """Compute the girth (length of shortest cycle) of a graph."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from each vertex to find shortest cycle
        # For efficiency, use BFS from each vertex and track parent
        min_cycle = float('inf')
        
        for start in range(n):
            # BFS from start
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            
            for u in queue:
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and parent[v] != u:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
            
            if min_cycle <= g:
                return min_cycle
        
        return min_cycle if min_cycle < float('inf') else float('inf')
    
    # First, try the LUW construction as a fallback (known to work)
    luw_result = _luw_construct(k, g)
    if luw_result is not None:
        best_n, best_edges = luw_result
    
    # Try random search to improve
    result = try_random_bfs()
    if result is not None:
        best_n, best_edges = result
    
    # If no improvement found, return LUW result
    if best_n is None:
        # Fallback: return a trivial valid graph if possible
        # For safety, return the LUW result or a simple cycle (if g allows)
        if k == 2:
            # 2-regular graph is a union of cycles
            n = max(g, 3)
            edges = [(i, (i+1) % n) for i in range(n)]
            return n, edges
        else:
            # Return a simple valid construction
            n = 2 * g  # A 2g-cycle has girth 2g >= g, but degree 2
            # For k>=3, we need a different approach
            # Just return a large enough graph
            n = k * g * 2
            # Build a k-regular graph: use a circulant
            edges = set()
            for i in range(n):
                for j in range(1, k//2 + 1):
                    a, b = i, (i+j) % n
                    edges.add((min(a,b), max(a,b)))
                    a, b = i, (i-j) % n
                    if a != b:
                        edges.add((min(a,b), max(a,b)))
            # Check degrees
            deg = [0]*n
            for u,v in edges:
                deg[u]+=1
                deg[v]+=1
            if all(d==k for d in deg):
                return n, list(edges)
            # Fallback: return empty
            return 0, []
    
    return best_n, best_edges


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q
    
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
    
    return len(order), edges

# EVOLVE-BLOCK-END
