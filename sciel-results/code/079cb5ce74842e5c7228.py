import time
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Uses a heuristic search: start with a random k-regular graph, then iteratively
    try to eliminate short cycles via edge swaps.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # leave 5s safety margin
    
    # Moore bound for k-regular graph with girth g
    # For odd g: n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
    # For even g: n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i
    if g % 2 == 1:
        moore = 1 + k * sum((k-1)**i for i in range((g-3)//2 + 1))
    else:
        moore = 2 * sum((k-1)**i for i in range(g//2))
    
    # Start with a reasonable n
    # The Moore bound is a lower bound; we need n >= moore
    # But the LUW construction gives much larger n. Let's try to start smaller.
    # For k=5, g=9: Moore bound is 1 + 5*(1 + 4 + 16 + 64 + 256) = 1 + 5*341 = 1706
    # The best known is 1116, which is LESS than the Moore bound? 
    # Wait, the Moore bound for girth 9 (odd) is:
    # 1 + k * (1 + (k-1) + (k-1)^2 + ... + (k-1)^((g-3)/2))
    # For g=9: (g-3)/2 = 3
    # 1 + 5 * (1 + 4 + 16 + 64) = 1 + 5 * 85 = 426
    # Ah, I miscalculated. Let me recompute.
    
    if g % 2 == 1:
        # n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        max_depth = (g - 3) // 2
        moore = 1 + k * sum((k-1)**i for i in range(max_depth + 1))
    else:
        # n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i
        max_depth = g // 2 - 1
        moore = 2 * sum((k-1)**i for i in range(max_depth + 1))
    
    # For k=5, g=9: moore = 1 + 5*(1+4+16+64) = 1 + 5*85 = 426
    # The problem statement says best lower bound is 428, which is close to Moore bound 426.
    # Best known is 1116.
    
    # We'll try to find a graph with n close to the Moore bound, but that's very hard.
    # Let's try a heuristic approach.
    
    # Actually, let's try a different strategy: use the LUW construction as a fallback,
    # but first try a random graph with local search.
    
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph construction."""
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
    
    # Try random graph construction with local search
    def random_k_regular(n, k):
        """Generate a random k-regular graph with n vertices."""
        # Use configuration model
        half_edges = [i for i in range(n) for _ in range(k)]
        random.shuffle(half_edges)
        adj = {i: set() for i in range(n)}
        for i in range(0, n * k, 2):
            u, v = half_edges[i], half_edges[i+1]
            if u != v and v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
            else:
                # Retry
                for j in range(i+1, n * k):
                    v2 = half_edges[j]
                    if u != v2 and v2 not in adj[u]:
                        adj[u].add(v2)
                        adj[v2].add(u)
                        half_edges[j] = v  # swap
                        break
        # Check if all vertices have degree k
        for i in range(n):
            if len(adj[i]) != k:
                return None
        return adj
    
    def find_shortest_cycle(adj, n, target_girth):
        """Find a cycle of length < target_girth using BFS from each vertex."""
        for start in range(n):
            # BFS to find shortest cycle back to start
            # We need to find a cycle of length < target_girth
            visited = {start: (0, -1)}  # vertex: (distance, parent)
            queue = deque([start])
            found_cycle = None
            while queue:
                u = queue.popleft()
                dist_u, parent_u = visited[u]
                for v in adj[u]:
                    if v == parent_u:
                        continue
                    if v == start:
                        # Found a cycle
                        cycle_len = dist_u + 1
                        if cycle_len < target_girth:
                            # Reconstruct cycle
                            cycle = [start]
                            cur = u
                            while cur != -1:
                                cycle.append(cur)
                                cur = visited[cur][1]
                            return cycle
                    elif v not in visited:
                        visited[v] = (dist_u + 1, u)
                        queue.append(v)
                    else:
                        # Back edge
                        dist_v, parent_v = visited[v]
                        if u != parent_v:
                            cycle_len = dist_u + dist_v + 1
                            if cycle_len < target_girth:
                                # Reconstruct cycle
                                cycle1 = [u]
                                cur = u
                                while cur != -1 and cur != start:
                                    cycle1.append(visited[cur][1])
                                    cur = visited[cur][1]
                                cycle2 = [v]
                                cur = v
                                while cur != -1 and cur != start:
                                    cycle2.append(visited[cur][1])
                                    cur = visited[cur][1]
                                cycle = list(reversed(cycle1)) + [start] + list(reversed(cycle2))[1:]
                                return cycle
            if time.time() > deadline:
                return None
        return None
    
    # Try several random graphs
    best_n = float('inf')
    best_edges = None
    
    # Start with n close to Moore bound but with some slack
    min_n = max(moore, k * (g - 1))  # heuristic lower bound
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Try different n values
        n = min_n + attempt * 50
        
        if n > 2000:  # Don't go too large
            break
            
        adj = random_k_regular(n, k)
        if adj is None:
            continue
        
        # Check if girth is sufficient
        # Try to improve by edge swaps
        improved = True
        iterations = 0
        while improved and time.time() < deadline:
            improved = False
            iterations += 1
            if iterations > 1000:
                break
                
            cycle = find_shortest_cycle(adj, n, g)
            if cycle is None:
                # No short cycle found
                if n < best_n:
                    best_n = n
                    best_edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                best_edges.append((u, v))
                break
            else:
                # Try to break the cycle by swapping edges
                # Find two edges in the cycle and swap
                cycle_set = set(cycle)
                edges_in_cycle = []
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i+1) % len(cycle)]
                    edges_in_cycle.append((u, v))
                
                # Try swapping
                swapped = False
                for i in range(len(edges_in_cycle)):
                    for j in range(i+1, len(edges_in_cycle)):
                        u1, v1 = edges_in_cycle[i]
                        u2, v2 = edges_in_cycle[j]
                        
                        # Try to swap (u1,v1) and (u2,v2) with (u1,v2) and (u2,v1)
                        if v2 in adj[u1] or v1 in adj[u2]:
                            continue
                        if u1 == u2 or v1 == v2:
                            continue
                            
                        # Remove old edges
                        adj[u1].remove(v1)
                        adj[v1].remove(u1)
                        adj[u2].remove(v2)
                        adj[v2].remove(u2)
                        
                        # Add new edges
                        adj[u1].add(v2)
                        adj[v2].add(u1)
                        adj[u2].add(v1)
                        adj[v1].add(u2)
                        
                        swapped = True
                        improved = True
                        break
                    if swapped:
                        break
                if not swapped:
                    # Try a different approach: remove an edge and reconnect
                    pass
    
    # If no good graph found, fall back to LUW construction
    if best_edges is None or time.time() > deadline - 5:
        best_n, best_edges = luw_construct(k, g)
    
    return best_n, best_edges
