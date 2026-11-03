import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Moore bound for odd girth g, degree k
    # n >= 1 + k * sum((k-1)^i for i in range((g-3)//2 + 1))
    if g % 2 == 1:
        half_g = (g - 3) // 2
        moore_bound = 1 + k * sum((k - 1) ** i for i in range(half_g + 1))
    else:
        half_g = g // 2 - 1
        moore_bound = 2 * sum((k - 1) ** i for i in range(half_g + 1))
    
    # We need a k-regular graph.
    # Try Circulant graphs C_n(S)
    # For k=5, n must be even to have a symmetric set of 5 elements including n/2.
    # S = {a, -a, b, -b, n/2}
    
    max_n = 5000  # Upper limit for search
    
    for n in range(moore_bound, max_n):
        if time.time() > deadline:
            break
        if n % 2 != 0:
            continue
            
        # Try several random shifts
        for _ in range(100):
            if time.time() > deadline:
                break
                
            # Choose random a, b
            # Ensure a, b are distinct and not 0 or n/2
            # And a != b, a != -b, etc.
            
            # Generate random candidates
            a = random.randint(1, n // 2 - 1)
            b = random.randint(1, n // 2 - 1)
            
            # Check distinctness
            if a == b or a == n - b or b == n - a or a == 0 or b == 0:
                continue
            if a == n // 2 or b == n // 2:
                continue
                
            S = {a, n - a, b, n - b, n // 2}
            
            # Build adjacency list
            adj = [[] for _ in range(n)]
            for u in range(n):
                for s in S:
                    v = (u + s) % n
                    adj[u].append(v)
            
            # Check girth
            # Girth is the length of the shortest cycle.
            # We need to ensure no cycle of length < g.
            # Use BFS from each vertex up to depth g//2 - 1
            # If we find a back-edge or a visited node at depth < g//2, it's a cycle.
            # Actually, for bipartite-like check, we just need to ensure no short cycles.
            
            # Efficient girth check:
            # For each vertex u, do BFS up to depth max_depth = g // 2 - 1
            # If we encounter a node that was visited in the same BFS tree at a depth 
            # such that depth + depth_of_other < g, we have a cycle.
            # More simply: if we find a node v such that dist[u][v] + 1 < g and v is an ancestor?
            # Standard method: 
            # For each edge (u,v), if dist[u][v] < g-1, then there is a cycle of length dist[u][v]+1.
            # But computing all-pairs shortest paths is O(n^2).
            # For n~500, n^2 = 250k. Very fast.
            
            # Let's do BFS from each vertex to find shortest cycle involving that vertex.
            # Actually, we can just check if any cycle of length < g exists.
            # A cycle of length L exists if there is a walk of length L from u to u.
            
            # Optimization: 
            # If g is small (9), we can just check for cycles of length 3, 4, 5, 6, 7, 8.
            
            valid = True
            # Check for cycles of length 3 to g-1
            # Since it's a circulant graph, we can check specific sums.
            # But BFS is safer.
            
            # BFS from each node, depth limit = g//2
            # If we find a neighbor that is the parent, skip.
            # If we find a neighbor that is already visited in the current BFS tree, 
            # and the path length + 1 < g, then invalid.
            
            # To speed up, we only need to check if girth >= g.
            # If we find ANY cycle of length < g, return False.
            
            for u in range(n):
                if time.time() > deadline:
                    valid = False
                    break
                # BFS
                visited = [-1] * n
                parent = [-1] * n
                queue = [(u, 0)]
                visited[u] = 0
                found_short_cycle = False
                
                while queue:
                    node, dist = queue.pop(0)
                    if dist >= g // 2:
                        continue
                    for v in adj[node]:
                        if visited[v] == -1:
                            visited[v] = dist + 1
                            parent[v] = node
                            queue.append((v, dist + 1))
                        elif v != parent[node]:
                            # Found a cycle
                            # The cycle length is dist + 1 + visited[v]
                            # Wait, this is not the shortest cycle necessarily, but if it's < g, it's bad.
                            cycle_len = dist + 1 + visited[v]
                            if cycle_len < g:
                                found_short_cycle = True
                                break
                    if found_short_cycle:
                        break
                if found_short_cycle:
                    valid = False
                    break
            
            if valid:
                # Construct edge list
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
    
    # Fallback: Return the LUW graph from seed (modified for minimal d)
    # Re-implementing the seed logic as fallback
    q = k
    # Minimal odd d such that d+5 >= g
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
        
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
