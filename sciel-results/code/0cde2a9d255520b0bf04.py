import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    # If k is not 3, we might need a different approach, but task implies k=3.
    # For robustness, if k != 3, we still try L-U-W with q=k.
    
    def luw_construct(k, g):
        q = k
        # Girth of L-U-W(q, d) is 2d + 2.
        # We need 2d + 2 >= g => d >= (g - 2) / 2.
        d_min = (g - 2 + 1) // 2
        d = max(1, d_min)
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
        # Rule for the linear recurrence
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
        
        n = len(order)
        return n, edges

    # Try to find a smaller graph using greedy backtracking for a short time?
    # The L-U-W graph with d=7 has n=4374.
    # The seed had n=354294.
    # This is a huge improvement.
    
    # We can attempt a greedy search for n in [400, 1000] with a strict time limit.
    # If it fails, return L-U-W.
    
    def try_greedy(n, time_limit):
        start = time.time()
        deg = [0] * n
        adj = [set() for _ in range(n)]
        total_edges = (n * k) // 2
        
        # Randomize vertex order to avoid bias
        vertices = list(range(n))
        random.shuffle(vertices)
        
        while total_edges > 0:
            if time.time() - start > time_limit:
                return None
            
            # Find u with min degree
            u = min(vertices, key=lambda x: deg[x])
            if deg[u] >= k:
                # All vertices have degree k?
                if all(d == k for d in deg):
                    break
                # If u is full, we need to pick another
                candidates_u = [i for i in vertices if deg[i] < k]
                if not candidates_u:
                    break
                u = random.choice(candidates_u)
            
            candidates_v = [i for i in vertices if deg[i] < k and i != u]
            if not candidates_v:
                return None
            
            v = random.choice(candidates_v)
            
            if v in adj[u]:
                continue
            
            # Check girth
            # BFS from u to v, avoid going back to u?
            # Actually, we check if there is a path of length < g-1 between u and v.
            # If such a path exists, adding (u,v) creates a cycle of length <= g-1.
            # We need cycle >= g.
            # So path length must be >= g-1.
            # If path length < g-1, reject.
            
            # Simple BFS
            visited = {u: 0}
            queue = deque([(u, 0)])
            found_short = False
            while queue:
                node, dist = queue.popleft()
                if dist >= g - 1:
                    continue
                for neighbor in adj[node]:
                    if neighbor == v:
                        if dist + 1 < g - 1:
                            found_short = True
                        break
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
                if found_short:
                    break
            
            if found_short:
                continue
            
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            total_edges -= 1
        
        if all(d == k for d in deg) and total_edges == 0:
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
        return None

    # Try a few small n values with a short time budget
    # The L-U-W graph has n=4374.
    # If we find any n < 4374, we win.
    # Let's try n=936 (known cage) with 5 seconds.
    # And n=1000, 1200, etc.
    
    best_result = None
    
    # Try known cage size first
    for n_try in [936, 1000, 1200, 1500, 2000, 3000, 4000]:
        if time.time() > deadline - 10:
            break
        res = try_greedy(n_try, 2.0) # 2 seconds per attempt
        if res:
            best_result = res
            break
        
    if best_result:
        return best_result
    
    # Fallback to L-U-W with minimal d
    return luw_construct(k, g)

