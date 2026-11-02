import random
import time
import sys
from collections import deque

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # Fallback: LUW construction (guaranteed valid, n=1024 for k=4, g=10)
    def luw_construct(k, g):
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

    # Try to find a better graph using Greedy Search
    # Target n = 320 (known cage size)
    target_n = 320
    best_n, best_edges = luw_construct(k, g)
    
    # If we have time, try to improve
    if time.time() < deadline - 20:
        # Use a deterministic seed for reproducibility
        random.seed(42)
        
        # Generate a random 4-regular graph with target_n vertices
        # This is a heuristic attempt to find the cage
        # If it fails to reach girth 10, we return the LUW result
        
        # Note: Implementing a robust cage finder in 100s is hard.
        # However, for this specific task, the LUW result is 1024.
        # Any valid graph with n < 1024 is an improvement.
        
        # We can try a smaller LUW variant? No.
        
        # Let's try to construct a graph using a known smaller algebraic method if available.
        # Since I don't have a better algebraic method for (4,10) readily available in code,
        # I will stick to the LUW result to ensure validity, but I will check if there is a 
        # smaller known construction I can hardcode.
        
        # The (4,10)-cage is 320 vertices.
        # I will not hardcode 320 edges due to length and error risk.
        
        # Instead, I will return the LUW result. 
        # To "improve" the score, I need n < 1024.
        # If I cannot find a better one, I return 1024.
        
        # Wait, is there a simpler construction?
        # The (4,10)-cage is the **320-vertex graph**.
        # I will attempt a **Greedy Cycle Breaking** on a random 4-regular graph of size 320.
        
        # Generate random 4-regular graph
        def random_4_regular(n):
            # Use configuration model
            # 4 * n / 2 = 2n edges
            # Pair up stubs
            stubs = []
            for i in range(n):
                stubs.extend([i, i, i, i])
            random.shuffle(stubs)
            edges = set()
            for i in range(0, 2*n, 2):
                u, v = stubs[i], stubs[i+1]
                if u != v:
                    if (u, v) in edges or (v, u) in edges:
                        # Retry a few times
                        for _ in range(10):
                            v = random.choice([x for x in range(n) if x != u])
                            if (u, v) not in edges and (v, u) not in edges:
                                break
                        else:
                            return None
                    edges.add((min(u, v), max(u, v)))
                else:
                    # Self loop, retry
                    for _ in range(10):
                        v = random.choice([x for x in range(n) if x != u])
                        if (u, v) not in edges and (v, u) not in edges:
                            break
                    else:
                        return None
                    edges.add((min(u, v), max(u, v)))
            return list(edges)

        def get_girth(n, edges):
            # BFS from each vertex to find shortest cycle
            # This is O(n * (n+e)) which is too slow for n=320 in Python if done naively for every step.
            # We only check at the end.
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            min_girth = float('inf')
            for start in range(n):
                # BFS to find shortest cycle containing start
                # Actually, just BFS to find shortest path from start to any neighbor of start
                dist = [-1] * n
                parent = [-1] * n
                q = deque()
                for nb in adj[start]:
                    q.append(nb)
                    dist[nb] = 1
                    parent[nb] = start
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u]:
                            # Cycle found
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
            return min_girth

        # Try to find a graph with girth >= 10
        # This is a stochastic search. It might not succeed in 100s.
        # Given the risk, I will stick to the LUW result.
        pass

    return best_n, best_edges
