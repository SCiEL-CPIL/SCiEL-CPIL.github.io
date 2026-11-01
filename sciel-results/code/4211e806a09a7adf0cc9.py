import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Determine lower bound
    if k == 2:
        n_min = g
    else:
        d = g // 2 - 1
        if d <= 0:
            n_min = k + 1
        else:
            # 1 + k * ((k-1)^d - 1) / (k-2)
            n_min = int(1 + k * (( (k-1)**d - 1 ) / (k-2))) + 1
            
    if k == 3 and g == 16:
        n_min = 512
        
    max_n_search = 2000
    current_n = n_min
    
    def generate_random_regular(n, k):
        if n * k % 2 != 0:
            return None
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        edges = []
        edge_set = set()
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                return None
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                return None
            edge_set.add((u, v))
            edges.append((u, v))
        return edges

    def has_girth_at_least_g(edges, n, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for u, v in edges:
            visited = {u: 0}
            queue = deque([(u, 0)])
            while queue:
                node, dist = queue.popleft()
                for neighbor in adj[node]:
                    if node == u and neighbor == v:
                        continue
                    if neighbor == u and node == v:
                        continue
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        if neighbor == v:
                            if dist + 1 < g - 1:
                                return False
                            queue.clear()
                            break
                        queue.append((neighbor, dist + 1))
        return True

    while current_n <= max_n_search:
        if time.time() > deadline:
            break
            
        attempts = 20
        for _ in range(attempts):
            edges = generate_random_regular(current_n, k)
            if edges is not None:
                if has_girth_at_least_g(edges, current_n, g):
                    return current_n, edges
                    
        current_n += 10
        
    # Fallback: find a valid graph with larger n
    while True:
        edges = generate_random_regular(current_n, k)
        if edges is not None:
            if has_girth_at_least_g(edges, current_n, g):
                return current_n, edges
        current_n += 10
        if time.time() > deadline + 5:
             # Emergency: return a trivially valid graph if possible? 
             # For k=3, we can't have trivial graph. 
             # We must hope we find one.
             pass
