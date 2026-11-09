import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 110.0

    if k != 3:
        # Fallback to a simple construction or seed logic
        # For now, assume k=3
        pass

    def has_short_cycle(adj, g):
        n = len(adj)
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            queue = deque()
            dist[start] = 0
            queue.append(start)
            while queue:
                u = queue.popleft()
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return True
        return False

    # Search for circulant graph C_N(1, N-1, a)
    for N in range(400, 5000, 2):
        if time.time() > deadline:
            break
        
        # Try random 'a'
        for _ in range(100):
            if time.time() > deadline:
                break
            a = random.randint(2, N-2)
            
            adj = [[] for _ in range(N)]
            for i in range(N):
                adj[i].append((i+1)%N)
                adj[i].append((i-1)%N)
                adj[i].append((i+a)%N)
            
            if not has_short_cycle(adj, g):
                edges = []
                for i in range(N):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return N, edges
                
    # If not found, try larger N or different structure
    # For safety, return a known valid graph if we had one, 
    # but we don't. 
    # We must return something.
    # If we run out of time, we might return an invalid graph?
    # No, we should try to find one.
    
    return 0, []
