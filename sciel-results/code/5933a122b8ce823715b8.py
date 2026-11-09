import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    timeout = 110.0
    
    if g % 2 == 0:
        r = g // 2
        moore = 1 + k * sum((k-1)**i for i in range(r))
    else:
        r = g // 2
        moore = 2 * sum((k-1)**i for i in range(r+1))
        
    n_start = max(moore, 2*k)
    step = 1 if k % 2 == 0 else 2
    
    best_n = None
    best_edges = None
    
    current_n = n_start
    max_n_to_try = n_start + 2000
    
    while current_n <= max_n_to_try:
        if time.time() - start_time > timeout:
            break
            
        success = False
        edges = None
        
        max_restarts = 10
        for _ in range(max_restarts):
            if time.time() - start_time > timeout:
                break
            
            edges = try_construct(k, g, current_n)
            if edges is not None:
                success = True
                break
                
        if success:
            best_n = current_n
            best_edges = edges
            break
            
        current_n += step

    if best_n is not None:
        return best_n, best_edges
    else:
        # Fallback
        n_fallback = 2000
        edges_fallback = try_construct(k, g, n_fallback)
        if edges_fallback:
            return n_fallback, edges_fallback
        return 0, []

def try_construct(k, g, n):
    adj = [[] for _ in range(n)]
    deg = [0] * n
    
    # To speed up finding candidates, we can maintain a list of vertices with deg < k
    # But random picking is fine for small n
    
    for _ in range(n * k // 2):
        u_found = False
        attempts = 0
        max_attempts = 100 # Increased attempts
        
        while attempts < max_attempts:
            u = random.randint(0, n-1)
            if deg[u] >= k:
                attempts += 1
                continue
                
            v = random.randint(0, n-1)
            if v == u or deg[v] >= k:
                attempts += 1
                continue
                
            if v in adj[u]:
                attempts += 1
                continue
                
            if has_short_path(adj, u, v, g):
                attempts += 1
                continue
                
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
            u_found = True
            break
            
        if not u_found:
            return None
            
    for i in range(n):
        if deg[i] != k:
            return None
            
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
                
    return edges

def has_short_path(adj, u, v, g):
    if u == v:
        return True
        
    target_dist = g - 1
    visited = [-1] * len(adj)
    visited[u] = 0
    queue = deque([(u, 0)])
    
    while queue:
        node, dist = queue.popleft()
        
        if node == v:
            return dist < target_dist
            
        if dist + 1 >= target_dist:
            continue
            
        for neighbor in adj[node]:
            if visited[neighbor] == -1:
                visited[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))
                
    return False
