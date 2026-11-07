import time
from collections import deque

def construct():
    deadline = time.time() + 80  # 80s deadline, 10s margin
    n = 24
    # Path edges
    path_edges = set()
    for i in range(n-1):
        path_edges.add((i, i+1))
        path_edges.add((i+1, i))
    
    adj = [set() for _ in range(n)]
    for u, v in path_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    added_edges = []
    added_set = set()
    
    def get_diameter():
        # BFS from each vertex
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        max_dist = max(max_dist, dist[v])
                        q.append(v)
        return max_dist
    
    def can_add(u, v):
        # Check triangle-free: no common neighbor
        for w in adj[u]:
            if w in adj[v]:
                return False
        return True
    
    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
        added_edges.append([u, v])
        added_set.add((min(u,v), max(u,v)))
    
    # Greedy: repeatedly find max-distance pair, try to reduce
    while time.time() < deadline:
        # Find max distance pair
        max_d = 0
        max_pair = None
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > max_d:
                            max_d = dist[v]
                            max_pair = (src, v)
                        q.append(v)
        
        if max_d <= 4:
            break
        
        if max_pair is None:
            break
        
        u, v = max_pair
        
        # Try adding edge (u,v)
        if can_add(u, v) and (min(u,v), max(u,v)) not in added_set and (u,v) not in path_edges and (v,u) not in path_edges:
            add_edge(u, v)
            continue
        
        # Try adding edges from u to neighbors of v's far side, or v to u's far side
        # Try all candidate edges that might help
        best_gain = 0
        best_edge = None
        for a in range(n):
            for b in range(a+1, n):
                key = (a, b)
                if key in added_set or key in path_edges or (b,a) in path_edges:
                    continue
                if not can_add(a, b):
                    continue
                # Estimate: does this edge reduce max distance?
                # Quick check: is a near u and b near v, or vice versa?
                # For speed, just try a subset
                pass
        
        # Fallback: try adding (u, v) if possible, else try (u, v-1), (u, v+1), etc.
        found = False
        for delta in range(1, n):
            for candidate in [(u, v-delta), (u, v+delta), (u-delta, v), (u+delta, v)]:
                a, b = candidate
                if 0 <= a < n and 0 <= b < n and a != b:
                    key = (min(a,b), max(a,b))
                    if key in added_set or key in path_edges or (b,a) in path_edges:
                        continue
                    if can_add(a, b):
                        add_edge(a, b)
                        found = True
                        break
            if found:
                break
        
        if not found:
            # Try any valid edge
            found = False
            for a in range(n):
                for b in range(a+1, n):
                    key = (a, b)
                    if key in added_set or key in path_edges or (b,a) in path_edges:
                        continue
                    if can_add(a, b):
                        add_edge(a, b)
                        found = True
                        break
                if found:
                    break
            if not found:
                break
    
    # Verify
    final_diam = get_diameter()
    # Return as list of [u,v]
    return [list(e) for e in added_edges]
