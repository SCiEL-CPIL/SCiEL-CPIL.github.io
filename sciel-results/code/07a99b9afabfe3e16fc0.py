import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 110.0
    best_n = float('inf')
    best_edges = []
    
    def try_construct():
        n = 1
        edges = []
        adj = [[] for _ in range(n)]
        edge_set = set()
        
        def add_edge(u, v):
            nonlocal edges, adj, edge_set
            if u == v:
                return False
            if (u, v) in edge_set or (v, u) in edge_set:
                return False
            
            # BFS to find distance
            visited = {u: 0}
            q = deque([u])
            dist = -1
            while q:
                curr = q.popleft()
                if curr == v:
                    dist = visited[curr]
                    break
                for nei in adj[curr]:
                    if nei not in visited:
                        visited[nei] = visited[curr] + 1
                        q.append(nei)
            
            if dist < g - 1:
                return False
            
            edges.append((u, v))
            edge_set.add((u, v))
            edge_set.add((v, u))
            adj[u].append(v)
            adj[v].append(u)
            return True

        def add_node():
            nonlocal n, adj
            adj.append([])
            n += 1

        while True:
            if time.time() > deadline:
                return None
            
            candidates = [i for i in range(n) if len(adj[i]) < k]
            if not candidates:
                break
            
            u = random.choice(candidates)
            
            # Compute distances from u to all
            visited = {u: 0}
            q = deque([u])
            while q:
                curr = q.popleft()
                for nei in adj[curr]:
                    if nei not in visited:
                        visited[nei] = visited[curr] + 1
                        q.append(nei)
            
            valid_neighbors = []
            for v in range(n):
                if v == u:
                    continue
                if (u, v) in edge_set:
                    continue
                if v in visited and visited[v] >= g - 1:
                    valid_neighbors.append((visited[v], v))
            
            if valid_neighbors:
                valid_neighbors.sort()
                dist, v = valid_neighbors[0]
                add_edge(u, v)
            else:
                new_node = n
                add_node()
                add_edge(u, new_node)
                
        return n, edges

    attempts = 0
    while time.time() < deadline - 5:
        res = try_construct()
        if res:
            n, edges = res
            if n < best_n:
                best_n = n
                best_edges = edges
        attempts += 1
        if best_n < 800:
            break

    if best_n == float('inf'):
        # Fallback: Return a simple graph if possible
        # For k=3, g=4, K4 is not girth 4 (it has triangles).
        # C6 is not 3-regular.
        # If we fail, we return a dummy?
        # We'll just return the best found.
        pass

    return best_n, best_edges
