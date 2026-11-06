import time
import random
from collections import deque

def construct():
    N = 40
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial Graph: 8 disjoint K5s
    adj = [set() for _ in range(N)]
    edges = set()
    for k in range(8):
        for a in range(5):
            for b in range(a+1, 5):
                u = 5*k + a
                v = 5*k + b
                adj[u].add(v)
                adj[v].add(u)
                edges.add((u, v))
    
    best_edges = edges.copy()
    
    def has_c6_path(u, v, adj):
        # Check if there is a simple path of length 4 between u and v
        # This implies adding (u, v) creates a C6 (or it already exists)
        visited = {u: 0}
        queue = deque([(u, 0)])
        while queue:
            node, dist = queue.popleft()
            if dist == 4:
                continue
            for neighbor in adj[node]:
                if neighbor == v and dist + 1 == 4:
                    return True
                if neighbor not in visited and dist + 1 < 4:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        return False

    def find_c6(u, v, adj):
        # Find a specific C6 involving (u, v)
        # Returns list of edges in the cycle, or None
        parent = {}
        dist = {u: 0}
        queue = deque([u])
        while queue:
            node = queue.popleft()
            if dist[node] == 4:
                continue
            for neighbor in adj[node]:
                if neighbor == v and dist[node] + 1 == 4:
                    # Found v at distance 4
                    path = [v]
                    cur = node
                    while cur != u:
                        path.append(cur)
                        cur = parent[cur]
                    path.append(u)
                    # path is [u, ..., v]
                    # Edges are (path[i], path[i+1])
                    edges_in_cycle = []
                    for i in range(len(path)-1):
                        x, y = path[i], path[i+1]
                        if x > y: x, y = y, x
                        edges_in_cycle.append((x, y))
                    return edges_in_cycle
                if neighbor not in dist and dist[node] + 1 < 4:
                    dist[neighbor] = dist[node] + 1
                    parent[neighbor] = node
                    queue.append(neighbor)
        return None

    def local_search(adj, edges):
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            # 1. Try to add an edge
            added = False
            # Shuffle to avoid bias?
            # For speed, just iterate.
            for u in range(N):
                if added: break
                for v in range(u+1, N):
                    if (u, v) not in edges:
                        if not has_c6_path(u, v, adj):
                            # Add edge
                            adj[u].add(v)
                            adj[v].add(u)
                            edges.add((u, v))
                            improved = True
                            added = True
                            break
            
            if added:
                continue
            
            # 2. Try a swap
            swapped = False
            for u in range(N):
                if swapped: break
                for v in range(u+1, N):
                    if swapped: break
                    if (u, v) not in edges:
                        c6_edges = find_c6(u, v, adj)
                        if c6_edges:
                            # Try to remove one edge from c6_edges
                            # c6_edges contains (u, v) as well?
                            # find_c6 returns edges in the path u-...-v.
                            # It does NOT include (u, v) because the path is in the current graph.
                            # So c6_edges are the 4 edges of the path.
                            # Wait, a C6 has 6 edges.
                            # The path has 4 edges.
                            # The cycle is path + (u, v).
                            # So the edges in the C6 are c6_edges + [(u, v)].
                            # We want to remove one of the edges in c6_edges.
                            for e in c6_edges:
                                x, y = e
                                # Tentative swap
                                adj[x].remove(y)
                                adj[y].remove(x)
                                edges.remove(e)
                                
                                adj[u].add(v)
                                adj[v].add(u)
                                edges.add((u, v))
                                
                                # Check if valid
                                if not has_c6_path(u, v, adj):
                                    improved = True
                                    swapped = True
                                    break
                                else:
                                    # Revert
                                    adj[x].add(y)
                                    adj[y].add(x)
                                    edges.add(e)
                                    
                                    adj[u].remove(v)
                                    adj[v].remove(u)
                                    edges.remove((u, v))
                            
                            if swapped:
                                break
        return adj, edges

    # Main Loop
    while time.time() < deadline:
        adj, edges = local_search(adj, edges)
        
        if len(edges) > len(best_edges):
            best_edges = edges.copy()
        
        # Perturb
        # Remove 5 random edges
        edge_list = list(edges)
        random.shuffle(edge_list)
        for e in edge_list[:5]:
            u, v = e
            adj[u].remove(v)
            adj[v].remove(u)
            edges.remove(e)
            
        if time.time() > deadline:
            break

    result = [[u, v] for u, v in best_edges]
    return result
