import time
import random
import numpy as np
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n = 40
    best_edges = 0
    best_graph = None

    def greedy_search(start_adj=None):
        if start_adj is None:
            adj = np.zeros((n, n), dtype=int)
        else:
            adj = start_adj.copy()
        
        # Compute initial distances
        dist = np.full((n, n), 100, dtype=int)
        for i in range(n):
            dist[i][i] = 0
            q = deque([i])
            visited = np.zeros(n, dtype=bool)
            visited[i] = True
            while q:
                u = q.popleft()
                for v in range(n):
                    if adj[u][v] and not visited[v]:
                        visited[v] = True
                        dist[i][v] = dist[i][u] + 1
                        q.append(v)
        
        edge_count = np.sum(adj) // 2
        
        while time.time() < deadline:
            # Find candidate edges
            candidates = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i][j] == 0 and dist[i][j] > 5:
                        candidates.append((i, j))
            
            if not candidates:
                break
            
            # Pick a random candidate
            i, j = random.choice(candidates)
            
            # Add edge
            adj[i][j] = 1
            adj[j][i] = 1
            edge_count += 1
            
            # Update distances using old values
            old_dist = dist.copy()
            for k in range(n):
                for l in range(n):
                    new_d1 = old_dist[k][i] + 1 + old_dist[j][l]
                    if new_d1 < dist[k][l]:
                        dist[k][l] = new_d1
                    new_d2 = old_dist[k][j] + 1 + old_dist[i][l]
                    if new_d2 < dist[k][l]:
                        dist[k][l] = new_d2
            
            if time.time() > deadline:
                break
        
        return adj, edge_count

    while time.time() < deadline:
        # Strategy 1: Empty start
        adj, count = greedy_search()
        if count > best_edges:
            best_edges = count
            best_graph = adj
        
        # Strategy 2: 8*K5 start
        adj_start = np.zeros((n, n), dtype=int)
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    adj_start[u][v] = 1
                    adj_start[v][u] = 1
        adj, count = greedy_search(adj_start)
        if count > best_edges:
            best_edges = count
            best_graph = adj
        
        # Strategy 3: Random sparse start
        if random.random() < 0.5:
            adj_start = np.zeros((n, n), dtype=int)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.05:
                        adj_start[i][j] = 1
                        adj_start[j][i] = 1
            adj, count = greedy_search(adj_start)
            if count > best_edges:
                best_edges = count
                best_graph = adj

    edges = []
    if best_graph is not None:
        for i in range(n):
            for j in range(i + 1, n):
                if best_graph[i][j]:
                    edges.append([i, j])
    else:
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    edges.append([5 * k + a, 5 * k + b])

    return edges
