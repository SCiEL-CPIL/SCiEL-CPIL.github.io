import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    n_start = 512
    n_max = 650

    def generate_random_regular_graph(n, k):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        edges = []
        adj = [[] for _ in range(n)]
        edge_set = set()
        
        random.shuffle(stubs)
        
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None, None
            if (u, v) in edge_set or (v, u) in edge_set:
                return None, None
            edges.append((u, v))
            edge_set.add((u, v))
            edge_set.add((v, u))
            adj[u].append(v)
            adj[v].append(u)
            
        return edges, adj

    def get_bad_edges(adj, n, g):
        bad_edges = set()
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            bad_edges.add((min(u, v), max(u, v)))
        return list(bad_edges)

    best_n = 0
    best_edges = []

    for n in range(n_start, n_max + 1):
        if remaining() < 10:
            break
        
        for restart in range(5):
            if remaining() < 5:
                break
                
            edges, adj = generate_random_regular_graph(n, k)
            if edges is None:
                continue
                
            bad_edges = get_bad_edges(adj, n, g)
            cost = len(bad_edges)
            
            if cost == 0:
                return (n, edges)
                
            T = 5.0
            cooling = 0.995
            min_T = 0.5
            max_iters = 3000
            
            for it in range(max_iters):
                if remaining() < 2:
                    break
                if cost == 0:
                    return (n, edges)
                    
                bad_edges = get_bad_edges(adj, n, g)
                if not bad_edges:
                    return (n, edges)
                    
                e1 = random.choice(bad_edges)
                u1, v1 = e1
                
                e2_idx = random.randint(0, len(edges) - 1)
                u2, v2 = edges[e2_idx]
                
                if (min(u2, v2), max(u2, v2)) == e1:
                    continue
                
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    continue
                if u2 in adj[u1] or v2 in adj[v1]:
                    continue
                    
                adj[u1].remove(v1)
                adj[v1].remove(u1)
                adj[u2].remove(v2)
                adj[v2].remove(u2)
                
                adj[u1].append(u2)
                adj[u2].append(u1)
                adj[v1].append(v2)
                adj[v2].append(v1)
                
                # Update edges list
                edges[e2_idx] = (u2, v2) # This is wrong, we need to update both
                # Actually, we are swapping e1 and e2.
                # e1 was (u1, v1). e2 was (u2, v2).
                # New edges: (u1, u2) and (v1, v2).
                # We need to find the index of e1 in edges to update it.
                # This is O(N). 
                # To avoid this, we can just maintain the adj matrix and reconstruct edges at the end.
                # But we need to pick e2 randomly from the current edges.
                # So we need the list.
                # Let's just find the index of e1.
                idx1 = -1
                for i, e in enumerate(edges):
                    if (e[0] == u1 and e[1] == v1) or (e[0] == v1 and e[1] == u1):
                        idx1 = i
                        break
                
                if idx1 != -1:
                    edges[idx1] = (u1, u2)
                    edges[e2_idx] = (v1, v2)
                
                new_bad = get_bad_edges(adj, n, g)
                new_cost = len(new_bad)
                
                delta = new_cost - cost
                if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                    cost = new_cost
                else:
                    # Revert
                    adj[u1].remove(u2)
                    adj[u2].remove(u1)
                    adj[v1].remove(v2)
                    adj[v2].remove(v1)
                    
                    adj[u1].append(v1)
                    adj[v1].append(u1)
                    adj[u2].append(v2)
                    adj[v2].append(u2)
                    
                    if idx1 != -1:
                        edges[idx1] = (u1, v1)
                        edges[e2_idx] = (u2, v2)
                        
                T *= cooling

    return (best_n, best_edges)

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
