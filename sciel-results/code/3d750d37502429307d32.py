import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 15
    total_edges = n * (n - 1) // 2
    
    # Precompute edge list
    all_edges = []
    edge_idx = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
            edge_idx[(u, v)] = idx
            edge_idx[(v, u)] = idx
            idx += 1
    
    def get_adj(adj_matrix):
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if adj_matrix[u][v]:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj
    
    def check_valid(adj):
        """Check if any path s-u-v-t has deg(s) == deg(t) where s != t"""
        deg = [len(adj[i]) for i in range(n)]
        for u in range(n):
            for v in adj[u]:
                # v is neighbor of u
                # For each s in adj[u] (s != v), and t in adj[v] (t != u, t != s)
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True
    
    def count_violations(adj):
        """Count number of violating paths"""
        deg = [len(adj[i]) for i in range(n)]
        count = 0
        for u in range(n):
            for v in adj[u]:
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            count += 1
        return count
    
    def edge_count(adj_matrix):
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj_matrix[u][v]:
                    count += 1
        return count
    
    best_edges = 0
    best_adj_matrix = None
    
    # Try multiple random starts
    random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline - 5.0:
            break
        
        # Start with random graph at varying densities
        density = 0.3 + 0.5 * (attempt / 50.0)
        adj_matrix = [[False] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < density:
                    adj_matrix[u][v] = True
                    adj_matrix[v][u] = True
        
        adj = get_adj(adj_matrix)
        
        # Remove violating edges greedily
        removed = True
        while removed and time.time() < deadline - 5.0:
            removed = False
            deg = [len(adj[i]) for i in range(n)]
            # Find a violating edge and remove it
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        # Check if removing this edge helps
                        # Just check if there's a violation involving this edge
                        for s in adj[u]:
                            if s == v:
                                continue
                            for t in adj[v]:
                                if t == u or t == s:
                                    continue
                                if deg[s] == deg[t]:
                                    # Remove edge (u,v)
                                    adj_matrix[u][v] = False
                                    adj_matrix[v][u] = False
                                    adj[u].remove(v)
                                    adj[v].remove(u)
                                    removed = True
                                    break
                            if removed:
                                break
                        if removed:
                            break
                if removed:
                    break
        
        # Now try to add edges back
        if time.time() < deadline - 5.0:
            adj = get_adj(adj_matrix)
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                # Try adding each missing edge
                for u in range(n):
                    for v in range(u + 1, n):
                        if not adj_matrix[u][v]:
                            # Try adding
                            adj_matrix[u][v] = True
                            adj_matrix[v][u] = True
                            adj[u].add(v)
                            adj[v].add(u)
                            if check_valid(adj):
                                improved = True
                            else:
                                adj_matrix[u][v] = False
                                adj_matrix[v][u] = False
                                adj[u].remove(v)
                                adj[v].remove(u)
        
        ec = edge_count(adj_matrix)
        if ec > best_edges:
            best_edges = ec
            best_adj_matrix = [row[:] for row in adj_matrix]
        
        if best_edges >= 40:  # Good enough
            break
    
    # Convert to edge list
    if best_adj_matrix is None:
        # Fallback: star graph
        return [[0, v] for v in range(1, n)]
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj_matrix[u][v]:
                edges.append([int(u), int(v)])
    
    return edges
