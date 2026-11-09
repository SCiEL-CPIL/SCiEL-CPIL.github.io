import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_edges = []
    best_count = 0
    
    n = 20
    
    def count_edges(adj):
        return np.sum(adj) // 2

    def is_valid_add(adj, common, u, v):
        if adj[u][v]:
            return False
        # Check common[u][v] <= 2
        if common[u][v] > 2:
            return False
        
        # Check max(common[u][w] for w in N(v)) <= 1
        # N(v) are indices where adj[v][w] is 1
        # We need common[u][w] <= 1 for all w in N(v)
        # And common[v][w] <= 1 for all w in N(u)
        
        # Efficient check using numpy
        # N_v = np.where(adj[v] == 1)[0]
        # if np.any(common[u][N_v] > 1): return False
        
        # But we must be careful with u and v themselves?
        # adj[v][u] is 0 initially.
        # adj[u][v] is 0 initially.
        
        # Let's just use loops for simplicity and speed for n=20
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w]:
                if common[u][w] > 1:
                    return False
            if adj[u][w]:
                if common[v][w] > 1:
                    return False
        return True

    def add_edge(adj, common, u, v):
        adj[u][v] = 1
        adj[v][u] = 1
        
        # Update common
        # common[u][v] and common[v][u] do NOT change value? 
        # Wait, common[u][v] is |N(u) cap N(v)|.
        # As derived before, it doesn't change.
        
        # But we need to update common[u][w] for w in N(v)
        # and common[v][w] for w in N(u)
        
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w]:
                common[u][w] += 1
                common[w][u] += 1
            if adj[u][w]:
                common[v][w] += 1
                common[w][v] += 1
                
    def remove_edge(adj, common, u, v):
        adj[u][v] = 0
        adj[v][u] = 0
        
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w]:
                common[u][w] -= 1
                common[w][u] -= 1
            if adj[u][w]:
                common[v][w] -= 1
                common[w][v] -= 1

    def greedy_fill(adj, common, max_time=None):
        changed = True
        while changed:
            if max_time and time.time() > max_time:
                return
            changed = False
            # Try to add any valid edge
            # Randomize order to avoid bias
            edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
            random.shuffle(edges_to_try)
            for u, v in edges_to_try:
                if is_valid_add(adj, common, u, v):
                    add_edge(adj, common, u, v)
                    changed = True
                    if max_time and time.time() > max_time:
                        return

    def local_search(adj, common, max_time=None):
        # Try to improve by swapping edges?
        # Or just greedy fill is enough if we start from different points.
        # Let's try a simple hill climbing: remove an edge, try to add two.
        # This is complex. Let's stick to multiple greedy restarts.
        pass

    # Multiple restarts
    # We increase restart count because starting from empty is faster per restart
    # and guarantees validity.
    for restart in range(5000):
        if time.time() > deadline:
            break
            
        # Initialize empty graph
        adj = np.zeros((n, n), dtype=int)
        common = np.zeros((n, n), dtype=int)
        
        # Greedy fill from empty state
        # This ensures the graph is always K2,3-free
        greedy_fill(adj, common, max_time=deadline)
        
        count = count_edges(adj)
        if count > best_count:
            best_count = count
            best_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        best_edges.append([i, j])

    return best_edges
