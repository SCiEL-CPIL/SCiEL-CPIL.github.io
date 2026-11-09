import random
import time
import itertools
import numpy as np

def construct():
    n = 16
    target_edges = 65
    
    def get_edges(adj_matrix):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj_matrix[i][j]:
                    edges.append([i, j])
        return edges

    def calculate_score_np(adj_np):
        """Calculate max triangle degree sum using NumPy for speed."""
        deg = adj_np.sum(axis=1)
        # Find all triangles: for each edge (u,v), check common neighbors
        max_sum = 0
        for u in range(n):
            for v in range(u+1, n):
                if adj_np[u, v]:
                    # Common neighbors of u and v
                    common = np.where(adj_np[u] & adj_np[v] > 0)[0]
                    for w in common:
                        if w > v:  # avoid duplicates
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    def make_baseline():
        adj = [[0]*n for _ in range(n)]
        for u in range(8):
            for v in range(8, 16):
                adj[u][v] = 1
                adj[v][u] = 1
        u, v = random.sample(range(8), 2)
        adj[u][v] = 1
        adj[v][u] = 1
        return adj

    def make_random():
        adj = [[0]*n for _ in range(n)]
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges[:65]:
            adj[u][v] = 1
            adj[v][u] = 1
        return adj

    def adj_to_np(adj):
        return np.array(adj, dtype=np.int8)

    def np_to_adj(adj_np):
        return adj_np.tolist()

    best_graph = None
    best_score = float('inf')
    
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s kill

    # Candidates: baseline + several random starts
    candidates = [make_baseline()]
    for _ in range(10):
        candidates.append(make_random())
    
    for start_adj in candidates:
        if time.time() - start_time > timeout:
            break
            
        current_adj = [row[:] for row in start_adj]
        current_np = adj_to_np(current_adj)
        current_score = calculate_score_np(current_np)
        
        if current_score < best_score:
            best_score = current_score
            best_graph = [row[:] for row in current_adj]
            
        if best_score <= 23:
            break

        # Simulated annealing local search
        T = 2.0  # initial temperature
        cooling = 0.995
        T_min = 0.01
        
        no_improve_count = 0
        
        while time.time() - start_time < timeout and T > T_min:
            # Pick random edge to remove
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if current_np[i, j]:
                        edges.append((i, j))
            
            if not edges:
                break
                
            u, v = random.choice(edges)
            
            # Pick random non-edge to add
            non_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if not current_np[i, j]:
                        non_edges.append((i, j))
                        
            if not non_edges:
                break
                
            x, y = random.choice(non_edges)
            
            # Apply swap
            current_np[u, v] = 0
            current_np[v, u] = 0
            current_np[x, y] = 1
            current_np[y, x] = 1
            
            new_score = calculate_score_np(current_np)
            delta = new_score - current_score
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                # Accept move
                current_score = new_score
                no_improve_count = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_graph = np_to_adj(current_np)
                    if best_score <= 23:
                        break
            else:
                # Revert
                current_np[u, v] = 1
                current_np[v, u] = 1
                current_np[x, y] = 0
                current_np[y, x] = 0
                no_improve_count += 1
            
            T *= cooling
            
            # If stuck for too long, restart with higher T
            if no_improve_count > 300:
                T = max(T, 1.0)
                no_improve_count = 0

    # Convert best_graph to edge list
    if best_graph is None:
        best_graph = make_baseline()
        
    edges = get_edges(best_graph)
    return {'n': 16, 'edges': edges}
