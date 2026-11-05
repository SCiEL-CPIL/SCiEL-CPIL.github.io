import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0
    
    def time_left():
        return deadline - time.time()
    
    def solve(n, max_steps=50000):
        m = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Build edge index lookup
        edge_to_idx = {}
        for idx, (i, j) in enumerate(edges):
            edge_to_idx[(i, j)] = idx
            edge_to_idx[(j, i)] = idx
        
        # Initialize
        color = np.random.randint(0, 5, size=m)
        
        # Precompute for each vertex, list of (neighbor, edge_idx)
        vertex_edges = [[] for _ in range(n)]
        for idx, (i, j) in enumerate(edges):
            vertex_edges[i].append((j, idx))
            vertex_edges[j].append((i, idx))
        
        def compute_energy(color):
            total = 0
            for c in range(5):
                adj = np.zeros((n, n), dtype=np.uint8)
                for idx, (i, j) in enumerate(edges):
                    if color[idx] == c:
                        adj[i, j] = 1
                        adj[j, i] = 1
                common = adj @ adj
                for i in range(n):
                    for j in range(i + 1, n):
                        if common[i, j] >= 2:
                            total += 1
            return total
        
        best_energy = compute_energy(color)
        best_color = color.copy()
        
        T = 5.0
        cooling = 0.9995
        min_T = 0.001
        
        for step in range(max_steps):
            if time_left() < 3:
                break
            if T < min_T:
                break
            
            idx = random.randint(0, m - 1)
            old_c = color[idx]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            color[idx] = new_c
            new_energy = compute_energy(color)
            delta = new_energy - best_energy
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                if delta < 0:
                    best_energy = new_energy
                    best_color = color.copy()
            else:
                color[idx] = old_c
            
            T *= cooling
        
        return best_energy, best_color
    
    # Try n=26
    energy, color = solve(26, max_steps=30000)
    if energy == 0:
        return {'n': 26, 'colors': color.tolist()}
    
    # Try n=25
    energy, color = solve(25, max_steps=30000)
    if energy == 0:
        return {'n': 25, 'colors': color.tolist()}
    
    # Fallback
    n = 11
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            colors.append((d * 3) % 5)
    return {'n': n, 'colors': colors}
