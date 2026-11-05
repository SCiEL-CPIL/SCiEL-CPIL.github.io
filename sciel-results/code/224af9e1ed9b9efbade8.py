import numpy as np
import time
import random

def construct():
    def try_solve(n, time_limit=100):
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        
        # Initialize
        colors = np.random.randint(0, 5, size=num_edges)
        adj = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
        
        for k, (i, j) in enumerate(edges):
            c = colors[k]
            adj[c][i, j] = 1
            adj[c][j, i] = 1
            
        def calc_cost():
            cost = 0
            for c in range(5):
                A = adj[c]
                A2 = A @ A
                trace4 = np.trace(A2 @ A2)
                degrees = np.sum(A, axis=1)
                sum_d_d_minus_1 = np.sum(degrees * (degrees - 1))
                cost += (trace4 - 2 * sum_d_d_minus_1) / 8
            return cost

        current_cost = calc_cost()
        start_time = time.time()
        
        while current_cost > 0 and time.time() - start_time < time_limit:
            k = np.random.randint(0, num_edges)
            old_color = colors[k]
            i, j = edges[k]
            
            best_cost = current_cost
            best_color = old_color
            
            for new_color in range(5):
                if new_color == old_color:
                    continue
                
                # Swap
                adj[old_color][i, j] = 0
                adj[old_color][j, i] = 0
                adj[new_color][i, j] = 1
                adj[new_color][j, i] = 1
                
                new_cost = calc_cost()
                
                # Revert
                adj[new_color][i, j] = 0
                adj[new_color][j, i] = 0
                adj[old_color][i, j] = 1
                adj[old_color][j, i] = 1
                
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_color = new_color
                    
            if best_cost < current_cost:
                # Apply best
                adj[old_color][i, j] = 0
                adj[old_color][j, i] = 0
                adj[best_color][i, j] = 1
                adj[best_color][j, i] = 1
                colors[k] = best_color
                current_cost = best_cost
            else:
                # Random move with probability 0.1
                if random.random() < 0.1:
                    new_color = np.random.randint(0, 5)
                    adj[old_color][i, j] = 0
                    adj[old_color][j, i] = 0
                    adj[new_color][i, j] = 1
                    adj[new_color][j, i] = 1
                    colors[k] = new_color
                    current_cost = calc_cost()
                    
        if current_cost == 0:
            return {'n': n, 'colors': colors.tolist()}
        return None

    # Try n=26
    res = try_solve(26, time_limit=100)
    if res:
        return res
        
    # Try n=25
    res = try_solve(25, time_limit=100)
    if res:
        return res
        
    # Try n=24
    res = try_solve(24, time_limit=100)
    if res:
        return res
        
    # Fallback
    n_seed = 11
    return {'n': n_seed, 'colors': [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]}
