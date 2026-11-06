import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def get_violations(colors, n, edge_indices):
        adj = [[0]*n for _ in range(5)]
        for idx, (i, j) in enumerate(edge_indices):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
        
        v_count = 0
        for c in range(5):
            for i in range(n):
                adj_i = adj[c][i]
                for j in range(i+1, n):
                    common = (adj_i & adj[c][j]).bit_count()
                    if common >= 2:
                        v_count += common - 1
        return v_count

    def try_solve(n, time_limit):
        edge_indices = [(i, j) for i in range(n) for j in range(i+1, n)]
        num_edges = len(edge_indices)
        
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        current_v = get_violations(colors, n, edge_indices)
        if current_v == 0:
            return colors
        
        best_v = current_v
        best_colors = colors[:]
        
        T = 10.0
        cooling = 0.9995
        min_T = 0.1
        
        while time.time() < time_limit:
            idx = random.randint(0, num_edges - 1)
            i, j = edge_indices[idx]
            
            old_c = colors[idx]
            
            # Try a random different color
            new_c = random.randint(0, 4)
            if new_c == old_c:
                new_c = (new_c + 1) % 5
            
            colors[idx] = new_c
            new_v = get_violations(colors, n, edge_indices)
            
            if new_v < current_v:
                current_v = new_v
                if current_v < best_v:
                    best_v = current_v
                    best_colors = colors[:]
                    if best_v == 0:
                        return best_colors
            else:
                delta = new_v - current_v
                if delta > 0:
                    if T > min_T:
                        # Avoid exp of very negative numbers
                        if delta < 50 * T:
                            if random.random() < math.exp(-delta / T):
                                current_v = new_v
                            else:
                                colors[idx] = old_c
                        else:
                            colors[idx] = old_c
                    else:
                        colors[idx] = old_c
                else:
                    # delta == 0, accept to move around
                    current_v = new_v
            
            T *= cooling
            if T < min_T:
                T = min_T

        if best_v == 0:
            return best_colors
        return None

    # Try n from 26 down to 11
    n_values = [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]
    
    for n in n_values:
        if time.time() > deadline - 2:
            break
            
        attempts = 0
        while time.time() < deadline - 2:
            attempts += 1
            res = try_solve(n, deadline - 2)
            if res is not None:
                return {'n': n, 'colors': res}
            
            if n == 26 and attempts >= 5:
                break
            if n < 26 and attempts >= 3:
                break

    # Fallback: Return the seed solution
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
