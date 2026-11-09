import time
import random
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 115.0

    def solve_n(n):
        m = n * (n - 1) // 2
        if m == 0:
            return []
        
        # Precompute edge mappings
        edge_to_uv = np.zeros((m, 2), dtype=np.int8)
        uv_to_edge = np.zeros((n, n), dtype=np.int32)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_to_uv[idx] = [i, j]
                uv_to_edge[i, j] = idx
                uv_to_edge[j, i] = idx
                idx += 1

        colors = np.zeros(m, dtype=np.int8)
        adj = [ [set() for _ in range(n)] for _ in range(5) ]
        N = [ np.zeros((n, n), dtype=np.int16) for _ in range(5) ]
        total_cost = 0

        def get_cost_delta(c, u, v, adding):
            delta = 0
            for w in range(n):
                if w == u or w == v:
                    continue
                if w in adj[c][v]:
                    if adding:
                        delta += N[c][u][w]
                    else:
                        delta -= (N[c][u][w] - 1)
                if w in adj[c][u]:
                    if adding:
                        delta += N[c][v][w]
                    else:
                        delta -= (N[c][v][w] - 1)
            return delta

        def apply_color(u, v, c, adding):
            nonlocal total_cost
            delta = get_cost_delta(c, u, v, adding)
            total_cost += delta
            if adding:
                adj[c][u].add(v)
                adj[c][v].add(u)
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if w in adj[c][v]:
                        N[c][u][w] += 1
                        N[c][w][u] += 1
                    if w in adj[c][u]:
                        N[c][v][w] += 1
                        N[c][w][v] += 1
            else:
                adj[c][u].remove(v)
                adj[c][v].remove(u)
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if w in adj[c][v]:
                        N[c][u][w] -= 1
                        N[c][w][u] -= 1
                    if w in adj[c][u]:
                        N[c][v][w] -= 1
                        N[c][w][v] -= 1

        # Greedy Init
        edge_indices = np.arange(m)
        random.shuffle(edge_indices)
        for idx in edge_indices:
            u, v = edge_to_uv[idx]
            best_c = 0
            best_d = float('inf')
            for c in range(5):
                d = get_cost_delta(c, u, v, True)
                if d < best_d:
                    best_d = d
                    best_c = c
            apply_color(u, v, best_c, True)
            colors[idx] = best_c

        # Local Search
        # Use a simple kick strategy
        steps = 0
        max_steps = 2000
        kick_interval = 200
        kick_size = 50
        last_improvement = 0
        current_best_cost = total_cost
        
        while total_cost > 0 and steps < max_steps:
            if time.time() > deadline - 2:
                break
            
            # Check if time for a kick
            if steps - last_improvement > kick_interval and total_cost > 0:
                # Perform random moves
                for _ in range(kick_size):
                    idx = random.randint(0, m - 1)
                    u, v = edge_to_uv[idx]
                    old_c = colors[idx]
                    new_c = random.randint(0, 4)
                    if new_c == old_c:
                        continue
                    apply_color(u, v, old_c, False)
                    apply_color(u, v, new_c, True)
                    colors[idx] = new_c
                last_improvement = steps
            
            idx = random.randint(0, m - 1)
            u, v = edge_to_uv[idx]
            old_c = colors[idx]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            d_remove = get_cost_delta(old_c, u, v, False)
            d_add = get_cost_delta(new_c, u, v, True)
            d_total = d_remove + d_add
            
            if d_total < 0:
                apply_color(u, v, old_c, False)
                apply_color(u, v, new_c, True)
                colors[idx] = new_c
                last_improvement = steps
                if total_cost < current_best_cost:
                    current_best_cost = total_cost
            elif d_total == 0:
                # Accept neutral moves to explore
                apply_color(u, v, old_c, False)
                apply_color(u, v, new_c, True)
                colors[idx] = new_c
            # else: reject
            
            steps += 1

        if total_cost == 0:
            return colors.tolist()
        return None

    # Try n from high to low
    for n in [28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
        if time.time() > deadline - 2:
            break
        
        found = None
        # Increase attempts for smaller n if larger n fails quickly?
        # For n=26, 20 attempts should be enough.
        attempts = 30 if n <= 26 else 10
        for seed in range(attempts):
            if time.time() > deadline - 2:
                break
            random.seed(seed + n * 100)
            res = solve_n(n)
            if res is not None:
                found = res
                break
        
        if found is not None:
            return {'n': n, 'colors': found}
    
    return {'n': 1, 'colors': []}
