"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import time

def construct():
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def simulate_annealing(edges, n, time_limit=30):
        start_time = time.time()
        
        # Build adjacency
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        E = len(edges)
        if E == 0:
            return []
            
        # Precompute conflict matrix
        conflict_matrix = [[False] * E for _ in range(E)]
        for i in range(E):
            u1, v1 = edges[i]
            for j in range(i + 1, E):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict_matrix[i][j] = True
                    conflict_matrix[j][i] = True
                else:
                    if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                        conflict_matrix[i][j] = True
                        conflict_matrix[j][i] = True
        
        K_max = 60
        W_conf = 10000
        W_sum = 1
        
        rng_sa = random.Random(42 + n)
        colors = [rng_sa.randint(0, K_max - 1) for _ in range(E)]
        
        color_edges = [set() for _ in range(K_max)]
        for i, c in enumerate(colors):
            color_edges[c].add(i)
            
        conflict_count = [0] * K_max
        for c in range(K_max):
            ce = color_edges[c]
            ce_list = list(ce)
            for idx, i in enumerate(ce_list):
                for j in ce_list[idx+1:]:
                    if conflict_matrix[i][j]:
                        conflict_count[c] += 1
                        
        total_conflicts = sum(conflict_count)
        total_sum_colors = sum(colors)
        
        def energy(total_conf, total_sum):
            return total_conf * W_conf + total_sum * W_sum
            
        current_energy = energy(total_conflicts, total_sum_colors)
        
        T = 100.0
        T_min = 0.1
        alpha = 0.9995
        
        best_energy = current_energy
        best_colors = colors[:]
        
        while time.time() - start_time < time_limit:
            i = rng_sa.randint(0, E - 1)
            c_old = colors[i]
            c_new = rng_sa.randint(0, K_max - 1)
            
            if c_old == c_new:
                continue
                
            delta_conf_remove = 0
            for j in color_edges[c_old]:
                if j != i and conflict_matrix[i][j]:
                    delta_conf_remove += 1
                    
            delta_conf_add = 0
            for j in color_edges[c_new]:
                if conflict_matrix[i][j]:
                    delta_conf_add += 1
                    
            delta_conflicts = delta_conf_add - delta_conf_remove
            delta_sum = c_new - c_old
            delta_energy = delta_conflicts * W_conf + delta_sum * W_sum
            
            if delta_energy < 0 or rng_sa.random() < math.exp(-delta_energy / T):
                colors[i] = c_new
                color_edges[c_old].remove(i)
                color_edges[c_new].add(i)
                total_conflicts += delta_conflicts
                total_sum_colors += delta_sum
                current_energy += delta_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_colors = colors[:]
                    
            T *= alpha
            if T < T_min:
                T = T_min
                
        return best_colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        colors = simulate_annealing(edges, n, time_limit=30)
        out.append(colors)
    return out
# EVOLVE-BLOCK-END
