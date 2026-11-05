import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_n = 11
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_colors = colors_fallback

    def solve_n(n):
        num_edges = n * (n - 1) // 2
        
        # Precompute edge index as 2D array for O(1) lookup
        edge_index = np.zeros((n, n), dtype=np.int32)
        endpoints = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[i, j] = idx
                edge_index[j, i] = idx
                endpoints.append((i, j))
                idx += 1
        
        max_restarts = 30
        max_iter = 50000
        
        for restart in range(max_restarts):
            if time.time() > deadline - 3:
                return None
            
            colors = np.random.randint(0, 5, size=num_edges, dtype=np.int8)
            
            # Symmetric CN matrices
            CN = [np.zeros((n, n), dtype=np.int16) for _ in range(5)]
            
            # Build neighbor lists
            neighbors = [ [list() for _ in range(5)] for _ in range(n) ]
            for e in range(num_edges):
                u, v = endpoints[e]
                c = colors[e]
                neighbors[u][c].append(v)
                neighbors[v][c].append(u)
            
            # Init CN
            for c in range(5):
                for k in range(n):
                    nbrs = neighbors[k][c]
                    for i in range(len(nbrs)):
                        u = nbrs[i]
                        for j in range(i + 1, len(nbrs)):
                            v = nbrs[j]
                            CN[c][u, v] += 1
                            CN[c][v, u] += 1
            
            # Initial cost
            current_cost = 0
            for c in range(5):
                tri_u, tri_v = np.triu_indices(n, 1)
                vals = CN[c][tri_u, tri_v]
                current_cost += int(np.sum(vals * (vals - 1) // 2))
            
            if current_cost == 0:
                return colors.tolist()
            
            # Steepest descent hill climbing
            for i in range(max_iter):
                if time.time() > deadline - 3:
                    break
                if current_cost == 0:
                    return colors.tolist()
                
                e_idx = np.random.randint(num_edges)
                u, v = endpoints[e_idx]
                old_c = int(colors[e_idx])
                
                # Try all 4 other colors, pick best
                best_delta = float('inf')
                best_new_c = -1
                for new_c in range(5):
                    if new_c == old_c:
                        continue
                    
                    delta = 0
                    for x in range(n):
                        if x == u or x == v:
                            continue
                        
                        c_vx = int(colors[edge_index[v, x]])
                        c_ux = int(colors[edge_index[u, x]])
                        
                        if c_vx == old_c:
                            delta -= (int(CN[old_c][u, x]) - 1)
                        if c_ux == old_c:
                            delta -= (int(CN[old_c][v, x]) - 1)
                        if c_vx == new_c:
                            delta += int(CN[new_c][u, x])
                        if c_ux == new_c:
                            delta += int(CN[new_c][v, x])
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_new_c = new_c
                
                if best_new_c < 0 or best_delta >= 0:
                    continue
                
                current_cost += best_delta
                colors[e_idx] = best_new_c
                
                # Update CN
                for x in range(n):
                    if x == u or x == v:
                        continue
                    
                    c_vx = int(colors[edge_index[v, x]])
                    c_ux = int(colors[edge_index[u, x]])
                    
                    if c_vx == old_c:
                        CN[old_c][u, x] -= 1
                        CN[old_c][x, u] -= 1
                    if c_ux == old_c:
                        CN[old_c][v, x] -= 1
                        CN[old_c][x, v] -= 1
                    if c_vx == best_new_c:
                        CN[best_new_c][u, x] += 1
                        CN[best_new_c][x, u] += 1
                    if c_ux == best_new_c:
                        CN[best_new_c][v, x] += 1
                        CN[best_new_c][x, v] += 1
                
                # Update neighbors
                neighbors[u][old_c].remove(v)
                neighbors[v][old_c].remove(u)
                neighbors[u][best_new_c].append(v)
                neighbors[v][best_new_c].append(u)
        
        return None

    # Try increasing n, starting from 12
    for n in range(12, 27):
        if time.time() > deadline - 5:
            break
        res = solve_n(n)
        if res is not None:
            best_n = n
            best_colors = res
    
    return {'n': best_n, 'colors': best_colors}
