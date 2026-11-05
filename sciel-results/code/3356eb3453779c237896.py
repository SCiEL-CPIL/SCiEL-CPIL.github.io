import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def build_edge_data(n):
        num_edges = n * (n - 1) // 2
        edge_index = np.zeros((n, n), dtype=np.int32)
        endpoints = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[i, j] = idx
                edge_index[j, i] = idx
                endpoints.append((i, j))
                idx += 1
        return edge_index, endpoints, num_edges

    def build_cn_and_cost(n, colors, edge_index, endpoints):
        num_edges = len(colors)
        neighbors = [[[] for _ in range(5)] for _ in range(n)]
        for e in range(num_edges):
            u, v = endpoints[e]
            c = int(colors[e])
            neighbors[u][c].append(v)
            neighbors[v][c].append(u)
        
        CN = [np.zeros((n, n), dtype=np.int16) for _ in range(5)]
        for c in range(5):
            for k in range(n):
                nbrs = neighbors[k][c]
                ln = len(nbrs)
                for i in range(ln):
                    u = nbrs[i]
                    for j in range(i + 1, ln):
                        v = nbrs[j]
                        CN[c][u, v] += 1
                        CN[c][v, u] += 1
        
        cost = 0
        for c in range(5):
            tri_u, tri_v = np.triu_indices(n, 1)
            vals = CN[c][tri_u, tri_v]
            cost += int(np.sum(vals * (vals - 1) // 2))
        
        return CN, neighbors, cost

    def solve_n(n, max_restarts=15):
        edge_index, endpoints, num_edges = build_edge_data(n)
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                return None
            
            colors = np.random.randint(0, 5, size=num_edges, dtype=np.int8)
            CN, neighbors, current_cost = build_cn_and_cost(n, colors, edge_index, endpoints)
            
            if current_cost == 0:
                return colors.tolist()
            
            best_cost = current_cost
            best_colors = colors.copy()
            
            # Hill climbing with random restarts within
            temperature = 10.0
            cooling = 0.99
            max_iter = 100000
            
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                if current_cost == 0:
                    return colors.tolist()
                
                e_idx = np.random.randint(num_edges)
                u, v = endpoints[e_idx]
                old_c = int(colors[e_idx])
                
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
                
                if best_new_c < 0:
                    continue
                
                # Accept with SA probability
                accept = False
                if best_delta < 0:
                    accept = True
                elif temperature > 0.1:
                    if random.random() < np.exp(-best_delta / temperature):
                        accept = True
                
                if accept:
                    current_cost += best_delta
                    colors[e_idx] = best_new_c
                    
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
                    
                    neighbors[u][old_c].remove(v)
                    neighbors[v][old_c].remove(u)
                    neighbors[u][best_new_c].append(v)
                    neighbors[v][best_new_c].append(u)
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors.copy()
                        if best_cost == 0:
                            return best_colors.tolist()
                
                temperature *= cooling
            
            if best_cost == 0:
                return best_colors.tolist()
        
        return None

    best_n = 11
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_colors = colors_fallback

    for n in range(12, 27):
        if time.time() > deadline - 10:
            break
        res = solve_n(n, max_restarts=15)
        if res is not None:
            best_n = n
            best_colors = res

    return {'n': best_n, 'colors': best_colors}
