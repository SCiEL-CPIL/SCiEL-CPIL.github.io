import time
import random
import math
import sys

def construct():
    """
    Find a 5-coloring of K_n with no monochromatic C4, maximizing n.
    Uses the Lazebnik-Woldar algebraic construction for n=26 as baseline,
    then attempts to extend to n=27 via local search.
    """
    deadline = time.time() + 115
    
    def time_left():
        return deadline - time.time()
    
    def edge_idx(i, j, n):
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    def compute_energy(adj, n):
        total = 0
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    common = adj[c][i] & adj[c][j]
                    k = common.bit_count()
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total
    
    def build_adj(colors, n):
        adj = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj
    
    # Lazebnik-Woldar construction for n=26
    # Vertices: (i, j) for i, j in Z_5 (25 vertices) + vertex 25
    n26 = 26
    colors26 = [0] * (n26 * (n26 - 1) // 2)
    
    for i1 in range(5):
        for j1 in range(5):
            v1 = i1 * 5 + j1
            for i2 in range(5):
                for j2 in range(5):
                    v2 = i2 * 5 + j2
                    if v1 >= v2:
                        continue
                    di = (i2 - i1) % 5
                    dj = (j2 - j1) % 5
                    c = (di + 2 * dj) % 5
                    colors26[edge_idx(v1, v2, n26)] = c
    
    for i in range(5):
        for j in range(5):
            v = i * 5 + j
            c = i
            colors26[edge_idx(v, 25, n26)] = c
    
    adj26 = build_adj(colors26, n26)
    energy26 = compute_energy(adj26, n26)
    
    best_n = 1
    best_colors = [0]
    
    if energy26 == 0:
        best_n = n26
        best_colors = colors26[:]
        
        # Try to extend to n=27
        if time_left() > 15:
            n27 = 27
            num_edges27 = n27 * (n27 - 1) // 2
            
            colors27 = [0] * num_edges27
            for i in range(n26):
                for j in range(i + 1, n26):
                    colors27[edge_idx(i, j, n27)] = colors26[edge_idx(i, j, n26)]
            
            for i in range(n26):
                colors27[edge_idx(i, 26, n27)] = random.randint(0, 4)
            
            adj27 = build_adj(colors27, n27)
            energy27 = compute_energy(adj27, n27)
            
            edge_list27 = []
            for i in range(n27):
                for j in range(i + 1, n27):
                    edge_list27.append((i, j, edge_idx(i, j, n27)))
            
            temp = 1.0
            cooling = 0.9999
            min_temp = 0.01
            max_steps = 300000
            stuck_count = 0
            max_stuck = 15000
            best_energy27 = energy27
            best_colors27 = colors27[:]
            
            for step in range(max_steps):
                if time_left() < 5:
                    break
                if temp < min_temp:
                    break
                if stuck_count > max_stuck:
                    break
                
                ei = random.randint(0, num_edges27 - 1)
                u, v, e_idx = edge_list27[ei]
                c_old = colors27[e_idx]
                c_new = random.randint(0, 4)
                if c_new == c_old:
                    continue
                
                delta = 0
                for w in range(n27):
                    if w == u or w == v:
                        continue
                    mask_w = adj27[c_old][w]
                    if mask_w & (1 << v):
                        k = (adj27[c_old][u] & mask_w).bit_count()
                        delta -= (k - 1)
                    if mask_w & (1 << u):
                        k = (adj27[c_old][v] & mask_w).bit_count()
                        delta -= (k - 1)
                
                for w in range(n27):
                    if w == u or w == v:
                        continue
                    mask_w = adj27[c_new][w]
                    if mask_w & (1 << v):
                        k = (adj27[c_new][u] & mask_w).bit_count()
                        delta += k
                    if mask_w & (1 << u):
                        k = (adj27[c_new][v] & mask_w).bit_count()
                        delta += k
                
                new_energy = energy27 + delta
                
                if delta <= 0 or random.random() < math.exp(-delta / temp):
                    energy27 = new_energy
                    adj27[c_old][u] &= ~(1 << v)
                    adj27[c_old][v] &= ~(1 << u)
                    adj27[c_new][u] |= (1 << v)
                    adj27[c_new][v] |= (1 << u)
                    colors27[e_idx] = c_new
                    
                    if energy27 < best_energy27:
                        best_energy27 = energy27
                        best_colors27 = colors27[:]
                        stuck_count = 0
                        if energy27 == 0:
                            best_n = n27
                            best_colors = colors27[:]
                            break
                else:
                    stuck_count += 1
                
                temp *= cooling
    
    # Fallback: search from scratch if construction failed
    if best_n < 26:
        for target_n in range(8, min(29, best_n + 15)):
            if time_left() < 10:
                break
            
            n = target_n
            num_edges = n * (n - 1) // 2
            
            edge_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    edge_list.append((i, j, edge_idx(i, j, n)))
            
            found = False
            max_restarts = 100
            
            for restart in range(max_restarts):
                if time_left() < 5:
                    break
                
                colors = [random.randint(0, 4) for _ in range(num_edges)]
                adj = build_adj(colors, n)
                energy = compute_energy(adj, n)
                
                if energy == 0:
                    best_n = n
                    best_colors = colors[:]
                    found = True
                    break
                
                best_energy = energy
                best_colors_local = colors[:]
                
                temp = 2.0
                cooling = 0.99995
                min_temp = 0.05
                max_steps = 200000
                stuck_count = 0
                max_stuck = 10000
                
                for step in range(max_steps):
                    if time_left() < 3:
                        break
                    if temp < min_temp:
                        break
                    if stuck_count > max_stuck:
                        break
                    
                    ei = random.randint(0, num_edges - 1)
                    u, v, e_idx = edge_list[ei]
                    c_old = colors[e_idx]
                    c_new = random.randint(0, 4)
                    if c_new == c_old:
                        continue
                    
                    delta = 0
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        mask_w = adj[c_old][w]
                        if mask_w & (1 << v):
                            k = (adj[c_old][u] & mask_w).bit_count()
                            delta -= (k - 1)
                        if mask_w & (1 << u):
                            k = (adj[c_old][v] & mask_w).bit_count()
                            delta -= (k - 1)
                    
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        mask_w = adj[c_new][w]
                        if mask_w & (1 << v):
                            k = (adj[c_new][u] & mask_w).bit_count()
                            delta += k
                        if mask_w & (1 << u):
                            k = (adj[c_new][v] & mask_w).bit_count()
                            delta += k
                    
                    new_energy = energy + delta
                    
                    if delta <= 0 or random.random() < math.exp(-delta / temp):
                        energy = new_energy
                        adj[c_old][u] &= ~(1 << v)
                        adj[c_old][v] &= ~(1 << u)
                        adj[c_new][u] |= (1 << v)
                        adj[c_new][v] |= (1 << u)
                        colors[e_idx] = c_new
                        
                        if energy < best_energy:
                            best_energy = energy
                            best_colors_local = colors[:]
                            stuck_count = 0
                            if energy == 0:
                                best_n = n
                                best_colors = colors[:]
                                found = True
                                break
                    else:
                        stuck_count += 1
                    
                    temp *= cooling
                
                if found:
                    break
            
            if found:
                continue
    
    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    result = construct()
    print(f"Found n = {result['n']}")
