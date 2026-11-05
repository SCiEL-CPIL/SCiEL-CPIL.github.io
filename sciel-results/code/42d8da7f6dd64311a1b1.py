import time
import random
import math
import sys

def construct():
    """
    Find a 5-coloring of K_n with no monochromatic C4, maximizing n.
    Uses incremental energy update for fast local search.
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

    best_n = 1
    best_colors = [0]
    
    for target_n in range(8, 29):
        if time_left() < 10:
            break
        
        n = target_n
        num_edges = n * (n - 1) // 2
        
        # Precompute edge list
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_list.append((i, j, edge_idx(i, j, n)))
        
        found = False
        max_restarts = 200
        
        for restart in range(max_restarts):
            if time_left() < 5:
                break
            
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            
            adj = [[0] * n for _ in range(5)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    idx += 1
            
            energy = compute_energy(adj, n)
            
            if energy == 0:
                best_n = n
                best_colors = colors[:]
                found = True
                break
            
            best_energy = energy
            best_colors_local = colors[:]
            
            # Phase 1: Simulated annealing with faster cooling
            temp = 3.0
            cooling = 0.9998
            min_temp = 0.1
            max_steps = 300000
            stuck_count = 0
            max_stuck = 15000
            
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
                
                # Calculate delta
                delta = 0
                
                # Remove from c_old
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
                
                # Add to c_new
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
            
            # Phase 2: Polish - focus on edges in the most violated color classes
            if time_left() > 5 and best_energy > 0:
                # Compute per-color energy to identify worst colors
                color_energy = [0] * 5
                for c in range(5):
                    for i in range(n):
                        for j in range(i + 1, n):
                            common = adj[c][i] & adj[c][j]
                            k = common.bit_count()
                            if k >= 2:
                                color_energy[c] += k * (k - 1) // 2
                
                # Sort edges by "violatedness" - edges in colors with high energy
                # For polish, we'll do targeted moves on edges in the worst colors
                polish_steps = 50000
                polish_temp = 0.5
                polish_cooling = 0.9999
                polish_stuck = 0
                polish_max_stuck = 5000
                
                # Build list of edges grouped by their current color
                edges_by_color = [[] for _ in range(5)]
                for ei in range(num_edges):
                    c = colors[ei]
                    edges_by_color[c].append(ei)
                
                for pstep in range(polish_steps):
                    if time_left() < 3:
                        break
                    if polish_temp < 0.05:
                        break
                    if polish_stuck > polish_max_stuck:
                        break
                    
                    # Pick an edge from the color class with highest energy
                    # Weighted random selection based on color energy
                    total_ce = sum(color_energy)
                    if total_ce == 0:
                        break
                    
                    r = random.random() * total_ce
                    cum = 0
                    chosen_color = 0
                    for c in range(5):
                        cum += color_energy[c]
                        if r <= cum:
                            chosen_color = c
                            break
                    
                    if not edges_by_color[chosen_color]:
                        continue
                    
                    ei = random.choice(edges_by_color[chosen_color])
                    u, v, e_idx = edge_list[ei]
                    c_old = colors[ei]
                    c_new = random.randint(0, 4)
                    if c_new == c_old:
                        continue
                    
                    # Calculate delta
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
                    
                    if delta <= 0 or random.random() < math.exp(-delta / polish_temp):
                        # Update color_energy before and after
                        # This is approximate; we'll recompute at end of polish
                        energy = new_energy
                        
                        adj[c_old][u] &= ~(1 << v)
                        adj[c_old][v] &= ~(1 << u)
                        adj[c_new][u] |= (1 << v)
                        adj[c_new][v] |= (1 << u)
                        
                        # Update edges_by_color
                        edges_by_color[c_old].remove(ei)
                        edges_by_color[c_new].append(ei)
                        colors[e_idx] = c_new
                        
                        if energy < best_energy:
                            best_energy = energy
                            best_colors_local = colors[:]
                            polish_stuck = 0
                            if energy == 0:
                                best_n = n
                                best_colors = colors[:]
                                found = True
                                break
                    else:
                        polish_stuck += 1
                    
                    polish_temp *= polish_cooling
                
                if found:
                    break
                
                # Recompute exact energy and color energies after polish
                energy = compute_energy(adj, n)
                if energy < best_energy:
                    best_energy = energy
                    best_colors_local = colors[:]
                
                if best_energy == 0:
                    best_n = n
                    best_colors = colors[:]
                    found = True
                    break

        if found:
            continue

    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    result = construct()
    print(f"Found n = {result['n']}")
