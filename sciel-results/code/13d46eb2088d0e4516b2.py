import random
import time
import itertools
import math

def construct():
    deadline = time.time() + 115
    N = 16
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    E = len(edges)
    edge_idx = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    k4s = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        k4_edges = [edge_idx[(a,b)], edge_idx[(a,c)], edge_idx[(a,d)],
                    edge_idx[(b,c)], edge_idx[(b,d)], edge_idx[(c,d)]]
        k4s.append(k4_edges)
        
    edge_to_k4s = [[] for _ in range(E)]
    for k4_idx, k4_edges in enumerate(k4s):
        for e in k4_edges:
            edge_to_k4s[e].append(k4_idx)
            
    # Helper to get distinct colors in a K4
    def get_k4_distinct(colors, k4_edges):
        return len(set(colors[e] for e in k4_edges))

    # Calculate total violations and palette size
    def get_metrics(colors):
        violations = 0
        for k4e in k4s:
            d = get_k4_distinct(colors, k4e)
            if d < 5:
                violations += (5 - d)
        
        color_counts = {}
        for c in colors:
            color_counts[c] = color_counts.get(c, 0) + 1
        palette_size = len(color_counts)
        return violations, palette_size

    # Initial state
    # Try a few inits including multiplicative modular scheme
    inits = [
        [(i + j) % 17 for (i, j) in edges],
        [(i * 3 + j) % 15 for (i, j) in edges],
        [(i * 7 + j) % 13 for (i, j) in edges],
        # Multiplicative modular scheme with prime p >= 16
        [(i * j) % 17 for (i, j) in edges],
        [(i * j) % 19 for (i, j) in edges],
        [(i * j) % 23 for (i, j) in edges],
        # Mixed additive-multiplicative
        [(i * j + i + j) % 17 for (i, j) in edges],
        [(i * j + 2*i + 3*j) % 19 for (i, j) in edges],
    ]
    
    rng = random.Random(42)
    for _ in range(8):
        p = rng.randint(13, 15)
        inits.append([rng.randint(0, p-1) for _ in range(E)])
        
    best_colors = None
    best_palette = 17
    best_violations = 0
    
    for init in inits:
        v, p = get_metrics(init)
        if v == 0:
            if p < best_palette:
                best_palette = p
                best_colors = init[:]
                best_violations = 0
            if best_palette == 13:
                return best_colors
                
    if best_colors is None:
        # Fallback to a known valid one, e.g., (i+j)%17
        best_colors = [(i + j) % 17 for (i, j) in edges]
        best_palette = 17
        best_violations = 0
        
    current_colors = best_colors[:]
    current_violations = best_violations
    current_palette = best_palette
    
    # Maintain color counts for fast palette update
    color_counts = [0] * 120
    for c in current_colors:
        color_counts[c] += 1
        
    def update_color_counts(c_old, c_new):
        color_counts[c_old] -= 1
        color_counts[c_new] += 1
        delta_pal = 0
        if color_counts[c_old] == 0:
            delta_pal -= 1
        if color_counts[c_new] == 1:
            delta_pal += 1
        return delta_pal

    # Simulated Annealing / Random Walk
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    rng_search = random.Random(123)
    
    # Precompute for faster delta calculation
    # For each edge, store the k4 indices and the position of the edge in each k4
    edge_k4_info = []
    for e in range(E):
        info = []
        for k4_idx in edge_to_k4s[e]:
            k4e = k4s[k4_idx]
            # Find position of e in k4e
            pos = k4e.index(e)
            info.append((k4_idx, pos))
        edge_k4_info.append(info)
    
    while time.time() < deadline - 2:
        if T < T_min:
            T = T_min
            
        # Pick random edge
        e = rng_search.randint(0, E - 1)
        c_old = current_colors[e]
        
        # Pick random new color from 0 to 15 (encourages small palette)
        c_new = rng_search.randint(0, 15)
        
        if c_new == c_old:
            continue
            
        # Calculate delta violations using precomputed info
        delta_viol = 0
        for k4_idx, pos in edge_k4_info[e]:
            k4e = k4s[k4_idx]
            
            # Get old distinct count
            old_set = set()
            for idx, ee in enumerate(k4e):
                old_set.add(current_colors[ee])
            old_dist = len(old_set)
            
            # Get new distinct count
            new_set = set()
            for idx, ee in enumerate(k4e):
                if idx == pos:
                    new_set.add(c_new)
                else:
                    new_set.add(current_colors[ee])
            new_dist = len(new_set)
            
            old_viol = max(0, 5 - old_dist)
            new_viol = max(0, 5 - new_dist)
            delta_viol += (new_viol - old_viol)
            
        # Calculate delta palette
        delta_pal = update_color_counts(c_old, c_new)
        
        delta_cost = delta_viol * 100 + delta_pal
        
        # Metropolis acceptance
        accept = False
        if delta_cost < 0:
            accept = True
        else:
            # Probability
            # exp(-delta_cost / T)
            if delta_cost > 50 * T: # heuristic cutoff
                accept = False
            else:
                ratio = delta_cost / T
                if ratio > 700: # exp(-700) is ~0
                    accept = False
                else:
                    accept = rng_search.random() < math.exp(-ratio)
                    
        if accept:
            current_colors[e] = c_new
            current_violations += delta_viol
            current_palette += delta_pal
            
            if current_violations == 0:
                if current_palette < best_palette:
                    best_palette = current_palette
                    best_colors = current_colors[:]
                    if best_palette == 13:
                        break
        else:
            # Revert color counts
            update_color_counts(c_new, c_old)
            
        # Cool down
        T *= cooling_rate
    
    # Final local search polish: try to reduce palette by recoloring edges
    # that don't violate the K4 constraint
    if best_violations == 0 and best_palette > 13:
        polish_colors = best_colors[:]
        polish_palette = best_palette
        polish_violations = 0
        
        # Try to eliminate unused colors by recoloring edges with that color
        color_counts_polish = [0] * 120
        for c in polish_colors:
            color_counts_polish[c] += 1
        
        for target_color in range(120):
            if color_counts_polish[target_color] == 0:
                continue
                
            # Try to recolor all edges with this color to other colors
            edges_with_color = [e for e in range(E) if polish_colors[e] == target_color]
            
            for e in edges_with_color:
                c_old = polish_colors[e]
                if c_old != target_color:
                    continue
                    
                # Try each possible new color from 0 to target_color-1
                found = False
                for c_new in range(target_color):
                    # Check if recoloring e to c_new keeps all K4s valid
                    valid = True
                    for k4_idx, pos in edge_k4_info[e]:
                        k4e = k4s[k4_idx]
                        new_set = set()
                        for idx, ee in enumerate(k4e):
                            if idx == pos:
                                new_set.add(c_new)
                            else:
                                new_set.add(polish_colors[ee])
                        if len(new_set) < 5:
                            valid = False
                            break
                    
                    if valid:
                        # Apply the change
                        polish_colors[e] = c_new
                        color_counts_polish[c_old] -= 1
                        color_counts_polish[c_new] += 1
                        found = True
                        break
                
                if not found:
                    # Keep the original color for this edge
                    pass
        
        # Recalculate palette after polish
        new_palette = len(set(polish_colors))
        if new_palette < polish_palette:
            polish_palette = new_palette
            best_palette = new_palette
            best_colors = polish_colors[:]
    
    return best_colors
