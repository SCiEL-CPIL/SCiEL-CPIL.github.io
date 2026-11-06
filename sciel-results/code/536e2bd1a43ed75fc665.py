"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools

def construct():
    deadline = time.time() + 115
    N = 16
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    E = len(edges)
    edge_idx = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    # Pre-compute K4 edge sets
    k4s = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        k4_edges = [edge_idx[(a,b)], edge_idx[(a,c)], edge_idx[(a,d)],
                    edge_idx[(b,c)], edge_idx[(b,d)], edge_idx[(c,d)]]
        k4s.append(k4_edges)
        
    # Map each edge to the K4s it belongs to
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
    # Try a few inits
    inits = [
        [(i + j) % 17 for (i, j) in edges],
        [(i * 3 + j) % 15 for (i, j) in edges],
        [(i * 7 + j) % 13 for (i, j) in edges],
    ]
    
    rng = random.Random(42)
    for _ in range(5):
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
    cooling_rate = 0.999
    
    rng_search = random.Random(123)
    
    while time.time() < deadline - 2:
        if T < T_min:
            T = T_min
            
        # Pick random edge
        e = rng_search.randint(0, E - 1)
        c_old = current_colors[e]
        
        # Pick random new color from 0 to 15 (encourages small palette)
        # Sometimes pick from 0 to 119 to allow escape?
        # Let's stick to 0..15 for now, as we want to minimize palette.
        c_new = rng_search.randint(0, 15)
        
        if c_new == c_old:
            continue
            
        # Calculate delta violations
        delta_viol = 0
        for k4_idx in edge_to_k4s[e]:
            k4e = k4s[k4_idx]
            # Get colors of other 5 edges
            # We can optimize this by precomputing, but 6 elements is small.
            # Just recompute distinct for old and new.
            # To avoid creating sets repeatedly, we can use a small list and sort?
            # Or just set comprehension.
            
            old_colors = [current_colors[ee] for ee in k4e]
            new_colors = old_colors[:]
            # Find index of e in k4e? 
            # It's easier to just build the set.
            # Optimization: 
            # old_set = set(old_colors)
            # old_dist = len(old_set)
            # new_set = set(new_colors) # where new_colors has c_new at e's position
            # But we don't know e's position in k4e easily without search.
            # Let's just iterate.
            
            old_dist = len(set(old_colors))
            
            # Construct new list
            # We know e is in k4e.
            # Let's just replace and set.
            new_colors_list = []
            for ee in k4e:
                if ee == e:
                    new_colors_list.append(c_new)
                else:
                    new_colors_list.append(current_colors[ee])
            new_dist = len(set(new_colors_list))
            
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
            # If delta_cost is large, prob is 0.
            # To avoid underflow/overflow, just check if rng < exp(...)
            # Since T is small, delta_cost/T can be large.
            if delta_cost > 50 * T: # heuristic cutoff
                accept = False
            else:
                # Use log to avoid exp overflow? 
                # log(p) = -delta_cost / T
                # log(rng) < -delta_cost / T  => rng < exp(-delta_cost/T)
                # We can just compute exp if delta_cost/T is not too big.
                ratio = delta_cost / T
                if ratio > 700: # exp(-700) is ~0
                    accept = False
                else:
                    accept = rng_search.random() < (2.718281828 ** (-ratio))
                    
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
        
    return best_colors
# EVOLVE-BLOCK-END
