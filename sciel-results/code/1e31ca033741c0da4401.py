import random
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    N = 16
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    E = len(edges)
    edge_idx = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    # Precompute all K4s and their edge lists
    k4s = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        k4_edges = [edge_idx[(a,b)], edge_idx[(a,c)], edge_idx[(a,d)],
                    edge_idx[(b,c)], edge_idx[(b,d)], edge_idx[(c,d)]]
        k4s.append(k4_edges)
    
    K = len(k4s)
    
    # For each edge, which K4s contain it
    edge_to_k4s = [[] for _ in range(E)]
    for k4_idx, k4_edges in enumerate(k4s):
        for e in k4_edges:
            edge_to_k4s[e].append(k4_idx)
    
    # Initialize with a structured coloring using modular arithmetic
    # Try (i * 7 + j) % 13 for initial diversity with 13 colors
    colors = [(i * 7 + j) % 13 for (i, j) in edges]
    
    # Track color usage counts
    color_counts = {}
    for c in colors:
        color_counts[c] = color_counts.get(c, 0) + 1
    
    # Track distinct color count per K4
    k4_distinct = [0] * K
    total_badness = 0
    for k4_idx, k4e in enumerate(k4s):
        distinct = len(set(colors[e] for e in k4e))
        k4_distinct[k4_idx] = distinct
        if distinct < 5:
            total_badness += (5 - distinct)
    
    palette_size = len(color_counts)
    
    best_colors = colors[:]
    best_badness = total_badness
    best_palette = palette_size
    
    rng = random.Random(42)
    
    def score(badness, palette):
        # Weight badness heavily to ensure validity first
        return badness * 10000 + palette
    
    def evaluate_state(colors_arr):
        """Evaluate the state and return badness and palette size"""
        cc = {}
        for c in colors_arr:
            cc[c] = cc.get(c, 0) + 1
        
        badness = 0
        for k4e in k4s:
            distinct = len(set(colors_arr[e] for e in k4e))
            if distinct < 5:
                badness += (5 - distinct)
        
        return badness, len(cc)
    
    # Simulated Annealing approach
    current_temp = 10.0
    cooling_rate = 0.9995
    min_temp = 0.01
    
    current_badness, current_palette = evaluate_state(colors)
    current_score = score(current_badness, current_palette)
    
    best_score = current_score
    best_colors = colors[:]
    best_badness = current_badness
    best_palette = current_palette
    
    iteration = 0
    while time.time() < deadline - 5:
        iteration += 1
        
        # Pick a random edge
        e = rng.randrange(E)
        old_c = colors[e]
        
        # Determine candidate colors
        if current_badness > 0:
            # When invalid, try to fix by using existing colors or new ones
            candidates = list(color_counts.keys())
            # Add some new colors to explore
            for _ in range(5):
                candidates.append(rng.randint(0, 20))
        else:
            # When valid, try to reduce palette by using existing colors
            candidates = [c for c in color_counts if c != old_c]
            if not candidates:
                candidates = [old_c]
        
        if not candidates:
            candidates = [0]
        
        nc = rng.choice(candidates)
        if nc == old_c:
            continue
        
        # Apply the move
        colors[e] = nc
        
        # Update color counts
        color_counts[old_c] -= 1
        if color_counts[old_c] == 0:
            del color_counts[old_c]
        color_counts[nc] = color_counts.get(nc, 0) + 1
        
        # Compute new state
        new_badness, new_palette = evaluate_state(colors)
        new_score = score(new_badness, new_palette)
        
        # Accept or reject based on simulated annealing
        delta = new_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / max(current_temp, 1e-10)):
            # Accept the move
            current_badness = new_badness
            current_palette = new_palette
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_colors = colors[:]
                best_badness = current_badness
                best_palette = current_palette
        else:
            # Reject the move and revert
            colors[e] = old_c
            color_counts[nc] -= 1
            if color_counts[nc] == 0:
                del color_counts[nc]
            color_counts[old_c] = color_counts.get(old_c, 0) + 1
        
        # Cool down
        current_temp *= cooling_rate
    
    # Final local search polish to try to reduce palette further
    if time.time() < deadline - 2:
        # Try to eliminate each color one by one
        for target_color in sorted(set(best_colors), key=lambda c: -best_colors.count(c)):
            if time.time() > deadline - 2:
                break
            
            # Try to recolor all edges with target_color
            success = True
            for e in range(E):
                if best_colors[e] != target_color:
                    continue
                
                # Try to find an alternative color
                found = False
                for alt_c in range(0, 20):
                    if alt_c == target_color:
                        continue
                    
                    # Check if changing this edge to alt_c maintains validity
                    # and doesn't increase palette
                    temp_colors = best_colors[:]
                    temp_colors[e] = alt_c
                    
                    # Check if target_color is still used elsewhere
                    if any(c == target_color for c in temp_colors):
                        # Can't eliminate target_color yet
                        continue
                    
                    # Check validity
                    valid = True
                    for k4e in k4s:
                        distinct = len(set(temp_colors[ee] for ee in k4e))
                        if distinct < 5:
                            valid = False
                            break
                    
                    if valid:
                        best_colors[e] = alt_c
                        found = True
                        break
                
                if not found:
                    success = False
                    break
            
            if not success:
                continue
    
    # Verify final result
    final_badness, final_palette = evaluate_state(best_colors)
    
    if final_badness > 0:
        # If invalid, try a simpler fallback
        # Use a known valid construction
        best_colors = [(i + j) % 17 for (i, j) in edges]
    
    return best_colors
