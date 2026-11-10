"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def construct():
    n = 16
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    # Precompute edge indices in lexicographic order
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)  # 120
    edge_to_idx = {e: idx for idx, e in enumerate(edges)}
    idx_to_edge = edges  # direct lookup
    
    # Precompute all K4s and their 6 edges
    all_k4s = list(itertools.combinations(range(n), 4))
    k4_edges = []
    for k4 in all_k4s:
        e_list = [edge_to_idx[(a, b)] for a, b in itertools.combinations(k4, 2)]
        k4_edges.append(e_list)
    
    # For each edge, which K4s contain it
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges):
        for e in e_list:
            edge_k4s[e].append(k4_idx)
    
    # Initial coloring: (i+j) mod 17
    colors = [(i + j) % 17 for i, j in edges]
    
    def k4_color_count(k4_idx, colors):
        e_list = k4_edges[k4_idx]
        return len({colors[e] for e in e_list})
    
    def is_valid(colors):
        for k4_idx in range(len(all_k4s)):
            if k4_color_count(k4_idx, colors) < 5:
                return False
        return True
    
    # Check initial validity
    if not is_valid(colors):
        return colors
    
    best_colors = colors[:]
    best_num_colors = len(set(colors))
    
    # Helper: check if recoloring edge `eidx` to `new_color` keeps all its K4s valid
    # Returns True if valid
    def can_recolor(eidx, new_color, colors, target_color):
        for k4_idx in edge_k4s[eidx]:
            e_list = k4_edges[k4_idx]
            seen = set()
            for e in e_list:
                if e == eidx:
                    seen.add(new_color)
                else:
                    seen.add(colors[e])
            if len(seen) < 5:
                return False
        return True
    
    random.seed(42)
    
    # Strategy: iteratively try to eliminate colors
    # Use a smarter approach: for each color, try to recolor all its edges
    # to other colors, prioritizing edges that are "flexible"
    
    max_rounds = 200
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        
        current_used = set(colors)
        if len(current_used) <= 13:
            break
        
        if len(current_used) < best_num_colors:
            best_colors = colors[:]
            best_num_colors = len(current_used)
        
        # Pick target color: prefer one with fewer edges (easier to eliminate)
        color_counts = {}
        for c in colors:
            color_counts[c] = color_counts.get(c, 0) + 1
        
        # Sort colors by count ascending - try to eliminate rare colors first
        target_color = min(current_used, key=lambda c: color_counts[c])
        
        edges_with_target = [idx for idx, c in enumerate(colors) if c == target_color]
        
        # Try to recolor each edge of target_color to another color
        success = True
        for eidx in edges_with_target:
            if time.time() > deadline:
                success = False
                break
            
            # Find valid new colors
            valid_colors = []
            for new_c in current_used:
                if new_c == target_color:
                    continue
                if can_recolor(eidx, new_c, colors, target_color):
                    valid_colors.append(new_c)
            
            if valid_colors:
                # Prefer colors that are more common (to consolidate)
                valid_colors.sort(key=lambda c: -color_counts.get(c, 0))
                colors[eidx] = valid_colors[0]
            else:
                success = False
                break
        
        if success:
            # Removed target_color entirely
            current_used.discard(target_color)
        else:
            # Try a different strategy: random recoloring attempts
            # Try several random target colors
            for _ in range(5):
                if time.time() > deadline:
                    break
                target = random.choice(list(current_used))
                edges_t = [idx for idx, c in enumerate(colors) if c == target]
                if not edges_t:
                    current_used.discard(target)
                    continue
                
                # Try to recolor all edges of this color
                all_ok = True
                for eidx in edges_t:
                    valid_colors = []
                    for new_c in current_used:
                        if new_c == target:
                            continue
                        if can_recolor(eidx, new_c, colors, target):
                            valid_colors.append(new_c)
                    if valid_colors:
                        colors[eidx] = random.choice(valid_colors)
                    else:
                        all_ok = False
                        break
                
                if all_ok:
                    current_used.discard(target)
                    break
    
    # Final check and return best
    if is_valid(colors):
        nc = len(set(colors))
        if nc < best_num_colors:
            best_colors = colors[:]
            best_num_colors = nc
    
    return best_colors
# EVOLVE-BLOCK-END
