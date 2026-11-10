"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def construct():
    n = 16
    deadline = time.time() + 130.0  # safety margin before 150s kill
    
    # Build edge index map: (i,j) -> index in colors list
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_idx[(i, j)] = idx
            idx_to_edge.append((i, j))
            idx += 1
    num_edges = idx  # 120
    
    # Precompute all K4s and their 6 edge indices
    all_k4s = list(itertools.combinations(range(n), 4))
    k4_edge_indices = []
    for k4 in all_k4s:
        eidxs = [edge_to_idx[(a, b)] for a, b in itertools.combinations(k4, 2)]
        k4_edge_indices.append(eidxs)
    
    # Precompute for each edge, the list of K4s containing it
    edge_k4s = [[] for _ in range(num_edges)]
    for ki, k4 in enumerate(all_k4s):
        for eidx in k4_edge_indices[ki]:
            edge_k4s[eidx].append(ki)
    
    # Precompute for each K4, the other 5 edges for each of its 6 edges
    # k4_other_edges[ki][e] = list of 5 edge indices in K4 ki excluding edge e
    k4_other_edges = []
    for ki, eidxs in enumerate(k4_edge_indices):
        others = []
        for e in eidxs:
            others.append([x for x in eidxs if x != e])
        k4_other_edges.append(others)
    
    # Initial coloring: (i+j) mod 17
    colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    best_colors = colors[:]
    best_num_colors = len(set(colors))
    
    def is_valid(colors):
        for ki in range(len(all_k4s)):
            eidxs = k4_edge_indices[ki]
            cset = set(colors[e] for e in eidxs)
            if len(cset) < 5:
                return False
        return True
    
    # Check initial validity
    if not is_valid(colors):
        return colors
    
    used_colors = set(colors)
    color_counts = {}
    for c in colors:
        color_counts[c] = color_counts.get(c, 0) + 1
    
    random.seed(42)
    
    def try_eliminate_color(target_color, colors, used_colors, color_counts):
        """Try to recolor all edges of target_color to other colors. Returns True if eliminated."""
        edges_with_color = [idx for idx, c in enumerate(colors) if c == target_color]
        if not edges_with_color:
            used_colors.discard(target_color)
            return True
        
        # Sort edges by number of affected K4s (ascending) to try easier ones first
        edges_with_color.sort(key=lambda e: len(edge_k4s[e]))
        
        for idx in edges_with_color:
            affected_k4s = edge_k4s[idx]
            
            # Find valid new colors: for each candidate color, check all affected K4s
            valid_new_colors = []
            for new_color in used_colors:
                if new_color == target_color:
                    continue
                
                valid = True
                for ki in affected_k4s:
                    # Get the 5 other edges in this K4
                    other_eidxs = k4_other_edges[ki][k4_edge_indices[ki].index(idx)]
                    # Count distinct colors among other 5 edges
                    other_colors = set(colors[e] for e in other_eidxs)
                    # If new_color is already in other_colors, the K4 still has at least
                    # len(other_colors) colors; we need >= 5 total
                    if new_color in other_colors:
                        if len(other_colors) < 5:
                            valid = False
                            break
                    else:
                        if len(other_colors) < 4:
                            valid = False
                            break
                
                if valid:
                    valid_new_colors.append(new_color)
            
            if valid_new_colors:
                # Prefer colors that are most common (to consolidate)
                valid_new_colors.sort(key=lambda c: -color_counts.get(c, 0))
                new_color = valid_new_colors[0]
                colors[idx] = new_color
                color_counts[target_color] = color_counts.get(target_color, 0) - 1
                color_counts[new_color] = color_counts.get(new_color, 0) + 1
            else:
                return False
        
        # All edges recolored successfully
        used_colors.discard(target_color)
        if color_counts.get(target_color, 0) == 0:
            del color_counts[target_color]
        return True
    
    # Local search: try to reduce number of distinct colors
    max_iterations = 100000
    iteration = 0
    stuck_count = 0
    
    while iteration < max_iterations and time.time() < deadline:
        iteration += 1
        
        current_num = len(used_colors)
        if current_num < best_num_colors:
            best_colors = colors[:]
            best_num_colors = current_num
        
        if best_num_colors <= 13:
            break
        
        # Pick a target color to eliminate: prefer least-used colors (easier to eliminate)
        target_color = min(used_colors, key=lambda c: color_counts.get(c, 0))
        
        success = try_eliminate_color(target_color, colors, used_colors, color_counts)
        
        if success:
            stuck_count = 0
        else:
            stuck_count += 1
            # If stuck many times, try a random color instead
            if stuck_count > 50:
                stuck_count = 0
                target_color = random.choice(list(used_colors))
                success = try_eliminate_color(target_color, colors, used_colors, color_counts)
                if success:
                    stuck_count = 0
    
    # Final check: ensure validity
    if is_valid(colors):
        final_num = len(set(colors))
        if final_num < best_num_colors:
            best_colors = colors[:]
            best_num_colors = final_num
    
    return best_colors
# EVOLVE-BLOCK-END
