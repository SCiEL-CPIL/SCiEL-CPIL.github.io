import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 16
    edges = []
    edge_index = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = len(edges)
            edges.append((i, j))
            
    num_edges = len(edges)
    
    # Pre-compute K4s and their edges
    k4s = []
    k4_edges = []
    for comb in itertools.combinations(range(n), 4):
        v0, v1, v2, v3 = comb
        e_indices = [
            edge_index[(v0, v1)],
            edge_index[(v0, v2)],
            edge_index[(v0, v3)],
            edge_index[(v1, v2)],
            edge_index[(v1, v3)],
            edge_index[(v2, v3)]
        ]
        k4s.append(comb)
        k4_edges.append(e_indices)
        
    num_k4s = len(k4s)
    
    # For each edge, list of K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges):
        for e_idx in e_list:
            edge_to_k4s[e_idx].append(k4_idx)
            
    def is_valid_coloring(colors):
        for k4_idx in range(num_k4s):
            e_list = k4_edges[k4_idx]
            if len({colors[e] for e in e_list}) < 5:
                return False
        return True
        
    def count_colors(colors):
        return len(set(colors))
        
    def compact_colors(colors):
        unique_c = sorted(set(colors))
        if len(unique_c) <= 1:
            return colors
        mapping = {old: new for new, old in enumerate(unique_c)}
        new_colors = [mapping[c] for c in colors]
        return new_colors

    # 1. Try 1-factorization (15 colors)
    # Standard 1-factorization of K_16
    # Vertices 0..15.
    # Let's use the cyclic method.
    # Fix vertex 0.
    # Remaining vertices 1..15.
    # For k in 0..14:
    #   Matching M_k contains edge (0, k+1)
    #   And edges (k+1+i, k+1-i) for i=1..7, mod 15, mapped to 1..15
    # Wait, the standard cyclic construction for K_{2m} fixes infinity.
    # Here m=8. Vertices 0..14 for the cycle, 15 for infinity?
    # Let's use vertices 0..14 for the cycle, 15 as the fixed point.
    
    colors_15 = [0] * num_edges
    fixed_vertex = 15
    cycle_vertices = list(range(15))
    
    for k in range(15):
        color_id = k
        # Edge to fixed vertex
        v_cycle = cycle_vertices[k]
        e_idx = edge_index[(fixed_vertex, v_cycle)] if fixed_vertex < v_cycle else edge_index[(v_cycle, fixed_vertex)]
        colors_15[e_idx] = color_id
        
        # Other edges in the matching
        for i in range(1, 8):
            v1_idx = (k + i) % 15
            v2_idx = (k - i) % 15
            v1 = cycle_vertices[v1_idx]
            v2 = cycle_vertices[v2_idx]
            e_idx = edge_index[(v1, v2)] if v1 < v2 else edge_index[(v2, v1)]
            colors_15[e_idx] = color_id
            
    # Verify 15-coloring
    if is_valid_coloring(colors_15):
        # We have a valid 15-coloring.
        # Try to reduce to 14 or 13 using local search.
        best_colors = colors_15
        best_count = 15
        
        # Local search to reduce colors
        # Strategy: Try to merge two colors by recoloring edges of one color to the other,
        # or more generally, try to change an edge's color to an existing lower color.
        
        current_colors = colors_15[:]
        current_count = 15
        
        # Pre-compute for speed: for each K4, track the set of colors?
        # Given the small size, set intersection is fast enough.
        
        def can_change_edge(e_idx, new_color, colors):
            old_color = colors[e_idx]
            if old_color == new_color:
                return True
            
            for k4_idx in edge_to_k4s[e_idx]:
                e_list = k4_edges[k4_idx]
                other_colors = set()
                for e in e_list:
                    if e != e_idx:
                        other_colors.add(colors[e])
                
                # If we change e_idx to new_color:
                # The new set of colors in this K4 will be:
                # (other_colors U {new_color}) - {old_color if old_color not in other_colors}
                # Actually, simpler:
                # Let S = other_colors.
                # If new_color is in S:
                #   The new set is S (if old_color is not in S) or S - {old_color} (if old_color is in S and no other edge has it? No, other_colors is just the set of colors on OTHER edges).
                #   Wait, other_colors is the set of colors on the 5 other edges.
                #   If new_color is in other_colors, the new set of colors is other_colors (plus new_color which is already there, minus old_color if it was unique to e_idx).
                #   This logic is tricky. Let's just compute the new set.
                
                new_set = set(other_colors)
                new_set.add(new_color)
                if old_color not in other_colors:
                    # old_color was only on e_idx, so it disappears
                    # But we don't need to remove it from new_set if it's not there.
                    # We just need to check if len(new_set) >= 5.
                    pass
                else:
                    # old_color is still present on other edges, so it remains in the set.
                    # But wait, new_set already contains other_colors, which includes old_color.
                    # So new_set is correct.
                    pass
                
                if len(new_set) < 5:
                    return False
            return True

        # Local search loop
        no_improve = 0
        while time.time() < deadline - 5.0:
            if no_improve > 500:
                break
                
            improved = False
            # Try to reduce color count by changing an edge to a smaller color
            # Iterate edges in random order
            for e_idx in random.sample(range(num_edges), num_edges):
                current_c = current_colors[e_idx]
                # Try colors 0 to current_c - 1
                for c in range(current_c):
                    if can_change_edge(e_idx, c, current_colors):
                        current_colors[e_idx] = c
                        improved = True
                        break
                if time.time() > deadline - 5.0:
                    break
                    
            new_count = count_colors(current_colors)
            if new_count < best_count:
                best_count = new_count
                best_colors = current_colors[:]
                no_improve = 0
            elif improved:
                no_improve += 1
            else:
                no_improve += 1
                
        if best_count < 15:
            return compact_colors(best_colors)
        else:
            return compact_colors(best_colors)
            
    # If 15-coloring is invalid (shouldn't happen, but just in case)
    # Fall back to random search with 17 colors
    best_colors = None
    best_count = 18
    
    for restart in range(10):
        if time.time() > deadline - 10.0:
            break
            
        colors = [0] * num_edges
        for i, (u, v) in enumerate(edges):
            colors[i] = (u + v) % 17
            
        if not is_valid_coloring(colors):
            continue
            
        current_colors = colors[:]
        current_count = count_colors(current_colors)
        
        if current_count < best_count:
            best_count = current_count
            best_colors = current_colors[:]
            
        # Simple local search
        no_improve = 0
        while time.time() < deadline - 10.0:
            if no_improve > 200:
                break
            improved = False
            for _ in range(100):
                e_idx = random.randint(0, num_edges - 1)
                current_c = current_colors[e_idx]
                for c in range(current_c):
                    if can_change_edge(e_idx, c, current_colors):
                        current_colors[e_idx] = c
                        improved = True
                        break
                        
            new_count = count_colors(current_colors)
            if new_count < best_count:
                best_count = new_count
                best_colors = current_colors[:]
                no_improve = 0
            elif improved:
                no_improve += 1
            else:
                no_improve += 1
                
    if best_colors is None:
        # Fallback
        colors = [0] * num_edges
        for i, (u, v) in enumerate(edges):
            colors[i] = (u + v) % 17
        return compact_colors(colors)
        
    return compact_colors(best_colors)
