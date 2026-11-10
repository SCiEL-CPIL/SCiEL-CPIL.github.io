"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    """
    Returns 120 integer colours in 0..119 for the pairs (i,j), 0<=i<j<16, in lexicographic order.
    Every set of four vertices must span edges of at least five distinct colours.
    Minimize the number of colours used.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin under 150s limit
    
    n = 16
    num_edges = n * (n - 1) // 2  # 120 edges
    
    # Precompute edge index mapping
    edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {(i, j): k for k, (i, j) in enumerate(edge_list)}
    
    # Precompute all K4s and their edge indices
    k4_list = list(itertools.combinations(range(n), 4))
    k4_edges = []  # list of 6 edge indices for each K4
    edge_to_k4 = [[] for _ in range(num_edges)]  # for each edge, list of K4 indices
    
    for k4_idx, v in enumerate(k4_list):
        a, b, c, d = v
        edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
        e_indices = []
        for e in edges:
            i, j = e
            if i > j:
                i, j = j, i
            ei = edge_index[(i, j)]
            e_indices.append(ei)
            edge_to_k4[ei].append(k4_idx)
        k4_edges.append(e_indices)
    
    num_k4 = len(k4_list)
    
    def compute_quad_distinct(colours, k4_idx):
        """Compute number of distinct colours in a K4."""
        e_indices = k4_edges[k4_idx]
        return len(set(colours[ei] for ei in e_indices))
    
    def full_violation_count(colours):
        """Count K4s with < 5 distinct colours."""
        v = 0
        for k4_idx in range(num_k4):
            if compute_quad_distinct(colours, k4_idx) < 5:
                v += 1
        return v
    
    # Base solution: (i+j) mod 17
    base_colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    # Verify base is valid
    assert full_violation_count(base_colors) == 0, "Base solution invalid"
    
    best_colors = base_colors.copy()
    best_num = len(set(base_colors))  # 17
    
    # Local search with incremental violation tracking
    def local_search(initial_colors, target_palette, max_time):
        """
        Try to find a valid coloring using only colours in [0, target_palette).
        Returns (colors, palette_size) or None if not found.
        """
        c = list(initial_colors)
        # Project to target palette
        for e in range(num_edges):
            c[e] = c[e] % target_palette
        
        # Initialize quad distinct counts
        quad_dist = [compute_quad_distinct(c, k4_idx) for k4_idx in range(num_k4)]
        viol = sum(1 for x in quad_dist if x < 5)
        
        if viol == 0:
            palette = len(set(c))
            return c, palette
        
        rng = random.Random()
        current_temp = 1.0
        cooling = 0.9995
        min_temp = 0.01
        
        steps = 0
        max_steps = 500000
        
        while viol > 0 and steps < max_steps and time.time() < max_time:
            steps += 1
            
            # Pick a random edge and new colour
            e = rng.randrange(num_edges)
            new_col = rng.randrange(target_palette)
            
            if c[e] == new_col:
                continue
            
            # Evaluate move: check affected K4s
            old_viol = viol
            affected_k4s = edge_to_k4[e]
            
            # Check if move is acceptable
            acceptable = True
            for k4_idx in affected_k4s:
                old_dist = quad_dist[k4_idx]
                # Compute new distinct count
                e_indices = k4_edges[k4_idx]
                new_colors_in_k4 = []
                for ei in e_indices:
                    if ei == e:
                        new_colors_in_k4.append(new_col)
                    else:
                        new_colors_in_k4.append(c[ei])
                new_dist = len(set(new_colors_in_k4))
                
                if old_dist < 5 and new_dist >= 5:
                    # Fixing a violation
                    pass
                elif old_dist >= 5 and new_dist < 5:
                    # Creating a violation
                    # Accept with probability based on temperature
                    if rng.random() > (current_temp / max(1.0, old_viol)):
                        acceptable = False
                        break
                # else: no change in violation status
        
            if not acceptable:
                continue
            
            # Apply move
            old_col = c[e]
            c[e] = new_col
            
            # Update quad_dist and viol
            for k4_idx in affected_k4s:
                old_dist = quad_dist[k4_idx]
                e_indices = k4_edges[k4_idx]
                new_colors_in_k4 = []
                for ei in e_indices:
                    if ei == e:
                        new_colors_in_k4.append(new_col)
                    else:
                        new_colors_in_k4.append(c[ei])
                new_dist = len(set(new_colors_in_k4))
                
                if old_dist < 5 and new_dist >= 5:
                    viol -= 1
                elif old_dist >= 5 and new_dist < 5:
                    viol += 1
                
                quad_dist[k4_idx] = new_dist
            
            # Cool down
            current_temp *= cooling
            if current_temp < min_temp:
                current_temp = min_temp
        
        if viol == 0:
            palette = len(set(c))
            return c, palette
        return None
    
    # Try to find solutions with decreasing palette sizes
    for target in [15, 14, 13]:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random restarts
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random perturbation of base
            perturbed = base_colors.copy()
            rng = random.Random(restart * 1000 + target)
            num_changes = rng.randint(10, 30)
            for _ in range(num_changes):
                e = rng.randrange(num_edges)
                perturbed[e] = rng.randrange(17)
            
            result = local_search(perturbed, target, min(deadline - 10, time.time() + 20))
            if result is not None:
                colors, palette = result
                if palette < best_num:
                    best_colors = colors
                    best_num = palette
                    print(f"Found {palette}-colour solution with target {target}")
                    break  # Found a solution for this target, move to next
    
    # If we haven't improved, try one more approach: direct local search on palette reduction
    if best_num == 17 and time.time() < deadline - 15:
        # Try to reduce from 17 by recoloring edges
        c = base_colors.copy()
        quad_dist = [compute_quad_distinct(c, k4_idx) for k4_idx in range(num_k4)]
        viol = sum(1 for x in quad_dist if x < 5)
        
        # Try to eliminate colour 16 (the highest)
        target_col = 16
        edges_with_target = [e for e in range(num_edges) if c[e] == target_col]
        
        success = True
        for e in edges_with_target:
            found = False
            for new_col in range(16):
                if new_col == c[e]:
                    continue
                
                # Check if valid
                ok = True
                for k4_idx in edge_to_k4[e]:
                    e_indices = k4_edges[k4_idx]
                    new_colors_in_k4 = []
                    for ei in e_indices:
                        if ei == e:
                            new_colors_in_k4.append(new_col)
                        else:
                            new_colors_in_k4.append(c[ei])
                    if len(set(new_colors_in_k4)) < 5:
                        ok = False
                        break
                
                if ok:
                    c[e] = new_col
                    found = True
                    break
            
            if not found:
                success = False
                break
        
        if success and target_col not in c:
            palette = len(set(c))
            if palette < best_num:
                best_colors = c
                best_num = palette
    
    # Final verification
    final_viol = full_violation_count(best_colors)
    if final_viol > 0:
        # Fall back to base
        best_colors = base_colors
        best_num = len(set(base_colors))
    
    # Renormalize colours to 0..best_num-1
    col_map = {}
    new_colors = []
    next_col = 0
    for c in best_colors:
        if c not in col_map:
            col_map[c] = next_col
            next_col += 1
        new_colors.append(col_map[c])
    
    return new_colors

# EVOLVE-BLOCK-END
