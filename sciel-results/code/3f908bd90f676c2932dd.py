import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 16
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute K4s and their edge indices
    k4s = []
    k4_edge_indices = []
    for quad in combinations(range(n), 4):
        k4s.append(quad)
        e_indices = [edge_index[(quad[i], quad[j])] for i in range(4) for j in range(i+1, 4)]
        k4_edge_indices.append(e_indices)
        
    num_k4s = len(k4s)
    
    # Map edge to list of K4 indices containing it
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4_edge_indices):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)
            
    # Precompute for each K4, the position of each of its 6 edges in the k4_edge_indices list
    # This allows O(1) access to the color of a specific edge within a K4
    k4_edge_pos = []
    for e_indices in k4_edge_indices:
        pos_map = [0]*6
        for i, e_idx in enumerate(e_indices):
            pos_map[e_idx] = i
        k4_edge_pos.append(pos_map)
        
    def count_distinct(e_indices, colors):
        # e_indices: list of 6 edge indices
        # Returns bitmask of distinct colors (0..15 assumed, or we can just count)
        # Since we want to minimize colors, we restrict to 0..15 initially, but task allows 0..119.
        # Actually, task says "Minimize the number of colours used".
        # If we find a solution with 13 colors, we use 0..12.
        # Let's use a simple set or bitmask. Bitmask is faster if colors < 32.
        # Let's assume we search in palette 0..15.
        mask = 0
        for e_idx in e_indices:
            mask |= 1 << colors[e_idx]
        return mask
        
    def popcount(x):
        # Python 3.8+ has int.bit_count()
        return x.bit_count()
        
    def is_violating(k4_idx, colors):
        mask = count_distinct(k4_edge_indices[k4_idx], colors)
        return popcount(mask) < 5
        
    def get_violations(colors):
        # Returns list of violating K4 indices
        v = []
        for k4_idx in range(num_k4s):
            if is_violating(k4_idx, colors):
                v.append(k4_idx)
        return v
        
    def delta_cost_for_edge_color(e_idx, new_color, colors, violating_set):
        # Calculate change in number of violations if we change e_idx to new_color
        # violating_set: set of currently violating K4 indices
        # We only care about K4s containing e_idx
        delta = 0
        for k4_idx in edge_to_k4s[e_idx]:
            e_indices = k4_edge_indices[k4_idx]
            pos = k4_edge_pos[k4_idx][e_idx] # Position of e_idx in this K4's edge list
            
            # Get current mask
            # To speed up, we can precompute masks for all K4s? 
            # No, colors change. 
            # But we can compute mask for this K4 quickly.
            
            # Current mask
            mask = 0
            for i, ei in enumerate(e_indices):
                if i == pos:
                    mask |= 1 << new_color
                else:
                    mask |= 1 << colors[ei]
                    
            new_distinct = popcount(mask)
            new_violating = 1 if new_distinct < 5 else 0
            
            # Old state
            if k4_idx in violating_set:
                old_violating = 1
            else:
                old_violating = 0
            
            delta += new_violating - old_violating
        return delta
        
    best_colors = None
    best_num_colors = 16 # Max palette size we try
    
    # Try to find a solution with as few colors as possible
    # Lower bound is 13.
    for num_colors in range(13, 16):
        # Random initialization
        colors = [random.randint(0, num_colors - 1) for _ in range(num_edges)]
        
        # Calculate initial violations
        violating_set = set(get_violations(colors))
        current_violations = len(violating_set)
        
        if current_violations == 0:
            best_colors = colors[:]
            best_num_colors = num_colors
            break
            
        # Min-conflicts style repair
        # We track the set of violating K4s to avoid recalculating everything from scratch
        # But delta_cost needs to know if a K4 was violating before.
        # So we maintain violating_set.
        
        no_improvement_count = 0
        max_no_improvement = 5000 # Restart threshold
        
        while time.time() < deadline:
            if current_violations == 0:
                best_colors = colors[:]
                best_num_colors = num_colors
                break
                
            if no_improvement_count > max_no_improvement:
                break # Restart with new random init
                
            # Select a random violating K4
            if not violating_set:
                break
            target_k4 = random.choice(list(violating_set))
            
            # Select one of its 6 edges
            target_edge = random.choice(k4_edge_indices[target_k4])
            
            # Try all colors in the palette for this edge
            old_color = colors[target_edge]
            best_delta = 0
            best_new_color = old_color
            
            for c in range(num_colors):
                if c == old_color:
                    continue
                d = delta_cost_for_edge_color(target_edge, c, colors, violating_set)
                if d < best_delta:
                    best_delta = d
                    best_new_color = c
                    
            if best_delta < 0:
                # Accept the move
                colors[target_edge] = best_new_color
                current_violations += best_delta
                no_improvement_count = 0
                
                # Update violating_set
                # We need to update the status of all K4s containing target_edge
                for k4_idx in edge_to_k4s[target_edge]:
                    if is_violating(k4_idx, colors):
                        violating_set.add(k4_idx)
                    else:
                        violating_set.discard(k4_idx)
            else:
                # No improving move found for this edge
                no_improvement_count += 1
                
        if best_colors is not None:
            break
            
    if best_colors is not None:
        return best_colors
    else:
        # Fallback: Return a valid coloring if we can find one with more colors, or just a seed
        # The seed (i+j)%17 uses 17 colors. But we want to minimize.
        # If we failed to find 13-15, try 16?
        # Or just return the seed.
        # Actually, the seed is (i+j)%17. Let's check if it's valid.
        # It might not be. 
        # Let's try one last desperate attempt with 16 colors.
        colors = [random.randint(0, 15) for _ in range(num_edges)]
        violating_set = set(get_violations(colors))
        current_violations = len(violating_set)
        
        while time.time() < deadline - 5:
            if current_violations == 0:
                return colors
            if not violating_set:
                break
            target_k4 = random.choice(list(violating_set))
            target_edge = random.choice(k4_edge_indices[target_k4])
            old_color = colors[target_edge]
            best_delta = 0
            best_new_color = old_color
            for c in range(16):
                if c == old_color:
                    continue
                d = delta_cost_for_edge_color(target_edge, c, colors, violating_set)
                if d < best_delta:
                    best_delta = d
                    best_new_color = c
            if best_delta < 0:
                colors[target_edge] = best_new_color
                current_violations += best_delta
                for k4_idx in edge_to_k4s[target_edge]:
                    if is_violating(k4_idx, colors):
                        violating_set.add(k4_idx)
                    else:
                        violating_set.discard(k4_idx)
        return colors

