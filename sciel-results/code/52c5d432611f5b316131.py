import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 150s limit
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Generate all edges (i, j) with i < j
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # Generate all K4s (sets of 4 vertices)
    k4s = []
    k4_edge_indices = []
    for quad in combinations(range(n), 4):
        k4s.append(quad)
        e_indices = []
        for i in range(4):
            for j in range(i + 1, 4):
                e_indices.append(edge_index[(quad[i], quad[j])])
        k4_edge_indices.append(e_indices)
        
    num_k4s = len(k4s)
    
    # Map edge to list of K4 indices containing it
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4_edge_indices):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)
    
    # Precompute for each K4, the position of each of its 6 edges
    # k4_edge_pos[k4_idx] = list of (edge_idx, position_in_k4)
    k4_edge_pos = []
    for e_indices in k4_edge_indices:
        k4_edge_pos.append([(ei, pos) for pos, ei in enumerate(e_indices)])
    
    # Fast distinct count using bitmask (colors 0..15 fit in 16 bits)
    def distinct_count_bitmask(colors, e_indices):
        mask = 0
        for ei in e_indices:
            mask |= (1 << colors[ei])
        return mask.bit_count()
    
    def get_cost(colors):
        cost = 0
        for e_indices in k4_edge_indices:
            if distinct_count_bitmask(colors, e_indices) < 5:
                cost += 1
        return cost
    
    def delta_cost(e_idx, new_color, colors, num_colors):
        """Compute change in cost when changing edge e_idx to new_color."""
        old_color = colors[e_idx]
        if old_color == new_color:
            return 0
        
        delta = 0
        for k4_idx in edge_to_k4s[e_idx]:
            e_indices = k4_edge_indices[k4_idx]
            
            # Compute old distinct count
            old_mask = 0
            for ei in e_indices:
                old_mask |= (1 << colors[ei])
            old_distinct = old_mask.bit_count()
            
            # Compute new distinct count
            # Remove old color contribution, add new color
            new_mask = 0
            for ei in e_indices:
                if ei == e_idx:
                    new_mask |= (1 << new_color)
                else:
                    new_mask |= (1 << colors[ei])
            new_distinct = new_mask.bit_count()
            
            old_bad = 1 if old_distinct < 5 else 0
            new_bad = 1 if new_distinct < 5 else 0
            delta += new_bad - old_bad
            
        return delta
    
    def minimize_colors(colors, num_colors, max_time):
        """Try to reduce the number of distinct colors used."""
        # Find which colors are actually used
        used_colors = set(colors)
        if len(used_colors) <= num_colors:
            return colors, len(used_colors)
        
        # Try to remap colors to use fewer
        # This is a constraint satisfaction problem
        # For now, just relabel to use contiguous colors 0..k-1
        color_map = {}
        new_colors = []
        next_color = 0
        for c in colors:
            if c not in color_map:
                color_map[c] = next_color
                next_color += 1
            new_colors.append(color_map[c])
        return new_colors, next_color
    
    def local_search(colors, num_colors, max_time, target_cost=0):
        """Run local search to minimize cost."""
        current_cost = get_cost(colors)
        
        if current_cost <= target_cost:
            return colors, current_cost
        
        # Track best found
        best_colors = colors[:]
        best_cost = current_cost
        
        # Use simulated annealing
        T = 2.0
        cool_rate = 0.999
        min_T = 0.1
        
        no_improve_steps = 0
        max_no_improve = 2000
        
        while time.time() < max_time and T > min_T:
            if current_cost <= target_cost:
                best_colors = colors[:]
                best_cost = current_cost
                break
            
            # Pick a random edge
            e_idx = random.randint(0, num_edges - 1)
            new_color = random.randint(0, num_colors - 1)
            
            if new_color == colors[e_idx]:
                continue
            
            dc = delta_cost(e_idx, new_color, colors, num_colors)
            
            # Accept with probability based on SA
            if dc <= 0 or random.random() < np.exp(-dc / T):
                colors[e_idx] = new_color
                current_cost += dc
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                
                if no_improve_steps > max_no_improve:
                    # Restart with current best
                    colors = best_colors[:]
                    current_cost = best_cost
                    no_improve_steps = 0
                    T = max(T * 0.5, min_T)
            else:
                no_improve_steps += 1
                if no_improve_steps > max_no_improve:
                    colors = best_colors[:]
                    current_cost = best_cost
                    no_improve_steps = 0
                    T = max(T * 0.5, min_T)
            
            T *= cool_rate
        
        return best_colors, best_cost
    
    # Main algorithm: try to find valid coloring with minimum colors
    best_solution = None
    best_num_colors = float('inf')
    
    # Try from 13 colors upward
    for num_colors in range(13, 16):
        if time.time() > deadline - 10:
            break
        
        # Multiple restarts
        max_restarts = 3 if num_colors == 13 else 2
        found_valid = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            colors = [random.randint(0, num_colors - 1) for _ in range(num_edges)]
            
            # Run local search
            remaining_time = deadline - time.time()
            if remaining_time < 5:
                break
            
            colors, cost = local_search(colors, num_colors, time.time() + min(remaining_time - 2, 30), target_cost=0)
            
            if cost == 0:
                # Valid solution found
                colors, actual_colors = minimize_colors(colors, num_colors, time.time())
                if actual_colors < best_num_colors:
                    best_num_colors = actual_colors
                    best_solution = colors
                found_valid = True
                break
        
        if found_valid:
            # Found valid solution with num_colors, can stop trying more colors
            break
    
    # If no valid solution found, return a fallback
    if best_solution is not None:
        return best_solution
    else:
        # Fallback: use a simple pattern that might work
        # Return seed-like solution
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
