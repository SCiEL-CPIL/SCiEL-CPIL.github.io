import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
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
            
    # Precompute edge positions within each K4 for faster mask updates
    k4_edge_pos = [{} for _ in range(num_k4s)]
    for k4_idx, e_indices in enumerate(k4_edge_indices):
        for pos, ei in enumerate(e_indices):
            k4_edge_pos[k4_idx][ei] = pos

    # Helper: Get distinct count from mask
    def mask_count(mask):
        return mask.bit_count()

    # Helper: Calculate cost (number of violated K4s)
    def get_cost(colors):
        cost = 0
        for k4_idx in range(num_k4s):
            e_indices = k4_edge_indices[k4_idx]
            mask = 0
            for ei in e_indices:
                mask |= (1 << colors[ei])
            if mask_count(mask) < 5:
                cost += 1
        return cost

    # Helper: Calculate delta cost for changing edge e_idx to new_color
    def delta_cost(e_idx, new_color, colors, k4_masks):
        old_color = colors[e_idx]
        if old_color == new_color:
            return 0, {}
        
        delta = 0
        updates = {}
        
        for k4_idx in edge_to_k4s[e_idx]:
            old_mask = k4_masks[k4_idx]
            old_distinct = mask_count(old_mask)
            old_bad = 1 if old_distinct < 5 else 0
            
            e_indices = k4_edge_indices[k4_idx]
            new_mask = 0
            for ei in e_indices:
                if ei == e_idx:
                    new_mask |= (1 << new_color)
                else:
                    new_mask |= (1 << colors[ei])
            
            new_distinct = mask_count(new_mask)
            new_bad = 1 if new_distinct < 5 else 0
            
            delta += new_bad - old_bad
            updates[k4_idx] = new_mask
            
        return delta, updates

    # Helper: Get list of violating K4 indices
    def get_violating_k4s(k4_masks):
        violating = []
        for k4_idx in range(num_k4s):
            if mask_count(k4_masks[k4_idx]) < 5:
                violating.append(k4_idx)
        return violating

    # Helper: Targeted local search on violating K4s
    def targeted_local_search(colors, k4_masks, num_colors, max_iters=2000):
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            violating = get_violating_k4s(k4_masks)
            if not violating:
                break
            
            # Pick a random violating K4
            k4_idx = random.choice(violating)
            e_indices = k4_edge_indices[k4_idx]
            
            # Try to fix by changing one of the 6 edges
            random.shuffle(e_indices)
            fixed = False
            for e_idx in e_indices:
                old_color = colors[e_idx]
                # Try all colors
                best_c = -1
                best_delta = float('inf')
                for c in range(num_colors):
                    if c == old_color:
                        continue
                    d, _ = delta_cost(e_idx, c, colors, k4_masks)
                    if d < best_delta:
                        best_delta = d
                        best_c = c
                
                if best_c >= 0 and best_delta < 0:
                    _, updates = delta_cost(e_idx, best_c, colors, k4_masks)
                    colors[e_idx] = best_c
                    for k4_i, new_mask in updates.items():
                        k4_masks[k4_i] = new_mask
                    fixed = True
                    break
            
            if not fixed:
                # Try a random edge change to escape local minimum
                e_idx = random.randint(0, num_edges - 1)
                new_color = random.randint(0, num_colors - 1)
                if new_color != colors[e_idx]:
                    d, updates = delta_cost(e_idx, new_color, colors, k4_masks)
                    if d <= 0 or random.random() < 0.3:
                        colors[e_idx] = new_color
                        for k4_i, new_mask in updates.items():
                            k4_masks[k4_i] = new_mask
        
        return colors, k4_masks

    # Helper: Minimize colors by removing unused colors
    def minimize_colors(colors, k4_masks, num_colors):
        # Try to remove each color if possible
        for target_color in range(num_colors - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            
            # Check if this color is actually used
            used = False
            for c in colors:
                if c == target_color:
                    used = True
                    break
            
            if not used:
                continue
            
            # Try to remap all edges with target_color to other colors
            edges_with_target = [i for i, c in enumerate(colors) if c == target_color]
            if not edges_with_target:
                continue
            
            # Try to remap one edge at a time
            success = True
            for e_idx in edges_with_target:
                # Try to change this edge to a different color
                best_c = -1
                best_delta = float('inf')
                for c in range(num_colors):
                    if c == target_color:
                        continue
                    d, _ = delta_cost(e_idx, c, colors, k4_masks)
                    if d < best_delta:
                        best_delta = d
                        best_c = c
                
                if best_c >= 0 and best_delta <= 0:
                    _, updates = delta_cost(e_idx, best_c, colors, k4_masks)
                    colors[e_idx] = best_c
                    for k4_i, new_mask in updates.items():
                        k4_masks[k4_i] = new_mask
                else:
                    success = False
                    break
            
            if not success:
                # Revert: we need to restore, but for simplicity, just stop
                # Actually, we should track and revert, but this is complex
                # Let's just break and try the next color
                break
        
        return colors, k4_masks

    # Helper: Relabel colors to 0..k-1
    def relabel_colors(colors):
        used = sorted(set(colors))
        mapping = {c: i for i, c in enumerate(used)}
        return [mapping[c] for c in colors], len(used)

    # Phase 1: Greedy Initialization
    def greedy_init(num_colors):
        colors = [0] * num_edges
        k4_masks = [1] * num_k4s  # All edges color 0, so mask = 1
        
        edge_order = list(range(num_edges))
        random.shuffle(edge_order)
        
        for e_idx in edge_order:
            best_c = 0
            best_delta = float('inf')
            
            for c in range(num_colors):
                if c == colors[e_idx]:
                    continue
                d, _ = delta_cost(e_idx, c, colors, k4_masks)
                if d < best_delta:
                    best_delta = d
                    best_c = c
            
            if best_c != colors[e_idx]:
                _, updates = delta_cost(e_idx, best_c, colors, k4_masks)
                colors[e_idx] = best_c
                for k4_idx, new_mask in updates.items():
                    k4_masks[k4_idx] = new_mask
        
        return colors, k4_masks

    # Phase 2: Simulated Annealing with targeted local search
    def simulated_annealing(initial_colors, initial_masks, num_colors, max_time):
        colors = initial_colors[:]
        k4_masks = initial_masks[:]
        
        current_cost = get_cost(colors)
        
        if current_cost == 0:
            return colors, current_cost
        
        T = 3.0
        cool_rate = 0.9995
        min_T = 0.05
        no_improve = 0
        max_no_improve = 3000
        
        best_colors = colors[:]
        best_cost = current_cost
        
        while time.time() < max_time:
            if current_cost == 0:
                best_colors = colors[:]
                best_cost = 0
                break
            
            # With some probability, do targeted local search
            if random.random() < 0.1:
                colors, k4_masks = targeted_local_search(colors, k4_masks, num_colors, max_iters=100)
                current_cost = get_cost(colors)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                    no_improve = 0
            else:
                # Pick a random edge
                e_idx = random.randint(0, num_edges - 1)
                new_color = random.randint(0, num_colors - 1)
                
                if new_color == colors[e_idx]:
                    continue
                
                d, updates = delta_cost(e_idx, new_color, colors, k4_masks)
                
                if d <= 0 or random.random() < np.exp(-d / T):
                    colors[e_idx] = new_color
                    for k4_idx, new_mask in updates.items():
                        k4_masks[k4_idx] = new_mask
                    current_cost += d
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors[:]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                
                if no_improve > max_no_improve:
                    T *= 0.5
                    no_improve = 0
                    if random.random() < 0.3:
                        colors = best_colors[:]
                        for k4_idx in range(num_k4s):
                            e_indices = k4_edge_indices[k4_idx]
                            mask = 0
                            for ei in e_indices:
                                mask |= (1 << colors[ei])
                            k4_masks[k4_idx] = mask
                        current_cost = best_cost
            
            T *= cool_rate
            if T < min_T:
                break
                
        return best_colors, best_cost

    # Try to find solution with 13 colors
    found_valid = False
    best_solution_13 = None
    
    if time.time() < deadline - 10:
        init_colors, init_masks = greedy_init(13)
        init_cost = get_cost(init_colors)
        
        if init_cost == 0:
            best_solution_13 = init_colors
            found_valid = True
        else:
            remaining = deadline - time.time()
            if remaining > 5:
                sa_colors, sa_cost = simulated_annealing(init_colors, init_masks, 13, time.time() + min(remaining - 2, 50))
                if sa_cost == 0:
                    best_solution_13 = sa_colors
                    found_valid = True
                else:
                    # Try targeted local search on the best SA result
                    tl_colors, tl_masks = targeted_local_search(sa_colors, init_masks, 13, max_iters=500)
                    tl_cost = get_cost(tl_colors)
                    if tl_cost == 0:
                        best_solution_13 = tl_colors
                        found_valid = True
    
    if found_valid and best_solution_13:
        # Minimize distinct colors
        used = set(best_solution_13)
        if len(used) < 13:
            mapping = {c: i for i, c in enumerate(sorted(used))}
            best_solution = [mapping[c] for c in best_solution_13]
            best_num_colors = len(used)
        else:
            best_solution = best_solution_13
            best_num_colors = 13
        
        return best_solution

    # If 13 failed, try 14
    if time.time() < deadline - 5:
        init_colors, init_masks = greedy_init(14)
        init_cost = get_cost(init_colors)
        
        if init_cost == 0:
            best_solution = init_colors
            used = set(best_solution)
            if len(used) < 14:
                mapping = {c: i for i, c in enumerate(sorted(used))}
                best_solution = [mapping[c] for c in best_solution]
            return best_solution
        else:
            remaining = deadline - time.time()
            if remaining > 5:
                sa_colors, sa_cost = simulated_annealing(init_colors, init_masks, 14, time.time() + min(remaining - 2, 30))
                if sa_cost == 0:
                    best_solution = sa_colors
                    used = set(best_solution)
                    if len(used) < 14:
                        mapping = {c: i for i, c in enumerate(sorted(used))}
                        best_solution = [mapping[c] for c in best_solution]
                    return best_solution
                else:
                    tl_colors, tl_masks = targeted_local_search(sa_colors, init_masks, 14, max_iters=300)
                    tl_cost = get_cost(tl_colors)
                    if tl_cost == 0:
                        best_solution = tl_colors
                        used = set(best_solution)
                        if len(used) < 14:
                            mapping = {c: i for i, c in enumerate(sorted(used))}
                            best_solution = [mapping[c] for c in best_solution]
                        return best_solution

    # Fallback: Return seed-like 17 color solution if all else fails
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
