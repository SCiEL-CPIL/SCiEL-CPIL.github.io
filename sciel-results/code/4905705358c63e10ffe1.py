import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n = 16
    num_edges = n * (n - 1) // 2  # 120
    
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
        
    num_k4s = len(k4s)  # 1820
    
    # Map edge to list of K4 indices containing it
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4_edge_indices):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)
            
    # Precompute for each K4: the 6 edge indices and their positions
    # k4_edge_pos[k4_idx] is a dict: edge_idx -> position (0-5)
    k4_edge_pos = [{} for _ in range(num_k4s)]
    for k4_idx, e_indices in enumerate(k4_edge_indices):
        for pos, ei in enumerate(e_indices):
            k4_edge_pos[k4_idx][ei] = pos

    def mask_count(mask):
        return mask.bit_count()

    def get_cost_and_masks(colors):
        """Calculate cost (number of violated K4s) and return masks."""
        cost = 0
        k4_masks = [0] * num_k4s
        for k4_idx in range(num_k4s):
            e_indices = k4_edge_indices[k4_idx]
            mask = 0
            for ei in e_indices:
                mask |= (1 << colors[ei])
            k4_masks[k4_idx] = mask
            if mask_count(mask) < 5:
                cost += 1
        return cost, k4_masks

    def delta_cost(e_idx, new_color, colors, k4_masks):
        """
        Calculate delta cost for changing edge e_idx to new_color.
        Returns (delta, updated_masks_dict)
        """
        old_color = colors[e_idx]
        if old_color == new_color:
            return 0, {}
        
        delta = 0
        updates = {}
        
        for k4_idx in edge_to_k4s[e_idx]:
            old_mask = k4_masks[k4_idx]
            old_distinct = mask_count(old_mask)
            old_bad = 1 if old_distinct < 5 else 0
            
            # Calculate new mask by recomputing for this specific K4 (6 edges)
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

    def greedy_init(num_colors, seed=None):
        """Greedy initialization with random edge ordering."""
        if seed is not None:
            random.seed(seed)
        
        colors = [0] * num_edges
        k4_masks = [1] * num_k4s  # All edges color 0 initially
        
        edge_order = list(range(num_edges))
        random.shuffle(edge_order)
        
        for e_idx in edge_order:
            best_c = colors[e_idx]
            best_delta = 0
            
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

    def targeted_local_search(colors, k4_masks, num_colors, max_time, seed=None):
        """
        Targeted local search focusing on violating K4s.
        When a K4 is violated, try to fix it by changing one of its edges.
        """
        if seed is not None:
            random.seed(seed)
        
        current_cost, k4_masks = get_cost_and_masks(colors)
        best_colors = colors[:]
        best_cost = current_cost
        
        if current_cost == 0:
            return best_colors, 0
        
        T = 3.0
        cool_rate = 0.9995
        min_T = 0.05
        no_improve = 0
        max_no_improve = 3000
        
        while time.time() < max_time:
            if current_cost == 0:
                best_colors = colors[:]
                best_cost = 0
                break
            
            # Find violating K4s
            violating_k4s = []
            for k4_idx in range(num_k4s):
                if mask_count(k4_masks[k4_idx]) < 5:
                    violating_k4s.append(k4_idx)
            
            if not violating_k4s:
                current_cost = 0
                best_colors = colors[:]
                best_cost = 0
                break
            
            # Pick a random violating K4
            k4_idx = random.choice(violating_k4s)
            e_indices = k4_edge_indices[k4_idx]
            
            # Try to fix by changing one of the 6 edges in this K4
            edge_order = e_indices[:]
            random.shuffle(edge_order)
            
            found_improvement = False
            for e_idx in edge_order:
                # Try each color for this edge
                color_order = list(range(num_colors))
                random.shuffle(color_order)
                
                best_d = 0
                best_c = colors[e_idx]
                
                for c in color_order:
                    if c == colors[e_idx]:
                        continue
                    d, _ = delta_cost(e_idx, c, colors, k4_masks)
                    if d < best_d:
                        best_d = d
                        best_c = c
                
                if best_c != colors[e_idx] and (best_d < 0 or random.random() < np.exp(-best_d / T)):
                    _, updates = delta_cost(e_idx, best_c, colors, k4_masks)
                    colors[e_idx] = best_c
                    for k4_i, new_mask in updates.items():
                        k4_masks[k4_i] = new_mask
                    current_cost += best_d
                    found_improvement = True
                    break
            
            if not found_improvement:
                # Try random edge change
                e_idx = random.randint(0, num_edges - 1)
                new_color = random.randint(0, num_colors - 1)
                if new_color == colors[e_idx]:
                    continue
                
                d, updates = delta_cost(e_idx, new_color, colors, k4_masks)
                if d <= 0 or random.random() < np.exp(-d / T):
                    colors[e_idx] = new_color
                    for k4_i, new_mask in updates.items():
                        k4_masks[k4_i] = new_mask
                    current_cost += d
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors[:]
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > max_no_improve:
                T *= 0.7
                no_improve = 0
                if random.random() < 0.3:
                    # Revert to best
                    colors = best_colors[:]
                    current_cost, k4_masks = get_cost_and_masks(colors)
            
            T *= cool_rate
            if T < min_T:
                T = min_T
        
        return best_colors, best_cost

    def minimize_distinct_colors(colors):
        """Relabel colors to use minimum distinct values."""
        used = set(colors)
        if len(used) <= 1:
            return [0] * len(colors)
        mapping = {c: i for i, c in enumerate(sorted(used))}
        return [mapping[c] for c in colors]

    # Main search strategy
    best_solution = None
    best_num_colors = float('inf')
    
    # Try 13 colors first (theoretical minimum)
    for attempt in range(3):
        if time.time() > deadline - 20:
            break
        
        seed = attempt * 12345 + 67890
        init_colors, init_masks = greedy_init(13, seed=seed)
        init_cost, _ = get_cost_and_masks(init_colors)
        
        if init_cost == 0:
            best_solution = minimize_distinct_colors(init_colors)
            best_num_colors = len(set(best_solution))
            break
        
        # Targeted local search
        remaining = deadline - time.time() - 15
        if remaining > 10:
            sa_colors, sa_cost = targeted_local_search(
                init_colors, init_masks, 13, 
                time.time() + min(remaining, 40),
                seed=seed
            )
            if sa_cost == 0:
                best_solution = minimize_distinct_colors(sa_colors)
                best_num_colors = len(set(best_solution))
                break
    
    if best_solution is not None:
        return best_solution
    
    # Try 14 colors
    for attempt in range(3):
        if time.time() > deadline - 15:
            break
        
        seed = attempt * 54321 + 11111
        init_colors, init_masks = greedy_init(14, seed=seed)
        init_cost, _ = get_cost_and_masks(init_colors)
        
        if init_cost == 0:
            best_solution = minimize_distinct_colors(init_colors)
            best_num_colors = len(set(best_solution))
            break
        
        remaining = deadline - time.time() - 10
        if remaining > 10:
            sa_colors, sa_cost = targeted_local_search(
                init_colors, init_masks, 14,
                time.time() + min(remaining, 35),
                seed=seed
            )
            if sa_cost == 0:
                best_solution = minimize_distinct_colors(sa_colors)
                best_num_colors = len(set(best_solution))
                break
    
    if best_solution is not None:
        return best_solution
    
    # Try 15 colors
    for attempt in range(3):
        if time.time() > deadline - 10:
            break
        
        seed = attempt * 98765 + 22222
        init_colors, init_masks = greedy_init(15, seed=seed)
        init_cost, _ = get_cost_and_masks(init_colors)
        
        if init_cost == 0:
            best_solution = minimize_distinct_colors(init_colors)
            best_num_colors = len(set(best_solution))
            break
        
        remaining = deadline - time.time() - 5
        if remaining > 8:
            sa_colors, sa_cost = targeted_local_search(
                init_colors, init_masks, 15,
                time.time() + min(remaining, 30),
                seed=seed
            )
            if sa_cost == 0:
                best_solution = minimize_distinct_colors(sa_colors)
                best_num_colors = len(set(best_solution))
                break
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: construct a valid 17-color solution
    # Use a structured coloring that guarantees validity
    # Color edge (i,j) based on (i mod 4, j mod 4) pattern
    # Actually, let's use a known valid construction
    # Simple approach: use enough colors to ensure any K4 has 5+
    
    # Fallback with 17 colors using modular arithmetic
    # This should be valid but not minimal
    fallback = []
    for i in range(16):
        for j in range(i + 1, 16):
            # Use a hash-like function to distribute colors
            c = (i * 3 + j * 7 + i * j) % 17
            fallback.append(c)
    
    # Verify and if invalid, use even more colors
    cost, _ = get_cost_and_masks(fallback)
    if cost > 0:
        # Use 20 colors with better distribution
        fallback = []
        for i in range(16):
            for j in range(i + 1, 16):
                c = (i * 5 + j * 11 + i * j * 3) % 20
                fallback.append(c)
        cost, _ = get_cost_and_masks(fallback)
    
    if cost > 0:
        # Last resort: assign unique color to each edge type
        # This guarantees validity with 120 colors
        fallback = list(range(120))
    
    return minimize_distinct_colors(fallback)
