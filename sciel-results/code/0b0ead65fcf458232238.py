import numpy as np
import time
import random

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 285  # Safety margin
    
    num_cells = N * N
    
    # Precompute squared distances as int32 to save memory and enable fast hashing
    coords = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    diff = coords[:, np.newaxis, :].astype(np.int32) - coords[np.newaxis, :, :].astype(np.int32)
    dist_sq = (diff[:, :, 0]**2 + diff[:, :, 1]**2).astype(np.int32)
    
    def simulated_annealing(seed_val, time_limit):
        random.seed(seed_val)
        
        # Initialize
        selected = set()
        # dist_counts[i] = {dist_value: count} for i in range(num_cells)
        dist_counts = [None] * num_cells
        # tri_counts[i] = number of isosceles triangles involving i
        tri_counts = [0] * num_cells
        num_violating = 0
        violating = set()  # cells with at least one duplicate distance
        
        def add_cell(j):
            nonlocal num_violating
            # Update j's perspective: j sees all currently selected cells
            j_dists = {}
            j_tri_count = 0
            for i in selected:
                d = dist_sq[j, i]
                if d in j_dists:
                    old_c = j_dists[d]
                    j_dists[d] = old_c + 1
                    # Triangles added for j: binom(c+1, 2) - binom(c, 2) = c
                    j_tri_count += old_c
                else:
                    j_dists[d] = 1
            
            j_now_violating = j_tri_count > 0
            tri_counts[j] = j_tri_count
            
            # Update all existing selected cells' perspective
            for i in selected:
                d = dist_sq[i, j]
                if dist_counts[i] is None:
                    dist_counts[i] = {}
                old_count = dist_counts[i].get(d, 0)
                dist_counts[i][d] = old_count + 1
                new_count = old_count + 1
                
                # Update tri_counts[i]
                # Delta is old_count
                tri_counts[i] += old_count
                
                if old_count == 1 and new_count == 2:
                    # i just became violating
                    if i not in violating:
                        violating.add(i)
                        num_violating += 1
                elif old_count == 0 and new_count == 1:
                    # No change in violation status for i (wasn't violating due to this dist)
                    pass
            
            selected.add(j)
            
            if j_now_violating:
                violating.add(j)
                num_violating += 1
            # Store j's dist_counts
            dist_counts[j] = j_dists

        def remove_cell(j):
            nonlocal num_violating
            # Remove j from all selected cells' counts
            # First, update j's own status (it will be removed, so we just clear)
            j_was_violating = tri_counts[j] > 0
            if j_was_violating and j in violating:
                violating.discard(j)
                num_violating -= 1
            
            # Update all other selected cells
            for i in selected:
                if i == j:
                    continue
                d = dist_sq[i, j]
                if dist_counts[i] is not None:
                    old_count = dist_counts[i].get(d, 0)
                    if old_count > 0:
                        new_count = old_count - 1
                        dist_counts[i][d] = new_count
                        # Update tri_counts[i]
                        # Delta is -(old_count - 1)
                        tri_counts[i] -= (old_count - 1)
                        
                        if old_count == 2:
                            # i might no longer be violating due to this distance
                            # Check if any other distance has count >= 2
                            # We can check if tri_counts[i] is now 0
                            if tri_counts[i] == 0:
                                if i in violating:
                                    violating.discard(i)
                                    num_violating -= 1
                            # Note: tri_counts[i] > 0 implies violation. 
                            # tri_counts[i] == 0 implies no isosceles triangles involving i.
            
            selected.discard(j)
            dist_counts[j] = None
            tri_counts[j] = 0

        def greedy_polish():
            # Remove cells with highest tri_count until no violations
            while violating:
                # Find cell with max tri_count
                # Since violating set is small, we can just check those
                max_tri = -1
                max_cell = -1
                for v in violating:
                    if tri_counts[v] > max_tri:
                        max_tri = tri_counts[v]
                        max_cell = v
                if max_cell == -1:
                    break
                remove_cell(max_cell)

        # Initialize with random sparse set
        initial_density = 0.003
        num_initial = int(num_cells * initial_density)
        initial_indices = random.sample(range(num_cells), num_initial)
        for idx in initial_indices:
            add_cell(idx)
        
        current_count = len(selected)
        current_violations = num_violating
        penalty_weight = 5
        
        best_count = current_count
        best_violations = current_violations
        best_selected = set(selected)
        
        # Simulated Annealing
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.9997
        
        start_sa = time.time()
        steps = 0
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() - start_sa < time_limit:
            # Choose move type
            r = random.random()
            if r < 0.5:
                move_type = 'add'
            elif r < 0.75:
                move_type = 'remove'
            else:
                move_type = 'swap'
            
            if move_type == 'add' and len(selected) < num_cells - 1:
                # Pick random unselected cell
                idx = random.randint(0, num_cells - 1)
                if idx in selected:
                    continue
                
                # Compute delta violations if we add idx
                idx_dist_counts = {}
                idx_tri_count = 0
                for i in selected:
                    d = dist_sq[idx, i]
                    if d in idx_dist_counts:
                        old_c = idx_dist_counts[d]
                        idx_dist_counts[d] = old_c + 1
                        idx_tri_count += old_c
                    else:
                        idx_dist_counts[d] = 1
                
                idx_violates = idx_tri_count > 0
                
                # For each selected i, check if adding idx makes i violate
                new_i_violations = 0
                delta_tri_total = idx_tri_count # Triangles formed by idx itself
                for i in selected:
                    d = dist_sq[i, idx]
                    if dist_counts[i] is not None:
                        old_c = dist_counts[i].get(d, 0)
                        if old_c == 1:
                            new_i_violations += 1
                        # Triangles added for i: old_c
                        delta_tri_total += old_c
                
                delta_viol = (1 if idx_violates else 0) + new_i_violations
                delta_score = 1 - penalty_weight * delta_viol
                
                if delta_score > 0 or random.random() < np.exp(delta_score / T):
                    add_cell(idx)
                    current_count += 1
                    current_violations = num_violating
                    
                    if current_count - penalty_weight * current_violations > best_count - penalty_weight * best_violations:
                        best_count = current_count
                        best_violations = current_violations
                        best_selected = set(selected)
                        no_improve = 0
                    else:
                        no_improve += 1
                    
            elif move_type == 'remove' and len(selected) > 0:
                if len(selected) == 0:
                    continue
                
                # If there are violations, pick a violating cell (greedy hint)
                # Otherwise, pick random
                if violating:
                    # Pick cell with highest tri_count among violating
                    max_tri = -1
                    idx = -1
                    for v in violating:
                        if tri_counts[v] > max_tri:
                            max_tri = tri_counts[v]
                            idx = v
                else:
                    idx = random.choice(list(selected))
                
                # Compute delta
                idx_was_violating = tri_counts[idx] > 0
                
                if idx_was_violating:
                    # Likely beneficial
                    if random.random() < 0.8: # High acceptance
                         remove_cell(idx)
                         current_count -= 1
                         current_violations = num_violating
                         if current_count - penalty_weight * current_violations > best_count - penalty_weight * best_violations:
                             best_count = current_count
                             best_violations = current_violations
                             best_selected = set(selected)
                             no_improve = 0
                         else:
                             no_improve += 1
                else:
                    # Less likely beneficial
                    if random.random() < np.exp(-1.0 / T):
                         remove_cell(idx)
                         current_count -= 1
                         current_violations = num_violating
                         if current_count - penalty_weight * current_violations > best_count - penalty_weight * best_violations:
                             best_count = current_count
                             best_violations = current_violations
                             best_selected = set(selected)
                             no_improve = 0
                         else:
                             no_improve += 1

            elif move_type == 'swap' and len(selected) > 0:
                # Skip swap for now to simplify and speed up
                continue

            T *= cooling_rate
            steps += 1
            
            if no_improve > max_no_improve:
                # Restart with fresh random init
                selected.clear()
                dist_counts = [None] * num_cells
                violating.clear()
                num_violating = 0
                tri_counts = [0] * num_cells
                num_initial = int(num_cells * 0.003)
                initial_indices = random.sample(range(num_cells), num_initial)
                for idx in initial_indices:
                    add_cell(idx)
                current_count = len(selected)
                current_violations = num_violating
                best_count = current_count
                best_violations = current_violations
                best_selected = set(selected)
                no_improve = 0
                T = 5.0
        
        # Clean up: Use greedy polish on best
        # Rebuild state for best_selected
        selected.clear()
        dist_counts = [None] * num_cells
        violating.clear()
        num_violating = 0
        tri_counts = [0] * num_cells
        
        for idx in best_selected:
            add_cell(idx)
            
        # Polish
        greedy_polish()
        
        points = [(idx // N, idx % N) for idx in selected]
        return points, len(selected)
    
    # Run multiple trials
    best_points = []
    best_count = 0
    trial_count = 0
    
    while time.time() < deadline - 20:
        trial_count += 1
        seed_val = trial_count * 7919 + 12345
        
        time_limit = min(15.0, deadline - 20 - time.time())
        if time_limit < 1.0:
            break
        
        points, count = simulated_annealing(seed_val, time_limit)
        
        if count > best_count:
            best_points = points
            best_count = count
    
    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
