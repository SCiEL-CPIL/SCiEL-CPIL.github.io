import numpy as np
import time
import random
from collections import defaultdict

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 285  # Safety margin
    
    num_cells = N * N
    
    # Precompute squared distances as int32 to save memory and enable fast hashing
    coords = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    diff = coords[:, np.newaxis, :].astype(np.int32) - coords[np.newaxis, :, :].astype(np.int32)
    dist_sq = (diff[:, :, 0]**2 + diff[:, :, 1]**2).astype(np.int32)
    
    # Precompute for each cell, list of other cells sorted by distance
    # This helps with faster local search
    cell_neighbors = []
    for i in range(num_cells):
        d = dist_sq[i].copy()
        d[i] = -1  # Mark self
        order = np.argsort(d)
        cell_neighbors.append(order)
    
    # Precompute for each cell, the sorted list of (distance, neighbor_cell) pairs
    # This allows fast incremental violation checking
    # neighbor_dists[i] = array of shape (num_cells-1,) with distances from i to all other cells
    # We need to quickly check: among selected cells, are any distances from i duplicated?
    
    # Strategy: For each cell i, maintain a dict mapping distance -> count of selected cells at that distance
    # A cell i is "violating" if any distance has count >= 2
    # When we add/remove a cell j, only cells i where dist_sq[i,j] could create/break a duplicate are affected
    # Specifically, adding j affects all selected cells i (their distance to j might duplicate an existing one)
    # And removing j affects all selected cells i (their distance to j was contributing to a count)
    
    # For efficiency, we track:
    # - selected: set of selected cell indices
    # - dist_counts[i]: dict {distance: count} for selected cells at that distance from i
    # - num_violating: number of cells with any distance count >= 2
    # - violating_cells: set of cells that are currently violating
    
    def simulated_annealing(seed_val, time_limit):
        random.seed(seed_val)
        
        # Initialize
        selected = set()
        # dist_counts[i] = {dist_value: count} for i in range(num_cells)
        # Only track for cells that have selected neighbors
        dist_counts = [None] * num_cells
        num_violating = 0
        violating = set()  # cells with at least one duplicate distance
        
        def add_cell(j):
            nonlocal num_violating
            # Update j's perspective: j sees all currently selected cells
            j_dists = {}
            for i in selected:
                d = dist_sq[j, i]
                j_dists[d] = j_dists.get(d, 0) + 1
            
            j_now_violating = any(c >= 2 for c in j_dists.values())
            
            # Update all existing selected cells' perspective
            for i in list(selected):
                d = dist_sq[i, j]
                if dist_counts[i] is None:
                    dist_counts[i] = {}
                old_count = dist_counts[i].get(d, 0)
                dist_counts[i][d] = old_count + 1
                new_count = old_count + 1
                
                if old_count == 1 and new_count == 2:
                    if i not in violating:
                        violating.add(i)
                        num_violating += 1
            
            selected.add(j)
            
            if j_now_violating:
                violating.add(j)
                num_violating += 1
            dist_counts[j] = j_dists
        
        def remove_cell(j):
            nonlocal num_violating
            j_dists = dist_counts[j]
            if j_dists is not None:
                j_was_violating = any(c >= 2 for c in j_dists.values())
                if j_was_violating and j in violating:
                    violating.discard(j)
                    num_violating -= 1
            
            for i in list(selected):
                if i == j:
                    continue
                d = dist_sq[i, j]
                if dist_counts[i] is not None:
                    old_count = dist_counts[i].get(d, 0)
                    if old_count > 0:
                        dist_counts[i][d] = old_count - 1
                        if old_count == 2:
                            new_violating = any(c >= 2 for c in dist_counts[i].values())
                            if not new_violating and i in violating:
                                violating.discard(i)
                                num_violating -= 1
            
            selected.discard(j)
            dist_counts[j] = None
        
        def local_polish(max_iters=500):
            """Try to improve by removing violating cells and adding non-violating ones"""
            nonlocal num_violating
            improved = True
            iters = 0
            while improved and iters < max_iters and time.time() < start_sa + time_limit:
                improved = False
                iters += 1
                
                # Phase 1: Remove cells that are violating
                if violating:
                    # Pick the violating cell with most duplicate distances
                    to_remove = list(violating)
                    to_remove.sort(key=lambda j: sum(1 for c in (dist_counts[j] or {}).values() if c >= 2), reverse=True)
                    j = to_remove[0]
                    remove_cell(j)
                    improved = True
                
                # Phase 2: Try to add cells that don't create violations
                # Sample a few random candidates
                for _ in range(5):
                    idx = random.randint(0, num_cells - 1)
                    if idx in selected:
                        continue
                    
                    # Check if adding idx would create any violations
                    would_violate = False
                    for i in selected:
                        d = dist_sq[i, idx]
                        if dist_counts[i] is not None:
                            old_c = dist_counts[i].get(d, 0)
                            if old_c >= 1:
                                would_violate = True
                                break
                        else:
                            # No other selected cells at this distance from i, so ok
                            pass
                    
                    # Also check idx itself
                    if not would_violate:
                        idx_dists = {}
                        for i in selected:
                            d = dist_sq[idx, i]
                            idx_dists[d] = idx_dists.get(d, 0) + 1
                        if any(c >= 2 for c in idx_dists.values()):
                            would_violate = True
                    
                    if not would_violate:
                        add_cell(idx)
                        improved = True
        
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
        
        # Simulated annealing
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.9997
        
        start_sa = time.time()
        steps = 0
        no_improve = 0
        max_no_improve = 5000
        
        # Track best valid solution separately
        best_valid_count = 0
        best_valid_selected = set()
        
        # If current is valid, record it
        if current_violations == 0:
            best_valid_count = current_count
            best_valid_selected = set(selected)
        
        while time.time() - start_sa < time_limit:
            # Choose move type
            r = random.random()
            if r < 0.4:
                move_type = 'add'
            elif r < 0.7:
                move_type = 'remove'
            elif r < 0.9:
                move_type = 'swap'
            else:
                move_type = 'polish'
            
            if move_type == 'polish':
                # Run local polish
                local_polish(max_iters=100)
                current_count = len(selected)
                current_violations = num_violating
                
                if current_violations == 0 and current_count > best_valid_count:
                    best_valid_count = current_count
                    best_valid_selected = set(selected)
                
                if current_count - penalty_weight * current_violations > best_count - penalty_weight * best_violations:
                    best_count = current_count
                    best_violations = current_violations
                    best_selected = set(selected)
                    no_improve = 0
                else:
                    no_improve += 1
                
                T *= cooling_rate
                steps += 1
                continue
            
            if move_type == 'add' and len(selected) < num_cells - 1:
                # Pick random unselected cell
                idx = random.randint(0, num_cells - 1)
                if idx in selected:
                    continue
                
                idx_dist_counts = {}
                for i in selected:
                    d = dist_sq[idx, i]
                    idx_dist_counts[d] = idx_dist_counts.get(d, 0) + 1
                
                idx_violates = any(c >= 2 for c in idx_dist_counts.values())
                
                new_i_violations = 0
                for i in selected:
                    d = dist_sq[i, idx]
                    if dist_counts[i] is not None:
                        old_c = dist_counts[i].get(d, 0)
                        if old_c == 1:
                            new_i_violations += 1
                
                delta_viol = (1 if idx_violates else 0) + new_i_violations
                delta_score = 1 - penalty_weight * delta_viol
                
                if delta_score > 0 or random.random() < np.exp(delta_score / T):
                    add_cell(idx)
                    current_count += 1
                    current_violations = num_violating
                    
                    if current_violations == 0 and current_count > best_valid_count:
                        best_valid_count = current_count
                        best_valid_selected = set(selected)
                    
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
                idx = random.choice(list(selected))
                
                idx_dists = dist_counts[idx]
                idx_was_violating = False
                if idx_dists is not None:
                    idx_was_violating = any(c >= 2 for c in idx_dists.values())
                
                delta_viol = -(1 if idx_was_violating else 0)
                for i in selected:
                    if i == idx:
                        continue
                    d = dist_sq[i, idx]
                    if dist_counts[i] is not None:
                        old_c = dist_counts[i].get(d, 0)
                        if old_c == 2:
                            still_violating = False
                            for dd, cc in dist_counts[i].items():
                                if dd != d and cc >= 2:
                                    still_violating = True
                                    break
                            if not still_violating:
                                delta_viol -= 1
                
                delta_score = -1 - penalty_weight * delta_viol
                
                if delta_score > 0 or random.random() < np.exp(delta_score / T):
                    remove_cell(idx)
                    current_count -= 1
                    current_violations = num_violating
                    
                    if current_violations == 0 and current_count > best_valid_count:
                        best_valid_count = current_count
                        best_valid_selected = set(selected)
                    
                    if current_count - penalty_weight * current_violations > best_count - penalty_weight * best_violations:
                        best_count = current_count
                        best_violations = current_violations
                        best_selected = set(selected)
                        no_improve = 0
                    else:
                        no_improve += 1
                    
            elif move_type == 'swap' and len(selected) > 0:
                if len(selected) == 0:
                    continue
                idx_remove = random.choice(list(selected))
                idx_add = random.randint(0, num_cells - 1)
                if idx_add in selected:
                    continue
                
                # Save state for potential revert
                saved_selected = set(selected)
                saved_dist_counts = [dc.copy() if dc else None for dc in dist_counts]
                saved_violating = set(violating)
                saved_num_viol = num_violating
                
                remove_cell(idx_remove)
                add_cell(idx_add)
                
                new_count = len(selected)
                new_viol = num_violating
                delta_score = (new_count - current_count) - penalty_weight * (new_viol - current_violations)
                
                if delta_score > 0 or random.random() < np.exp(delta_score / T):
                    current_count = new_count
                    current_violations = new_viol
                    if current_violations == 0 and current_count > best_valid_count:
                        best_valid_count = current_count
                        best_valid_selected = set(selected)
                    if current_count - penalty_weight * current_violations > best_count - penalty_weight * best_violations:
                        best_count = current_count
                        best_violations = current_violations
                        best_selected = set(selected)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Revert
                    selected = saved_selected
                    dist_counts = saved_dist_counts
                    violating = saved_violating
                    num_violating = saved_num_viol
            
            T *= cooling_rate
            steps += 1
            
            if no_improve > max_no_improve:
                # Restart with fresh random init
                selected.clear()
                dist_counts = [None] * num_cells
                violating.clear()
                num_violating = 0
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
        
        # Use best valid solution if available, otherwise clean up best_selected
        if best_valid_count > 0:
            best_sel = set(best_valid_selected)
        else:
            # Clean up: if best has violations, remove violating cells greedily
            best_sel = set(best_selected)
            # Rebuild dist_counts for best_sel
            dist_counts2 = [None] * num_cells
            for j in best_sel:
                j_dists = {}
                for i in best_sel:
                    if i != j:
                        d = dist_sq[j, i]
                        j_dists[d] = j_dists.get(d, 0) + 1
                dist_counts2[j] = j_dists
            
            # Find violating cells and remove them
            changed = True
            while changed and best_sel:
                changed = False
                to_remove = []
                for j in list(best_sel):
                    if dist_counts2[j] is not None:
                        if any(c >= 2 for c in dist_counts2[j].values()):
                            to_remove.append(j)
                if to_remove:
                    to_remove.sort(key=lambda j: sum(1 for c in dist_counts2[j].values() if c >= 2), reverse=True)
                    j = to_remove[0]
                    best_sel.discard(j)
                    for i in best_sel:
                        d = dist_sq[i, j]
                        if dist_counts2[i] is not None and d in dist_counts2[i]:
                            dist_counts2[i][d] -= 1
                            if dist_counts2[i][d] <= 0:
                                del dist_counts2[i][d]
                    dist_counts2[j] = None
                    changed = True
        
        points = [(idx // N, idx % N) for idx in best_sel]
        return points, len(points)
    
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
