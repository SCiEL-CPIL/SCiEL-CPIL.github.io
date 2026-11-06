import numpy as np
import time
import random

N = 64
TOTAL_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Precompute grid points
    grid_points = [(x, y) for x in range(N) for y in range(N)]
    grid_array = np.array(grid_points, dtype=np.int64)
    
    # Precompute all pairwise squared distances
    # dist_sq[i, j] = squared distance between point i and point j
    diff = grid_array[:, np.newaxis, :] - grid_array[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)  # (4096, 4096) int64
    
    best_points = []
    best_count = 0
    
    def count_violations(selected):
        """Count number of isosceles triangles in the selected set."""
        k = len(selected)
        if k < 3:
            return 0
        
        # Get pairwise distances among selected points
        # selected is a list of point indices
        sel_dist = dist_sq[np.ix_(selected, selected)]
        
        violations = 0
        for i in range(k):
            # For each point i, check if any two distances from i are equal
            # and if that creates an isosceles triangle
            for j in range(i + 1, k):
                for l in range(j + 1, k):
                    d_ij = sel_dist[i, j]
                    d_il = sel_dist[i, l]
                    d_jl = sel_dist[j, l]
                    # Isosceles if any two sides are equal
                    if d_ij == d_il or d_ij == d_jl or d_il == d_jl:
                        violations += 1
        
        return violations
    
    def incremental_violation_change(selected, add_idx, remove_idx=None):
        """
        Compute change in violation count when adding add_idx to selected.
        If remove_idx is not None, also compute effect of removing it.
        Returns (delta_add, delta_remove) or just delta if remove is None.
        """
        # For adding add_idx:
        # New isosceles triangles involving add_idx
        # A triangle (add_idx, a, b) is isosceles if:
        #   dist(add_idx, a) == dist(add_idx, b)  [isosceles at add_idx]
        #   dist(add_idx, a) == dist(a, b)         [isosceles at a]
        #   dist(add_idx, b) == dist(a, b)         [isosceles at b]
        
        k = len(selected)
        if k < 2:
            # Can't form a triangle with fewer than 2 other points
            return 0
        
        # Get distances from add_idx to all selected points
        d_to_sel = dist_sq[add_idx, selected]  # shape (k,)
        
        # Count pairs (a, b) in selected such that triangle (add_idx, a, b) is isosceles
        new_violations = 0
        
        # For each pair (a, b) in selected
        for i in range(k):
            for j in range(i + 1, k):
                d_a = d_to_sel[i]
                d_b = d_to_sel[j]
                d_ab = dist_sq[selected[i], selected[j]]
                
                if d_a == d_b or d_a == d_ab or d_b == d_ab:
                    new_violations += 1
        
        return new_violations
    
    def run_simulated_annealing():
        """Simulated annealing to maximize set size while minimizing violations."""
        
        # Start with a random greedy construction
        selected = []
        used = np.zeros(TOTAL_POINTS, dtype=bool)
        
        # Greedy initialization
        candidates = np.arange(TOTAL_POINTS)
        random.shuffle(candidates.tolist())
        
        for idx in candidates:
            if time.time() > deadline - 10:
                break
            delta = incremental_violation_change(selected, idx)
            if delta == 0:
                selected.append(idx)
                used[idx] = True
        
        if len(selected) == 0:
            return []
        
        current_violations = count_violations(selected)
        current_size = len(selected)
        
        best_selected = list(selected)
        best_violations = current_violations
        best_size = current_size
        
        # Simulated annealing parameters
        T = 1.0  # Initial temperature
        cooling = 0.9995
        min_T = 0.01
        max_iterations = 50000
        
        iter_count = 0
        
        while T > min_T and iter_count < max_iterations:
            if time.time() > deadline - 10:
                break
            
            iter_count += 1
            
            # Choose a move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a random unselected point
                unselected = np.where(~used)[0]
                if len(unselected) == 0:
                    continue
                idx = random.choice(unselected.tolist())
                
                delta_v = incremental_violation_change(selected, idx)
                delta_size = 1
                
                # Accept if it improves or with probability exp(-delta_v/T)
                if delta_v == 0:
                    # Always accept if no new violations
                    selected.append(idx)
                    used[idx] = True
                    current_violations += delta_v
                    current_size += delta_size
                elif random.random() < np.exp(-delta_v / T):
                    selected.append(idx)
                    used[idx] = True
                    current_violations += delta_v
                    current_size += delta_size
                # else reject
                
            elif move_type < 0.7:
                # Try to remove a random selected point
                if len(selected) == 0:
                    continue
                pos = random.randint(0, len(selected) - 1)
                idx = selected[pos]
                
                # Compute violation change from removing idx
                # This is complex; approximate by recomputing
                old_violations = current_violations
                selected.pop(pos)
                used[idx] = False
                current_size -= 1
                new_violations = count_violations(selected)
                delta_v = new_violations - old_violations
                current_violations = new_violations
                
                # Accept if it improves or with probability
                if delta_v < 0 or random.random() < np.exp(delta_v / T):
                    # Keep the removal
                    pass
                else:
                    # Reject: put it back
                    selected.insert(pos, idx)
                    used[idx] = True
                    current_size += 1
                    current_violations = old_violations
                    
            else:
                # Try a swap: remove one, add another
                if len(selected) == 0 or np.sum(~used) == 0:
                    continue
                pos = random.randint(0, len(selected) - 1)
                idx_remove = selected[pos]
                
                unselected = np.where(~used)[0]
                idx_add = random.choice(unselected.tolist())
                
                # Compute net change
                old_violations = current_violations
                old_size = current_size
                
                # Temporarily remove
                selected.pop(pos)
                used[idx_remove] = False
                current_size -= 1
                
                # Temporarily add
                delta_add = incremental_violation_change(selected, idx_add)
                selected.append(idx_add)
                used[idx_add] = True
                current_size += 1
                
                new_violations = count_violations(selected)
                delta_v = new_violations - old_violations
                delta_size = 0  # Swap keeps size same
                
                if delta_v <= 0 or random.random() < np.exp(-delta_v / T):
                    current_violations = new_violations
                    current_size = old_size
                else:
                    # Reject: revert
                    selected.pop()
                    used[idx_add] = False
                    current_size -= 1
                    selected.insert(pos, idx_remove)
                    used[idx_remove] = True
                    current_size += 1
                    current_violations = old_violations
            
            # Track best valid solution
            if current_violations == 0 and current_size > best_size:
                best_size = current_size
                best_violations = 0
                best_selected = list(selected)
            elif current_violations == 0 and current_size == best_size:
                # Same size, but maybe different configuration
                pass
            
            # Cool down
            T *= cooling
        
        # Return best valid solution found
        if best_violations == 0:
            return best_selected
        else:
            # If no valid solution found, return empty or try to repair
            # For now, return empty
            return []
    
    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline - 5:
        num_restarts += 1
        pts_idx = run_simulated_annealing()
        if len(pts_idx) > best_count:
            best_points = [grid_points[i] for i in pts_idx]
            best_count = len(pts_idx)
        
        # Early exit if we've found a good solution
        if best_count >= 100:
            break
    
    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
