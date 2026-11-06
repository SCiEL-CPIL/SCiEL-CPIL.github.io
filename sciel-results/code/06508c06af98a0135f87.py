import numpy as np
import time
import random

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 290s deadline, 40s margin
    
    # Precompute all pairs and their squared distances for fast constraint checking
    # Represent points as indices 0..N*N-1
    num_cells = N * N
    
    # Precompute squared distances between all pairs
    # dist_matrix[i, j] = squared distance between cell i and cell j
    # This is a (num_cells x num_cells) matrix - 4096x4096 = 16M entries
    # Memory: 16M * 8 bytes = 128MB, acceptable
    coords = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.float64)
    # Compute pairwise squared distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # For each cell i, we need to track which distances are "used" by other selected cells
    # A cell j can be selected if:
    # 1. For all selected cells k != j: dist_sq[j,k] is not in the set of distances from k to other selected cells
    # 2. All distances from j to selected cells are distinct
    
    # We'll use simulated annealing on a binary vector of length num_cells
    # Objective: maximize sum(x) - penalty * (number of violated constraints)
    # A constraint is violated if three selected cells form an isosceles triangle
    
    # For efficiency, we'll track:
    # - selected: boolean array of length num_cells
    # - For each cell i, dist_to_selected[i]: set of distances from i to selected cells (excluding i itself)
    # - For each cell i, used_dists[i]: set of distances that are "blocked" for i because some selected cell k has two selected neighbors at the same distance
    
    def count_violations(selected):
        """Count the number of violated isosceles triangle constraints."""
        selected_indices = np.where(selected)[0]
        if len(selected_indices) < 3:
            return 0
        
        violations = 0
        for i in range(len(selected_indices)):
            idx_i = selected_indices[i]
            # Get distances from idx_i to all other selected cells
            dists = dist_sq[idx_i, selected_indices]
            dists = dists[dists > 0]  # Exclude self-distance (0)
            
            # Check for duplicate distances from idx_i (apex at idx_i)
            if len(dists) != len(np.unique(dists)):
                violations += 1
        
        return violations
    
    def is_valid_configuration(selected):
        """Check if the configuration has no isosceles triangles."""
        return count_violations(selected) == 0
    
    def simulated_annealing(seed_val, time_limit):
        """Run simulated annealing to find a large valid configuration."""
        random.seed(seed_val)
        
        # Initialize with a random sparse configuration
        selected = np.zeros(num_cells, dtype=bool)
        initial_density = 0.005  # Start with ~20 selected cells
        num_initial = int(num_cells * initial_density)
        initial_indices = random.sample(range(num_cells), num_initial)
        for idx in initial_indices:
            selected[idx] = True
        
        # Track current and best solutions
        current_violations = count_violations(selected)
        current_score = np.sum(selected) - current_violations * 10  # Heavy penalty for violations
        
        best_selected = selected.copy()
        best_score = current_score
        best_violations = current_violations
        
        # Simulated annealing parameters
        T_start = 10.0
        T_min = 0.01
        cooling_rate = 0.9995
        
        T = T_start
        step = 0
        
        start_sa = time.time()
        while time.time() - start_sa < time_limit and T > T_min:
            # Choose a random move: add, remove, or swap
            move_type = random.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add' and np.sum(selected) < num_cells - 1:
                # Try adding a random unselected cell
                candidates = np.where(~selected)[0]
                if len(candidates) == 0:
                    continue
                idx = random.choice(candidates)
                selected[idx] = True
                
                new_violations = count_violations(selected)
                new_score = np.sum(selected) - new_violations * 10
                
                delta = new_score - current_score
                if delta > 0 or random.random() < np.exp(delta / T):
                    current_score = new_score
                    current_violations = new_violations
                    if new_score > best_score:
                        best_score = new_score
                        best_violations = new_violations
                        best_selected = selected.copy()
                else:
                    selected[idx] = False  # Revert
                
            elif move_type == 'remove' and np.sum(selected) > 0:
                # Try removing a random selected cell
                candidates = np.where(selected)[0]
                if len(candidates) == 0:
                    continue
                idx = random.choice(candidates)
                selected[idx] = False
                
                new_violations = count_violations(selected)
                new_score = np.sum(selected) - new_violations * 10
                
                delta = new_score - current_score
                if delta > 0 or random.random() < np.exp(delta / T):
                    current_score = new_score
                    current_violations = new_violations
                    if new_score > best_score:
                        best_score = new_score
                        best_violations = new_violations
                        best_selected = selected.copy()
                else:
                    selected[idx] = True  # Revert
                    
            elif move_type == 'swap':
                # Try swapping a selected cell with an unselected cell
                sel_candidates = np.where(selected)[0]
                unsel_candidates = np.where(~selected)[0]
                if len(sel_candidates) == 0 or len(unsel_candidates) == 0:
                    continue
                idx_remove = random.choice(sel_candidates)
                idx_add = random.choice(unsel_candidates)
                
                selected[idx_remove] = False
                selected[idx_add] = True
                
                new_violations = count_violations(selected)
                new_score = np.sum(selected) - new_violations * 10
                
                delta = new_score - current_score
                if delta > 0 or random.random() < np.exp(delta / T):
                    current_score = new_score
                    current_violations = new_violations
                    if new_score > best_score:
                        best_score = new_score
                        best_violations = new_violations
                        best_selected = selected.copy()
                else:
                    selected[idx_remove] = True
                    selected[idx_add] = False  # Revert
            
            T *= cooling_rate
            step += 1
        
        # Return the best valid configuration found
        # If best has violations, we need to clean it up
        if best_violations > 0:
            # Greedily remove cells to eliminate violations
            best_selected = best_selected.copy()
            while count_violations(best_selected) > 0 and np.sum(best_selected) > 0:
                selected_indices = np.where(best_selected)[0]
                # Find a cell that's part of a violation and remove it
                # For simplicity, just remove random cells until valid
                idx = random.choice(selected_indices)
                best_selected[idx] = False
        
        # Convert back to list of (x, y) tuples
        selected_indices = np.where(best_selected)[0]
        points = [(idx // N, idx % N) for idx in selected_indices]
        
        return points, len(points)
    
    # Run multiple simulated annealing trials
    best_points = []
    best_count = 0
    trial_count = 0
    
    while time.time() < deadline - 30:
        trial_count += 1
        seed_val = trial_count * 7919 + 12345  # Different seed for each trial
        
        # Time limit for this trial: leave 30s margin
        time_limit = min(20.0, deadline - 30 - time.time())
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
