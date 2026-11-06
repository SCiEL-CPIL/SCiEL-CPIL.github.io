import numpy as np
import time
import random

N = 64
TOTAL_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    # Precompute coordinates
    coords = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)
    
    # Precompute distance matrix
    # This is 4096x4096 int32.
    dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    dist_matrix = dx*dx + dy*dy
    
    def is_valid_add(p, selected_indices, dist_sets):
        """
        Check if p can be added to the set defined by selected_indices.
        selected_indices: list of indices
        dist_sets: dict index -> set of distances
        """
        if not selected_indices:
            return True
        
        sel_np = np.array(selected_indices)
        dists = dist_matrix[p, sel_np]
        
        # Check 1: Distances from p to all selected must be distinct
        # Using a set is O(N)
        if len(set(dists.tolist())) != len(dists):
            return False
        
        # Check 2: Distances must not conflict with existing distances from selected points
        # For each selected point s, dist(p, s) must not be in dist_sets[s]
        for i, s in enumerate(selected_indices):
            if dists[i] in dist_sets[s]:
                return False
                
        return True

    def add_point(p, selected_indices, dist_sets):
        """
        Add p to the set. Assumes is_valid_add is true.
        Returns updated selected_indices and dist_sets.
        """
        new_selected = selected_indices + [p]
        
        # Calculate distances from p to all in new_selected (excluding p)
        # We already have dists from p to old selected
        sel_np = np.array(selected_indices)
        if selected_indices:
            dists = dist_matrix[p, sel_np]
        else:
            dists = np.array([])
        
        # Update dist_sets for p
        dist_sets[p] = set(dists.tolist())
        
        # Update dist_sets for existing points
        for i, s in enumerate(selected_indices):
            dist_sets[s].add(dists[i])
            
        return new_selected, dist_sets

    def try_greedy(seed):
        random.seed(seed)
        candidates = list(range(TOTAL_POINTS))
        random.shuffle(candidates)
        
        selected = []
        dist_sets = {}
        
        for p in candidates:
            if is_valid_add(p, selected, dist_sets):
                selected, dist_sets = add_point(p, selected, dist_sets)
                
        return selected, dist_sets

    def local_search(selected, dist_sets, time_limit):
        """
        Try to improve the set via swaps.
        """
        # Create a list of points not in selected
        selected_set = set(selected)
        non_selected = [p for p in range(TOTAL_POINTS) if p not in selected_set]
        
        # We will try to swap each point in selected with a point in non_selected
        # To be efficient, we might just try a few random swaps.
        
        # Sort non_selected by some heuristic? e.g., distance to center?
        # Just random for now.
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Shuffle selected to try different orders
            sel_copy = selected[:]
            random.shuffle(sel_copy)
            
            for p_out in sel_copy:
                if time.time() >= time_limit:
                    break
                
                # Temporarily remove p_out
                # We need to reconstruct dist_sets without p_out?
                # Or just simulate?
                # Simulation is complex. Let's just remove and re-add.
                
                # Remove p_out
                # This requires updating dist_sets for all other points.
                # This is O(N^2).
                
                # Alternative: Just try to add a new point to the set, 
                # if it fails, try removing one and adding another.
                
                # Let's stick to a simpler local search:
                # Try to add a random point from non_selected.
                # If it fails, pick a random point from selected, remove it, 
                # then try to add the new point.
                
                # Pick a candidate to add
                if not non_selected:
                    break
                    
                p_in = random.choice(non_selected)
                
                # Try adding p_in directly
                if is_valid_add(p_in, selected, dist_sets):
                    selected, dist_sets = add_point(p_in, selected, dist_sets)
                    non_selected.remove(p_in)
                    improved = True
                    continue
                
                # Try swapping
                # Pick a random point to remove
                if not selected:
                    break
                p_out = random.choice(selected)
                
                # Remove p_out
                # We need to update dist_sets.
                # Let's create a helper to remove.
                
                # Actually, removing is expensive.
                # Let's just restart greedy with a bias?
                # No, local search is better.
                
                # Implement remove_point
                # For each s in selected (excluding p_out), remove dist(p_out, s) from dist_sets[s]
                # Delete dist_sets[p_out]
                
                # This is O(N).
                
                # Let's do it.
                
                # 1. Remove p_out from selected
                selected.remove(p_out)
                
                # 2. Update dist_sets
                # Get distances from p_out to all remaining selected
                if selected:
                    sel_np = np.array(selected)
                    dists = dist_matrix[p_out, sel_np]
                    for i, s in enumerate(selected):
                        dist_sets[s].discard(dists[i])
                del dist_sets[p_out]
                
                # 3. Try to add p_in
                if is_valid_add(p_in, selected, dist_sets):
                    selected, dist_sets = add_point(p_in, selected, dist_sets)
                    non_selected.remove(p_in)
                    # p_out is now in non_selected?
                    non_selected.append(p_out)
                    improved = True
                else:
                    # Revert: Add p_out back
                    selected, dist_sets = add_point(p_out, selected, dist_sets)
                    # Note: add_point assumes it's valid. p_out was valid before.
                    # But we removed it and didn't add p_in.
                    # So adding p_out back should be valid.
                    
        return selected, dist_sets

    best_set = []
    best_dist_sets = {}
    restart_count = 0
    
    while time.time() < deadline:
        # Run greedy
        selected, dist_sets = try_greedy(restart_count)
        
        # Run local search if we have time
        # Allocate remaining time - 5s
        local_time_limit = deadline - 5
        if time.time() < local_time_limit and len(selected) > 0:
            selected, dist_sets = local_search(selected, dist_sets, local_time_limit)
            
        if len(selected) > len(best_set):
            best_set = selected
            best_dist_sets = dist_sets
            # print(f"New best: {len(best_set)}")
            
        restart_count += 1
        
        # If we are close to deadline, stop
        if time.time() > deadline - 10:
            break

    # Convert indices to (x, y) tuples
    return [tuple(coords[i]) for i in best_set]
