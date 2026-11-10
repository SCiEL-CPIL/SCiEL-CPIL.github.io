import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points_total = len(grid)
    deadline = time.time() + 590
    
    # Precompute squared norms for all grid points
    grid_sq_norms = (grid ** 2).sum(axis=1)
    
    # Map from tuple to index for fast lookup
    point_to_idx = {tuple(p): i for i, p in enumerate(grid)}
    
    def count_violations_for_point(p_idx, S_indices):
        """
        Count how many 4-subsets of S_indices are cospherical with p_idx.
        Returns the count.
        """
        m = len(S_indices)
        if m < 4:
            return 0
            
        p = grid[p_idx]
        p_sq = grid_sq_norms[p_idx]
        
        # Get the points in S
        S_pts = grid[S_indices]
        S_sqs = grid_sq_norms[S_indices]
        
        # Differences and squared norm diffs
        diffs = S_pts - p
        sq_diffs = S_sqs - p_sq
        
        # We need to check all combinations of 4
        # If m is small, we can just loop. If m is large, we can batch.
        # For m up to ~30, binom(m,4) is < 30k.
        # Let's generate the combinations of indices in S_indices
        # But we need to map these back to the rows of diffs/sq_diffs.
        
        # Actually, it's easier to work with the local indices 0..m-1
        # and then map to global indices if needed, but we only need the values.
        
        count = 0
        # Optimization: If m is small, itertools is fine.
        # If m is large, we might need to be careful.
        # Let's try vectorizing the determinant calculation for all combos at once if possible.
        # However, creating a huge array of matrices might be memory heavy.
        # For m=20, 4845 matrices of 4x4 is small.
        
        combos = list(itertools.combinations(range(m), 4))
        if not combos:
            return 0
            
        # Build the stack of matrices
        # Shape: (num_combos, 4, 4)
        num_combos = len(combos)
        mats = np.empty((num_combos, 4, 4), dtype=np.float64)
        
        for k, combo in enumerate(combos):
            for r, idx in enumerate(combo):
                mats[k, r, 0:3] = diffs[idx]
                mats[k, r, 3] = sq_diffs[idx]
                
        dets = np.linalg.det(mats)
        count = np.sum(np.abs(dets) < 0.5)
        return count

    def get_energy(S_indices):
        """
        Calculate total energy (number of bad 5-tuples) for set S_indices.
        This is O(m^5) if done naively, but we can use the incremental approach in SA.
        For initialization, we can just sum count_violations for each point?
        No, that would double count.
        A bad 5-tuple is counted 5 times if we sum for each point.
        So Energy = sum(count_violations_for_point(p, S \ {p})) / 5.
        """
        m = len(S_indices)
        if m < 5:
            return 0
        total = 0
        for i in range(m):
            p_idx = S_indices[i]
            rest = [S_indices[j] for j in range(m) if j != i]
            total += count_violations_for_point(p_idx, rest)
        return total // 5

    # --- Phase 1: Greedy Initialization ---
    best_set = np.empty((0, 3), dtype=np.int64)
    np.random.seed(42)
    
    # Try a few greedy restarts to get a good starting point
    for restart in range(10):
        if time.time() > deadline: break
        
        order = np.random.permutation(num_points_total)
        curr_set_indices = []
        
        for p_idx in order:
            # Check if p_idx can be added
            # We need to check if it creates any violations with the current set
            # Violations are 5-tuples. Adding p creates violations if there are 4 points in curr_set
            # that are cospherical with p.
            v = count_violations_for_point(p_idx, curr_set_indices)
            if v == 0:
                curr_set_indices.append(p_idx)
        
        curr_set = np.array([grid[i] for i in curr_set_indices])
        if len(curr_set) > len(best_set):
            best_set = curr_set.copy()
            
    # --- Phase 2: Simulated Annealing ---
    # State: list of indices
    current_indices = [point_to_idx[tuple(p)] for p in best_set]
    current_energy = get_energy(current_indices)
    
    best_indices = current_indices.copy()
    best_energy = current_energy
    best_size = len(current_indices)
    
    # Temperature parameters
    T = 10.0
    cooling_rate = 0.99
    min_T = 0.01
    max_steps = 50000
    
    # Precompute list of all indices for sampling
    all_indices = list(range(num_points_total))
    
    steps = 0
    while time.time() < deadline and T > min_T:
        # Choose a move
        move_type = np.random.choice(3, p=[0.4, 0.3, 0.3]) # Add, Remove, Swap
        
        if move_type == 0: # Add
            if len(current_indices) >= num_points_total: continue
            # Sample a point not in current set
            # Efficient sampling: pick random index, check if in set
            # For small sets, set lookup is fast.
            curr_set_set = set(current_indices)
            while True:
                p_idx = np.random.choice(num_points_total)
                if p_idx not in curr_set_set:
                    break
            
            # Calculate delta energy
            # New energy = old energy + count_violations_for_point(p_idx, current_indices)
            # Note: Adding p might also break existing valid 5-tuples? No, adding a point only ADDS tuples.
            # It doesn't change the cosphericity of existing tuples.
            delta_e = count_violations_for_point(p_idx, current_indices)
            
            # Acceptance probability
            if delta_e < 0:
                accept = True
            else:
                accept = np.random.random() < np.exp(-delta_e / T)
            
            if accept:
                current_indices.append(p_idx)
                current_energy += delta_e
                
        elif move_type == 1: # Remove
            if len(current_indices) <= 1: continue
            # Pick a random point to remove
            rem_idx_pos = np.random.randint(len(current_indices))
            p_idx = current_indices[rem_idx_pos]
            
            # Calculate delta energy
            # Removing p removes all tuples containing p.
            # Number of tuples containing p is count_violations_for_point(p_idx, current_indices \ {p_idx})
            # Wait, the energy counts bad 5-tuples.
            # If we remove p, we remove all bad 5-tuples that contain p.
            # The number of such tuples is exactly the number of 4-subsets of the remaining set
            # that are cospherical with p.
            rest_indices = [current_indices[j] for j in range(len(current_indices)) if j != rem_idx_pos]
            removed_violations = count_violations_for_point(p_idx, rest_indices)
            delta_e = -removed_violations
            
            if delta_e < 0:
                accept = True
            else:
                accept = np.random.random() < np.exp(-delta_e / T)
                
            if accept:
                current_indices.pop(rem_idx_pos)
                current_energy += delta_e
                
        else: # Swap
            if len(current_indices) == 0: continue
            # Pick point to remove
            rem_idx_pos = np.random.randint(len(current_indices))
            p_out = current_indices[rem_idx_pos]
            
            # Pick point to add
            curr_set_set = set(current_indices)
            while True:
                p_in = np.random.choice(num_points_total)
                if p_in not in curr_set_set:
                    break
                    
            # Calculate delta energy
            # New set = (S \ {p_out}) U {p_in}
            # E_new = E(S) - (tuples containing p_out) + (tuples containing p_in in S \ {p_out})
            # Note: Tuples containing both p_out and p_in are not possible since p_in not in S.
            # So the sets of tuples are disjoint.
            
            rest_indices = [current_indices[j] for j in range(len(current_indices)) if j != rem_idx_pos]
            
            # Tuples containing p_out in S
            # These are 4-subsets of rest_indices cospherical with p_out
            e_out = count_violations_for_point(p_out, rest_indices)
            
            # Tuples containing p_in in S \ {p_out}
            # These are 4-subsets of rest_indices cospherical with p_in
            e_in = count_violations_for_point(p_in, rest_indices)
            
            delta_e = e_in - e_out
            
            if delta_e < 0:
                accept = True
            else:
                accept = np.random.random() < np.exp(-delta_e / T)
                
            if accept:
                current_indices[rem_idx_pos] = p_in
                current_energy += delta_e
        
        # Update best
        if current_energy == 0 and len(current_indices) > best_size:
            best_size = len(current_indices)
            best_indices = current_indices.copy()
            best_energy = 0
            
        # Cool down
        T *= cooling_rate
        steps += 1
        
        # Optional: If we are stuck at a high energy for a long time, maybe restart?
        # Or just let it run.
        
    # Construct final array from best_indices
    best_points = np.array([grid[i] for i in best_indices], dtype=np.int64)
    
    # Final verification and cleanup
    # Ensure no duplicates (should be guaranteed by set logic)
    # Ensure valid coordinates (guaranteed by grid)
    
    return best_points
