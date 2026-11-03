"""Improved program: Expanded candidate pool + Max-Min Greedy + Remove-Reoptimize Local Search."""
import numpy as np
import itertools
import time


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    D = 11

    # Generate weight-4 vectors (norm=2): C(11,4)*16 = 5280
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Generate weight-2 vectors (norm=sqrt(2)): C(11,2)*4 = 220
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)  # 5500

    # Precompute norms squared for each candidate
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)
    max_norm_sq = norms_sq.max()

    # Precompute the full dot product matrix
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)

    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    # => v_i·v_j <= (||v_i||^2 + ||v_j||^2 - max_norm_sq) / 2
    norm_i = norms_sq[:, np.newaxis]
    norm_j = norms_sq[np.newaxis, :]
    threshold = (norm_i + norm_j - max_norm_sq) / 2.0
    compat = (dot <= threshold).astype(np.uint8)
    np.fill_diagonal(compat, 1)

    # Helper: Greedy Add
    # Given a set of selected indices (as a boolean mask), try to add more compatible vectors.
    # Returns the updated mask and the count of additions.
    def greedy_add(selected_mask, time_limit):
        """
        selected_mask: bool array of size n
        Returns: new_selected_mask
        """
        # Compute compatibility with ALL selected
        # If any selected is incompatible, it's not compatible with all.
        # compat_with_all[i] = 1 if compat[i, j] == 1 for all j where selected_mask[j]==1
        
        # Efficient way: 
        # For each i, check if (compat[i] & ~selected_mask).any() is False? No.
        # Check if (compat[i] & selected_mask).sum() == selected_mask.sum()
        
        # Vectorized:
        # sum_compat = (compat * selected_mask[np.newaxis, :]).sum(axis=1)
        # is_compatible_with_all = (sum_compat == selected_mask.sum()) & (~selected_mask)
        
        # This is O(N^2) per call if done naively.
        # But we can maintain a "conflict" count.
        # Let's stick to the O(N^2) check but only do it a few times or optimize.
        
        # Actually, since N=5500, N^2 is 30M ops. In numpy, this is very fast (~10-50ms).
        # We can run this loop many times.
        
        added_count = 0
        while time.time() < time_limit:
            if selected_mask.sum() == n:
                break
            
            # Check which unselected vectors are compatible with all selected
            # sum_compat[i] is the number of selected vectors compatible with i
            sum_compat = (compat * selected_mask[np.newaxis, :]).sum(axis=1)
            target_count = selected_mask.sum()
            
            # Candidates that are compatible with all selected
            potential = (sum_compat == target_count) & (~selected_mask)
            
            if not potential.any():
                break
            
            # Pick one. To be greedy, pick the one with highest degree in the remaining graph?
            # Or just pick the first one.
            # Let's pick the one with max degree among potentials to be "max-min" style.
            # deg[i] = sum(compat[i] & potential)
            # But potential is a mask.
            # We can just pick the first one to be fast, or randomize.
            # Let's pick the one with max degree in the full graph among potentials.
            degrees = (compat * potential[np.newaxis, :]).sum(axis=1)
            # Only consider potential ones
            degrees[~potential] = -1
            best_idx = np.argmax(degrees)
            
            if best_idx == -1 or not potential[best_idx]:
                break
                
            selected_mask[best_idx] = True
            added_count += 1
            
            # Optimization: If we added a lot, we might break early if we hit time limit
            # But the loop condition handles it.
            
        return selected_mask, added_count

    def local_search_remove_reoptimize(selected_indices, time_limit):
        """
        Start with a set of indices.
        Try removing each one and re-optimizing the rest.
        If the new set is larger, keep it.
        """
        selected_mask = np.zeros(n, dtype=bool)
        selected_mask[selected_indices] = True
        
        best_size = len(selected_indices)
        improved = True
        
        while improved and time.time() < time_limit:
            improved = False
            current_size = selected_mask.sum()
            
            # Try removing each selected index
            # To speed up, we can iterate through selected indices
            sel_idx_list = np.where(selected_mask)[0]
            
            # Randomize order to avoid bias? Or just iterate.
            # Iterating 5000 times * O(N^2) check might be too slow if O(N^2) is heavy.
            # O(N^2) is ~30M ops. 5000 * 30M = 150B ops. Too slow.
            
            # We need to be smarter.
            # Only try removing indices that are "bottlenecks".
            # Or, limit the number of removal attempts per epoch.
            
            # Alternative: Randomly pick a selected index to remove, re-optimize, check size.
            # This is a standard local search move.
            
            num_attempts = 0
            max_attempts_per_epoch = 100 # Adjust based on time
            
            while num_attempts < max_attempts_per_epoch and time.time() < time_limit:
                if current_size == 0:
                    break
                
                # Pick a random selected index to remove
                rand_idx = np.random.choice(sel_idx_list)
                
                # Remove it
                selected_mask[rand_idx] = False
                
                # Re-optimize (Greedy Add)
                new_mask, added = greedy_add(selected_mask.copy(), time_limit)
                new_size = new_mask.sum()
                
                if new_size > current_size:
                    selected_mask = new_mask
                    current_size = new_size
                    best_size = current_size
                    improved = True
                    # Update sel_idx_list
                    sel_idx_list = np.where(selected_mask)[0]
                else:
                    # Restore
                    selected_mask[rand_idx] = True
                
                num_attempts += 1
                
        return np.where(selected_mask)[0]

    # Strategy:
    # 1. Initial Greedy (Max-Min)
    # 2. Local Search (Remove-Reoptimize)
    
    np.random.seed(42)
    
    # 1. Initial Greedy
    def greedy_maxmin_initial(time_limit):
        remaining = np.ones(n, dtype=np.uint8)
        selected = []
        while time.time() < time_limit:
            if not remaining.any():
                break
            deg = (compat * remaining[np.newaxis, :]).sum(axis=1).astype(np.int32)
            deg[remaining == 0] = -1
            max_deg = deg.max()
            if max_deg < 0:
                break
            candidates_with_max = np.where((deg == max_deg) & (remaining > 0))[0]
            if len(candidates_with_max) == 0:
                break
            new_idx = candidates_with_max[0]
            selected.append(new_idx)
            remaining = remaining & compat[new_idx]
            remaining[new_idx] = 0
        return selected

    best_selected = greedy_maxmin_initial(deadline - 60.0)
    
    # 2. Local Search
    if time.time() < deadline - 10.0:
        improved_selected = local_search_remove_reoptimize(best_selected, deadline - 10.0)
        if len(improved_selected) > len(best_selected):
            best_selected = improved_selected

    # 3. If time remains, do more restarts or more local search
    # Let's do a few random restarts of the greedy + local search if time permits
    # But the current greedy is deterministic. Let's add randomness to the greedy tie-breaking.
    
    for trial in range(20):
        if time.time() > deadline - 20.0:
            break
            
        # Randomized Greedy
        remaining = np.ones(n, dtype=np.uint8)
        selected = []
        # Shuffle the "priority" by adding noise to degrees?
        # Or just randomize the order of candidates_with_max.
        
        while time.time() < deadline - 20.0:
            if not remaining.any():
                break
            deg = (compat * remaining[np.newaxis, :]).sum(axis=1).astype(np.int32)
            deg[remaining == 0] = -1
            max_deg = deg.max()
            if max_deg < 0:
                break
            candidates_with_max = np.where((deg == max_deg) & (remaining > 0))[0]
            if len(candidates_with_max) == 0:
                break
            # Random pick from max degree candidates
            new_idx = np.random.choice(candidates_with_max)
            selected.append(new_idx)
            remaining = remaining & compat[new_idx]
            remaining[new_idx] = 0
            
        # Local Search on this new start
        if time.time() < deadline - 10.0:
            improved_selected = local_search_remove_reoptimize(selected, deadline - 10.0)
            if len(improved_selected) > len(best_selected):
                best_selected = improved_selected

    if not best_selected:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    result = candidates[np.array(best_selected, dtype=np.int64)]
    return result.astype(np.int64)
