"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s limit
    
    # Generate all weight-4 vectors with ±1 entries
    # C(11,4) = 330 positions, 2^4 = 16 sign combos -> 5280 vectors
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int8)
    n_candidates = len(candidates)
    
    # Precompute the full dot product matrix (n x n)
    # Dot[i,j] = candidates[i] · candidates[j]
    # Constraint for valid pair: dot <= 2 (since norm^2 = 4, |u-v|^2 = 8-2*u.v >= 4 iff u.v <= 2)
    dot_matrix = candidates @ candidates.T  # (n, n) int array
    
    # Precompute conflict: conflict[i,j] = True if dot > 2
    conflict = (dot_matrix > 2)
    # Symmetric, diagonal is 4 > 2 so self-conflict, but we won't pick same index twice
    
    def greedy_select_vectorized(order, time_limit):
        """Fast greedy: for each candidate in order, check if it conflicts with any selected."""
        selected_indices = []
        # Track which candidates are still available (not conflicting with selected)
        # Use a boolean mask of compatible candidates
        available = np.ones(n_candidates, dtype=bool)
        
        for idx in order:
            if time.time() > time_limit:
                break
            if not available[idx]:
                continue
            # Select this vector
            selected_indices.append(idx)
            # Remove all candidates that conflict with this one
            available[conflict[idx]] = False
            # Also remove itself
            available[idx] = False
        
        return selected_indices
    
    def greedy_maxmin_vectorized(order, time_limit):
        """Max-min greedy: at each step, pick the available candidate that maximizes
        its minimum dot-product margin (i.e., minimizes max dot with selected)."""
        selected_indices = []
        available = np.ones(n_candidates, dtype=bool)
        
        # For efficiency, track max_dot_with_selected for each candidate
        # max_dot[i] = max over selected s of dot_matrix[s, i]
        max_dot = np.full(n_candidates, -10, dtype=np.int32)
        
        for step in range(len(order)):
            if time.time() > time_limit:
                break
            if not np.any(available):
                break
            
            # Find best candidate: among available, minimize max_dot (most negative = best)
            # Only consider candidates where max_dot <= 2 (compatible)
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break
            
            # Filter to compatible ones
            compatible = avail_indices[max_dot[avail_indices] <= 2]
            if len(compatible) == 0:
                break
            
            # Among compatible, pick the one with smallest max_dot (most room)
            best_idx = compatible[np.argmin(max_dot[compatible])]
            
            selected_indices.append(best_idx)
            # Update max_dot for all candidates
            max_dot = np.maximum(max_dot, dot_matrix[best_idx])
            available[best_idx] = False
            # Mark incompatible as unavailable
            available[conflict[best_idx]] = False
        
        return selected_indices
    
    def local_search_improve(selected_indices, time_limit, n_iterations=5000):
        """Try to improve the set by swapping or adding.
        Returns improved list of indices."""
        selected_set = set(selected_indices)
        # Build a quick lookup: for each candidate, is it compatible with all selected?
        # compatible[i] = all dot_matrix[s, i] <= 2 for s in selected_set
        sel_arr = np.array(selected_indices, dtype=np.int32)
        if len(sel_arr) == 0:
            return selected_indices
        
        # Compute max dot with selected for each candidate
        max_dot_sel = np.max(dot_matrix[sel_arr], axis=0)  # (n,)
        
        # Candidates that can be added
        addable = np.where(max_dot_sel <= 2)[0]
        addable = [i for i in addable if i not in selected_set]
        
        improved = list(selected_indices)
        improved_set = set(improved)
        
        # Try adding candidates one by one (greedy add)
        for idx in addable:
            if time.time() > time_limit:
                break
            if idx in improved_set:
                continue
            # Check compatibility with all current improved
            if np.all(dot_matrix[improved, idx] <= 2):
                improved.append(idx)
                improved_set.add(idx)
        
        # Try swap: remove one selected, add up to two new ones
        for _ in range(n_iterations):
            if time.time() > time_limit:
                break
            # Pick a random selected to remove
            rem_idx = improved[np.random.randint(len(improved))]
            rem_pos = improved.index(rem_idx)
            improved = improved[:rem_pos] + improved[rem_pos+1:]
            improved_set.discard(rem_idx)
            
            # Recompute max dot
            if len(improved) == 0:
                max_dot_sel = np.full(n_candidates, -10, dtype=np.int32)
            else:
                max_dot_sel = np.max(dot_matrix[np.array(improved, dtype=np.int32)], axis=0)
            
            # Find candidates that can now be added
            addable_now = np.where(max_dot_sel <= 2)[0]
            addable_now = [i for i in addable_now if i not in improved_set]
            
            if len(addable_now) >= 1:
                # Try to add the best few
                added = 0
                for idx in addable_now:
                    if time.time() > time_limit:
                        break
                    if idx in improved_set:
                        continue
                    if np.all(dot_matrix[improved, idx] <= 2):
                        improved.append(idx)
                        improved_set.add(idx)
                        added += 1
                    if added >= 2:
                        break
            
            # If we didn't improve, put the removed one back
            if len(improved) <= len(selected_indices) - 1:
                improved.append(rem_idx)
                improved_set.add(rem_idx)
        
        return improved
    
    # Main search
    best_selected_indices = []
    np.random.seed(42)
    
    # Strategy 1: Fixed order greedy (fast baseline)
    fixed_order = np.arange(n_candidates)
    sel = greedy_select_vectorized(fixed_order, deadline - 5.0)
    if len(sel) > len(best_selected_indices):
        best_selected_indices = sel
    
    # Strategy 2: Many random greedy restarts (vectorized, fast)
    n_trials = 200
    for trial in range(n_trials):
        if time.time() > deadline - 10.0:
            break
        perm = np.random.permutation(n_candidates)
        sel = greedy_select_vectorized(perm, deadline - 10.0)
        if len(sel) > len(best_selected_indices):
            best_selected_indices = sel
    
    # Strategy 3: Max-min greedy (slower per trial but may find better)
    if time.time() < deadline - 20.0:
        for trial in range(30):
            if time.time() > deadline - 10.0:
                break
            perm = np.random.permutation(n_candidates)
            sel = greedy_maxmin_vectorized(perm, deadline - 10.0)
            if len(sel) > len(best_selected_indices):
                best_selected_indices = sel
    
    # Strategy 4: Local search improvement on best found
    if time.time() < deadline - 5.0 and len(best_selected_indices) > 0:
        improved = local_search_improve(best_selected_indices, deadline - 2.0, n_iterations=10000)
        if len(improved) > len(best_selected_indices):
            best_selected_indices = improved
    
    if not best_selected_indices:
        # Fallback
        v = [0] * D
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)
    
    return candidates[best_selected_indices].astype(np.int64)
# EVOLVE-BLOCK-END
