import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points_total = len(grid)
    deadline = time.time() + 580  # Safety margin

    # Precompute squared norms
    grid_sq = (grid ** 2).sum(axis=1)

    def check_violation(S_idx, p_idx):
        """
        Checks if adding p_idx to S_idx creates a cospherical 5-tuple.
        Returns True if a violation is found (invalid), False if valid.
        """
        k = len(S_idx)
        if k < 4:
            return False
        
        S_pts = grid[S_idx]
        p = grid[p_idx]
        
        diff = S_pts - p
        sq_diff = grid_sq[S_idx] - grid_sq[p_idx]
        
        # Generate combinations of indices in S_idx (0 to k-1)
        # We only need to check if ANY combination of 4 points from S_idx 
        # forms a cospherical set with p.
        
        # For efficiency, we can vectorize the determinant calculation.
        # We need to check all C(k, 4) combinations.
        
        if k < 4:
            return False
            
        combos = list(itertools.combinations(range(k), 4))
        if not combos:
            return False
            
        combos_arr = np.array(combos, dtype=np.int32)
        
        # d_diff: (N, 4, 3)
        d_diff = diff[combos_arr]
        # d_sq: (N, 4)
        d_sq = sq_diff[combos_arr]
        
        # Combine to (N, 4, 4)
        mats = np.concatenate([d_diff, d_sq[:, :, None]], axis=2)
        
        dets = np.linalg.det(mats)
        
        return np.any(np.abs(dets) < 0.5)

    def is_valid_set(S_idx):
        """
        Checks if the entire set S_idx is valid (no 5 cospherical points).
        """
        # This is more expensive, O(k^5) if naive, but we can reuse logic.
        # Actually, we can just check all 5-tuples.
        # But for the swap logic, we only need to check if the NEW set is valid.
        # Since we only change 1 point, we only need to check 5-tuples involving the new point.
        # Wait, removing a point might break a violation that didn't involve the new point?
        # No, if the old set was valid, removing a point keeps it valid.
        # Adding a point p to a valid set S' only creates violations involving p.
        # So we only need to check 5-tuples containing p.
        # This is exactly what check_violation does.
        pass

    # 1. Greedy Construction
    best_set = []
    np.random.seed(42)
    
    # Try a few random orders for greedy
    for _ in range(5):
        if time.time() > deadline: break
        
        order = np.random.permutation(num_points_total)
        curr_set = []
        
        for p_idx in order:
            if time.time() > deadline: break
            if not check_violation(curr_set, p_idx):
                curr_set.append(p_idx)
        
        if len(curr_set) > len(best_set):
            best_set = curr_set.copy()

    current_set = best_set.copy()
    in_set = np.zeros(num_points_total, dtype=bool)
    for idx in current_set:
        in_set[idx] = True

    # 2. Guided Local Search
    # Calculate conflict scores for points in current_set
    # Conflict score of p: number of 4-subsets of S \ {p} that form a violation with p.
    
    def get_conflict_scores(S_idx):
        scores = np.zeros(len(S_idx), dtype=int)
        k = len(S_idx)
        if k < 5:
            return scores
            
        S_pts = grid[S_idx]
        S_sq = grid_sq[S_idx]
        
        # For each point i in S_idx, we need to count how many 4-subsets of S_idx \ {i} 
        # are cospherical with S_idx[i].
        
        # This is expensive: C(k-1, 4) determinants per point.
        # Total C(k, 4) * 4 determinants? No, each 4-subset is checked for each of its 5 points?
        # No, for point i, we check subsets of S \ {i}.
        
        # Optimization: Precompute all 5-tuples? No, too many.
        # Let's just do it directly if k is small enough.
        # k is around 40. C(39, 4) = 82,251.
        # 40 * 82,251 = 3.3 million determinants.
        # Each determinant is 4x4.
        # This might take a few seconds.
        
        for i in range(k):
            # Indices of other points
            others = [j for j in range(k) if j != i]
            if len(others) < 4:
                continue
                
            p = S_pts[i]
            p_sq = S_sq[i]
            
            # Subsets of others of size 4
            # We can vectorize this per point i?
            # Let's try.
            others_arr = np.array(others)
            other_pts = S_pts[others_arr]
            other_sq = S_sq[others_arr]
            
            diff = other_pts - p
            sq_diff = other_sq - p_sq
            
            combos = list(itertools.combinations(range(len(others)), 4))
            if not combos:
                continue
                
            combos_arr = np.array(combos, dtype=np.int32)
            d_diff = diff[combos_arr]
            d_sq = sq_diff[combos_arr]
            mats = np.concatenate([d_diff, d_sq[:, :, None]], axis=2)
            dets = np.linalg.det(mats)
            
            scores[i] = np.sum(np.abs(dets) < 0.5)
            
        return scores

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Get conflict scores
        scores = get_conflict_scores(current_set)
        
        # Find the point with the highest conflict score
        if len(scores) > 0:
            max_conf_idx = np.argmax(scores)
            if scores[max_conf_idx] > 0:
                # Try to swap out current_set[max_conf_idx]
                p_out_idx = current_set[max_conf_idx]
                
                # Try to find a p_in from unused points
                # We want to try points that are "compatible" with the rest of the set.
                # Let's just try random unused points first, or sorted by distance to p_out?
                # Let's try a few random unused points.
                
                unused_indices = np.where(~in_set)[0]
                np.random.shuffle(unused_indices)
                
                found_swap = False
                for p_in_idx in unused_indices[:50]: # Try top 50 random candidates
                    if time.time() > deadline: break
                    
                    # Check if (current_set - p_out) + p_in is valid
                    # This is equivalent to checking if p_in is valid with (current_set - p_out)
                    reduced_set = [x for x in current_set if x != p_out_idx]
                    
                    if not check_violation(reduced_set, p_in_idx):
                        # Valid swap!
                        current_set.remove(p_out_idx)
                        current_set.append(p_in_idx)
                        in_set[p_out_idx] = False
                        in_set[p_in_idx] = True
                        improved = True
                        found_swap = True
                        break
                        
                if found_swap:
                    continue

        # If no swap found, try to add a point?
        # The set size is fixed in swap. We should also try to grow the set.
        # But if we are stuck, maybe we need to remove 2 and add 3?
        # For now, let's try adding a point if possible.
        
        added = False
        unused_indices = np.where(~in_set)[0]
        np.random.shuffle(unused_indices)
        for p_in_idx in unused_indices[:50]:
            if time.time() > deadline: break
            if not check_violation(current_set, p_in_idx):
                current_set.append(p_in_idx)
                in_set[p_in_idx] = True
                improved = True
                added = True
                break
                
        if not added and not improved:
            # Stuck. Try a random kick?
            # Remove a random point and try to add 2?
            # This is complex. Let's just break or do random restarts.
            # Let's do a few random restarts of the greedy phase.
            if len(current_set) < 45: # If we are not doing well
                 # Restart greedy with a new seed
                 pass # Handled by outer loop? No, this is inner.
                 # Let's just break and let the outer loop handle it if we had one.
                 # Since we don't have an outer loop for restarts in this phase, 
                 # we can just break and return.
                 break

    return grid[np.array(current_set)]
