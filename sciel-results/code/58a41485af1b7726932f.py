# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Finds a large set of points in the 8x8x8 grid such that no 5 points are co-spherical.
    Uses greedy construction with random restarts, followed by local search with 
    systematic single-swap feasibility checks.
    """
    n = 8
    deadline = time.time() + 590.0  # Safety margin before 630s kill

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1)  # |p|^2 for each grid point
    num_points_total = len(grid)
    
    # Precompute lifted coordinates for all grid points?
    # We need |p|^2 and p.
    
    def get_lifted(p: np.ndarray) -> np.ndarray:
        """Returns the lifted row for point p: (p, |p|^2)"""
        return np.append(p, (p ** 2).sum())

    def can_add(arr: np.ndarray, p: np.ndarray, arr_sq: np.ndarray) -> bool:
        """
        Check if point p can be added to arr without creating 5 co-spherical points.
        """
        m = len(arr)
        if m < 4:
            return True
        
        # Build lifted matrix relative to p
        # Rows: (q_i - p, |q_i|^2 - |p|^2)
        # Shape: (m, 4)
        p_sq = (p ** 2).sum()
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq - p_sq
        
        # Generate all combinations of 4 indices
        combos = list(itertools.combinations(range(m), 4))
        if not combos:
            return True
            
        combo_arr = np.array(combos, dtype=np.int64)
        submats = w[combo_arr]
        dets = np.linalg.det(submats)
        
        if np.any(np.abs(dets) < 0.5):
            return False
            
        return True

    def find_bad_subsets(arr: np.ndarray, arr_sq: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Returns indices of 4-subsets of arr that form a bad sphere with p.
        """
        m = len(arr)
        if m < 4:
            return np.array([], dtype=np.int64)
        
        p_sq = (p ** 2).sum()
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq - p_sq
        
        combos = list(itertools.combinations(range(m), 4))
        if not combos:
            return np.array([], dtype=np.int64)
            
        combo_arr = np.array(combos, dtype=np.int64)
        submats = w[combo_arr]
        dets = np.linalg.det(submats)
        
        bad_mask = np.abs(dets) < 0.5
        return combo_arr[bad_mask]

    def greedy_fill(order: np.ndarray) -> tuple:
        """Greedy fill with given order."""
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.float64)
        for idx in order:
            if time.time() > deadline - 10:
                break
            p = grid[idx]
            if can_add(arr, p, arr_sq):
                if len(arr) == 0:
                    arr = p[None, :]
                    arr_sq = np.array([grid_sq[idx]], dtype=np.float64)
                else:
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, grid_sq[idx])
        return arr, arr_sq

    def local_search(arr: np.ndarray, arr_sq: np.ndarray) -> tuple:
        """
        Local search: Try to add points, then try systematic swaps.
        """
        in_arr_set = set(map(tuple, arr))
        remaining_indices = [i for i in range(num_points_total) if tuple(grid[i]) not in in_arr_set]
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            m = len(arr)
            
            # Phase 1: Try to add a point
            # We can check all remaining points for validity.
            # To speed up, we can just try them in random order and stop at first success.
            # Or check all and pick the one that allows future growth? 
            # For now, random order.
            np.random.shuffle(remaining_indices)
            for idx in remaining_indices:
                p = grid[idx]
                if can_add(arr, p, arr_sq):
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, grid_sq[idx])
                    in_arr_set.add(tuple(p))
                    remaining_indices.remove(idx)
                    improved = True
                    break # Restart add phase
            
            # Phase 2: Try systematic swaps if no add was possible
            if not improved and len(remaining_indices) > 0 and m > 0:
                # For each candidate p_out, check if a swap is possible.
                # We precompute the bad subsets for each p_out? 
                # That's expensive. 
                # Instead, for each p_out, find bad subsets in current arr.
                # If bad subsets intersect at a single point p_in, swap is valid.
                
                # To avoid O(|Remaining| * C(m,4)) which is slow, we can sample.
                # But let's try a limited systematic pass.
                # Shuffle remaining to vary the search.
                np.random.shuffle(remaining_indices)
                
                swap_found = False
                for out_idx in remaining_indices:
                    p_out = grid[out_idx]
                    bad_subs = find_bad_subsets(arr, arr_sq, p_out)
                    
                    if len(bad_subs) == 0:
                        # Can add directly? But Phase 1 failed, so this shouldn't happen.
                        # Unless Phase 1 didn't check this specific index? 
                        # Phase 1 checks all. So this is redundant.
                        pass
                    elif len(bad_subs) > 0:
                        # Check intersection of all bad subsets
                        # bad_subs is (K, 4)
                        # We want to know if there is a single element common to all rows.
                        
                        # Efficient check:
                        # For each column j in 0..3, check if all rows have the same value at j.
                        # If so, that value is the candidate p_in.
                        # Then verify that this p_in is in ALL rows (which is true if all rows have same value at j)
                        # AND that no other element is common to all rows?
                        # Actually, if all rows have the same value v at column j, then v is in the intersection.
                        # We need the intersection to be EXACTLY {v}.
                        
                        # Let's just compute the intersection.
                        # Since m is small (< 100), we can use sets.
                        # But bad_subs can be large.
                        
                        # Optimization: 
                        # If any column is constant, that value is a candidate.
                        # There can be at most 4 such candidates (one per column).
                        # Actually, if col 0 is constant v0, and col 1 is constant v1, then v0==v1 must hold for intersection to be non-empty?
                        # No, the intersection is the set of points that appear in ALL bad subsets.
                        
                        # Let's find the intersection of the sets represented by rows of bad_subs.
                        # We can do this by taking the first row as a set, and intersecting with others.
                        # But rows are tuples of indices.
                        
                        # Faster: 
                        # Count frequency of each index in bad_subs.
                        # An index is in the intersection if its count == len(bad_subs).
                        
                        flat_bad = bad_subs.flatten()
                        unique, counts = np.unique(flat_bad, return_counts=True)
                        intersection_indices = unique[counts == len(bad_subs)]
                        
                        if len(intersection_indices) == 1:
                            p_in_idx = intersection_indices[0]
                            p_in = arr[p_in_idx]
                            
                            # Perform swap
                            # Remove p_in
                            mask = np.ones(m, dtype=bool)
                            mask[p_in_idx] = False
                            arr = arr[mask]
                            arr_sq = arr_sq[mask]
                            
                            # Add p_out
                            arr = np.vstack([arr, p_out])
                            arr_sq = np.append(arr_sq, grid_sq[out_idx])
                            
                            # Update sets
                            in_arr_set.remove(tuple(p_in))
                            in_arr_set.add(tuple(p_out))
                            
                            # Update remaining
                            remaining_indices.remove(out_idx)
                            # Find grid index of p_in
                            # We can maintain a mapping or search.
                            # Since grid is small, search is ok.
                            in_grid_idx = np.where((grid == p_in).all(axis=1))[0][0]
                            if in_grid_idx not in remaining_indices:
                                remaining_indices.append(in_grid_idx)
                            
                            improved = True
                            swap_found = True
                            break # Restart local search loop

    np.random.seed(42)
    best_arr = np.empty((0, 3), dtype=np.int64)
    best_len = 0
    
    # Number of restarts depends on time.
    for restart in range(200):
        if time.time() > deadline - 30:
            break
        
        order = np.random.permutation(num_points_total)
        
        # Greedy phase
        curr_arr, curr_arr_sq = greedy_fill(order)
        
        # Local search phase
        curr_arr, curr_arr_sq = local_search(curr_arr, curr_arr_sq)
        
        if len(curr_arr) > best_len:
            best_arr = curr_arr.copy()
            best_len = len(curr_arr)
            
            if best_len > 100:
                break

    return best_arr.astype(np.int64)

# EVOLVE-BLOCK-END
