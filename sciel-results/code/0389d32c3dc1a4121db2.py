import numpy as np
import itertools
import time
import random

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    deadline = time.time() + 590
    
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    sq_norms = (grid ** 2).sum(axis=1)
    point_to_idx = {tuple(p): i for i, p in enumerate(grid)}
    
    def check_validity(indices):
        """Check if a set of point indices is valid (no 5 co-spherical/co-planar)."""
        m = len(indices)
        if m < 5:
            return True
        pts = grid[indices]
        sqs = sq_norms[indices]
        # For each pair of 5 points, check determinant
        # Use the 4x4 matrix test: for points p0..p4, 
        # matrix rows are (p_i - p_0, |p_i|^2 - |p_0|^2) for i=1..4
        for combo in itertools.combinations(range(m), 5):
            p0 = pts[combo[0]]
            s0 = sqs[combo[0]]
            M = np.zeros((4, 4), dtype=np.float64)
            for j in range(1, 5):
                pj = pts[combo[j]]
                M[j-1, 0:3] = pj - p0
                M[j-1, 3] = sqs[combo[j]] - s0
            if abs(np.linalg.det(M)) < 0.5:
                return False
        return True
    
    def can_add_fast(points_arr, sqs_arr, p_idx):
        """Check if adding grid[p_idx] to points_arr (with corresponding sqs_arr) is valid."""
        m = len(points_arr)
        if m < 4:
            return True
        
        p = grid[p_idx]
        ps = sq_norms[p_idx]
        
        # Build matrix of relative vectors and norm differences
        # For each existing point q: row = [q - p, |q|^2 - |p|^2]
        rel = points_arr - p  # (m, 3)
        norm_diffs = sqs_arr - ps  # (m,)
        
        # We need to check all 4-combinations of the m points
        # For a combination of 4 points, check if det of 4x4 matrix < 0.5
        # Optimize: if m is small, use itertools; if large, sample
        
        if m <= 50:
            for combo in itertools.combinations(range(m), 4):
                sub = np.column_stack([rel[list(combo)], norm_diffs[list(combo)]])
                if abs(np.linalg.det(sub)) < 0.5:
                    return False
            return True
        else:
            # For large m, use random sampling to speed up
            n_samples = min(2000, len(list(itertools.combinations(range(m), 4))))
            # But this is expensive to count. Let's just check all if m <= 80
            if m <= 80:
                for combo in itertools.combinations(range(m), 4):
                    sub = np.column_stack([rel[list(combo)], norm_diffs[list(combo)]])
                    if abs(np.linalg.det(sub)) < 0.5:
                        return False
                return True
            else:
                # Random sampling for very large sets
                for _ in range(n_samples):
                    combo = random.sample(range(m), 4)
                    sub = np.column_stack([rel[combo], norm_diffs[combo]])
                    if abs(np.linalg.det(sub)) < 0.5:
                        return False
                # If no violation found in samples, assume valid (risky but fast)
                return True
    
    # Phase 1: Build initial set using greedy with multiple strategies
    best_indices = None
    best_size = 0
    
    # Strategy 1: Greedy by various orderings
    orders_list = []
    # Random orders
    rng = np.random.default_rng(42)
    for i in range(30):
        orders_list.append(rng.permutation(N))
    
    # Structured orders
    orders_list.append(np.argsort(grid[:, 0] * 1000 + grid[:, 1] * 100 + grid[:, 2]))
    orders_list.append(np.argsort(-sq_norms))
    orders_list.append(np.argsort(sq_norms))
    
    # Space-filling curve order
    order_morton = np.argsort(grid[:, 0] + grid[:, 1] * 8 + grid[:, 2] * 64)
    orders_list.append(order_morton)
    
    for order in orders_list:
        if time.time() > deadline - 300:
            break
        
        current = []
        pts_arr = np.empty((0, 3), dtype=np.int64)
        sqs_arr = np.empty(0, dtype=np.int64)
        
        for idx in order:
            if can_add_fast(pts_arr, sqs_arr, idx):
                current.append(idx)
                pts_arr = np.vstack([pts_arr, grid[idx]]) if len(pts_arr) > 0 else grid[idx].reshape(1, 3)
                sqs_arr = np.append(sqs_arr, sq_norms[idx])
        
        if len(current) > best_size:
            best_size = len(current)
            best_indices = current.copy()
    
    if best_indices is None or len(best_indices) == 0:
        return np.empty((0, 3), dtype=np.int64)
    
    current_indices = best_indices.copy()
    
    # Phase 2: Local search to improve
    pts_arr = grid[current_indices]
    sqs_arr = sq_norms[current_indices]
    
    improvement_count = 0
    no_improve_count = 0
    
    while time.time() < deadline - 30:
        # Try to add a point
        added = False
        in_set = set(current_indices)
        candidates = [i for i in range(N) if i not in in_set]
        random.shuffle(candidates)
        
        for idx in candidates:
            if time.time() > deadline - 30:
                break
            if can_add_fast(pts_arr, sqs_arr, idx):
                current_indices.append(idx)
                pts_arr = np.vstack([pts_arr, grid[idx]])
                sqs_arr = np.append(sqs_arr, sq_norms[idx])
                added = True
                break
        
        if added:
            improvement_count += 1
            no_improve_count = 0
            continue
        
        # Try swap: remove one, add one
        swapped = False
        m = len(current_indices)
        if m > 0:
            swap_positions = list(range(m))
            random.shuffle(swap_positions)
            for pos in swap_positions:
                if time.time() > deadline - 30:
                    break
                old_idx = current_indices[pos]
                reduced_indices = current_indices[:pos] + current_indices[pos+1:]
                reduced_pts = np.delete(pts_arr, pos, axis=0)
                reduced_sqs = np.delete(sqs_arr, pos)
                
                reduced_set = set(reduced_indices)
                swap_candidates = [i for i in range(N) if i not in reduced_set and i != old_idx]
                random.shuffle(swap_candidates)
                
                for new_idx in swap_candidates:
                    if can_add_fast(reduced_pts, reduced_sqs, new_idx):
                        current_indices = reduced_indices + [new_idx]
                        pts_arr = np.vstack([reduced_pts, grid[new_idx]])
                        sqs_arr = np.append(reduced_sqs, sq_norms[new_idx])
                        swapped = True
                        break
                if swapped:
                    break
        
        if swapped:
            no_improve_count += 1
            if no_improve_count > 50:
                # Try kick: remove 2, add 2
                kicked = False
                if len(current_indices) >= 2:
                    kick_pos = random.sample(range(len(current_indices)), 2)
                    kick_set = set(kick_pos)
                    kept_indices = [idx for i, idx in enumerate(current_indices) if i not in kick_set]
                    kept_mask = [i for i in range(len(current_indices)) if i not in kick_set]
                    kept_pts = pts_arr[kept_mask]
                    kept_sqs = sqs_arr[kept_mask]
                    
                    kept_set = set(kept_indices)
                    kick_candidates = [i for i in range(N) if i not in kept_set]
                    random.shuffle(kick_candidates)
                    
                    added_pts = []
                    added_idx = []
                    for idx in kick_candidates:
                        if len(added_idx) >= 2:
                            break
                        temp_pts = np.vstack([kept_pts, grid[added_idx]]) if added_idx else kept_pts
                        temp_sqs = np.concatenate([kept_sqs, sq_norms[added_idx]]) if added_idx else kept_sqs
                        if can_add_fast(temp_pts, temp_sqs, idx):
                            added_idx.append(idx)
                    
                    if len(added_idx) == 2:
                        current_indices = kept_indices + added_idx
                        pts_arr = np.vstack([kept_pts, grid[added_idx]])
                        sqs_arr = np.concatenate([kept_sqs, sq_norms[added_idx]])
                        kicked = True
                        no_improve_count = 0
                if kicked:
                    continue
            else:
                continue
        else:
            no_improve_count += 1
            if no_improve_count > 100:
                # Try larger kick
                kicked = False
                if len(current_indices) >= 3:
                    kick_size = min(3, len(current_indices))
                    kick_pos = random.sample(range(len(current_indices)), kick_size)
                    kick_set = set(kick_pos)
                    kept_indices = [idx for i, idx in enumerate(current_indices) if i not in kick_set]
                    kept_mask = [i for i in range(len(current_indices)) if i not in kick_set]
                    kept_pts = pts_arr[kept_mask] if kept_mask else np.empty((0,3), dtype=np.int64)
                    kept_sqs = sqs_arr[kept_mask] if kept_mask else np.empty(0)
                    
                    kept_set = set(kept_indices)
                    kick_candidates = [i for i in range(N) if i not in kept_set]
                    random.shuffle(kick_candidates)
                    
                    added_idx = []
                    for idx in kick_candidates:
                        if len(added_idx) >= kick_size:
                            break
                        temp_pts = np.vstack([kept_pts, grid[added_idx]]) if added_idx else (kept_pts if len(kept_pts) > 0 else grid[added_idx].reshape(1,3))
                        temp_sqs = np.concatenate([kept_sqs, sq_norms[added_idx]]) if added_idx else (kept_sqs if len(kept_sqs) > 0 else sq_norms[added_idx].reshape(1))
                        if can_add_fast(temp_pts, temp_sqs, idx):
                            added_idx.append(idx)
                    
                    if len(added_idx) == kick_size:
                        current_indices = kept_indices + added_idx
                        pts_arr = np.vstack([kept_pts, grid[added_idx]])
                        sqs_arr = np.concatenate([kept_sqs, sq_norms[added_idx]])
                        kicked = True
                        no_improve_count = 0
                if kicked:
                    continue
                break
    
    # Final verification
    final_indices = current_indices
    if final_indices and check_validity(final_indices):
        return grid[final_indices].copy()
    else:
        # Fallback: use best known valid set
        if best_indices and check_validity(best_indices):
            return grid[best_indices].copy()
        return np.empty((0, 3), dtype=np.int64)
