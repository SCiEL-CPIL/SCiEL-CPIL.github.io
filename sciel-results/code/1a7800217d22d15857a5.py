import numpy as np
import itertools
import time
import sys

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    sq_norms = (grid ** 2).sum(axis=1)
    
    # Precompute R matrix for determinant if needed, but we will use constraint equations
    # R_i = [x, y, z, 1] for plane/sphere solving
    R_base = np.hstack([grid.astype(np.float64), np.ones((N, 1))])
    
    # Time limit
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 10.0
    
    def get_constraints(indices):
        """
        Compute plane/sphere constraints for all 4-tuples in the set defined by indices.
        Returns:
        planes: list of (normal, d) tuples
        spheres: list of (A, B, C, D) tuples
        """
        planes = []
        spheres = []
        k = len(indices)
        if k < 4:
            return planes, spheres
        
        # Get the 4x4 matrices M for all 4-tuples
        # M rows: [x, y, z, 1]
        # We need to solve M * [A, B, C, D]^T = -sq_norms
        
        combos = list(itertools.combinations(range(k), 4))
        if not combos:
            return planes, spheres
            
        # Batch processing for speed
        # We will process in chunks to avoid memory issues if k is large
        chunk_size = 1000
        for start in range(0, len(combos), chunk_size):
            chunk = combos[start:start+chunk_size]
            # Build M_chunk: (C, 4, 4)
            # Get indices of the 4 points in the original grid
            # chunk[i] is a tuple of 4 indices into 'indices'
            # We need the grid points
            # points_in_chunk = indices[list(itertools.chain.from_iterable(chunk))] # Slow
            
            # Faster:
            # Create an array of shape (C, 4) containing the grid indices
            # Then use fancy indexing
            idx_arr = np.array(chunk, dtype=np.int64) # (C, 4)
            # Map local indices to global grid indices
            global_idx = indices[idx_arr] # (C, 4)
            
            # Get R_base for these points: (C, 4, 4)
            M_chunk = R_base[global_idx] # (C, 4, 4)
            # Get sq_norms: (C, 4)
            sq_chunk = sq_norms[global_idx]
            
            # Check determinants to distinguish plane vs sphere
            # det(M) ~ 0 means coplanar
            dets = np.linalg.det(M_chunk)
            
            # Threshold for coplanar
            # Since coordinates are integers, det is integer.
            # If det == 0, coplanar.
            # But we use float.
            is_plane = np.abs(dets) < 1e-9
            
            # Process Planes
            if np.any(is_plane):
                plane_indices = np.where(is_plane)[0]
                M_planes = M_chunk[plane_indices]
                # For planes, we solve M * n = 0? No.
                # Plane equation: n . x + d = 0.
                # We can find normal by cross product of two vectors.
                # Or solve M * [nx, ny, nz, d]^T = 0?
                # M is 4x4. If det=0, the system M * v = 0 has non-trivial solution.
                # We want v = [nx, ny, nz, d].
                # Use SVD to find null space.
                # For each plane:
                for i in plane_indices:
                    M_i = M_chunk[i]
                    # SVD
                    U, s, Vt = np.linalg.svd(M_i)
                    # Null vector is last row of Vt
                    n_d = Vt[-1]
                    # Normalize? Not strictly necessary for check, but good.
                    # Check: n_d . [x, y, z, 1] = 0
                    planes.append(n_d)
            
            # Process Spheres
            if np.any(~is_plane):
                sphere_indices = np.where(~is_plane)[0]
                M_spheres = M_chunk[sphere_indices]
                sq_spheres = sq_chunk[sphere_indices]
                
                # Solve M * coef = -sq
                # coef = [A, B, C, D]
                # Use np.linalg.solve for batch?
                # np.linalg.solve expects (..., M, M) and (..., M)
                try:
                    coef_spheres = np.linalg.solve(M_spheres, -sq_spheres)
                    # coef_spheres shape: (C, 4)
                    for i in range(len(coef_spheres)):
                        spheres.append(coef_spheres[i])
                except np.linalg.LinAlgError:
                    # Fallback to lstsq if singular (should not happen if det != 0)
                    coef_spheres = np.linalg.lstsq(M_spheres, -sq_spheres, rcond=None)[0]
                    for i in range(len(coef_spheres)):
                        spheres.append(coef_spheres[i])
        
        return planes, spheres

    def check_candidates(indices, planes, spheres, candidates):
        """
        Check which candidates are valid given the constraints.
        Returns list of valid candidate indices.
        """
        if not candidates:
            return []
        
        valid = []
        if not planes and not spheres:
            return candidates
        
        # Get grid points for candidates
        cand_points = grid[candidates] # (M, 3)
        cand_sq = sq_norms[candidates] # (M,)
        
        # Check Planes
        if planes:
            # planes is list of (4,) vectors [nx, ny, nz, d]
            P = np.array(planes) # (P, 4)
            # For each candidate, compute P @ [x, y, z, 1]
            # cand_homog = [x, y, z, 1]
            cand_homog = np.hstack([cand_points, np.ones((len(candidates), 1))]) # (M, 4)
            # Distances: (M, P)
            dists = cand_homog @ P.T
            # If any dist is close to 0, invalid
            # Use min abs distance
            min_dists = np.min(np.abs(dists), axis=1)
            invalid_mask = min_dists < 1e-6
        else:
            invalid_mask = np.zeros(len(candidates), dtype=bool)
        
        # Check Spheres
        if spheres:
            # spheres is list of (4,) vectors [A, B, C, D]
            S = np.array(spheres) # (S, 4)
            # Equation: x^2+y^2+z^2 + Ax + By + Cz + D = 0
            # val = cand_sq + cand_points @ [A, B, C]^T + D
            # cand_points @ S[:, :3].T -> (M, S)
            # S[:, 3] -> (S,)
            linear_part = cand_points @ S[:, :3].T # (M, S)
            const_part = S[:, 3] # (S,)
            vals = cand_sq[:, np.newaxis] + linear_part + const_part[np.newaxis, :] # (M, S)
            min_sphere_dists = np.min(np.abs(vals), axis=1)
            invalid_mask |= (min_sphere_dists < 1e-6)
        
        valid_indices = [candidates[i] for i in range(len(candidates)) if not invalid_mask[i]]
        return valid_indices

    def greedy_init(seed):
        np.random.seed(seed)
        order = np.random.permutation(N)
        selected_indices = []
        planes = []
        spheres = []
        
        for idx in order:
            # Check if idx can be added
            if len(selected_indices) < 4:
                selected_indices.append(idx)
                # No constraints yet
                continue
            
            # Recompute constraints?
            # For greedy, we can just check against existing constraints.
            # But adding a point creates NEW constraints with the existing points.
            # The check "can_add" only needs to ensure the NEW point doesn't form a bad 5-tuple
            # with ANY 4 existing points.
            # This is exactly what check_candidates does with the current constraints.
            
            # However, the current constraints are for 4-tuples in selected_indices.
            # If we add idx, we must ensure idx is not on any sphere/plane defined by 4-tuples in selected_indices.
            # This is sufficient.
            
            valid = check_candidates(selected_indices, planes, spheres, [idx])
            if valid:
                selected_indices.append(idx)
                # We don't need to update planes/spheres immediately for the next check
                # UNLESS the next check depends on the new constraints.
                # But check_candidates uses the constraints of the CURRENT set.
                # If we add idx, the set changes.
                # So we MUST recompute constraints for the next iteration.
                # This is expensive O(k^4).
                # For k < 40, it's fine.
                
                # Optimization: Only recompute if we are going to check more.
                # We will recompute at the start of the next loop iteration.
        
        return selected_indices

    def local_search(initial_indices, max_time):
        current_indices = list(initial_indices)
        best_indices = list(current_indices)
        best_size = len(current_indices)
        
        start_ls = time.time()
        
        while time.time() - start_ls < max_time:
            # 1. Try to add
            if len(current_indices) < N:
                # Compute constraints for current set
                planes, spheres = get_constraints(current_indices)
                
                candidates = [i for i in range(N) if i not in set(current_indices)]
                if candidates:
                    valid_adds = check_candidates(current_indices, planes, spheres, candidates)
                    if valid_adds:
                        # Pick one (random for diversity)
                        np.random.shuffle(valid_adds)
                        current_indices.append(valid_adds[0])
                        if len(current_indices) > best_size:
                            best_indices = list(current_indices)
                            best_size = len(current_indices)
                        continue
            
            # 2. Try 1-swap
            # Pick a random point to remove
            out_idx = np.random.choice(current_indices)
            remaining = [i for i in current_indices if i != out_idx]
            
            # Compute constraints for remaining
            planes, spheres = get_constraints(remaining)
            
            candidates = [i for i in range(N) if i not in set(remaining)]
            valid_adds = check_candidates(remaining, planes, spheres, candidates)
            
            if valid_adds:
                # Swap
                np.random.shuffle(valid_adds)
                in_idx = valid_adds[0]
                current_indices = remaining + [in_idx]
                # Size is same, so best_size doesn't change, but we might unblock
                # We don't update best_indices here unless size increases (which it doesn't)
                # But we continue searching.
            else:
                # If no swap found, try a few more random removals
                found_swap = False
                for _ in range(5):
                    out_idx = np.random.choice(current_indices)
                    remaining = [i for i in current_indices if i != out_idx]
                    planes, spheres = get_constraints(remaining)
                    candidates = [i for i in range(N) if i not in set(remaining)]
                    valid_adds = check_candidates(remaining, planes, spheres, candidates)
                    if valid_adds:
                        np.random.shuffle(valid_adds)
                        in_idx = valid_adds[0]
                        current_indices = remaining + [in_idx]
                        found_swap = True
                        break
                
                if not found_swap:
                    # Stuck. Break or try 2-swap?
                    # For now, break to restart
                    break

        return best_indices

    # Run multiple restarts
    best_overall = []
    
    num_restarts = 0
    while time.time() - start_time < time_limit - safety_margin:
        num_restarts += 1
        seed = num_restarts
        init_indices = greedy_init(seed)
        
        elapsed = time.time() - start_time
        remaining = time_limit - safety_margin - elapsed
        if remaining < 5:
            break
            
        ls_time = min(remaining * 0.5, 30.0)
        if ls_time < 1:
            break
            
        search_result = local_search(init_indices, ls_time)
        
        if len(search_result) > len(best_overall):
            best_overall = search_result
            # Polish
            current = list(best_overall)
            current_set = set(current)
            # Try adding in sorted order
            for idx in range(N):
                if idx not in current_set:
                    planes, spheres = get_constraints(current)
                    valid = check_candidates(current, planes, spheres, [idx])
                    if valid:
                        current.append(idx)
                        current_set.add(idx)
            
            if len(current) > len(best_overall):
                best_overall = current

    if not best_overall:
        return np.empty((0, 3), dtype=np.int64)
        
    return grid[best_overall].astype(np.int64)

if __name__ == "__main__":
    # Quick test
    res = no_5_on_a_sphere8()
    print(f"Found {len(res)} points")
