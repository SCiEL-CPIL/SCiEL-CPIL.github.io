import numpy as np
import time
import random

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290

    # Precompute full distance matrix
    all_points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    num_points = len(all_points)
    
    all_pts64 = all_points.astype(np.int64)
    diff = all_pts64[:, np.newaxis, :] - all_pts64[np.newaxis, :, :]
    dist_matrix = np.sum(diff ** 2, axis=2).astype(np.int32)
    del diff

    def is_valid_addition(p_idx, current_indices, dist_sets):
        """
        Check if point p_idx can be added to current set.
        """
        k = len(current_indices)
        if k < 2:
            return True
        
        # Get distances from p_idx to all current points
        d_p = dist_matrix[p_idx, current_indices]
        
        # Check for isosceles at p: any two distances equal
        if len(set(d_p.tolist())) != len(d_p):
            return False
        
        # Check for isosceles at existing points
        for i in range(k):
            if d_p[i] in dist_sets[i]:
                return False
        
        return True

    def add_point(p_idx, current_indices, dist_sets):
        """Add point p_idx to the set and update dist_sets."""
        k = len(current_indices)
        d_p = dist_matrix[p_idx, current_indices]
        new_set = set(d_p.tolist())
        for i in range(k):
            dist_sets[i].add(d_p[i])
        current_indices.append(p_idx)
        dist_sets.append(new_set)

    def remove_point(p_idx, current_indices, dist_sets):
        """Remove point p_idx from the set and update dist_sets."""
        k = len(current_indices)
        if k == 0:
            return
        if k == 1:
            current_indices.clear()
            dist_sets.clear()
            return
        
        # Find position of p_idx in current_indices
        pos = current_indices.index(p_idx)
        
        # Get distances from p_idx to all other points
        d_p = dist_matrix[p_idx, current_indices]
        
        # Remove p_idx from all other points' dist_sets
        for i in range(k):
            if i != pos:
                dist_sets[i].discard(d_p[i])
        
        # Remove the point and its dist_set
        current_indices.pop(pos)
        dist_sets.pop(pos)

    def local_search(current_indices, dist_sets, max_iterations=200):
        """
        Local search: try to swap points to increase set size or maintain size with better structure.
        Returns improved (current_indices, dist_sets) or original if no improvement.
        """
        best_size = len(current_indices)
        best_indices = current_indices[:]
        best_dist_sets = [s.copy() for s in dist_sets]
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            if time.time() > deadline:
                break
            
            k = len(current_indices)
            
            # Try adding a random point not in the set
            if k < num_points:
                # Sample some candidate points to add
                candidates = []
                num_candidates = min(50, num_points - k)
                for _ in range(num_candidates):
                    c = random.randint(0, num_points - 1)
                    if c not in current_indices:
                        candidates.append(c)
                
                for c in candidates:
                    if is_valid_addition(c, current_indices, dist_sets):
                        add_point(c, current_indices, dist_sets)
                        best_size = len(current_indices)
                        best_indices = current_indices[:]
                        best_dist_sets = [s.copy() for s in dist_sets]
                        improved = True
                        break
            
            # Try swapping: remove one point, add another
            if k > 0 and len(current_indices) > 0:
                # Try removing a random point and adding a different one
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    if not current_indices:
                        break
                    
                    # Pick a point to remove
                    rem_pos = random.randint(0, len(current_indices) - 1)
                    rem_idx = current_indices[rem_pos]
                    
                    # Try to find a replacement
                    for _ in range(30):
                        add_idx = random.randint(0, num_points - 1)
                        if add_idx == rem_idx or add_idx in current_indices:
                            continue
                        
                        # Remove rem_idx
                        remove_point(rem_idx, current_indices, dist_sets)
                        
                        # Try to add add_idx
                        if is_valid_addition(add_idx, current_indices, dist_sets):
                            add_point(add_idx, current_indices, dist_sets)
                            if len(current_indices) > best_size:
                                best_size = len(current_indices)
                                best_indices = current_indices[:]
                                best_dist_sets = [s.copy() for s in dist_sets]
                                improved = True
                            break
                        else:
                            # Revert: add rem_idx back
                            add_point(rem_idx, current_indices, dist_sets)
        
        return best_indices, best_dist_sets

    # Baseline: 1D Sidon-like set
    xs, used = [], set()
    x = 0
    while x < N:
        diffs = {abs(x - y) for y in xs}
        if not (diffs & used):
            used |= diffs
            xs.append(x)
        x += 1
    best_set = [(x, 0) for x in xs]
    best_size = len(best_set)

    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Use a smarter ordering
        if restart % 4 == 0:
            # Random
            perm = np.random.permutation(num_points)
        elif restart % 4 == 1:
            # Prioritize corners
            scores = all_points[:, 0].astype(float) * all_points[:, 1].astype(float)
            perm = np.argsort(-scores) + np.random.randn(num_points) * 10
            perm = np.argsort(perm)
        elif restart % 4 == 2:
            # Prioritize edges
            edge_score = np.maximum.reduce([
                all_points[:, 0].astype(float),
                (N - 1 - all_points[:, 0]).astype(float),
                all_points[:, 1].astype(float),
                (N - 1 - all_points[:, 1]).astype(float)
            ])
            perm = np.argsort(-edge_score) + np.random.randn(num_points) * 5
            perm = np.argsort(perm)
        else:
            # Diagonal priority
            diag_score = np.abs(all_points[:, 0].astype(float) - all_points[:, 1].astype(float))
            perm = np.argsort(diag_score) + np.random.randn(num_points) * 3
            perm = np.argsort(perm)
        
        current_indices = []
        dist_sets = []
        current_list = []

        for idx in perm:
            if time.time() > deadline:
                break
            if is_valid_addition(idx, current_indices, dist_sets):
                add_point(idx, current_indices, dist_sets)
                current_list.append((int(all_points[idx, 0]), int(all_points[idx, 1])))

        # Apply local search
        if time.time() < deadline - 5:
            current_indices, dist_sets = local_search(current_indices, dist_sets)
            current_list = [(int(all_points[i, 0]), int(all_points[i, 1])) for i in current_indices]

        if len(current_list) > best_size:
            best_size = len(current_list)
            best_set = current_list[:]

    return best_set
# EVOLVE-BLOCK-END
