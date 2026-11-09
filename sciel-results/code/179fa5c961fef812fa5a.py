# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points such that the minimum
    squared distance between any two points is >= the maximum squared norm of any point.
    
    Strategy:
    1. Build base configuration from E8 (240 pts) and D3 (12 pts) lattices, all norm^2 = 8.
    2. Find all additional integer vectors with norm^2 = 8 that are "compatible" with the
       base set (distance >= 8 to every base point).
    3. Greedily expand: iteratively add vectors with norm^2 <= 8 that maintain the constraint.
    4. Also try vectors with norm^2 = 16 if they don't violate the constraint.
    """
    start_time = time.time()
    deadline = start_time + 580.0

    d = 11

    # ---------------------------------------------------------------
    # Step 1: Build E8 lattice minimal vectors (240 points, norm^2 = 8)
    # ---------------------------------------------------------------
    def build_e8_minimal():
        pts = []
        # Type 1: (±2, ±2, 0, ..., 0)
        for i in range(8):
            for j in range(i + 1, 8):
                for si in [2, -2]:
                    for sj in [2, -2]:
                        v = np.zeros(8, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        pts.append(v)

        # Type 2: (±1, ±1, ±1, ±1, ±1, ±1, ±1, ±1) with even number of -1's
        for signs in itertools.product([1, -1], repeat=8):
            if sum(1 for s in signs if s == -1) % 2 == 0:
                pts.append(np.array(signs, dtype=np.int64))

        return pts

    e8_pts = build_e8_minimal()  # 240 points in R^8, all norm^2 = 8

    # ---------------------------------------------------------------
    # Step 2: Build D3 lattice minimal vectors (12 points, norm^2 = 8)
    # ---------------------------------------------------------------
    def build_d3_minimal():
        pts = []
        for i in range(3):
            for j in range(i + 1, 3):
                for si in [2, -2]:
                    for sj in [2, -2]:
                        v = np.zeros(3, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        pts.append(v)
        return pts

    d3_pts = build_d3_minimal()  # 12 points in R^3, all norm^2 = 8

    # ---------------------------------------------------------------
    # Step 3: Build base configuration (union of E8 and D3 embedded in R^11)
    # All points have norm^2 = 8. Min dist^2 = 8. Valid: 8 >= 8.
    # ---------------------------------------------------------------
    base_points = []
    for v8 in e8_pts:
        v11 = np.zeros(11, dtype=np.int64)
        v11[:8] = v8
        base_points.append(v11)
    for v3 in d3_pts:
        v11 = np.zeros(11, dtype=np.int64)
        v11[8:] = v3
        base_points.append(v11)

    best_points = base_points[:]
    best_k = len(base_points)  # 252

    # ---------------------------------------------------------------
    # Step 4: Generate all integer vectors in Z^11 with norm^2 <= 16
    # Coordinates in range [-4, 4] (since 4^2 = 16)
    # ---------------------------------------------------------------
    def generate_all_vectors(max_norm_sq):
        """Generate all integer vectors in Z^11 with norm^2 <= max_norm_sq."""
        candidates = []
        max_coord = int(np.sqrt(max_norm_sq))
        
        # Use recursive generation with pruning
        def gen(idx, current, norm_sq):
            if norm_sq > max_norm_sq:
                return
            if idx == d:
                if norm_sq > 0:
                    candidates.append(current.copy())
                return
            # Pruning: remaining coords can contribute at most (d-idx)*max_coord^2
            # But we just iterate and check
            for val in range(-max_coord, max_coord + 1):
                new_norm = norm_sq + val * val
                if new_norm > max_norm_sq:
                    continue
                current[idx] = val
                gen(idx + 1, current, new_norm)
            current[idx] = 0

        gen(0, np.zeros(d, dtype=np.int64), 0)
        return candidates

    # Generate candidates with norm^2 <= 16
    all_candidates = generate_all_vectors(16)
    
    # Remove base points from candidates
    base_set = set(tuple(p) for p in base_points)
    candidates = [p for p in all_candidates if tuple(p) not in base_set]
    
    # Sort by norm^2 (prefer smaller norms first)
    candidates.sort(key=lambda p: int(np.sum(p ** 2)))

    # ---------------------------------------------------------------
    # Step 5: Greedy expansion
    # For each candidate, check if it can be added while maintaining
    # min_dist^2 >= max_norm^2
    # ---------------------------------------------------------------
    current_set = best_points[:]
    current_max_norm_sq = 8  # All base points have norm^2 = 8

    for p in candidates:
        if time.time() > deadline - 30:
            break
        
        p_norm_sq = int(np.sum(p ** 2))
        
        # If adding p would increase max_norm_sq, we need to be more careful
        new_max_norm_sq = max(current_max_norm_sq, p_norm_sq)
        
        # Check distance to all current points
        # Need ||p - v||^2 >= new_max_norm_sq for all v in current_set
        # This is equivalent to: for all v, ||p||^2 + ||v||^2 - 2<p,v> >= new_max_norm_sq
        # i.e., 2<p,v> <= ||p||^2 + ||v||^2 - new_max_norm_sq
        
        # Vectorized check
        diffs = np.array(current_set, dtype=np.int64) - p[np.newaxis, :]
        dists_sq = np.sum(diffs ** 2, axis=1)
        min_dist = np.min(dists_sq)
        
        if min_dist >= new_max_norm_sq:
            current_set.append(p)
            current_max_norm_sq = new_max_norm_sq

    if len(current_set) > best_k:
        best_points = current_set
        best_k = len(current_set)

    # ---------------------------------------------------------------
    # Step 6: Second pass - try to add more vectors with higher norms
    # if the constraint allows (i.e., if min_dist^2 is large enough)
    # ---------------------------------------------------------------
    if time.time() < deadline - 60:
        # Compute current min_dist^2 and max_norm^2
        current_max_norm_sq = 0
        for p in best_points:
            n_sq = int(np.sum(p ** 2))
            if n_sq > current_max_norm_sq:
                current_max_norm_sq = n_sq

        # Generate candidates with norm^2 up to min_dist^2 (if we know it)
        # For now, try norm^2 up to 32
        if current_max_norm_sq < 32:
            all_candidates_32 = generate_all_vectors(32)
            base_set = set(tuple(p) for p in best_points)
            candidates_32 = [p for p in all_candidates_32 if tuple(p) not in base_set]
            candidates_32.sort(key=lambda p: int(np.sum(p ** 2)))

            current_set = best_points[:]
            for p in candidates_32:
                if time.time() > deadline - 30:
                    break
                p_norm_sq = int(np.sum(p ** 2))
                new_max_norm_sq = max(current_max_norm_sq, p_norm_sq)
                
                diffs = np.array(current_set, dtype=np.int64) - p[np.newaxis, :]
                dists_sq = np.sum(diffs ** 2, axis=1)
                min_dist = np.min(dists_sq)
                
                if min_dist >= new_max_norm_sq:
                    current_set.append(p)
                    current_max_norm_sq = new_max_norm_sq

            if len(current_set) > best_k:
                best_points = current_set
                best_k = len(current_set)

    # ---------------------------------------------------------------
    # Final verification and return
    # ---------------------------------------------------------------
    points_arr = np.array(best_points, dtype=np.int64)
    points_arr = np.around(points_arr).astype(np.int64)

    if len(points_arr) < 2 or points_arr.shape[1] != 11:
        points_arr = np.array([np.ones(11, dtype=np.int64), -np.ones(11, dtype=np.int64)])
        return points_arr

    zero_mask = np.all(points_arr == 0, axis=1)
    if np.any(zero_mask):
        points_arr = points_arr[~zero_mask]

    if len(points_arr) < 2:
        points_arr = np.array([np.ones(11, dtype=np.int64), -np.ones(11, dtype=np.int64)])
        return points_arr

    k = len(points_arr)
    min_dist_sq = float('inf')
    max_norm_sq = 0

    for i in range(k):
        norm_sq = int(np.sum(points_arr[i] ** 2))
        if norm_sq > max_norm_sq:
            max_norm_sq = norm_sq
        for j in range(i + 1, k):
            diff = points_arr[i] - points_arr[j]
            dist_sq = int(np.sum(diff ** 2))
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq

    if min_dist_sq < max_norm_sq:
        # Fallback to base configuration
        fallback = []
        for v8 in e8_pts:
            v11 = np.zeros(11, dtype=np.int64)
            v11[:8] = v8
            fallback.append(v11)
        for v3 in d3_pts:
            v11 = np.zeros(11, dtype=np.int64)
            v11[8:] = v3
            fallback.append(v11)
        points_arr = np.array(fallback, dtype=np.int64)

    return points_arr

# EVOLVE-BLOCK-END
