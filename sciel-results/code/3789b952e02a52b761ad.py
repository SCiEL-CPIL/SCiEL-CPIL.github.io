# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points such that the minimum
    squared distance between any two points is >= the maximum squared norm of any point.
    Uses a lattice-based construction (E8 ⊕ D3 union) followed by greedy expansion.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill

    d = 11

    # ---------------------------------------------------------------
    # Step 1: Build E8 lattice points (240 points, all norm^2 = 8)
    # Scaled E8 roots:
    #   Type 1: (±2, ±2, 0, ..., 0): 112 points
    #   Type 2: (±1, ..., ±1) with even number of -1's: 128 points
    # Total: 240 points, all norm^2 = 8
    # Min distance^2 between distinct points: 8
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
    # Scaled D3 roots: (±2, ±2, 0) and permutations
    # Total: 12 points, all norm^2 = 8
    # Min distance^2: 8
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
    # Step 3: Combine E8 ⊕ D3 as a UNION (not product)
    # E8 points embedded in R^11 (padded with 0s in last 3 coords): 240 points, norm^2 = 8
    # D3 points embedded in R^11 (padded with 0s in first 8 coords): 12 points, norm^2 = 8
    # Union: 252 points, all norm^2 = 8.
    # Min distance: within E8 part: >= 8. Within D3 part: >= 8.
    # Between E8 point (a, 0) and D3 point (0, b): dist^2 = ||a||^2 + ||b||^2 = 16 >= 8. ✓
    # So min_dist^2 = 8, max_norm^2 = 8. Valid!
    # ---------------------------------------------------------------

    base_points = []

    # E8 part: 240 points in first 8 coordinates, last 3 are 0
    for v8 in e8_pts:
        v11 = np.zeros(11, dtype=np.int64)
        v11[:8] = v8
        base_points.append(v11)

    # D3 part: 12 points in last 3 coordinates, first 8 are 0
    for v3 in d3_pts:
        v11 = np.zeros(11, dtype=np.int64)
        v11[8:] = v3
        base_points.append(v11)

    best_points = base_points[:]
    best_k = len(base_points)  # 252

    # ---------------------------------------------------------------
    # Step 4: Greedy expansion - try to add more integer points
    # Current state: max_norm^2 = 8, min_dist^2 = 8
    # To add point p: need ||p||^2 <= 8 and min_{v in S} ||p-v||^2 >= 8
    # ---------------------------------------------------------------

    if time.time() < deadline - 120:
        # Generate all integer vectors in Z^11 with norm^2 <= 8
        # Use recursive generation with pruning
        def generate_candidates(max_norm_sq=8, d=11):
            candidates = []
            coord_range = list(range(-2, 3))  # -2, -1, 0, 1, 2

            def gen(idx, current, norm_sq):
                if norm_sq > max_norm_sq:
                    return
                if idx == d:
                    if norm_sq > 0:  # exclude zero vector
                        candidates.append(current.copy())
                    return
                for val in coord_range:
                    new_norm = norm_sq + val * val
                    if new_norm > max_norm_sq:
                        continue
                    current[idx] = val
                    gen(idx + 1, current, new_norm)
                current[idx] = 0

            gen(0, np.zeros(d, dtype=np.int64), 0)
            return candidates

        try:
            candidates = generate_candidates(8, 11)

            # Remove candidates already in base_points
            base_set = set(tuple(p) for p in base_points)
            candidates = [p for p in candidates if tuple(p) not in base_set]

            # Sort candidates by norm^2 (prefer smaller norms for easier addition)
            candidates.sort(key=lambda p: int(np.sum(p ** 2)))

            # Greedily add candidates
            current_set = base_points[:]
            for p in candidates:
                if time.time() > deadline - 60:
                    break
                # Check minimum distance to all points in current_set
                # Use vectorized computation for speed
                diffs = np.array(current_set) - p[np.newaxis, :]
                dists_sq = np.sum(diffs ** 2, axis=1)
                min_dist = np.min(dists_sq)
                if min_dist >= 8:
                    current_set.append(p)

            if len(current_set) > best_k:
                best_points = current_set
                best_k = len(current_set)
        except Exception:
            pass

    # ---------------------------------------------------------------
    # Step 5: Try a different strategy - use weight-2 vectors (norm^2=2)
    # All 220 vectors with exactly two non-zero entries (±1, ±1) have norm^2=2
    # and min distance^2 = 2. This is a valid configuration with 220 points.
    # We can try to combine this with other points.
    # ---------------------------------------------------------------

    if time.time() < deadline - 90:
        # Generate all weight-2 vectors in Z^11
        weight2_vectors = []
        for i in range(d):
            for j in range(i + 1, d):
                for si in [-1, 1]:
                    for sj in [-1, 1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        weight2_vectors.append(v)

        # Check if we can add some of these to the main set
        # But their norm^2 = 2 < 8, so max_norm would still be 8
        # Need min_dist >= 8 from all existing points
        # This is unlikely to help much, but let's try a few

        # Actually, a better approach: try a completely different configuration
        # using weight-4 vectors (norm^2=4) with greedy selection
        # This was the inspiration 2 approach

        # Generate all weight-4 vectors in 11 dimensions
        # C(11,4) * 2^4 = 330 * 16 = 5280 vectors
        supports = list(itertools.combinations(range(d), 4))
        sign_combinations = list(itertools.product([-1, 1], repeat=4))

        all_vectors = []
        for sup in supports:
            for signs in sign_combinations:
                v = np.zeros(d, dtype=np.int64)
                for idx, val in zip(sup, signs):
                    v[idx] = val
                all_vectors.append(v)

        # Greedily select a subset where dot(u,v) <= 2 for all distinct pairs
        # This ensures dist^2 = 8 - 2*dot >= 4 = norm^2
        import random
        best_w4_set = []
        best_w4_size = 0

        for restart in range(10):
            if time.time() > deadline - 30:
                break

            indices = list(range(len(all_vectors)))
            random.shuffle(indices)
            current_w4 = []

            for idx in indices:
                v = all_vectors[idx]
                valid = True
                # Check against all in current_w4
                for existing in current_w4:
                    dot = int(np.dot(v, existing))
                    if dot > 2:
                        valid = False
                        break

                if valid:
                    current_w4.append(v)

            if len(current_w4) > best_w4_size:
                best_w4_size = len(current_w4)
                best_w4_set = current_w4[:]

        # If the weight-4 configuration is larger than our current best, use it
        if best_w4_size > best_k:
            best_points = best_w4_set
            best_k = best_w4_size

    # ---------------------------------------------------------------
    # Final verification and return
    # ---------------------------------------------------------------
    points_arr = np.array(best_points, dtype=np.int64)
    points_arr = np.around(points_arr).astype(np.int64)

    # Verify validity
    if len(points_arr) < 2 or points_arr.shape[1] != 11:
        points_arr = np.array([np.ones(11, dtype=np.int64), -np.ones(11, dtype=np.int64)])
        return points_arr

    # Check no zero vector
    zero_mask = np.all(points_arr == 0, axis=1)
    if np.any(zero_mask):
        points_arr = points_arr[~zero_mask]

    if len(points_arr) < 2:
        points_arr = np.array([np.ones(11, dtype=np.int64), -np.ones(11, dtype=np.int64)])
        return points_arr

    # Check the distance constraint
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
        # Invalid - fallback to the base 252-point configuration
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
