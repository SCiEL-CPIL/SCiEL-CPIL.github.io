# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm
    is smaller than their minimum pairwise distance, aiming to maximize the number of points.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin: kill at 630s, return well before

    d = 11

    # Start with the trivial configuration: two antipodal points
    points = [np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)]
    best_points = points.copy()
    best_k = 2

    # Strategy: Greedily add points that satisfy the constraint
    # For a set S, a new point p is valid if:
    #   1. p != 0
    #   2. ||p||^2 < min_{i<j in S} ||v_i - v_j||^2
    #   3. For all v in S: ||p - v||^2 >= max_{u in S} ||u||^2
    #
    # Actually, the constraint is:
    #   min pairwise distance squared >= max squared norm
    #
    # Let M = max_{v in S} ||v||^2
    # Let D = min_{i<j} ||v_i - v_j||^2
    # We need D >= M.
    #
    # When adding a new point p:
    #   New M' = max(M, ||p||^2)
    #   New D' = min(D, min_{v in S} ||p - v||^2)
    #   Need D' >= M'
    #
    # So we need:
    #   min(D, min_v ||p-v||^2) >= max(M, ||p||^2)
    #
    # This means:
    #   D >= max(M, ||p||^2) AND min_v ||p-v||^2 >= max(M, ||p||^2)
    #
    # Since D >= M already holds for the current set, we need:
    #   D >= ||p||^2 AND min_v ||p-v||^2 >= ||p||^2 AND min_v ||p-v||^2 >= M
    #
    # So: ||p||^2 <= D AND min_v ||p-v||^2 >= max(M, ||p||^2)

    # Let's try a different approach: use known good configurations
    # The kissing number problem in 11D is hard, but we can use lattice-based constructions

    # Start with the Leech lattice or similar, but in 11D we can use simpler constructions
    # Let's try to build a set of points with small norms and large mutual distances

    # Approach: Start with points on a sphere of radius r, and find many points with mutual distance >= 2r (squared: 4r^2)
    # But we need integer coordinates.

    # Let's try a greedy approach with random integer vectors
    np.random.seed(42)

    # Precompute some candidate points
    # Try points with small coordinates
    # First, let's try the cross-polytope: e_i and -e_i for i=1..11
    # That gives 22 points, all with norm 1, and minimum distance between e_i and e_j is sqrt(2) for i!=j
    # Between e_i and -e_i it's sqrt(2*2)=2
    # Wait: ||e_i - e_j||^2 = 2 for i != j, ||e_i - (-e_i)||^2 = 4
    # So min distance squared = 2, max norm squared = 1. 2 >= 1. Valid!
    # That gives us 22 points.

    cross_polytope = []
    for i in range(d):
        v = np.zeros(d, dtype=np.int64)
        v[i] = 1
        cross_polytope.append(v)
        v_neg = -v
        cross_polytope.append(v_neg)

    # Verify and use this as our starting point
    points = cross_polytope
    best_points = points.copy()
    best_k = len(points)

    # Now try to add more points
    # Current max norm squared = 1
    # Current min pairwise distance squared = 2 (between e_i and e_j for i != j)
    # To add a new point p, we need:
    #   ||p||^2 <= 2 (so ||p||^2 is 1 or 2)
    #   min_v ||p - v||^2 >= max(1, ||p||^2)
    #
    # If ||p||^2 = 1: p is one of the e_i or -e_i, already in the set
    # If ||p||^2 = 2: p has two non-zero entries, each ±1
    #   For p with p_i = 1, p_j = 1 (i != j):
    #     ||p - e_i||^2 = 1 (since p - e_i has 1 at position j only)
    #     But we need min_v ||p-v||^2 >= 2, and we got 1. Fail.
    #
    # So we can't add any more points to the cross-polytope with the same constraints.
    #
    # Let's try a different base: use points with larger norms but also larger minimum distances.
    #
    # Alternative: Use the root lattice A_11 or similar.
    # In A_n, points are integer vectors with sum of coordinates = 0.
    # The minimum distance in A_n is sqrt(2), and the norm of e_i - e_j is sqrt(2).
    # But A_n points can have various norms.
    #
    # Let's try: all permutations of (1, -1, 0, ..., 0) in 11D.
    # Number of such points: 11 * 10 * 2 = 220 (choose position of 1, position of -1, but that's 11*10=110, times 2 for sign? No)
    # Actually: choose i for +1, j for -1, i != j: 11 * 10 = 110 points.
    # Norm squared of each = 2.
    # Distance between two such points:
    #   If they share the same +1 position: e.g., e_i - e_j and e_i - e_k (j != k)
    #     Difference: (e_i - e_j) - (e_i - e_k) = -e_j + e_k, norm squared = 2
    #   If they share the same -1 position: similar, norm squared = 2
    #   If they are different: e_i - e_j and e_k - e_l with {i,j} != {k,l}
    #     If i=k, j=l: same point
    #     If i=l, j=k: negative, distance squared = 4*2 = 8? No: (e_i - e_j) - (e_j - e_i) = 2e_i - 2e_j, norm squared = 8
    #     Generally, the minimum distance is 2 (squared).
    # So min distance squared = 2, max norm squared = 2. 2 >= 2. Valid!
    # That gives us 110 points.

    a11_points = []
    for i in range(d):
        for j in range(d):
            if i != j:
                v = np.zeros(d, dtype=np.int64)
                v[i] = 1
                v[j] = -1
                a11_points.append(v)

    # Verify this is valid
    points = a11_points
    best_points = points.copy()
    best_k = len(points)

    # Can we add more?
    # Max norm squared = 2, min distance squared = 2
    # To add p: ||p||^2 <= 2, and min_v ||p-v||^2 >= 2
    # If ||p||^2 = 0: p = 0, not allowed
    # If ||p||^2 = 1: p = e_i or -e_i
    #   ||e_i - (e_i - e_j)||^2 = ||e_j||^2 = 1 < 2. Fail.
    # If ||p||^2 = 2: p is already in the set or is a permutation
    #   All vectors with norm squared 2 and integer coords are permutations of (±1, ±1, 0, ..., 0)
    #   Those are exactly the A_11 points (with sum 0) or not
    #   Wait: (1, 1, 0, ..., 0) has norm squared 2 but sum = 2, not in A_11
    #   Let's check: ||(1,1,0,...,0) - (1,-1,0,...,0)||^2 = ||(0,2,0,...,0)||^2 = 4 >= 2
    #   But ||(1,1,0,...,0) - (-1,1,0,...,0)||^2 = ||(2,0,0,...,0)||^2 = 4 >= 2
    #   ||(1,1,0,...,0) - (1,0,-1,0,...,0)||^2 = ||(0,1,1,0,...,0)||^2 = 2 >= 2
    #   So it might work!
    #
    # Let's try to add all vectors of norm squared 2 that are NOT in the A_11 set
    # Vectors of norm squared 2: exactly two non-zero entries, each ±1
    # Total: C(11,2) * 4 = 55 * 4 = 220
    # A_11 has 110 of them (those with sum 0, i.e., +1 and -1)
    # The other 110 have sum ±2 (i.e., +1,+1 or -1,-1)

    all_norm2_points = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    all_norm2_points.append(v)

    # Now try to add these to the A_11 set
    # Start with A_11 and greedily add norm-2 points
    current_set = a11_points[:]
    for p in all_norm2_points:
        if time.time() > deadline - 30:
            break
        # Check if p is already in current_set
        if any(np.array_equal(p, q) for q in current_set):
            continue
        # Check if adding p maintains validity
        # Current max norm squared = 2
        # Need: min_{v in current_set} ||p - v||^2 >= 2 AND ||p||^2 <= 2 (already true)
        min_dist = min(np.sum((p - v) ** 2) for v in current_set)
        if min_dist >= 2:
            current_set.append(p)

    if len(current_set) > best_k:
        best_points = current_set
        best_k = len(current_set)

    # Try to add more points with norm squared 1 or 0? Norm 0 is zero vector, not allowed.
    # Norm squared 1: e_i or -e_i
    # Check if we can add any e_i
    for i in range(d):
        if time.time() > deadline - 30:
            break
        for sign in [1, -1]:
            p = np.zeros(d, dtype=np.int64)
            p[i] = sign
            if any(np.array_equal(p, q) for q in current_set):
                continue
            min_dist = min(np.sum((p - v) ** 2) for v in current_set)
            max_norm = max(np.sum(v ** 2) for v in current_set)
            if min_dist >= max(max_norm, np.sum(p ** 2)):
                current_set.append(p)

    if len(current_set) > best_k:
        best_points = current_set
        best_k = len(current_set)

    # Now try a completely different approach: use a lattice with larger minimum distance
    # Consider the lattice generated by vectors with larger norms
    # For example, in 11D, consider points with coordinates in {-1, 0, 1} with even weight
    # Or use the D_n lattice

    # D_11: integer vectors with even sum of coordinates
    # Minimum distance squared = 2
    # Let's try to find a large set of D_11 points with mutual distance >= 2 and norm <= 2
    # That's essentially what we did above with A_11 plus some extra points

    # Let's try a random search for more points
    if time.time() < deadline - 60:
        # Try random integer vectors with small coordinates
        np.random.seed(123)
        current_set2 = a11_points[:]
        for _ in range(10000):
            if time.time() > deadline - 30:
                break
            # Generate a random vector with coordinates in {-2, -1, 0, 1, 2}
            p = np.random.randint(-2, 3, size=d)
            if np.all(p == 0):
                continue
            norm_sq = np.sum(p ** 2)
            if norm_sq > 4:  # Limit to small norms
                continue
            if any(np.array_equal(p, q) for q in current_set2):
                continue
            min_dist = min(np.sum((p - v) ** 2) for v in current_set2)
            max_norm = max(np.sum(v ** 2) for v in current_set2)
            if min_dist >= max(max_norm, norm_sq):
                current_set2.append(p)

        if len(current_set2) > best_k:
            best_points = current_set2
            best_k = len(current_set2)

    # Final verification
    points_arr = np.array(best_points, dtype=np.int64)
    points_arr = np.around(points_arr).astype(np.int64)

    # Verify validity
    if len(points_arr) < 2 or points_arr.shape[1] != 11:
        # Fallback to the trivial 2-point solution
        points_arr = np.array([np.ones(11, dtype=np.int64), -np.ones(11, dtype=np.int64)])

    # Check no zero vector
    if any(np.all(p == 0) for p in points_arr):
        # Remove zero vectors
        mask = ~np.all(points_arr == 0, axis=1)
        points_arr = points_arr[mask]

    # Check the distance constraint
    if len(points_arr) >= 2:
        min_dist_sq = float('inf')
        max_norm_sq = 0
        for i in range(len(points_arr)):
            for j in range(i + 1, len(points_arr)):
                diff = points_arr[i] - points_arr[j]
                dist_sq = np.sum(diff ** 2)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
            norm_sq = np.sum(points_arr[i] ** 2)
            if norm_sq > max_norm_sq:
                max_norm_sq = norm_sq

        if min_dist_sq < max_norm_sq:
            # Invalid, fallback to cross-polytope
            points_arr = np.array(cross_polytope, dtype=np.int64)

    return points_arr

# EVOLVE-BLOCK-END
