# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates
    using the 2E_8 ⊕ 2D_3 lattice structure to maximize the number of points
    while satisfying the kissing configuration constraints.

    Returns:
        points: np.ndarray of shape (num_points, 11)
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second deadline

    # Generate E8 lattice minimal vectors (norm^2 = 2)
    # E8 has 240 minimal vectors with coordinates:
    # Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0) - 112 vectors
    # Type 2: (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2) with even number of -1/2s - 128 vectors
    
    # For integer coordinates, we need to scale. The minimal vectors of E8 have norm^2 = 2.
    # But we need integer coordinates. Let's use the standard E8 lattice points with integer coords.
    
    # Actually, E8 lattice points with integer coordinates include:
    # (±1, ±1, 0, 0, 0, 0, 0, 0) and permutations: 8 * C(8,2) * 2^2 = 8*28*4 = 896... that's too many
    # Wait, the minimal vectors of E8 are specifically 240 vectors.
    
    # Let me reconsider. The E8 lattice in Z^8 contains vectors with integer coordinates.
    # The minimal vectors (shortest non-zero vectors) have squared length 2.
    # These are: (±1, ±1, 0, ..., 0) and (±1/2, ..., ±1/2) with even number of minus signs.
    
    # For integer coordinates only, we can use: (±1, ±1, 0, 0, 0, 0, 0, 0) and permutations
    # Number: choose 2 positions out of 8 for the ±1's: C(8,2)=28, then 2^2=4 sign choices = 112
    
    e8_vectors = []
    for i in range(8):
        for j in range(i+1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0]*8
                    v[i] = si
                    v[j] = sj
                    e8_vectors.append(v)
    
    # That gives 112 vectors with norm^2 = 2
    
    # Now for D3 lattice: D_n = {x in Z^n : sum(x_i) is even}
    # D3 minimal vectors have norm^2 = 2: (±1, ±1, 0) and permutations
    # Number: C(3,2)*2^2 = 3*4 = 12
    
    d3_vectors = []
    for i in range(3):
        for j in range(i+1, 3):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0]*3
                    v[i] = si
                    v[j] = sj
                    d3_vectors.append(v)
    
    # Now construct 2E8 ⊕ 2D3 in 11 dimensions:
    # First 8 dims: E8, next 3 dims: D3
    # But we want to combine them to get more vectors.
    
    # Actually, let's think about this differently. We want vectors in Z^11.
    # The idea is to take vectors that are either in E8⊕{0} or {0}⊕D3 or combinations.
    
    # Let's try: points are (e8_vec, d3_vec) where e8_vec is from E8 minimal set and d3_vec is from D3 minimal set,
    # but we need to be careful about the norm and distance constraints.
    
    # Norm of (e8_vec, d3_vec) = norm(e8_vec)^2 + norm(d3_vec)^2 = 2 + 2 = 4
    # Distance between two such points: if they differ in both components, distance^2 >= 2+2=4
    # If they differ only in E8 part, distance^2 >= 2
    # If they differ only in D3 part, distance^2 >= 2
    
    # So min distance^2 could be 2, and max norm^2 is 4. That doesn't satisfy the constraint!
    # We need min_dist^2 >= max_norm^2.
    
    # Let me reconsider. We need all points to have the same norm, and pairwise distances >= that norm.
    
    # For a kissing configuration on a sphere of radius r, we need:
    # - All points have norm^2 = r^2
    # - All pairwise distances^2 >= r^2
    
    # So if we use vectors of norm^2 = 4, we need pairwise distances^2 >= 4.
    
    # Let's use: points are (e8_vec, 0) where e8_vec has norm^2 = 4, or (0, d3_vec) where d3_vec has norm^2 = 4.
    # But D3 doesn't have vectors of norm^2 = 4 with integer coords easily... (±2,0,0) has norm^2=4 but sum=2 which is even, so yes.
    
    # Actually, let's just use the 220 vectors of D11 lattice directly, or find a better construction.
    
    # D_n lattice: {x in Z^n : sum(x_i) is even}
    # Minimal vectors of D_n have norm^2 = 2: (±1, ±1, 0, ..., 0)
    # Number: n*(n-1)*2 = n(n-1)*2... wait, C(n,2)*4 = n(n-1)*2
    
    # For D11: 11*10*2 = 220 vectors with norm^2 = 2.
    # Pairwise distances: between two distinct minimal vectors, distance^2 >= 2 (since they're in a lattice).
    # Actually, we need to check: can two distinct minimal vectors have distance^2 = 2?
    # If v and w are both (±1,±1,0,...,0) type, and they share one ±1 position, then distance^2 = 1+1 = 2.
    # So min distance^2 = 2 = max norm^2. This satisfies the constraint!
    
    # So D11 gives us 220 points. Can we do better?
    
    # Let's try to add more vectors. What about vectors of norm^2 = 4 in D11?
    # If we add a vector of norm^2 = 4, then max_norm^2 becomes 4, and we need all pairwise distances^2 >= 4.
    # But the existing 220 vectors have pairwise distances as low as 2, so that fails.
    
    # So we're stuck with norm^2 = 2 vectors for the D11 approach, giving 220 points.
    
    # Let's try the 2E8 ⊕ 2D3 approach more carefully.
    # Dimension 8 + 3 = 11. 
    # Use vectors of the form (e8, d3) where e8 is in E8 lattice and d3 is in D3 lattice.
    # We want norm^2 = e8_norm^2 + d3_norm^2 to be constant, and pairwise distances >= that.
    
    # If we use e8 vectors of norm^2 = 2 and d3 vectors of norm^2 = 2, total norm^2 = 4.
    # Pairwise distance between (e8_a, d3_a) and (e8_b, d3_b) = dist(e8_a,e8_b)^2 + dist(d3_a,d3_b)^2.
    # Min dist in E8 between distinct minimal vectors is sqrt(2), so dist^2 >= 2.
    # Min dist in D3 between distinct minimal vectors is sqrt(2), so dist^2 >= 2.
    # So total dist^2 >= 2+2 = 4 if they differ in both, or >= 2 if they differ in only one.
    # Problem: if two points have the same E8 component but different D3 components, dist^2 = 0 + 2 = 2 < 4.
    
    # So we can't just take the Cartesian product. We need a more careful construction.
    
    # Alternative: just use the 220 D11 vectors. That's a solid baseline.
    
    d = 11
    points = []
    
    # Generate D11 minimal vectors: (±1, ±1, 0, ..., 0) with even sum (automatically satisfied since sum = ±2 or 0)
    for i in range(d):
        for j in range(i+1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0]*d
                    v[i] = si
                    v[j] = sj
                    points.append(v)
    
    # That's 11*10*4/2... wait, C(11,2)*4 = 55*4 = 220 vectors.
    
    points_arr = np.array(points, dtype=np.int64)
    
    # Verify constraints
    k = len(points_arr)
    if k >= 2:
        # Check no zero vector
        norms = np.sum(points_arr**2, axis=1)
        if np.all(norms > 0):
            # Check min pairwise distance^2 >= max norm^2
            max_norm_sq = np.max(norms)
            min_dist_sq = np.inf
            for i in range(k):
                for j in range(i+1, k):
                    diff = points_arr[i] - points_arr[j]
                    dist_sq = np.sum(diff**2)
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
            
            if min_dist_sq >= max_norm_sq:
                return points_arr
    
    # Fallback: return the two-point configuration
    return np.array([np.ones(11), -np.ones(11)], dtype=np.int64)


# EVOLVE-BLOCK-END
