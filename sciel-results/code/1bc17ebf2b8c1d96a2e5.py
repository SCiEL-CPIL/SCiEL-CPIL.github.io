# EVOLVE-BLOCK-START
import numpy as np
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points using the 2E8 ⊕ 2D3 lattice
    to maximize the kissing number configuration.
    
    The construction uses:
    - 240 minimal vectors of 2*E8 (squared norm 8)
    - 12 minimal vectors of 2*D3 (squared norm 8)
    Total: 252 points, all with squared norm 8, minimum pairwise squared distance 8.
    
    Returns:
        points: np.ndarray of shape (252, 11) with int64 entries
    """
    start_time = time.time()
    deadline = 630.0  # safety margin before 630s wall-clock kill
    
    # --- Construct 2*E8 minimal vectors (240 vectors in 8D) ---
    # E8 minimal vectors: all vectors with squared norm 2 in E8.
    # 2*E8: scale by 2, so squared norm becomes 8.
    # 
    # E8 minimal vectors can be generated as:
    # Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0) permutations: C(8,2)*4 = 28*4 = 112
    # Type 2: (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2) with even number of minus signs: 128
    # Total: 240
    #
    # For 2*E8, multiply by 2:
    # Type 1: (±2, ±2, 0, 0, 0, 0, 0, 0) permutations: 112
    # Type 2: (±1, ±1, ±1, ±1, ±1, ±1, ±1, ±1) with even number of minus signs: 128
    # Total: 240, all with squared norm 8.
    
    e8_vectors = []
    
    # Type 1: (±2, ±2, 0, 0, 0, 0, 0, 0) with all permutations of positions
    for i in range(8):
        for j in range(i + 1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * 8
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    e8_vectors.append(v)
    
    # Type 2: (±1, ±1, ±1, ±1, ±1, ±1, ±1, ±1) with even number of minus signs
    # There are 2^8 = 256 sign combinations, half have even number of minuses = 128
    for signs in range(256):
        vec = []
        minus_count = 0
        for bit in range(8):
            if (signs >> bit) & 1:
                vec.append(-1)
                minus_count += 1
            else:
                vec.append(1)
        if minus_count % 2 == 0:
            e8_vectors.append(vec)
    
    # --- Construct 2*D3 minimal vectors (12 vectors in 3D) ---
    # D3 is isomorphic to A3. D3 minimal vectors (squared norm 1):
    # (±1, ±1, 0) and permutations: 3 * 4 = 12
    # 2*D3: scale by 2, squared norm becomes 4... wait, that's not 8.
    # 
    # Let me reconsider. We need all points to have the same squared norm for the kissing condition.
    # Actually, the condition is: min pairwise squared distance >= max squared norm.
    # 
    # For 2*E8 ⊕ 2*D3:
    # 2*E8 vectors have squared norm 8 in their 8D subspace.
    # 2*D3 vectors: D3 minimal vectors have squared norm 1, so 2*D3 has squared norm 4.
    # That doesn't match.
    #
    # Let me reconsider the lattice structure. The knowledge says "2E8 ⊕ 2D3 lattice, 
    # all 252 vectors achieve a uniform squared norm of 8."
    #
    # So the D3 part must also have squared norm 8. D3 minimal vectors have norm squared 1.
    # To get norm squared 8, we need to scale by sqrt(8), but that's not integer.
    #
    # Alternative: maybe the D3 part uses vectors with squared norm 8 in 3D.
    # In Z^3, vectors with squared norm 8: (±2, ±2, 0) and permutations: 3*4 = 12.
    # Yes! These are the 12 vectors.
    
    d3_vectors = []
    for i in range(3):
        for j in range(i + 1, 3):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * 3
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    d3_vectors.append(v)
    
    # --- Combine into 11D points ---
    # E8 part occupies dimensions 0-7, D3 part occupies dimensions 8-10.
    points = []
    
    # Add 2*E8 vectors (240 points): 8D E8 part + 3D zeros
    for v in e8_vectors:
        point = v + [0, 0, 0]
        points.append(point)
    
    # Add 2*D3 vectors (12 points): 8D zeros + 3D D3 part
    for v in d3_vectors:
        point = [0] * 8 + v
        points.append(point)
    
    # Convert to numpy array
    points_arr = np.array(points, dtype=np.int64)
    points_arr = np.around(points_arr).astype(np.int64)
    
    # Verify: check that all have squared norm 8
    norms_sq = np.sum(points_arr ** 2, axis=1)
    
    # Verify: check minimum pairwise squared distance
    # For 252 points, that's 252*251/2 = 31626 pairs - manageable
    min_pair_dist_sq = np.inf
    n = len(points_arr)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points_arr[i] - points_arr[j]
            dist_sq = np.sum(diff ** 2)
            if dist_sq < min_pair_dist_sq:
                min_pair_dist_sq = dist_sq
    
    max_norm_sq = np.max(norms_sq)
    
    # Validity check
    assert points_arr.shape == (252, 11), f"Shape is {points_arr.shape}"
    assert min_pair_dist_sq >= max_norm_sq, f"min_dist_sq={min_pair_dist_sq}, max_norm_sq={max_norm_sq}"
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        raise TimeoutError("Computation exceeded deadline")
    
    return points_arr


# EVOLVE-BLOCK-END
