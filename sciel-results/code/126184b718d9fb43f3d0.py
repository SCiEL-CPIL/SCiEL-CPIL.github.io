# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590
    
    def generate_vectors(n_nonzero):
        """Generate all vectors with exactly n_nonzero non-zero entries (each ±1)."""
        vecs = []
        for combo in itertools.combinations(range(d), n_nonzero):
            for signs in itertools.product([1, -1], repeat=n_nonzero):
                v = np.zeros(d, dtype=np.int64)
                for idx, val in zip(combo, signs):
                    v[idx] = val
                vecs.append(v)
        return vecs
    
    # Generate candidates by norm level
    # Norm squared: 2, 4, 6, 8, 10, 11
    # For a set of points with min squared distance >= max squared norm:
    # If all points have norm^2 = N, then for any pair u,v: ||u-v||^2 = 2N - 2*u.v >= N
    # => u.v <= N/2
    # For mixed norms, need: ||u-v||^2 >= max(||u||^2, ||v||^2)
    
    # Strategy: Use vectors with norm^2 = 4 (4 non-zero ±1 entries)
    # Constraint: u.v <= 2 for any pair
    # Also add norm^2 = 6 vectors (6 non-zero ±1 entries)
    # For 4-norm vs 6-norm: ||u-v||^2 >= 6 => 4 + 6 - 2*u.v >= 6 => u.v <= 2
    # For 6-norm vs 6-norm: ||u-v||^2 >= 6 => 12 - 2*u.v >= 6 => u.v <= 3
    
    # Generate 4-norm candidates
    candidates_4 = generate_vectors(4)
    
    # Generate 6-norm candidates  
    candidates_6 = generate_vectors(6)
    
    # Generate 2-norm candidates
    candidates_2 = generate_vectors(2)
    
    # Greedy selection - process in order of increasing norm to maximize packing
    # Start with 4-norm (most constrained, need dot <= 2)
    sel4 = []
    sel4_arr = np.zeros((0, d), dtype=np.int64)
    
    for v in candidates_4:
        if time.time() > deadline - 200:
            break
        if not sel4:
            sel4.append(v)
            sel4_arr = v.reshape(1, -1).astype(np.int64)
            continue
        dots = sel4_arr @ v
        if np.max(dots) <= 2:
            sel4.append(v)
            sel4_arr = np.vstack((sel4_arr, v.reshape(1, -1).astype(np.int64)))
    
    # Add 6-norm vectors
    # For 6-norm vs 4-norm selected: need dot <= 2
    # For 6-norm vs 6-norm selected: need dot <= 3
    sel6 = []
    sel6_arr = np.zeros((0, d), dtype=np.int64)
    
    for v in candidates_6:
        if time.time() > deadline - 100:
            break
        valid = True
        if len(sel4) > 0:
            dots4 = sel4_arr @ v
            if np.max(dots4) > 2:
                valid = False
        if valid and len(sel6) > 0:
            dots6 = sel6_arr @ v
            if np.max(dots6) > 3:
                valid = False
        if valid:
            sel6.append(v)
            sel6_arr = np.vstack((sel6_arr, v.reshape(1, -1).astype(np.int64)))
    
    # Add 2-norm vectors
    # For 2-norm vs 4-norm: need dot <= 1 (||u-v||^2 >= 4 => 2+4-2dot >= 4 => dot <= 1)
    # For 2-norm vs 6-norm: need dot <= 1 (||u-v||^2 >= 6 => 2+6-2dot >= 6 => dot <= 1)
    # For 2-norm vs 2-norm: need dot <= 0
    sel2 = []
    sel2_arr = np.zeros((0, d), dtype=np.int64)
    
    for v in candidates_2:
        if time.time() > deadline - 30:
            break
        valid = True
        if len(sel4) > 0:
            dots4 = sel4_arr @ v
            if np.max(dots4) > 1:
                valid = False
        if valid and len(sel6) > 0:
            dots6 = sel6_arr @ v
            if np.max(dots6) > 1:
                valid = False
        if valid and len(sel2) > 0:
            dots2 = sel2_arr @ v
            if np.max(dots2) > 0:
                valid = False
        if valid:
            sel2.append(v)
            sel2_arr = np.vstack((sel2_arr, v.reshape(1, -1).astype(np.int64)))
    
    # Combine all selected points
    all_points = sel4 + sel6 + sel2
    
    if not all_points:
        # Fallback
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                p1 = np.zeros(d, dtype=np.int64); p1[i] = 1; p1[j] = 1; points.append(p1)
                p2 = np.zeros(d, dtype=np.int64); p2[i] = -1; p2[j] = -1; points.append(p2)
                p3 = np.zeros(d, dtype=np.int64); p3[i] = 1; p3[j] = -1; points.append(p3)
                p4 = np.zeros(d, dtype=np.int64); p4[i] = -1; p4[j] = 1; points.append(p4)
        return np.array(points, dtype=np.int64)
    
    result = np.array(all_points, dtype=np.int64)
    return np.around(result).astype(np.int64)

# EVOLVE-BLOCK-END
