# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580
    
    def get_scaled_e8_vectors():
        """
        Generate the 240 minimal vectors of the scaled E8 lattice (integer coordinates).
        Type 1: +/-2e_i +/-2e_j (112 vectors)
        Type 2: (+/-1, ..., +/-1) with even number of minuses (128 vectors)
        Padded to 11 dimensions (last 3 are 0).
        """
        vecs = []
        # Type 1
        for i in range(8):
            for j in range(i + 1, 8):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        v = np.zeros(11, dtype=np.int64)
                        v[i] = 2 * si
                        v[j] = 2 * sj
                        vecs.append(v)
        
        # Type 2
        # All combinations of +/-1 in first 8 dimensions
        for signs in itertools.product([1, -1], repeat=8):
            # Check if number of -1s is even
            if signs.count(-1) % 2 == 0:
                v = np.zeros(11, dtype=np.int64)
                for k, s in enumerate(signs):
                    v[k] = s
                vecs.append(v)
        
        return vecs

    def get_a3_vectors():
        """
        Generate the 12 minimal vectors of A3 (D3) lattice.
        +/-e_i +/-e_j for i != j in 3 dims.
        Padded to 11 dimensions (first 8 are 0).
        """
        vecs = []
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                for si in [1, -1]:
                    for sj in [1, -1]:
                        v = np.zeros(11, dtype=np.int64)
                        v[8 + i] = si
                        v[8 + j] = sj
                        vecs.append(v)
        return vecs

    def validate_and_count(points):
        """Validate the set and return count if valid, else 0."""
        if len(points) < 2:
            return 0
        arr = np.array(points, dtype=np.int64)
        norms = np.sum(arr * arr, axis=1)
        max_norm = int(np.max(norms))
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                diff = arr[i] - arr[j]
                sq_dist = int(np.sum(diff * diff))
                if sq_dist < max_norm:
                    return 0
        return n

    # Construct the high-symmetry configuration
    e8_vecs = get_scaled_e8_vectors()
    a3_vecs = get_a3_vectors()
    
    # We can use all 240 E8 vectors.
    # We can add A3 vectors, but they must be mutually far apart.
    # In A3, max dist sq is 8 (between opposites).
    # Norm of E8 is 8. Norm of A3 is 2.
    # Max norm in set will be 8.
    # Dist between A3 vectors must be >= 8.
    # Only opposites satisfy this.
    # So we can pick one pair of opposites from A3.
    
    # Let's pick the first two opposite vectors from A3
    # a3_vecs[0] is (0,0,0,0,0,0,0,0, 1, 1, 0)
    # a3_vecs[1] is (0,0,0,0,0,0,0,0, 1, -1, 0) -> Not opposite.
    # Opposite of (1,1,0) is (-1,-1,0).
    # Let's find them.
    a3_set = []
    v0 = a3_vecs[0]
    v0_neg = -v0
    # Check if v0_neg is in a3_vecs
    # It should be.
    a3_set.append(v0)
    a3_set.append(v0_neg)
    
    candidate_points = e8_vecs + a3_set
    
    best_points = []
    cnt = validate_and_count(candidate_points)
    if cnt > 0:
        best_points = list(candidate_points)
    
    # Fallback to greedy search if the lattice construction fails or is small
    if len(best_points) < 100:
        # Re-implement greedy search briefly to ensure we don't regress if lattice logic has a bug
        def generate_vectors(n_nonzero):
            vecs = []
            for combo in itertools.combinations(range(d), n_nonzero):
                for signs in itertools.product([1, -1], repeat=n_nonzero):
                    v = np.zeros(d, dtype=np.int64)
                    for idx, val in zip(combo, signs):
                        v[idx] = val
                    vecs.append(v)
            return vecs
        
        candidates_2 = generate_vectors(2)
        candidates_4 = generate_vectors(4)
        
        def greedy_select(candidate_list, norm_sq, existing_points, existing_norms, max_norm):
            selected = []
            for v in candidate_list:
                if time.time() > deadline - 5:
                    break
                valid = True
                if len(existing_points) > 0:
                    ex_arr = np.array(existing_points, dtype=np.int64)
                    dots = ex_arr @ v
                    for i, dot in enumerate(dots):
                        allowed = (existing_norms[i] + norm_sq - max_norm) / 2.0
                        if dot > allowed:
                            valid = False
                            break
                if valid and len(selected) > 0:
                    sel_arr = np.array(selected, dtype=np.int64)
                    dots = sel_arr @ v
                    for dot in dots:
                        allowed = (norm_sq + norm_sq - max_norm) / 2.0
                        if dot > allowed:
                            valid = False
                            break
                if valid:
                    selected.append(v)
            return selected

        existing_points = []
        existing_norms = []
        max_norm = 4
        new_sel = greedy_select(candidates_4, 4, existing_points, existing_norms, max_norm)
        existing_points.extend(new_sel)
        existing_norms.extend([4] * len(new_sel))
        new_sel = greedy_select(candidates_2, 2, existing_points, existing_norms, max_norm)
        existing_points.extend(new_sel)
        existing_norms.extend([2] * len(new_sel))
        
        cnt = validate_and_count(existing_points)
        if cnt > len(best_points):
            best_points = list(existing_points)

    if not best_points:
        # Absolute fallback
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                p1 = np.zeros(d, dtype=np.int64); p1[i] = 1; p1[j] = 1; points.append(p1)
                p2 = np.zeros(d, dtype=np.int64); p2[i] = -1; p2[j] = -1; points.append(p2)
        return np.array(points, dtype=np.int64)
    
    result = np.array(best_points, dtype=np.int64)
    return np.around(result).astype(np.int64)

# EVOLVE-BLOCK-END
