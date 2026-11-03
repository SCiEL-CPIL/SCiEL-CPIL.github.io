"""Construct nonzero integer vectors in R^11 maximizing kissing number."""
import numpy as np
import time
import itertools

D = 11
TIME_LIMIT = 110  # seconds, safety margin before 120s kill

def construct():
    start_time = time.time()
    
    # Start with the seed: all vectors with exactly two non-zero entries ±1
    # These have norm sqrt(2), pairwise distances >= sqrt(2)
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
    
    # Add vectors with a single non-zero entry ±1 (norm 1)
    # Wait, these would have distance 1 from each other, but max norm is sqrt(2)
    # So min pairwise distance must be >= max norm = sqrt(2)
    # Distance between two single-entry vectors: if same position, dist=2 or 0 (if same sign)
    # If different positions, dist=sqrt(2)
    # But distance 0 (same vector) is not allowed since we need distinct vectors
    # Actually let me re-read: "minimum pairwise distance is at least the maximum vector norm"
    # So for any two distinct vectors v, w: ||v - w|| >= max_i ||v_i||
    
    # The seed vectors all have norm sqrt(2). 
    # Distance between two seed vectors:
    # - Same sign pattern, overlapping positions: e.g., (1,1,0...) and (1,1,0...) same -> dist 0 (not allowed, same vector)
    # - Let's think more carefully. Two vectors with entries in positions {i,j} and {k,l}.
    #   If {i,j} == {k,l}: 
    #     Same signs: distance 0 (identical) - not allowed
    #     Different signs in one position: distance 2
    #     Different signs in both positions: distance 2*sqrt(2)
    #   If |{i,j} ∩ {k,l}| = 1: distance depends, can be 1 or sqrt(5) etc.
    #   If disjoint: distance sqrt(4) = 2
    #   
    #   Min distance among distinct seed vectors: when they share one position with opposite sign,
    #   e.g., v=(1,1,0...), w=(1,-1,0...): dist = sqrt(0+4+0...) = 2 >= sqrt(2) ✓
    #   Or v=(1,1,0...), w=(1,0,1,0...): dist = sqrt(0+1+1+0...) = sqrt(2) >= sqrt(2) ✓
    
    # So the seed works with k = C(11,2) * 4 = 55 * 4 = 220
    
    base_vectors = vectors.copy()
    
    # Try to add more vectors. Let's try vectors with one entry ±2 and others 0.
    # Norm = 2. Need distance to all base vectors >= 2.
    # Distance from (2*e_i) to a base vector v with v[i] = s (s ∈ {1,-1,0}):
    #   If v[i] = 0: dist = sqrt(4 + ||v||^2 - 0) = sqrt(4+2) = sqrt(6) >= 2 ✓
    #   If v[i] = 1: dist = sqrt((2-1)^2 + sum_{j≠i}v[j]^2) = sqrt(1 + 1) = sqrt(2) < 2 ✗
    #   If v[i] = -1: dist = sqrt((2-(-1))^2 + 1) = sqrt(9+1) = sqrt(10) >= 2 ✓
    # So (2*e_i) conflicts with base vectors that have v[i]=1.
    
    # What about (-2*e_i)? 
    #   If v[i] = 0: dist = sqrt(4+2) = sqrt(6) >= 2 ✓
    #   If v[i] = 1: dist = sqrt((-2-1)^2 + 1) = sqrt(9+1) = sqrt(10) >= 2 ✓  
    #   If v[i] = -1: dist = sqrt((-2-(-1))^2 + 1) = sqrt(1+1) = sqrt(2) < 2 ✗
    
    # So 2*e_i conflicts with half the base vectors touching i, and -2*e_i conflicts with the other half.
    # We can't add both for the same i.
    
    # Let's try a different approach: use a lattice-based construction.
    # In dimension 11, the Leech lattice doesn't directly apply, but we can try other constructions.
    
    # Alternative: try to add vectors of the form e_i - e_j (already in seed) and e_i + e_j (already in seed)
    # What about vectors with three non-zero entries?
    
    # Let's try a greedy approach: start with seed, then try to add candidate vectors.
    
    candidates = []
    
    # Try vectors with entries in {-2, -1, 0, 1, 2}
    # Focus on sparse vectors (2-4 non-zero entries)
    
    # First, let's try adding single-entry vectors with value ±2
    # As analyzed, 2*e_i conflicts with vectors having v[i]=1
    # -2*e_i conflicts with vectors having v[i]=-1
    
    # For each i, check if we can add 2*e_i or -2*e_i
    # But we need to check against ALL current vectors
    
    def min_dist_to_set(v, vec_list):
        """Compute minimum distance from v to any vector in vec_list."""
        v_arr = np.array(v, dtype=np.float64)
        min_d = float('inf')
        for w in vec_list:
            w_arr = np.array(w, dtype=np.float64)
            d = np.linalg.norm(v_arr - w_arr)
            if d < min_d:
                min_d = d
        return min_d
    
    def max_norm(vec_list):
        """Compute maximum norm in vec_list."""
        if not vec_list:
            return 0
        return max(np.linalg.norm(v) for v in vec_list)
    
    def is_valid_addition(v, vec_list):
        """Check if adding v to vec_list maintains the constraint."""
        v_arr = np.array(v, dtype=np.float64)
        v_norm = np.linalg.norm(v_arr)
        if v_norm == 0:
            return False
        # New max norm
        new_max_norm = max(v_norm, max_norm(vec_list))
        # Check distance from v to all existing vectors
        for w in vec_list:
            w_arr = np.array(w, dtype=np.float64)
            d = np.linalg.norm(v_arr - w_arr)
            if d < new_max_norm - 1e-9:
                return False
        return True
    
    # Convert to numpy for efficiency
    vec_array = np.array(base_vectors, dtype=np.int64)
    vec_norms = np.sqrt(np.sum(vec_array**2, axis=1))
    current_max_norm = np.max(vec_norms)
    
    def is_valid_addition_fast(v, vec_array, current_max_norm):
        v_arr = np.array(v, dtype=np.float64)
        v_norm = np.linalg.norm(v_arr)
        if v_norm == 0:
            return False
        new_max_norm = max(v_norm, current_max_norm)
        # Compute distances from v to all vectors in vec_array
        diffs = vec_array.astype(np.float64) - v_arr
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        if np.min(dists) < new_max_norm - 1e-9:
            return False
        return True
    
    def try_add(v, vec_array, current_max_norm):
        if is_valid_addition_fast(v, vec_array, current_max_norm):
            v_arr = np.array(v, dtype=np.int64)
            vec_array = np.vstack([vec_array, v_arr])
            new_norm = np.linalg.norm(v_arr.astype(np.float64))
            current_max_norm = max(current_max_norm, new_norm)
            return vec_array, current_max_norm, True
        return vec_array, current_max_norm, False
    
    # Generate candidate vectors
    # Try sparse vectors with entries in {-2, -1, 1, 2}
    
    # Candidate 1: single entry ±2
    for i in range(D):
        for val in (2, -2):
            v = [0] * D
            v[i] = val
            vec_array, current_max_norm, added = try_add(v, vec_array, current_max_norm)
    
    # Candidate 2: two entries, one ±2 and one ±1
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            for vi in (2, -2):
                for vj in (1, -1):
                    v = [0] * D
                    v[i] = vi
                    v[j] = vj
                    vec_array, current_max_norm, added = try_add(v, vec_array, current_max_norm)
    
    # Candidate 3: three entries, all ±1
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    vec_array, current_max_norm, added = try_add(v, vec_array, current_max_norm)
    
    # Candidate 4: two entries, both ±2
    for i, j in itertools.combinations(range(D), 2):
        for si in (2, -2):
            for sj in (2, -2):
                v = [0] * D
                v[i], v[j] = si, sj
                vec_array, current_max_norm, added = try_add(v, vec_array, current_max_norm)
    
    # Candidate 5: three entries, one ±2 and two ±1
    for i, j, k in itertools.combinations(range(D), 3):
        for vi in (2, -2):
            for vj in (1, -1):
                for vk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = vi, vj, vk
                    vec_array, current_max_norm, added = try_add(v, vec_array, current_max_norm)
    
    # Check time
    if time.time() - start_time > TIME_LIMIT:
        return vec_array
    
    # Candidate 6: vectors from the Gosset lattice or similar
    # Try some hand-crafted vectors based on known constructions
    
    # Let's also try removing some vectors to allow adding more
    # This is more complex; skip for now
    
    return vec_array
