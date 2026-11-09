"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm. Score is the number of vectors k.
    
    Strategy:
    1. Start with a known good configuration (e.g., all vectors with two non-zero
       entries of ±1, giving norm sqrt(2) and pairwise distances either 0, sqrt(2), 
       or 2. But wait, if two vectors share one coordinate, distance is sqrt(1^2+1^2)=sqrt(2) 
       only if they differ in both, or if one has (1,0) and other (0,1) in same pair... 
       Actually the seed program uses pairs of coordinates with ±1. Let's analyze:
       v_ij^±± has norm sqrt(2). Two such vectors:
       - Same pair, same signs: distance 0 (but we only include distinct sign combos, so not an issue)
       - Same pair, different signs: e.g., (1,1) vs (1,-1) -> distance sqrt(0+4)=2 >= sqrt(2) ✓
       - Different pairs sharing one coordinate: e.g., (1,0,0,...) wait no, each vector has exactly two 1s.
         v1 = e_i + e_j, v2 = e_i + e_k (j≠k): distance = ||(0,1,0,...,0,1,0)|| = sqrt(1+1) = sqrt(2) ✓
         v1 = e_i + e_j, v2 = -e_i + e_k: distance = ||(2,1,0,...,0,1)|| = sqrt(4+1+1) = sqrt(6) ✓
       - Disjoint pairs: distance = sqrt(4) = 2 ✓
       
       So the seed configuration is valid! It has C(11,2)*4 = 55*4 = 220 vectors.
       
    2. Try to add more vectors. We can try vectors with different norms.
       The constraint is: for all pairs, ||v_i - v_j|| >= max_norm.
       
       Let's try adding vectors of norm 1 (unit vectors e_i and -e_i).
       If we add e_i, then for any existing vector v with norm sqrt(2):
       ||e_i - v|| must be >= sqrt(2).
       If v has a 1 in position i: ||e_i - v|| = ||v - e_i||. If v = e_i + e_j, then ||v - e_i|| = ||e_j|| = 1 < sqrt(2). Violation!
       So we can't add unit vectors if we have the pair vectors.
       
    3. Alternative: Use a different base configuration.
       Consider vectors with one non-zero entry of ±1: 2*11 = 22 vectors, norm 1.
       Pairwise distances: ||e_i - e_j|| = sqrt(2) >= 1 ✓, ||e_i - (-e_i)|| = 2 >= 1 ✓.
       So 22 vectors is valid but worse than 220.
       
    4. What about mixing? We need to find a set where min_pairwise_dist >= max_norm.
       Let M = max norm. All pairwise distances >= M.
       
       The seed gives 220 vectors with M = sqrt(2).
       
       Can we do better? The kissing number in dimension 11 is known to be at least 504 (from Leech lattice constructions).
       Actually, the kissing number in 11 dimensions is known to be 504.
       
       The Leech lattice is in 24 dimensions, but we can project or use constructions.
       
       A better approach: use the construction from the D_11 lattice or similar.
       
    5. Let's try a greedy approach:
       - Start with a candidate pool of integer vectors with small norms.
       - Greedily add vectors that satisfy the constraint.
       
    6. Even better: use known constructions.
       For dimension 11, we can use the following:
       - All vectors with exactly one non-zero entry of ±1: 22 vectors, norm 1.
       - All vectors with exactly two non-zero entries of ±1: 220 vectors, norm sqrt(2).
       
       But these can't be combined as shown above.
       
    7. Let's try vectors of the form: (±1, ±1, ±1, 0, ..., 0) with three non-zero entries.
       Norm = sqrt(3).
       Count: C(11,3) * 2^3 = 165 * 8 = 1320 vectors.
       
       Check pairwise distances:
       - Two vectors sharing 2 coordinates: e.g., (1,1,1,0,...) and (1,1,-1,0,...): distance = sqrt(0+0+4+0+...) = 2. Is 2 >= sqrt(3)? Yes, 2 > 1.732.
       - Two vectors sharing 1 coordinate: e.g., (1,1,1,0,...) and (1,1,0,1,0,...): wait, the second has non-zero in positions 1,2,4.
         distance = ||(0,0,1,-1,0,...)|| = sqrt(1+1) = sqrt(2) < sqrt(3). Violation!
       
       So this doesn't work directly.
       
    8. Let's go back to the seed and try to enhance it.
       The seed has 220 vectors. Can we add more?
       
       What if we add vectors of the form (1,1,1,0,...,0) but only in a way that they don't conflict?
       Actually, let's just implement a greedy search with a larger candidate pool.
       
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Start with the seed configuration
    vectors = []
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(tuple(v))
    
    vectors = list(set(vectors))  # Remove duplicates
    max_norm = np.sqrt(2)
    
    # Generate candidate vectors with small norms
    # Try vectors with 1, 2, 3 non-zero entries
    candidates = []
    
    # Already have 2-nonzero vectors
    # Try 3-nonzero vectors
    for combo in product(range(D), repeat=3):
        i, j, k = combo
        if len(set([i, j, k])) == 3:
            for signs in product([1, -1], repeat=3):
                v = [0] * D
                v[i] = signs[0]
                v[j] = signs[1]
                v[k] = signs[2]
                candidates.append(tuple(v))
    
    # Try 1-nonzero vectors
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = sign
            candidates.append(tuple(v))
    
    # Remove candidates that are already in vectors
    existing = set(vectors)
    candidates = [c for c in candidates if c not in existing]
    
    # Greedily add candidates
    def is_compatible(v, vec_list, max_n):
        """Check if v is compatible with all vectors in vec_list given max norm max_n."""
        v_arr = np.array(v, dtype=np.float64)
        v_norm_sq = np.dot(v_arr, v_arr)
        if v_norm_sq > max_n * max_n + 1e-9:
            return False
        for u in vec_list:
            u_arr = np.array(u, dtype=np.float64)
            diff = v_arr - u_arr
            dist_sq = np.dot(diff, diff)
            if dist_sq < max_n * max_n - 1e-9:
                return False
        return True
    
    # Sort candidates by norm (prefer smaller norms to allow more additions)
    candidates.sort(key=lambda v: np.dot(np.array(v), np.array(v)))
    
    added = 0
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        
        cand_arr = np.array(cand, dtype=np.float64)
        cand_norm_sq = np.dot(cand_arr, cand_arr)
        
        # Check compatibility with all existing vectors
        compatible = True
        for existing_v in vectors:
            existing_arr = np.array(existing_v, dtype=np.float64)
            diff = cand_arr - existing_arr
            dist_sq = np.dot(diff, diff)
            if dist_sq < max_norm * max_norm - 1e-9:
                compatible = False
                break
        
        if compatible:
            vectors.append(cand)
            added += 1
    
    if len(vectors) == 0:
        return np.array([[1] + [0]*10], dtype=np.int64)
    
    return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
