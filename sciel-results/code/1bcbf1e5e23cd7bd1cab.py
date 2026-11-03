import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    D = 11

    # Phase 1: Generate all minimal vectors of the D_11 root lattice.
    # These are vectors with exactly two non-zero entries, each +/-1.
    # Total: C(11,2) * 4 = 55 * 4 = 220 vectors.
    # Norm squared = 2, so norm = sqrt(2).
    # Minimum pairwise distance in this set is sqrt(2).
    # This satisfies: min_dist >= max_norm.
    
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for s_i in (-1, 1):
                for s_j in (-1, 1):
                    if time.time() > deadline - 10.0:
                        return np.array(vectors, dtype=np.int8)
                    v = np.zeros(D, dtype=np.int8)
                    v[i] = s_i
                    v[j] = s_j
                    vectors.append(v)
        
        if time.time() > deadline - 10.0:
            break

    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)

    base_vectors = np.array(vectors, dtype=np.int8)
    n_base = len(base_vectors)
    
    # Phase 2: Try to add unit vectors (norm 1) that are compatible.
    # A unit vector e_i has norm 1. Max norm becomes max(sqrt(2), 1) = sqrt(2).
    # Condition: dist(e_i, u) >= sqrt(2) for all u in current set.
    # dist^2(e_i, u) = 1 + ||u||^2 - 2*e_i.u
    # For u with norm^2 = 2: dist^2 = 3 - 2*e_i.u
    # e_i.u is the i-th component of u, which is -1, 0, or 1.
    # If e_i.u = 1: dist^2 = 1 < 2. Conflict.
    # If e_i.u = -1: dist^2 = 5 >= 2. OK.
    # If e_i.u = 0: dist^2 = 3 >= 2. OK.
    # So e_i conflicts only with vectors that have +1 in position i.
    
    # Let's build a conflict matrix for unit vectors vs base vectors.
    # For each unit vector e_i, check if it conflicts with any base vector.
    # If it doesn't conflict with ANY, we can add it.
    # But it might conflict with some. In that case, we'd need to remove those.
    # Removing a base vector loses 1 vector, adding e_i gains 1. Net change = 0 at best.
    # Unless removing one base vector allows adding multiple unit vectors?
    # No, unit vectors e_i and e_j (i != j) have dist^2 = 2. OK.
    # But each e_i conflicts independently with specific base vectors.
    
    # Actually, let's check: can we add ALL 11 unit vectors if we remove conflicting base vectors?
    # e_i conflicts with base vectors having +1 at position i.
    # For position i, the base vectors with +1 at i are:
    #   e_i + e_j for all j != i (10 vectors)
    #   e_i - e_j for all j != i (10 vectors)
    # Total 20 vectors per position.
    # But these vectors are shared: e_i + e_j has +1 at i AND +1 at j.
    # It conflicts with both e_i and e_j.
    
    # Let's think about this differently.
    # If we want to add all 11 unit vectors, we need to remove all base vectors
    # that have +1 in any position.
    # A base vector e_i + e_j has +1 in positions i and j.
    # A base vector e_i - e_j has +1 in position i, -1 in position j.
    # So e_i - e_j conflicts with e_i but NOT with e_j (since e_j . (e_i - e_j) = -1, dist^2 = 3).
    # Wait: dist^2(e_j, e_i - e_j) = 1 + 2 - 2*(e_j . (e_i - e_j)) = 3 - 2*(0 - 1) = 3 + 2 = 5 >= 2. OK.
    # So e_i - e_j conflicts only with e_i, not with e_j.
    
    # If we add all 11 unit vectors, we need to remove:
    # - All e_i + e_j (have +1 in both i and j): 55 vectors
    # - All e_i - e_j (have +1 in i): 55 vectors
    # Total removed: 110.
    # Added: 11.
    # Net change: 11 - 110 = -99. Terrible.
    
    # What if we add a subset of unit vectors?
    # Suppose we add unit vectors for a set S of indices.
    # We need to remove base vectors that have +1 in any index in S.
    # A base vector e_i + e_j is removed if i in S or j in S.
    # A base vector e_i - e_j is removed if i in S.
    
    # Let |S| = s.
    # Number of e_i + e_j removed: those with at least one index in S.
    # Total e_i + e_j: 55.
    # Those with neither index in S: C(11-s, 2).
    # Removed: 55 - C(11-s, 2).
    
    # Number of e_i - e_j removed: those with i in S.
    # For each i in S, there are 10 vectors e_i - e_j.
    # Total: 10s.
    
    # Total removed: 55 - C(11-s, 2) + 10s.
    # Added: s.
    # Net change: s - (55 - C(11-s, 2) + 10s) = s - 55 + C(11-s, 2) - 10s = C(11-s, 2) - 9s - 55.
    
    # For s = 1: C(10,2) - 9 - 55 = 45 - 9 - 55 = -19. Bad.
    # For s = 0: C(11,2) - 0 - 55 = 55 - 55 = 0. (No change, as expected)
    # For s = 2: C(9,2) - 18 - 55 = 36 - 18 - 55 = -37. Bad.
    
    # So adding unit vectors is never beneficial. The D_11 set of 220 is locally optimal
    # with respect to adding unit vectors.
    
    # Phase 3: Try weight-3 vectors (norm^2 = 3).
    # If we add weight-3 vectors, max norm becomes sqrt(3).
    # Min dist must be >= sqrt(3), so dist^2 >= 3.
    # For two base vectors u, v: dist^2 = 4 - 2 u.v.
    # Need 4 - 2 u.v >= 3 => u.v <= 0.5 => u.v <= 0.
    # Base vectors with u.v > 0 exist (e.g., e_i+e_j and e_i+e_k have dot 1).
    # So we must remove all pairs of base vectors with dot > 0.
    # This is a massive removal.
    
    # For base vector u and weight-3 vector v: dist^2 = 2 + 3 - 2 u.v = 5 - 2 u.v.
    # Need 5 - 2 u.v >= 3 => u.v <= 1.
    
    # For two weight-3 vectors v1, v2: dist^2 = 3 + 3 - 2 v1.v2 = 6 - 2 v1.v2.
    # Need 6 - 2 v1.v2 >= 3 => v1.v2 <= 1.5 => v1.v2 <= 1.
    
    # This is a complex constraint satisfaction problem.
    # Given the time budget and the strong local optimality of the 220-vector set,
    # I will stick with the D_11 root lattice minimal vectors.
    
    # Final check: verify the set is valid.
    if n_base < 2:
        return base_vectors
    
    # Compute all pairwise distances
    # dist^2(i,j) = ||v_i||^2 + ||v_j||^2 - 2 v_i.v_j
    norms_sq = np.sum(base_vectors * base_vectors, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Compute dot products
    dot = base_vectors @ base_vectors.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dot
    np.fill_diagonal(dist_sq, 0)
    
    min_dist_sq = np.min(dist_sq[dist_sq > 0]) if np.any(dist_sq > 0) else 0
    
    # Check validity
    if min_dist_sq < max_norm_sq:
        # Should not happen for D_11 minimal vectors, but just in case
        # Fall back to a simpler valid set
        pass
    
    if time.time() > deadline - 5.0:
        return base_vectors
    
    return base_vectors
