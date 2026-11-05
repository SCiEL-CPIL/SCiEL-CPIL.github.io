def check_validity_vectorized(candidates, S, D, dist2, max_dist):
    """
    candidates: np.array of indices to test
    S: list of indices in current set
    D: list of sets of distances for each point in S
    Returns: boolean array of same length as candidates, True if valid
    """
    if len(S) == 0:
        return np.ones(len(candidates), dtype=bool)
    if len(S) == 1:
        # Only need to check distance to the single point? No, just 1 point, no triangle.
        # Wait, if S has 1 point, adding P makes S size 2. No triangle.
        # So any P is valid?
        # Yes, 2 points cannot form a triangle.
        return np.ones(len(candidates), dtype=bool)
    
    K = len(S)
    # Get distances from candidates to S
    # dist2 shape: (4096, 4096)
    # candidates shape: (M,)
    # S shape: (K,)
    # vals shape: (M, K)
    vals = dist2[candidates][:, S]
    
    # Check 1: Distances from candidate to all points in S must be distinct
    sorted_vals = np.sort(vals, axis=1)
    has_dup = np.any(sorted_vals[:, :-1] == sorted_vals[:, 1:], axis=1)
    valid = ~has_dup
    
    # Check 2: For each point Q in S, dist(candidate, Q) must not be in D[Q]
    # We iterate over Q in S
    for j in range(K):
        q_idx = S[j]
        d_set = D[j]
        # Create a lookup array for this specific set
        # Instead of creating a full max_dist array every time, 
        # we can use np.isin or a smaller lookup if possible.
        # Given max_dist ~ 8000, creating a bool array is fast.
        lookup = np.zeros(max_dist + 1, dtype=bool)
        for d in d_set:
            lookup[d] = True
        
        # vals[:, j] are the distances to Q
        # We need ~lookup[vals[:, j]]
        # Mask out invalid candidates
        valid &= ~lookup[vals[:, j]]
        
    return valid
