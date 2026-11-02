"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

D = 11

# EVOLVE-BLOCK-START

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Uses a constructive approach based on the E11 root lattice structure.
    We generate all vectors with squared norm 2 (i.e., vectors with exactly two
    non-zero entries, each being ±1) and verify the constraint.
    
    For vectors with squared norm 2:
    - Each vector has exactly two non-zero coordinates, each ±1
    - The distance between two such vectors depends on their overlap
    
    Let's verify: if we take all such vectors, the minimum pairwise distance
    and maximum norm need to satisfy: min_dist >= max_norm
    
    Max norm = sqrt(2)
    
    For two vectors v, w with squared norm 2:
    - If they share no non-zero coordinates: dist^2 = 4, dist = 2 > sqrt(2) ✓
    - If they share one non-zero coordinate:
      - If same sign: dist^2 = (0)^2 + (1-1)^2 + ... = 2 (from the other coords)
        Actually: v has (a,b) at positions (i,j), w has (a,c) at positions (i,k)
        dist^2 = 0^2 + (b-c)^2 + ... where b,c ∈ {±1}
        If b=c: dist^2 = 0, that's zero distance - BAD
        If b≠c: dist^2 = 4
      - So sharing one coordinate with same sign gives distance 0 (same vector!)
      
    So we need to be more careful. Let's think about this differently.
    
    Actually, the constraint is: min pairwise distance >= max norm.
    
    If all vectors have the same norm r, then we need min_dist >= r.
    
    Let's use vectors of squared norm 4 (norm 2). These have entries that sum of squares = 4.
    Options: one entry ±2, or two entries ±√2 (not integer), or four entries ±1.
    
    Let's try vectors with four ±1 entries. Norm = 2.
    Then we need min pairwise distance >= 2.
    
    For two such vectors, dist^2 = sum of (v_i - w_i)^2.
    Each term is 0 (if same), 1 (if one is 0, other is ±1), 4 (if opposite signs), 1 (if one is ±1, other is 0).
    
    Actually let me reconsider. Let's just use the simple construction and verify.
    
    Strategy: Generate all vectors with exactly two non-zero entries (±1), giving norm sqrt(2).
    Then filter to ensure min pairwise distance >= sqrt(2).
    
    Two such vectors:
    - Disjoint support: dist^2 = 4, dist = 2 >= sqrt(2) ✓
    - Share exactly one position: 
      * Same value at shared position: the other positions differ by 1 each, so dist^2 = 1 + 1 = 2, dist = sqrt(2) >= sqrt(2) ✓ (tight)
      * Opposite values at shared position: dist^2 = 4 + 1 + 1 = 6, dist = sqrt(6) >= sqrt(2) ✓
    
    Wait, if they share one position with the same value, say both have +1 at position i.
    v has +1 at i, +1 at j. w has +1 at i, +1 at k (j≠k).
    dist^2 = (1-1)^2 + (1-0)^2 + (0-1)^2 + ... = 1 + 1 = 2. dist = sqrt(2). OK, equals max norm.
    
    v has +1 at i, +1 at j. w has +1 at i, -1 at k.
    dist^2 = 0 + 1 + 1 + 1 = 3. dist = sqrt(3) > sqrt(2). OK.
    
    v has +1 at i, -1 at j. w has +1 at i, +1 at k.
    dist^2 = 0 + 1 + 1 + 1 = 3. OK.
    
    v has +1 at i, -1 at j. w has -1 at i, +1 at k.
    dist^2 = 4 + 1 + 1 = 6. OK.
    
    So ALL such vectors satisfy the constraint! The minimum distance is exactly sqrt(2) = max norm.
    
    Number of such vectors: C(11,2) * 4 = 55 * 4 = 220.
    
    But wait - can we do better? Let's try norm 2 vectors (squared norm 4).
    Vectors with four ±1 entries: C(11,4) * 2^4 = 330 * 16 = 5280 vectors.
    But we need to check if min dist >= 2.
    
    Actually, let's just go with the 220 vectors first, which are guaranteed valid.
    Then we can try to add more.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Generate all vectors with exactly two non-zero entries (±1)
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    
    # Verify the constraint
    if len(vectors) == 0:
        return vectors
    
    # Compute norms
    norms_sq = np.sum(vectors ** 2, axis=1)
    max_norm = np.sqrt(np.max(norms_sq))
    
    # Compute pairwise distances
    # For efficiency, compute in chunks
    n = len(vectors)
    min_dist_sq = float('inf')
    
    # Sample check for large sets
    if n > 500:
        # Use a subset for verification
        idx = np.random.choice(n, min(n, 500), replace=False)
        sub = vectors[idx]
        diff = sub[:, np.newaxis, :] - sub[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
    else:
        diff = vectors[:, np.newaxis, :] - vectors[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
    
    # Check constraint: min_dist >= max_norm
    if min_dist_sq >= max_norm ** 2 - 1e-9:
        return vectors
    
    # If constraint not met (shouldn't happen for our construction), fall back
    # Try a greedy approach to find a valid subset
    selected = [vectors[0]]
    for i in range(1, n):
        v = vectors[i]
        ok = True
        for u in selected:
            d_sq = np.sum((v - u) ** 2)
            if d_sq < max_norm ** 2 - 1e-9:
                ok = False
                break
        if ok:
            selected.append(v)
        if time.time() > deadline:
            break
    
    return np.array(selected, dtype=np.int64)

# EVOLVE-BLOCK-END
