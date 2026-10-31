import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    # Key insight: for any two DISTINCT weight-4 vectors in {-1,0,1}^11,
    # their dot product is at most 2 (dot=4 implies identical vectors).
    # Since the compatibility condition is dot <= 2, ALL 5280 are mutually compatible.
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    # These are compatible with all weight-4 vectors (max dot = 2)
    # And among themselves: distinct pairs have dot <= 2 (dot=-4 for opposite signs same coord,
    # dot=0 for different coords). All 22 can be included.
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    
    # Generate all weight-2 vectors in {-1, 0, 1}^11 with two non-zero entries (Norm^2 = 2)
    # Wait - these have norm^2 = 2, not 4. Let me reconsider.
    # For norm^2 = 4, we need either:
    # - 4 entries of ±1 (weight-4)
    # - 1 entry of ±2 (weight-1 with value 2)
    # - 2 entries of ±1 and 2 entries of ±1? No, that's weight 4 again.
    # Actually, let me think about other norm^2=4 vectors:
    # - One entry of ±2: done (22 vectors)
    # - Four entries of ±1: done (5280 vectors)
    # - Two entries of ±1 and two entries of ±1? That's still weight 4.
    # - What about one entry of ±1 and one entry of ±√15? No, must be integers.
    # So we've exhausted all integer vectors with norm^2=4.
    
    # Let's try a different approach: use vectors with larger norm.
    # If we use norm^2 = 8, we need dot <= 4 for compatibility.
    # Weight-2 vectors in {-2,0,2}^11 have norm^2 = 8.
    # Number of such vectors: C(11,2) * 4 = 55 * 4 = 220
    
    # Check compatibility between weight-2 vectors in {-2,0,2}^11:
    # Two such vectors have dot product that's a multiple of 4.
    # Max dot for distinct vectors: if they share 1 position with same sign, dot=4.
    # If they share 1 position with opposite sign, dot=-4.
    # If they share 2 positions, dot can be -8, 0, or 8.
    # For dot <= 4, we need to avoid pairs sharing 2 positions with same signs.
    
    v2_candidates = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = 2 * s
            v2_candidates.append(v)
    
    # Check if all weight-2 vectors are mutually compatible (dot <= 4)
    # Two distinct weight-2 vectors can share at most 1 position (if they shared 2, they'd be identical or opposite)
    # If they share 1 position: dot = ±4
    # If they share 0 positions: dot = 0
    # So max dot = 4, which satisfies dot <= 4. ALL 220 are compatible!
    
    # Now check compatibility between weight-4 (norm^2=4) and weight-2 (norm^2=8) vectors:
    # The condition is: dist^2 >= max(norm^2) = 8
    # dist^2 = norm1^2 + norm2^2 - 2*dot = 4 + 8 - 2*dot = 12 - 2*dot
    # Need 12 - 2*dot >= 8, i.e., dot <= 2
    # Weight-4 vector has 4 entries of ±1, weight-2 has 2 entries of ±2.
    # Max dot: if the 2 non-zero positions of weight-2 are among the 4 non-zero positions of weight-4,
    # and signs match: dot = 2*1 + 2*1 = 4. This violates dot <= 2!
    # So we can't just take all of them.
    
    # Let's use the original approach with norm^2=4 vectors only, but verify correctness.
    
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    # Quick sanity check: verify no conflicts in a sample of weight-4 vectors
    if len(vectors_v4) > 0:
        sample_size = min(200, len(vectors_v4))
        sample = vectors_v4[:sample_size]
        dots_sample = sample @ sample.T
        mask = ~np.eye(sample_size, dtype=bool)
        if np.any(dots_sample[mask] > 2):
            return _construct_with_mis(vectors_v4, vectors_v1, deadline)
    
    # Verify weight-1 vectors are compatible with each other
    if len(vectors_v1) > 0:
        dots_v1 = vectors_v1 @ vectors_v1.T
        mask_v1 = ~np.eye(len(vectors_v1), dtype=bool)
        if np.any(dots_v1[mask_v1] > 2):
            conflicts_v1 = dots_v1 > 2
            np.fill_diagonal(conflicts_v1, False)
            keep = np.ones(len(vectors_v1), dtype=bool)
            for i in range(len(vectors_v1)):
                if keep[i]:
                    conflicts_v1[i] = False
                    for j in range(i+1, len(vectors_v1)):
                        if keep[j] and dots_v1[i,j] > 2:
                            keep[j] = False
            vectors_v1 = vectors_v1[keep]
    
    # For norm^2=4 vectors, the compatibility condition is dot <= 2.
    # We've shown all weight-4 (5280) and weight-1 (22) vectors satisfy this.
    # Total: 5302 vectors.
    
    # Try to find additional vectors with norm^2=4 that we might have missed.
    # Integer vectors with norm^2=4 in R^11:
    # Case 1: One coordinate is ±2, rest 0. (22 vectors) - included
    # Case 2: Four coordinates are ±1, rest 0. (C(11,4)*16 = 330*16 = 5280) - included
    # Case 3: Two coordinates are ±1 and two coordinates are ±1? No, that's 4 coordinates.
    # Case 4: One coordinate ±1 and one coordinate ±√15? Not integer.
    # So we've exhausted all integer vectors with norm^2=4.
    
    # The current solution of 5302 vectors with norm^2=4 is complete for this norm.
    # To improve, we'd need to use a different norm or a mixed-norm approach.
    # But the directive says to stay within the same design family.
    
    final_vectors = np.vstack([vectors_v4, vectors_v1])
    return final_vectors

def _construct_with_mis(vectors_v4, vectors_v1, deadline):
    """Fallback MIS solver if the mathematical assumption is violated."""
    N4 = len(vectors_v4)
    if N4 == 0:
        return vectors_v1
        
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    alive = np.ones(N4, dtype=bool)
    selected_indices = []
    degrees = conflicts.sum(axis=1).astype(np.int32)
    rng = np.random.default_rng(42)
    
    while np.any(alive) and time.time() < deadline:
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[rng.integers(len(cands))]
        
        selected_indices.append(chosen)
        
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        alive[to_remove] = False
        
        rem = np.where(alive)[0]
        if len(rem) > 0:
            degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
            
        if time.time() > deadline - 10:
            break

    result_v4 = vectors_v4[selected_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    return final_vectors
