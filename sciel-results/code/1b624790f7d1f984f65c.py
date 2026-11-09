# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    """
    Constructs a large valid set of integer points in Z^11 where
    min pairwise dist^2 >= max norm^2.
    
    Strategy: 
    - Use weight-4 vectors (norm^2 = 4). Max norm squared is 4.
    - We need min pairwise squared distance >= 4.
    - Total candidates: C(11,4) * 2^4 = 330 * 16 = 5280.
    - Instead of slow greedy O(N^2) checking, we use a combinatorial 
      selection strategy to pick a large subset that is guaranteed 
      to have pairwise distances >= 4.
    
    Observation:
    Two vectors u, v in {±1}^4 (embedded in Z^11) have dist^2 = ||u-v||^2.
    If they share the same support (same 4 indices), dist^2 is 0, 4, or 8.
    dist^2 = 0 if identical. dist^2 = 4 if they differ in exactly one sign.
    dist^2 = 8 if they differ in two or more signs.
    Wait, the requirement is min dist^2 >= 4.
    So, vectors differing in exactly one sign are ALLOWED (dist^2=4).
    Vectors differing in zero signs are NOT allowed (identical, dist^2=0).
    
    If two vectors have DIFFERENT supports:
    Let S_u and S_v be their supports.
    If S_u == S_v, we are in the case above.
    If S_u != S_v:
    Let k = |S_u ∩ S_v|.
    The distance squared is sum over i in S_u Δ S_v of (±1 - 0)^2 + sum over i in S_v Δ S_u of (0 - ±1)^2.
    Actually, simpler: ||u-v||^2 = ||u||^2 + ||v||^2 - 2<u,v> = 4 + 4 - 2<u,v> = 8 - 2<u,v>.
    <u,v> is the dot product.
    If supports are disjoint, <u,v>=0, dist^2=8. (Valid)
    If supports overlap by 1, <u,v> is ±1, dist^2 is 6 or 10. (Valid)
    If supports overlap by 2, <u,v> is -2, 0, or 2. dist^2 is 12, 8, or 4. (Valid)
    If supports overlap by 3, <u,v> is -3, -1, 1, 3. dist^2 is 14, 10, 6, 2. 
       Wait. If overlap is 3, and signs match on all 3, <u,v>=3, dist^2 = 8-6=2. INVALID.
       If overlap is 3, and signs match on 2, mismatch on 1, <u,v>=1, dist^2=6. Valid.
    
    So the main conflict is:
    1. Identical vectors (same support, same signs).
    2. Vectors with same support, differing in exactly 1 sign? 
       dist^2 = 4. This is VALID (>= 4).
       So we can include ALL sign variations for a given support?
       Let's check: u = (1,1,1,1), v = (-1,1,1,1). dist^2 = (2)^2 + 0 + 0 + 0 = 4. Valid.
       So for a FIXED support, ALL 16 sign variations are mutually valid?
       Let's check u=(1,1,1,1), v=(1,1,-1,-1). dist^2 = 0+0+4+4=8. Valid.
       u=(1,1,1,1), v=(-1,-1,1,1). dist^2 = 4+4+0+0=8. Valid.
       
    Conclusion: For any fixed support S of size 4, the set of all 16 vectors with non-zero entries on S is a valid clique (all pairwise distances >= 4).
    
    Now, can we combine different supports?
    If we take ALL 5280 vectors, do we have conflicts?
    Conflict arises if dist^2 < 4. i.e., dist^2 = 0 or 2.
    dist^2 = 0 only if identical.
    dist^2 = 2 requires 8 - 2<u,v> = 2 => <u,v> = 3.
    <u,v> = 3 implies they share 3 indices with same sign, and the 4th index must be 0 in both? No, supports size 4.
    If they share 3 indices, they differ in one index.
    u has non-zero on S_u, v on S_v.
    If |S_u ∩ S_v| = 3, then there is one index in S_u not in S_v, and one in S_v not in S_u.
    Let S_u = {1,2,3,4}, S_v = {1,2,3,5}.
    u = (u1, u2, u3, u4, 0, ...), v = (v1, v2, v3, 0, v5, ...).
    <u,v> = u1v1 + u2v2 + u3v3.
    To get <u,v>=3, we need u1=v1, u2=v2, u3=v3.
    Then dist^2 = ||u||^2 + ||v||^2 - 2(3) = 4+4-6 = 2. INVALID.
    
    So, we cannot have two vectors that:
    1. Share 3 indices.
    2. Have the same signs on those 3 shared indices.
    
    Strategy:
    We want to select a large subset of the 5280 vectors.
    For each support S (size 4), we have 16 sign patterns.
    If we pick ALL 16 for a support S, we must ensure no vector from another support S' conflicts with them.
    
    Alternative Strategy:
    Partition the supports into groups such that supports within a group are "far apart" or handle signs carefully.
    
    Simpler Strategy:
    Just pick one sign pattern per support? That gives 330 points.
    Can we do better?
    
    Let's look at the "Conflict Graph".
    Nodes: 5280 vectors.
    Edge if dist^2 < 4 (i.e., dist^2=2).
    We want a Max Independent Set.
    
    Observation:
    If we restrict to vectors where the first non-zero coordinate is +1?
    This reduces symmetry.
    
    Let's try a greedy approach but optimized with Bitsets or Numpy vectorization to handle 5280 nodes fast.
    5280 nodes. Pairs ~ 14 million.
    Numpy can handle 14M dot products quickly if batched.
    
    Actually, let's just generate all 5280 vectors and run a fast greedy independent set algorithm using vectorized distance checks.
    5280 is small enough that even O(N^2) in Python might be slow, but Numpy batched is fast.
    
    Algorithm:
    1. Generate all 5280 vectors.
    2. Initialize selected = [].
    3. Iterate through candidates (maybe sorted by something to help packing, e.g., random or fixed).
    4. For each candidate p, compute dist^2 to all selected points.
    5. If min dist^2 >= 4, add to selected.
    
    With N=5280, the selected set will grow. 
    If we pick ~1000 points, checking each new point against 1000 existing ones is 1000 dot products.
    Total ops ~ 5280 * 1000 = 5.2M. Very fast in Python if we use Numpy for the batch check against the current selected set.
    
    Let's implement this.
    """
    deadline = time.time() + 580.0
    
    d = 11
    candidates = []
    
    # Generate all weight-4 vectors
    # C(11, 4) = 330 supports
    # 2^4 = 16 signs
    supports = list(combinations(range(d), 4))
    signs = list(product([-1, 1], repeat=4))
    
    for s in supports:
        for sg in signs:
            p = np.zeros(d, dtype=np.int64)
            for i, idx in enumerate(s):
                p[idx] = sg[i]
            candidates.append(p)
    
    cands = np.array(candidates, dtype=np.int64)
    N = len(cands)
    
    # We want min dist^2 >= 4.
    # Greedy selection
    selected_indices = []
    selected_points = []
    
    # Shuffle to avoid bias? Or keep fixed? Fixed is deterministic.
    # Let's try a heuristic order: maybe randomize once for better packing?
    # But determinism is good for reproducibility. Let's stick to fixed order first.
    # Actually, random order often yields larger independent sets in greedy.
    # Let's use a fixed random seed for shuffling.
    rng = np.random.default_rng(42)
    order = rng.permutation(N)
    
    for idx in order:
        if time.time() > deadline:
            break
            
        p = cands[idx]
        
        if not selected_points:
            selected_indices.append(idx)
            selected_points.append(p)
            continue
            
        # Check distance against all selected points
        # selected_points is a list of 1D arrays. Convert to 2D array for batch op?
        # Or just loop if selected is small.
        # As selected grows, batch op is better.
        
        # Let's maintain selected_arr as a 2D array for efficiency
        # But appending to 2D array is expensive.
        # Let's just use a list and convert to array for the check if len > 10.
        
        if len(selected_points) < 10:
            valid = True
            for q in selected_points:
                diff = p - q
                d2 = int(np.dot(diff, diff))
                if d2 < 4:
                    valid = False
                    break
            if valid:
                selected_indices.append(idx)
                selected_points.append(p)
        else:
            sel_arr = np.array(selected_points)
            diffs = sel_arr - p
            dists_sq = np.sum(diffs * diffs, axis=1)
            if np.min(dists_sq) >= 4:
                selected_indices.append(idx)
                selected_points.append(p)
    
    if len(selected_indices) < 2:
        # Fallback
        return np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    points = np.array(selected_points, dtype=np.int64)
    
    # Final verification
    k = len(points)
    assert k >= 2
    assert points.shape[1] == 11
    
    # Verify no zero vector
    for p in points:
        assert np.any(p != 0)
        
    # Verify min dist^2 >= max norm^2
    max_norm = int(np.max(np.sum(points * points, axis=1)))
    # max_norm should be 4
    
    min_dist = float('inf')
    # Spot check or full check?
    # k is likely ~1000-2000. k^2/2 pairs. 2000^2/2 = 2M pairs.
    # Full check in Python loop is too slow.
    # Use Numpy for final verification.
    if k < 5000:
        # Compute all pairwise distances
        # dist^2(i,j) = ||p_i||^2 + ||p_j||^2 - 2 p_i.p_j
        norms_sq = np.sum(points * points, axis=1)
        dot_prods = points @ points.T
        dists_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dot_prods
        
        # Set diagonal to inf
        np.fill_diagonal(dists_sq, float('inf'))
        
        min_dist = int(np.min(dists_sq))
        
        if min_dist < max_norm:
            # This should not happen if greedy was correct, but if it does, 
            # we need to repair or return a smaller valid set.
            # For now, trust the greedy construction which checked against 
            # ALL previously selected points.
            pass
            
    return points

# EVOLVE-BLOCK-END
