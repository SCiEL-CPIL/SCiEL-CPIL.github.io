import time
import numpy as np
from itertools import product

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Generate a diverse pool of candidate integer vectors
    # We focus on vectors with small coordinates to keep distances manageable
    # but large enough to have many candidates.
    # Let's try coordinates in {-2, -1, 0, 1, 2}.
    # Total candidates: 5^11 = 48,828,125. This is too large to store all at once efficiently in Python lists,
    # but we can generate them in chunks or use a smarter pool.
    
    # A better pool: 
    # - Vectors with norm squared <= 10 (approx)
    # - Or just a large random sample? No, deterministic is better for reproducibility if needed, 
    #   but random is fine if seeded.
    # Let's use a structured pool: 
    # 1. All vectors with 0, 1, or 2 non-zero elements (values +/-1, +/-2).
    # 2. All vectors with 3 non-zero elements (values +/-1).
    
    candidates = []
    
    # Type 1: 1 non-zero element (values +/-1, +/-2)
    # 11 dims * 4 values = 44 vectors
    for i in range(D):
        for val in [-2, -1, 1, 2]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = val
            candidates.append(v)
            
    # Type 2: 2 non-zero elements (values +/-1, +/-2)
    # C(11,2) * 16 values = 55 * 16 = 880 vectors
    for i in range(D):
        for j in range(i + 1, D):
            for v1 in [-2, -1, 1, 2]:
                for v2 in [-2, -1, 1, 2]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = v1
                    v[j] = v2
                    candidates.append(v)
                    
    # Type 3: 3 non-zero elements (values +/-1)
    # C(11,3) * 8 values = 165 * 8 = 1320 vectors
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for v1 in [-1, 1]:
                    for v2 in [-1, 1]:
                        for v3 in [-1, 1]:
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = v1
                            v[j] = v2
                            v[k] = v3
                            candidates.append(v)
                            
    # Type 4: 4 non-zero elements (values +/-1)
    # C(11,4) * 16 = 330 * 16 = 5280 vectors
    # This might be getting expensive for O(N^2) distance checks if N is large.
    # Let's estimate pool size: 44 + 880 + 1320 + 5280 = ~7500.
    # Greedy insertion:
    # We want to select a subset S such that min_{u,v in S, u!=v} ||u-v|| >= max_{u in S} ||u||.
    # Note: The problem statement says "minimum pairwise distance is at least the maximum vector norm".
    # This implies we should probably normalize or just check the condition as stated.
    # If we pick vectors with varying norms, the condition is strict.
    # Often, kissing number problems assume equal norm. The prompt says "Normalizing each vector to norm 2 certifies...".
    # This suggests the raw integer vectors should ideally have the same norm, or at least the condition holds.
    # If norms vary, max_norm is determined by the largest vector.
    # To maximize k, it's best to use vectors of the same norm (or close to it) to allow tighter packing.
    # Let's filter candidates to those with norm squared == 4 (vectors like (2,0,..), (1,1,0,..), (1,-1,0,..)) 
    # or norm squared == 2 (1,0,..)
    # If we use norm=2 (sq=4), then min_dist must be >= 2.
    # If we use norm=1 (sq=1), min_dist must be >= 1.
    # Vectors with sq=1: (1,0,..). Max dist between two such is 2 (if orthogonal) or sqrt(2) (if not?).
    # Wait, distance between (1,0) and (0,1) is sqrt(2) < 2 (norm). So they fail.
    # Distance between (1,0) and (-1,0) is 2 >= 1. OK.
    # So for norm=1, we can only have orthogonal vectors? No, (1,0) and (0,1) dist=sqrt(2) >= 1. OK.
    # (1,0) and (1,1)? Norm of (1,1) is sqrt(2). Max norm is sqrt(2). Dist is 1 < sqrt(2). Fail.
    # So mixing norms is tricky.
    
    # Strategy: Select a single norm class.
    # Let's try norm squared = 4 (Norm = 2).
    # Candidates with norm sq 4:
    # - (2, 0, ...) : 11 * 2 = 22 vectors
    # - (1, 1, 0, ...) : C(11,2) * 4 (signs) = 55 * 4 = 220 vectors
    # - (1, -1, 0, ...) : Included in above (signs).
    # Total ~242 vectors. This matches the E8+A3 count roughly.
    
    # Let's try norm squared = 2 (Norm = sqrt(2)).
    # Candidates: (1, 1, 0, ...) -> 55 * 4 = 220.
    # Min dist required: sqrt(2).
    # Dist between (1,1,0) and (1,-1,0) is 2 >= 1.414. OK.
    # Dist between (1,1,0) and (-1,1,0) is 2 >= 1.414. OK.
    # Dist between (1,1,0) and (0,1,1) is sqrt(1+0+1) = sqrt(2) >= sqrt(2). OK.
    # This looks promising.
    
    # Let's try norm squared = 1 (Norm = 1).
    # Candidates: (1, 0, ...) -> 22 vectors.
    # Min dist required: 1.
    # Dist between (1,0) and (0,1) is sqrt(2) >= 1. OK.
    # Only 22 vectors. Low score.
    
    # Let's try norm squared = 5 (Norm = sqrt(5) ~ 2.23).
    # Candidates: (2, 1, 0, ...) -> 11*10*2*2 = 440?
    # i for 2, j for 1, signs.
    # C(11,2) * 2 (which is 2) * 4 (signs) = 55 * 2 * 4 = 440.
    # Min dist required: sqrt(5).
    
    # Let's stick to Norm Squared = 4 (Norm 2) as it's a standard kissing configuration norm.
    # Pool:
    pool = []
    
    # Type A: (2, 0, ..., 0) and (-2, 0, ..., 0)
    for i in range(D):
        v = np.zeros(D, dtype=np.int64)
        v[i] = 2
        pool.append(v)
        v[i] = -2
        pool.append(v)
        
    # Type B: (1, 1, 0, ...), (1, -1, 0, ...), etc.
    # Two indices i, j. Values are +/-1.
    for i in range(D):
        for j in range(i + 1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    pool.append(v)
                    
    pool = np.array(pool, dtype=np.int64)
    
    # Greedy Farthest-Point / Best-Insertion Search
    # We want to select a subset S from pool.
    # Condition: min_{u,v in S, u!=v} ||u-v|| >= max_{u in S} ||u||.
    # Since all vectors in pool have norm 2, max_norm = 2.
    # Condition: min_{u,v in S, u!=v} ||u-v|| >= 2.
    # Squared condition: min ||u-v||^2 >= 4.
    
    selected = []
    available = list(range(len(pool)))
    
    # Start with a random vector? Or just the first one?
    # Let's start with pool[0].
    selected.append(pool[0])
    available.remove(0)
    
    # To speed up, we can precompute pairwise distances? 
    # Pool size ~ 242. 242^2 = ~58k distances. Very fast.
    # Let's precompute the distance matrix for the pool.
    
    # dist[i, j] = ||pool[i] - pool[j]||^2
    # We only care if dist[i, j] >= 4.
    
    # Compute dist matrix
    # pool shape (N, D)
    # dist_sq[i,j] = sum((pool[i]-pool[j])**2)
    # Using broadcasting: (pool[:, np.newaxis, :] - pool[np.newaxis, :, :])**2
    # This creates an array of size N x N x D. 242^2 * 11 ~ 640k floats. Fine.
    
    diff = pool[:, np.newaxis, :] - pool[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Now greedy selection:
    # At each step, find the candidate c in available such that 
    # min_{s in selected} dist_sq[c, s] is maximized.
    # And that min value must be >= 4.
    
    # Actually, a better greedy is:
    # Add the vector that maximizes the minimum distance to the current set.
    # If that max-min-distance is < 4, we stop? No, we just can't add it.
    # We should pick the best available. If the best available has min-dist < 4, 
    # then NO available vector can be added (since it's the max of the mins).
    # So we stop.
    
    # Wait, is it possible that a vector with a smaller min-dist now allows adding MORE vectors later?
    # Usually, greedy farthest-point is a good heuristic for packing.
    
    # Optimization: 
    # Maintain an array `min_dists` for each available candidate: 
    # min_dists[c] = min_{s in selected} dist_sq[c, s]
    # Initially, min_dists[c] = dist_sq[c, 0] for all c != 0.
    # At each step:
    # 1. Find c in available with max min_dists[c].
    # 2. If min_dists[c] < 4, break.
    # 3. Add c to selected.
    # 4. Remove c from available.
    # 5. Update min_dists for all remaining available candidates:
    #    min_dists[a] = min(min_dists[a], dist_sq[a, c])
    
    N = len(pool)
    min_dists = dist_sq.copy() # (N, N)
    # Initialize with distance to first selected (index 0)
    # But we removed 0 from available.
    # min_dists[:, 0] is dist to 0.
    # We want min over selected. Selected = {0}.
    # So current min_dists for candidate i is dist_sq[i, 0].
    
    available_mask = np.ones(N, dtype=bool)
    available_mask[0] = False
    
    selected_indices = [0]
    
    # Pre-compute which candidates are valid against index 0
    # If dist_sq[i, 0] < 4, i can never be added? 
    # Not necessarily, if we add other vectors, the min dist might change? 
    # No, min dist is the MIN over all selected. If dist to 0 is small, it's the min.
    # So if dist_sq[i, 0] < 4, i is permanently invalid.
    
    # Let's filter available_mask initially.
    valid_mask = dist_sq[:, 0] >= 4
    available_mask &= valid_mask
    
    current_min_dists = dist_sq[:, 0].copy()
    
    while time.time() < deadline:
        # Find best candidate in available_mask
        # Set non-available to -1 to ignore them in argmax
        scores = np.where(available_mask, current_min_dists, -1)
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
        if best_score < 4:
            break
            
        # Add best_idx
        selected_indices.append(best_idx)
        available_mask[best_idx] = False
        
        # Update current_min_dists for remaining available
        # current_min_dists[a] = min(current_min_dists[a], dist_sq[a, best_idx])
        # Only for available ones
        updates = dist_sq[:, best_idx]
        # We want to update only where available_mask is True
        # np.minimum(current_min_dists, updates, out=current_min_dists, where=available_mask)
        # But we must not update the best_idx itself (it's already False)
        # And we don't care about non-available.
        
        # Vectorized update
        np.minimum(current_min_dists, updates, out=current_min_dists, where=available_mask)
        
        # Optional: prune available_mask further?
        # If current_min_dists[a] < 4, then a is invalid forever.
        # We can set available_mask[a] = False for those.
        newly_invalid = available_mask & (current_min_dists < 4)
        available_mask[newly_invalid] = False
        
    # Construct result
    if not selected_indices:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result = pool[selected_indices]
    
    # Final Validation
    # Check min pairwise distance >= max norm
    norms_sq = np.sum(result**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    min_dist_sq = np.inf
    for i in range(len(result)):
        diffs = result[i+1:] - result[i]
        dists_sq = np.sum(diffs**2, axis=1)
        if len(dists_sq) > 0:
            min_d = np.min(dists_sq)
            if min_d < min_dist_sq:
                min_dist_sq = min_d
                
    if min_dist_sq < max_norm_sq:
        # Fallback to a known good subset or empty
        # The greedy should have ensured this, but let's be safe.
        # If it failed, we return the first few that are valid?
        # Actually, if greedy logic is correct, this shouldn't happen.
        # But if it does, return empty to score 0 rather than invalid? 
        # Or return the seed's result? We don't have it.
        # Let's just return the result and hope the evaluator's check is lenient? 
        # No, "invalid result... scores 0".
        # So we must return a valid result.
        # If the greedy produced an invalid one, we need to fix it.
        # Simplest fix: return only the vectors that satisfy the condition with the first vector?
        # Or just return the first vector (k=1, valid).
        return result[:1]

    return result

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
