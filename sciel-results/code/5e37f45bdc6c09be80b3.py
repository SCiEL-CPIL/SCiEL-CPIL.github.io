import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    """
    Multi-shell union strategy:
    1. Start with all 220 weight-2 vectors (norm²=2) - guaranteed valid set.
    2. Greedily add weight-4 vectors (norm²=6 or 8) that satisfy the global
       min-distance constraint against all existing points.
    
    The constraint: for all pairs (a,b), ||a-b||² >= max_norm_sq.
    For a point p with norm²=n_p and existing point q with norm²=n_q:
      ||p-q||² = n_p + n_q - 2*(p·q) >= max_norm_sq
    So we need p·q <= (n_p + n_q - max_norm_sq) / 2 for all existing q.
    """
    d = 11
    deadline = time.time() + 580  # safety margin
    
    # Phase 1: All weight-2 vectors (norm² = 2)
    base_points = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in (-1, 1):
                for sj in (-1, 1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    base_points.append(p)
    
    points = np.array(base_points, dtype=np.int64)
    max_norm_sq = 2
    n = len(points)
    
    # Precompute norms for existing points
    norms_sq = np.sum(points * points, axis=1)
    
    # Phase 2: Generate candidate weight-4 vectors
    # Weight-4 means exactly 4 non-zero coordinates, each +/-1
    # Norm² = 4 (if all same sign pattern doesn't matter, it's always 4)
    # Wait: weight-4 with entries +/-1 gives norm² = 4
    # Weight-4 with entries in {-1, 0, 1} and exactly 4 nonzeros: norm² = 4
    
    # Actually let's think more carefully:
    # Weight-2: exactly 2 nonzeros, each ±1 → norm² = 2
    # Weight-4: exactly 4 nonzeros, each ±1 → norm² = 4
    # We could also have weight-3: exactly 3 nonzeros → norm² = 3
    
    # Let's try adding weight-4 vectors (norm²=4) to the base set.
    # Constraint: for new point p (norm²=4) and existing q (norm²=2):
    #   ||p-q||² = 4 + 2 - 2*(p·q) = 6 - 2*(p·q) >= max_norm_sq
    # Initially max_norm_sq = 2, so need 6 - 2*(p·q) >= 2 → p·q <= 2
    # But as we add points with norm²=4, max_norm_sq becomes 4.
    # Then for two new points p1, p2 (both norm²=4):
    #   ||p1-p2||² = 8 - 2*(p1·p2) >= 4 → p1·p2 <= 2
    # And for new p (norm²=4) vs old q (norm²=2):
    #   ||p-q||² = 6 - 2*(p·q) >= 4 → p·q <= 1
    
    # So the constraint tightens. Let's build incrementally.
    
    # Generate all weight-4 vectors: C(11,4) * 2^4 = 330 * 16 = 5280
    candidates = []
    for coords in combinations(range(d), 4):
        for signs in product([-1, 1], repeat=4):
            p = np.zeros(d, dtype=np.int64)
            for idx, s in zip(coords, signs):
                p[idx] = s
            candidates.append(p)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Greedy: try to add candidates one by one
    # For efficiency, track the current max_norm_sq
    # A candidate p is valid if for all existing points q:
    #   ||p - q||² >= max_norm_sq_current
    # i.e., ||p||² + ||q||² - 2*(p·q) >= max_norm_sq_current
    # i.e., p·q <= (||p||² + ||q||² - max_norm_sq_current) / 2
    
    # Since all existing points have norm²=2 (initially), and new ones have norm²=4:
    # We need to check against all points in the current set.
    
    # For speed, we'll batch-check candidates against the current point set.
    # points has shape (n, 11), candidates has shape (n_cand, 11)
    # Dot products: candidates @ points.T → (n_cand, n)
    # norm_sq of candidates: all 4
    # norm_sq of points: varies (2 for base, 4 for added)
    
    # Let's do this in batches for memory
    batch_size = 512
    added_count = 0
    
    # Shuffle candidates for diversity
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_cand)
    
    for start in range(0, n_cand, batch_size):
        if time.time() > deadline:
            break
        end = min(start + batch_size, n_cand)
        batch_idx = perm[start:end]
        batch = candidates[batch_idx]
        n_batch = len(batch)
        
        # Current max norm squared
        curr_max_norm = int(np.max(norms_sq))
        
        # Check each candidate in batch
        # For candidate p (norm²=4), need for all q in points:
        #   4 + norms_sq[j] - 2*(p·q_j) >= curr_max_norm
        #   p·q_j <= (4 + norms_sq[j] - curr_max_norm) / 2
        
        # Compute dot products: batch @ points.T → (n_batch, n)
        dots = batch @ points.T  # (n_batch, n)
        
        # Threshold for each pair: (4 + norms_sq[j] - curr_max_norm) / 2
        # norms_sq shape: (n,)
        thresholds = (4 + norms_sq - curr_max_norm) / 2  # (n,)
        
        # Valid if all dots[i, j] <= thresholds[j]
        valid_mask = np.all(dots <= thresholds[np.newaxis, :], axis=1)
        
        # Add valid candidates
        valid_indices = np.where(valid_mask)[0]
        for vi in valid_indices:
            p = batch[vi]
            points = np.vstack([points, p[np.newaxis, :]])
            norms_sq = np.append(norms_sq, 4)
            n += 1
            added_count += 1
        
        # Also check: if we added any, the max_norm_sq might have changed
        # (it becomes max(2, 4) = 4 after first addition)
        # But we recompute curr_max_norm each batch, so it's fine.
    
    # Phase 3: Try weight-3 vectors (norm²=3) - these might fit
    # For p (norm²=3) vs q (norm²=2): ||p-q||² = 5 - 2*(p·q) >= max_norm
    # For p (norm²=3) vs r (norm²=4): ||p-r||² = 7 - 2*(p·r) >= max_norm
    # If max_norm = 4: need p·q <= 0.5 → p·q <= 0 (integers), p·r <= 1.5 → p·r <= 1
    
    # Generate weight-3 vectors: C(11,3) * 2^3 = 165 * 8 = 1320
    if time.time() < deadline - 30:
        cand3 = []
        for coords in combinations(range(d), 3):
            for signs in product([-1, 1], repeat=3):
                p = np.zeros(d, dtype=np.int64)
                for idx, s in zip(coords, signs):
                    p[idx] = s
                cand3.append(p)
        cand3 = np.array(cand3, dtype=np.int64)
        n_cand3 = len(cand3)
        perm3 = rng.permutation(n_cand3)
        
        for start in range(0, n_cand3, batch_size):
            if time.time() > deadline:
                break
            end = min(start + batch_size, n_cand3)
            batch_idx = perm3[start:end]
            batch = cand3[batch_idx]
            
            curr_max_norm = int(np.max(norms_sq))
            dots = batch @ points.T
            # p has norm²=3
            thresholds = (3 + norms_sq - curr_max_norm) / 2
            valid_mask = np.all(dots <= thresholds[np.newaxis, :], axis=1)
            
            valid_indices = np.where(valid_mask)[0]
            for vi in valid_indices:
                p = batch[vi]
                points = np.vstack([points, p[np.newaxis, :]])
                norms_sq = np.append(norms_sq, 3)
                n += 1
    
    # Final verification
    points = np.around(points).astype(np.int64)
    k = points.shape[0]
    
    # Verify: no zero vector
    assert np.all(np.any(points != 0, axis=1)), "Zero vector found"
    
    # Verify: min pairwise dist² >= max norm²
    max_norm = int(np.max(np.sum(points * points, axis=1)))
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            dsq = int(np.sum(diff * diff))
            if dsq < min_dist_sq:
                min_dist_sq = dsq
    
    assert min_dist_sq >= max_norm, f"Invalid: min_dist={min_dist_sq} < max_norm={max_norm}"
    
    return points
