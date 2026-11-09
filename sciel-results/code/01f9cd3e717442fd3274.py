# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product, combinations

def kissing_number11() -> np.ndarray:
    """
    Constructs a set of 11-dimensional integer points maximizing k, such that
    min_{i != j} ||p_i - p_j||^2 >= max_i ||p_i||^2.
    
    Strategy: Greedy incremental packing.
    1. Start with the 220 weight-2 vectors (norm^2=2).
    2. Iterate through candidate vectors with increasing squared norm (3, 4, ...).
    3. For each candidate, check if it satisfies the distance constraint with all
       existing points. If yes, add it.
    4. Stop when no more candidates can be added or time limit is approached.
    """
    start_time = time.time()
    time_limit = 580.0  # Safety margin for 600s limit
    
    d = 11
    points = []
    
    # Helper to check if a point can be added
    def can_add(p):
        p_norm_sq = np.dot(p, p)
        for q in points:
            diff = p - q
            dist_sq = np.dot(diff, diff)
            if dist_sq < p_norm_sq:
                return False
        return True
    
    # Step 1: Add all weight-2 vectors (norm^2 = 2)
    # These are (±1, ±1, 0, ..., 0)
    # There are C(11,2) * 4 = 55 * 4 = 220 such vectors.
    for i in range(d):
        for j in range(i + 1, d):
            for si in (1, -1):
                for sj in (1, -1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points.append(p)
    
    current_max_k = len(points)
    
    # Step 2: Try to add vectors with higher norms
    # We generate candidates by increasing norm squared.
    # Norm squared = sum(x_i^2).
    # We look for integer vectors with small norm.
    
    # To be efficient, we generate candidates in a structured way.
    # Norm^2 = 3: (±1, ±1, ±1, 0, ...) -> C(11,3)*8 = 165*8 = 1320
    # Norm^2 = 4: (±2, 0, ...) -> 11*2 = 22; (±1, ±1, ±1, ±1, 0...) -> C(11,4)*16 = 330*16 = 5280
    # Norm^2 = 5: (±1, ±1, ±1, ±1, ±1, 0...) -> C(11,5)*32 = 462*32 = 14784; (±2, ±1, 0...) -> 11*10*4 = 440
    
    # We will iterate through possible "weights" (non-zero positions) and values.
    # To keep it simple and fast, we'll generate candidates up to a certain norm.
    # However, the number of candidates grows rapidly.
    # Let's focus on adding a few high-norm vectors if possible.
    
    # Optimization: Pre-compute norms of existing points to check constraint faster?
    # Constraint: ||p-q||^2 >= ||p||^2  => ||q||^2 - 2p.q >= 0 => 2p.q <= ||q||^2.
    # Also symmetry: ||p-q||^2 >= ||q||^2 => ||p||^2 - 2p.q >= 0 => 2p.q <= ||p||^2.
    # So we need 2p.q <= min(||p||^2, ||q||^2).
    
    # Let's try adding vectors with norm^2 = 3 first.
    # These are permutations of (±1, ±1, ±1, 0, ..., 0).
    
    # To avoid generating millions of candidates blindly, we can use a heuristic:
    # Only try to add a candidate if it is "far enough" from the centroid or some subset.
    # But the simplest robust approach is just to try them all if the count is manageable.
    
    # Count of norm^2=3 vectors: 1320.
    # Check cost: 1320 * 220 dot products. ~300k ops. Very fast.
    
    # Count of norm^2=4 vectors:
    # Type A: (±2, 0, ...): 11 * 2 = 22
    # Type B: (±1, ±1, ±1, ±1, 0, ...): C(11,4) * 4^4 = 330 * 256 = 84480
    # Check cost: 84502 * 222 (approx) ~ 18M ops. Still fast in Python/NumPy?
    # 18M dot products of length 11 might take a few seconds in pure Python loops.
    # We should vectorize the check if possible, or limit the search.
    
    # Let's implement a vectorized check for a batch of candidates.
    
    def try_add_batch(candidates):
        """
        candidates: list of numpy arrays.
        Returns list of candidates that can be added.
        """
        if not candidates:
            return []
        
        cand_arr = np.array(candidates, dtype=np.int64)
        cand_norms_sq = np.sum(cand_arr * cand_arr, axis=1)
        
        added = []
        
        # If we have many existing points, we can't easily vectorize against 'points' 
        # unless we convert 'points' to an array too.
        # Let's maintain 'points' as a list for flexibility, but use an array for checks.
        
        if not points:
            added = candidates
            return added
            
        points_arr = np.array(points, dtype=np.int64)
        points_norms_sq = np.sum(points_arr * points_arr, axis=1)
        
        # For each candidate, check against all points.
        # This is a k x N matrix multiplication.
        # dist_sq = ||c||^2 + ||p||^2 - 2 c.p
        # We need dist_sq >= ||c||^2  => ||p||^2 - 2 c.p >= 0 => 2 c.p <= ||p||^2
        # AND dist_sq >= ||p||^2  => ||c||^2 - 2 c.p >= 0 => 2 c.p <= ||c||^2
        
        # So condition is: 2 c.p <= min(||c||^2, ||p||^2)
        
        # Compute dot products: M = cand_arr @ points_arr.T  (k x N)
        M = cand_arr @ points_arr.T
        
        # We need M[i, j] * 2 <= min(cand_norms_sq[i], points_norms_sq[j])
        
        # Create matrix of limits: L[i, j] = min(cand_norms_sq[i], points_norms_sq[j])
        # L = np.minimum(cand_norms_sq[:, np.newaxis], points_norms_sq[np.newaxis, :])
        
        # Check: 2 * M <= L
        # valid_mask = 2 * M <= L
        
        # To avoid huge memory usage if k and N are large, we can do this in chunks.
        # But k (candidates) and N (points) are small enough (< 10000) that 10000x10000 int64 is 800MB.
        # That's too big. We need to chunk.
        
        chunk_size = 1000
        for start in range(0, len(candidates), chunk_size):
            end = min(start + chunk_size, len(candidates))
            c_chunk = cand_arr[start:end]
            c_norms_chunk = cand_norms_sq[start:end]
            
            M_chunk = c_chunk @ points_arr.T
            L_chunk = np.minimum(c_norms_chunk[:, np.newaxis], points_norms_sq[np.newaxis, :])
            
            valid = np.all(2 * M_chunk <= L_chunk, axis=1)
            
            for idx in np.where(valid)[0]:
                added.append(candidates[start + idx])
                
            # Update points array for subsequent chunks? 
            # The problem is that if we add a point, it affects the check for later candidates in the same batch?
            # No, the check is against the *existing* set. If we add one, does it invalidate another?
            # The condition is pairwise. If c1 and c2 are both valid against 'points', are they valid against each other?
            # We must also check c1 vs c2.
            # This greedy approach is tricky if we add multiple at once.
            # Simpler: Add one by one, but that's slow.
            # Alternative: Only add if valid against current 'points'. Then re-verify pairwise among new ones?
            # Given the high density, likely few will be added.
            # Let's just add the valid ones to a temp list, then filter the temp list pairwise?
            # Actually, if c1 and c2 are both in 'candidates', and both are valid against 'points',
            # we still need ||c1-c2||^2 >= max(||c1||^2, ||c2||^2).
            
            # For now, let's just collect valid ones against 'points'.
            # Then we will filter the collected ones against each other in a second pass.
            
        return added

    # Generate candidates for norm^2 = 3
    candidates_norm3 = []
    for i in range(d):
        for j in range(i+1, d):
            for k in range(j+1, d):
                for s1 in (1, -1):
                    for s2 in (1, -1):
                        for s3 in (1, -1):
                            p = np.zeros(d, dtype=np.int64)
                            p[i] = s1
                            p[j] = s2
                            p[k] = s3
                            candidates_norm3.append(p)
    
    # Try to add norm3 candidates
    valid_norm3 = try_add_batch(candidates_norm3)
    
    # Filter valid_norm3 pairwise
    # If len(valid_norm3) is small, do it explicitly.
    final_new_points = []
    for p in valid_norm3:
        ok = True
        for q in final_new_points:
            diff = p - q
            if np.dot(diff, diff) < max(np.dot(p,p), np.dot(q,q)):
                ok = False
                break
        if ok:
            final_new_points.append(p)
            
    points.extend(final_new_points)
    
    # Check time
    if time.time() - start_time > time_limit * 0.5:
        # Skip heavier searches
        pass
    else:
        # Generate candidates for norm^2 = 4
        # Type A: (±2, 0, ...)
        candidates_norm4 = []
        for i in range(d):
            for s in (2, -2):
                p = np.zeros(d, dtype=np.int64)
                p[i] = s
                candidates_norm4.append(p)
        
        # Type B: (±1, ±1, ±1, ±1, 0, ...)
        for i in range(d):
            for j in range(i+1, d):
                for k in range(j+1, d):
                    for l in range(k+1, d):
                        for s1 in (1, -1):
                            for s2 in (1, -1):
                                for s3 in (1, -1):
                                    for s4 in (1, -1):
                                        p = np.zeros(d, dtype=np.int64)
                                        p[i] = s1
                                        p[j] = s2
                                        p[k] = s3
                                        p[l] = s4
                                        candidates_norm4.append(p)
                                        
        # This is ~84k candidates. 
        # try_add_batch might be slow if implemented naively with Python loops inside.
        # My try_add_batch uses vectorized ops. 
        # 84k * 220 (points) = 18M ops. 
        # NumPy handles this fast.
        
        valid_norm4 = try_add_batch(candidates_norm4)
        
        # Filter pairwise
        final_new_points4 = []
        for p in valid_norm4:
            ok = True
            for q in final_new_points4:
                diff = p - q
                if np.dot(diff, diff) < max(np.dot(p,p), np.dot(q,q)):
                    ok = False
                    break
            if ok:
                final_new_points4.append(p)
                
        points.extend(final_new_points4)
        
        # Check time again
        if time.time() - start_time > time_limit * 0.8:
            pass
        else:
            # Norm^2 = 5
            # Type A: (±2, ±1, 0, ...)
            candidates_norm5 = []
            for i in range(d):
                for j in range(i+1, d):
                    for s1 in (2, -2):
                        for s2 in (1, -1):
                            p = np.zeros(d, dtype=np.int64)
                            p[i] = s1
                            p[j] = s2
                            candidates_norm5.append(p)
            
            # Type B: (±1, ±1, ±1, ±1, ±1, 0, ...)
            for i in range(d):
                for j in range(i+1, d):
                    for k in range(j+1, d):
                        for l in range(k+1, d):
                            for m in range(l+1, d):
                                for s1 in (1, -1):
                                    for s2 in (1, -1):
                                        for s3 in (1, -1):
                                            for s4 in (1, -1):
                                                for s5 in (1, -1):
                                                    p = np.zeros(d, dtype=np.int64)
                                                    p[i] = s1
                                                    p[j] = s2
                                                    p[k] = s3
                                                    p[l] = s4
                                                    p[m] = s5
                                                    candidates_norm5.append(p)
                                                    
            # This is getting large. C(11,5)*32 = 14784 + 440 = ~15k.
            # 15k * 220+ = 3M ops. Fine.
            
            valid_norm5 = try_add_batch(candidates_norm5)
            
            final_new_points5 = []
            for p in valid_norm5:
                ok = True
                for q in final_new_points5:
                    diff = p - q
                    if np.dot(diff, diff) < max(np.dot(p,p), np.dot(q,q)):
                        ok = False
                        break
                if ok:
                    final_new_points5.append(p)
                    
            points.extend(final_new_points5)

    # Finalize
    points_arr = np.array(points, dtype=np.int64)
    points_arr = np.around(points_arr).astype(np.int64)
    
    return points_arr

# EVOLVE-BLOCK-END
