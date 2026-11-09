import numpy as np
import time
import math
import random

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 610  # Safety margin before 630s limit
    
    n = 33
    # Precompute indices for all 7-combinations
    # C(33, 7) = 4,292,145
    # Storing these as int32 arrays: 4.3M * 7 * 4 bytes = ~120 MB. Acceptable.
    # But generating them takes time. Let's use itertools.combinations in a list.
    # 4.3M tuples is heavy in Python memory. 
    # Alternative: Generate on the fly? No, too slow.
    # Alternative: Use a smaller subset? No, we need exact count.
    # Let's try to generate the index arrays once.
    
    # Actually, 4.3M combinations might be too slow to generate in Python loop.
    # Let's use a recursive generator or a precomputed list if possible.
    # Wait, 4.3M is not that large for numpy operations, but generating the list in Python is O(C(n,k)).
    # Let's estimate generation time. 4M iterations in Python is ~0.5-1s. That's fine.
    
    comb_indices = []
    from itertools import combinations
    for c in combinations(range(n), 7):
        comb_indices.append(c)
    
    # Convert to numpy array of shape (N_COMBS, 7)
    # N_COMBS = 4292145
    # Memory: 4.3M * 7 * 4 bytes = 119 MB. OK.
    comb_arr = np.array(comb_indices, dtype=np.int32)
    n_combs = len(comb_arr)
    
    # Delete comb_indices to save memory
    del comb_indices
    
    def count_convex_7goms_vectorized(points):
        """
        Vectorized count of convex 7-gons.
        points: (33, 2) array
        Returns: count
        """
        # Get coordinates for all 7-tuples
        # pts_comb: (N_COMBS, 7, 2)
        pts_comb = points[comb_arr]
        
        x = pts_comb[:, :, 0]
        y = pts_comb[:, :, 1]
        
        # To check if 7 points are in convex position:
        # 1. Compute convex hull of the 7 points.
        # 2. If the number of points on the hull is 7, it is convex.
        # 
        # Vectorized convex hull is tricky.
        # Alternative: A set of points is in convex position iff every point is a vertex of the convex hull.
        # Equivalent: For every point i in the set, it is NOT inside the convex hull of the other 6.
        # Or: For every point i, there exists a line through i such that all other 6 points are on one side.
        # This is equivalent to: The minimum angle subtended by any two other points from i is > 180? No.
        # 
        # Simpler check for small N (7):
        # A point p_i is on the hull of the 7 points if and only if it is not inside the convex hull of the other 6.
        # For 7 points, p_i is NOT on hull if p_i is inside the convex hull of the other 6.
        # 
        # How to check if p_i is inside convex hull of 6 other points efficiently?
        # This is still hard to vectorize fully without a hull algorithm.
        #
        # Let's use the property:
        # Points are in convex position iff the area of their convex hull is equal to the area of the polygon formed by the points sorted by angle around the centroid?
        # No, that's circular.
        #
        # Let's use the "turn" definition from the prompt.
        # "points are sorted lexicographically, for each pivot the later points are sorted by atan2 angle around it"
        # This description is for a specific DP.
        #
        # Let's stick to a robust geometric check.
        # For a set of 7 points, it is convex iff for every point i, the other 6 points do NOT contain i in their convex hull.
        #
        # Vectorized point-in-hull check for 6 points is complex.
        #
        # Alternative approach for vectorization:
        # Check if the 7 points form a convex polygon when sorted by angle around their centroid.
        # 1. Compute centroid (mean of 7 points).
        # 2. Compute angles of each point relative to centroid.
        # 3. Sort points by angle.
        # 4. Check if the polygon formed by these sorted points is convex (all cross products same sign).
        #
        # Sorting 7 elements per row in a (4.3M, 7) array?
        # np.sort is available.
        
        # 1. Centroid
        centroid = np.mean(pts_comb, axis=1) # (N_COMBS, 2)
        
        # 2. Vectors from centroid
        vecs = pts_comb - centroid[:, np.newaxis, :] # (N_COMBS, 7, 2)
        
        # 3. Angles
        angles = np.arctan2(vecs[:, :, 1], vecs[:, :, 0]) # (N_COMBS, 7)
        
        # 4. Sort by angle
        # argsort gives indices
        sort_idx = np.argsort(angles, axis=1) # (N_COMBS, 7)
        
        # Gather sorted points
        # This is a fancy indexing operation.
        # We need to index pts_comb with sort_idx.
        # pts_comb shape (N, 7, 2), sort_idx shape (N, 7)
        # We want result (N, 7, 2) where result[i, k] = pts_comb[i, sort_idx[i, k]]
        
        # Construct index array
        # This might be memory heavy. 4.3M * 7 indices.
        # Let's try.
        
        row_idx = np.arange(n_combs)[:, np.newaxis] # (N, 1)
        # sorted_pts = pts_comb[row_idx, sort_idx, :]
        # This creates a view or copy? Copy. 4.3M * 7 * 2 * 8 bytes = ~480 MB.
        # Total memory might be tight but likely OK on modern systems.
        
        sorted_pts = pts_comb[row_idx, sort_idx, :]
        
        x_s = sorted_pts[:, :, 0]
        y_s = sorted_pts[:, :, 1]
        
        # 5. Check convexity of the sorted polygon
        # For a convex polygon, all cross products of consecutive edges must have the same sign.
        # Edge i is p_i -> p_{i+1}.
        # Vector e_i = p_{i+1} - p_i.
        # Cross product e_i x e_{i+1}.
        
        # Wrap around
        x_next = np.roll(x_s, -1, axis=1)
        y_next = np.roll(y_s, -1, axis=1)
        
        # Edge vectors
        ex = x_next - x_s
        ey = y_next - y_s
        
        # Cross products
        # c_i = ex_i * ey_{i+1} - ey_i * ex_{i+1}
        ex_next = np.roll(ex, -1, axis=1)
        ey_next = np.roll(ey, -1, axis=1)
        
        crosses = ex * ey_next - ey * ex_next # (N, 7)
        
        # Check if all crosses are > 0 or all < 0 (with tolerance)
        # If any is near 0, it's degenerate (collinear), which is not strictly convex in the "convex position" sense usually, 
        # but the prompt says "convex position". Usually means vertices of convex hull.
        # The prompt's DP uses > 1e-9 for turn.
        # So we require strict convexity.
        
        # If the polygon is convex, all crosses have the same sign.
        # min_cross = np.min(crosses, axis=1)
        # max_cross = np.max(crosses, axis=1)
        
        # Condition: (min_cross > 1e-9) or (max_cross < -1e-9)
        # Note: If the points are sorted by angle, the polygon is simple.
        # If it's convex, the turns are all consistent.
        
        min_cross = np.min(crosses, axis=1)
        max_cross = np.max(crosses, axis=1)
        
        is_convex = (min_cross > 1e-9) | (max_cross < -1e-9)
        
        return np.sum(is_convex)

    def is_valid(points):
        if points.shape != (33, 2):
            return False
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return False
        if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
            return False
        # Distances
        # Vectorized distance check
        # Compute pairwise squared distances
        # (33, 1, 2) - (33, 2) -> (33, 33, 2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        # Mask diagonal
        np.fill_diagonal(dist_sq, np.inf)
        if np.min(dist_sq) < 1e-12:
            return False
        
        # Collinearity
        # Check all triples? 33C3 = 5456.
        # Vectorized?
        # We can check if any area is < 1e-9.
        # Area of triangle i, j, k = 0.5 * |cross(p_j - p_i, p_k - p_i)|
        # We can compute cross products for all pairs of edges?
        # Simpler: loop over i, vectorize j, k?
        # Or just use the fact that if points are in general position, this is rare.
        # But we must be strict.
        
        # Let's use a loop for i, and vectorize the rest.
        for i in range(n-2):
            p_i = points[i]
            # Vectors from i to all j > i
            others = points[i+1:]
            # Cross product of (p_j - p_i) and (p_k - p_i)
            # This is not straightforward vectorized for all j,k pairs without making a big matrix.
            # C(32, 2) = 496. Small enough to loop?
            # 33 * 496 = 16k operations. Very fast.
            for j in range(i+1, n-1):
                p_j = points[j]
                v1 = p_j - p_i
                # v2 for all k > j
                v2s = points[j+1:] - p_i
                crosses = v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0]
                if np.any(np.abs(crosses) < 1e-9):
                    return False
        return True

    # Initialization
    # Use a "corner clustering" strategy as per inspiration.
    # Place points in 4 corners, slightly jittered.
    # This minimizes convex hulls initially.
    
    random.seed(42)
    np.random.seed(42)
    
    # 4 corners: (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)
    # 33 points / 4 = 8.25.
    # 9, 8, 8, 8 points.
    
    corners = [
        (0.1, 0.1),
        (0.9, 0.1),
        (0.1, 0.9),
        (0.9, 0.9)
    ]
    
    pts = np.zeros((33, 2))
    idx = 0
    # Distribute points
    counts = [9, 8, 8, 8]
    for i, count in enumerate(counts):
        cx, cy = corners[i]
        # Small jitter
        jitters = np.random.uniform(-0.05, 0.05, (count, 2))
        pts[idx:idx+count] = np.array([cx, cy]) + jitters
        idx += count
        
    # Ensure within bounds
    pts = np.clip(pts, 0.0, 1.0)
    
    # Check validity, if not, spread out
    if not is_valid(pts):
        pts = np.random.rand(33, 2) * 0.9 + 0.05
        if not is_valid(pts):
             pts = np.random.rand(33, 2)

    best_pts = pts.copy()
    best_score = 1.0 / (1.0 + count_convex_7goms_vectorized(best_pts))
    
    # Simulated Annealing
    # T starts high, cools down.
    # Move: pick a random point, move it by a random Gaussian perturbation.
    # Batch: generate B candidate moves, evaluate all, pick the best (or accept worse with prob).
    # To be efficient, we can evaluate a batch of candidates and take the best one that improves, 
    # or use SA logic on the batch.
    
    # Let's do:
    # 1. Generate B candidates (e.g., 500).
    # 2. Evaluate C for each.
    # 3. Find the best candidate.
    # 4. Accept if better, or with probability exp(-(delta_score)/T) if worse.
    
    B = 200 # Batch size. 200 * 4.3M ops = 860M ops. NumPy can do this in ~1-2s.
    # 600s / 2s = 300 iterations.
    # Each iteration tries 200 moves. Total 60k moves.
    
    T = 0.1
    T_min = 1e-4
    cooling = 0.99
    
    current_pts = best_pts.copy()
    current_score = best_score
    
    while time.time() < deadline - 5:
        # Generate B candidate moves
        # Pick B random indices
        move_indices = np.random.randint(0, n, B)
        # Generate perturbations
        # Scale perturbation with T? Or fixed?
        # Fixed scale might get stuck. Scale with T.
        scale = 0.1 * math.sqrt(T)
        perturbations = np.random.normal(0, scale, (B, 2))
        
        candidates = np.tile(current_pts, (B, 1))
        candidates[np.arange(B), move_indices] += perturbations
        
        # Clip to [0, 1]
        candidates = np.clip(candidates, 0.0, 1.0)
        
        # Evaluate scores
        # We need to count convex 7-gons for each candidate.
        # This is the expensive part.
        # count_convex_7goms_vectorized expects a (33, 2) array.
        # We have a (B, 33, 2) array.
        # We need to modify the counting function to handle a batch.
        
        # Let's write a batched counter.
        # It will be slower per point but allows vectorization across the batch dimension?
        # No, the inner loops (combinations) are the same.
        # We can just loop over the batch?
        # B=200. 200 calls to count_convex_7goms_vectorized.
        # Each call takes ~0.5s? 200 * 0.5 = 100s per iteration. Too slow.
        
        # We need to vectorize across the batch dimension too.
        # This is complex.
        
        # Alternative: Reduce B.
        # If 1 call takes 0.5s, we can do 10 calls in 5s.
        # 600s / 5s = 120 iterations.
        # 120 * 10 moves = 1200 moves.
        
        # Let's measure the time of one count.
        
        # Revised Plan:
        # Use a smaller batch, e.g., B=10.
        # Or optimize the counter.
        
        # Let's try B=10.
        B = 10
        move_indices = np.random.randint(0, n, B)
        perturbations = np.random.normal(0, 0.05, (B, 2)) # Fixed scale 0.05
        
        candidates = np.tile(current_pts, (B, 1))
        candidates[np.arange(B), move_indices] += perturbations
        candidates = np.clip(candidates, 0.0, 1.0)
        
        best_cand_score = -np.inf
        best_cand_idx = -1
        
        for i in range(B):
            if not is_valid(candidates[i]):
                continue
            score_i = 1.0 / (1.0 + count_convex_7goms_vectorized(candidates[i]))
            if score_i > best_cand_score:
                best_cand_score = score_i
                best_cand_idx = i
        
        if best_cand_idx != -1:
            delta = best_cand_score - current_score
            # Accept if better, or with prob
            if delta > 0 or random.random() < math.exp(delta / T):
                current_pts = candidates[best_cand_idx].copy()
                current_score = best_cand_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min

    # Final check
    if not is_valid(best_pts):
        # Fallback
        return np.random.rand(33, 2) * 0.9 + 0.05

    return best_pts
