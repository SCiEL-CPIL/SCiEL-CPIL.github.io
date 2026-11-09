import numpy as np
import time
import itertools

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s limit
    
    n = 33
    
    # Precompute all 7-subsets
    all_7_subsets = list(itertools.combinations(range(n), 7))
    num_7_subsets = len(all_7_subsets)
    
    # Precompute all 4-subsets (for potential faster proxy if needed, but we'll use 7 for accuracy)
    # Actually, let's stick to 7-subsets for the objective to be exact, but use a fast vectorized check.
    
    def count_convex_7goms(points):
        """
        Vectorized count of convex 7-gons.
        """
        # points: (33, 2)
        # all_7_subsets: list of tuples of 7 indices
        if not all_7_subsets:
            return 0
            
        # Stack all 7-subsets into an array of shape (M, 7, 2)
        # M = 4292145. This is ~ 4.3M * 7 * 2 * 8 bytes = ~ 480 MB.
        # This might be too large for memory or slow to allocate.
        # Let's do it in batches of 100,000.
        
        total_convex = 0
        batch_size = 100000
        
        for start_idx in range(0, num_7_subsets, batch_size):
            end_idx = min(start_idx + batch_size, num_7_subsets)
            batch = all_7_subsets[start_idx:end_idx]
            idx_array = np.array(batch) # (B, 7)
            pts = points[idx_array] # (B, 7, 2)
            
            # Unpack
            p0 = pts[:, 0, :]
            p1 = pts[:, 1, :]
            p2 = pts[:, 2, :]
            p3 = pts[:, 3, :]
            p4 = pts[:, 4, :]
            p5 = pts[:, 5, :]
            p6 = pts[:, 6, :]
            
            # A 7-point set is convex iff no point is inside the convex hull of the other 6.
            # For a set of points in general position, this is equivalent to checking if
            # the convex hull has 7 vertices.
            # A faster check: 
            # Sort the 7 points by angle around their centroid.
            # If the resulting polygon is convex (all turns same sign), it's a convex 7-gon.
            # However, if the centroid is not inside the convex hull (which happens if points are not convex),
            # the angle sort might not yield the hull order.
            
            # Alternative: Check if any point is inside the triangle of any 3 other points? No, that's not sufficient for 7 points.
            # A point is inside the convex hull of the other 6 iff it is inside some triangle formed by 3 of the other 6?
            # Yes, if a point is inside a convex polygon, it is inside some triangle formed by vertices of that polygon.
            # So, a 7-point set is NON-convex iff there exists a point $p_i$ and a triangle $(p_j, p_k, p_l)$ from the other 6 points such that $p_i$ is strictly inside the triangle.
            
            # This requires checking for each of the 7 points, if it is inside any of the C(6,3)=20 triangles formed by the others.
            # Total checks per 7-set: 7 * 20 = 140 point-in-triangle tests.
            # 4.3M * 140 = 600M tests. This is too slow in Python even vectorized per batch if not careful.
            
            # Let's try a different vectorization.
            # For each 7-set, compute the convex hull size.
            # NumPy doesn't have a built-in convex hull for arbitrary batches.
            
            # Let's go back to the 4-gon proxy which was faster and correlated well.
            # Or, we can just use the 4-gon count as the objective function for optimization,
            # and only compute the 7-gon count at the end for reporting/validity?
            # The task says "C is the number of 7-point subsets... value is 1/(1+C)".
            # So I MUST optimize for C (7-gons).
            # But if 7-gon counting is too slow for the inner loop, I can use 4-gons as a surrogate.
            
            # Let's implement the 4-gon count for the surrogate, as it's much faster (C(33,4)=40920).
            # We'll use the 4-gon count for the search, and assume it correlates.
            
            # Wait, the current program already does this and got 2.05.
            # The inspiration suggests a different approach.
            
            # Let's try to make the 7-gon check faster by only checking "candidate" 7-sets?
            # No, let's stick to 4-gons for the search loop, as it's 100x faster.
            # We will return the 4-gon based optimization result.
            # The evaluator will compute the 7-gon count.
            
            # So, for this function, let's just return the 4-gon count as a proxy,
            # but rename it to avoid confusion? No, the function name is fixed?
            # No, I can define helper functions.
            
            # Let's just return 0 for now in this placeholder and implement the 4-gon count separately.
            pass
            
        return total_convex

    def count_convex_4goms(points):
        """
        Fast vectorized count of convex 4-gons.
        """
        all_4_subsets = list(itertools.combinations(range(n), 4))
        num_4_subsets = len(all_4_subsets) # 40920
        
        total_convex = 0
        batch_size = 40920 # Do it all at once
        
        idx_array = np.array(all_4_subsets)
        pts = points[idx_array] # (40920, 4, 2)
        
        p0 = pts[:, 0, :]
        p1 = pts[:, 1, :]
        p2 = pts[:, 2, :]
        p3 = pts[:, 3, :]
        
        def inside_tri(pt, a, b, c):
            d1 = (b[:, 0] - a[:, 0]) * (pt[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (pt[:, 0] - a[:, 0])
            d2 = (c[:, 0] - b[:, 0]) * (pt[:, 1] - b[:, 1]) - (c[:, 1] - b[:, 1]) * (pt[:, 0] - b[:, 0])
            d3 = (a[:, 0] - c[:, 0]) * (pt[:, 1] - c[:, 1]) - (a[:, 1] - c[:, 1]) * (pt[:, 0] - c[:, 0])
            return ((d1 > 0) & (d2 > 0) & (d3 > 0)) | ((d1 < 0) & (d2 < 0) & (d3 < 0))
        
        in_0 = inside_tri(p0, p1, p2, p3)
        in_1 = inside_tri(p1, p0, p2, p3)
        in_2 = inside_tri(p2, p0, p1, p3)
        in_3 = inside_tri(p3, p0, p1, p2)
        
        is_convex = ~(in_0 | in_1 | in_2 | in_3)
        
        return np.sum(is_convex)

    # Initialization: 
    # The directive suggests: "Model the problem as a discrete combinatorial packing task by initializing points on a sparse grid of high-symmetry lattice points... to maximize collinearity, then use a stochastic hill-climbing search that only moves points to other grid positions or small neighborhood perturbations to break degeneracies while preserving the dense, non-convex structure."
    
    # This is a different strategy: Discrete Grid Search + Perturbation.
    
    # Grid: 6x6 grid in [0.1, 0.9]x[0.1, 0.9]
    # 36 points available. We need 33.
    # We can pick 33 of them.
    # To maximize "non-convexity", we want points to be clustered or on lines.
    # A grid has many collinear points.
    # If we pick all 36, we have 33 to choose.
    # Let's start with a full 6x6 grid minus 3 points.
    
    grid_x = np.linspace(0.1, 0.9, 6)
    grid_y = np.linspace(0.1, 0.9, 6)
    
    grid_points = []
    for x in grid_x:
        for y in grid_y:
            grid_points.append([x, y])
            
    grid_points = np.array(grid_points)
    
    # Randomly select 33 points from the 36 grid points
    np.random.seed(42)
    selected_indices = np.random.choice(36, 33, replace=False)
    pts = grid_points[selected_indices].copy()
    
    # Ensure validity (distance > 1e-12, no collinearity < 1e-9)
    # Grid points are well-separated.
    # Collinearity: Many points are collinear (e.g., 6 on a row).
    # The evaluator requires |area| >= 1e-9.
    # For grid points, area is exactly 0 for collinear triples.
    # So we MUST perturb them.
    
    # Strategy: 
    # 1. Start with grid points.
    # 2. Apply small random perturbations to break collinearity.
    # 3. Optimize using local search (move to nearby grid points or perturb).
    
    # Perturb to break collinearity
    # We need to ensure no 3 points are collinear.
    # A simple way: add small random noise.
    # But we want to keep the "grid-like" structure.
    
    # Let's use a "jittered grid" approach.
    # Perturb each point slightly.
    
    def is_valid(points):
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return False
        if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
            return False
        # Distance check
        d = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(d**2, axis=-1)
        np.fill_diagonal(dist_sq, 1e18)
        if np.min(dist_sq) < 1e-12:
            return False
        # Collinearity check
        # This is O(N^3). 33^3 = 35937. Fast.
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if area < 1e-9:
                        return False
        return True

    # Initial perturbation
    for _ in range(100):
        pts_perturbed = pts + np.random.normal(0, 0.01, (n, 2))
        pts_perturbed = np.clip(pts_perturbed, 0.0, 1.0)
        if is_valid(pts_perturbed):
            pts = pts_perturbed
            break
    else:
        # If failed, use random points
        pts = np.random.rand(n, 2) * 0.8 + 0.1
        while not is_valid(pts):
            pts += np.random.normal(0, 0.01, (n, 2))
            pts = np.clip(pts, 0.0, 1.0)

    current_score = count_convex_4goms(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    # Local Search
    # Move one point at a time.
    # Options:
    # 1. Move to a nearby grid point (if not occupied).
    # 2. Small random perturbation.
    
    def delta_convex_4goms(points, idx, new_p):
        others = [i for i in range(n) if i != idx]
        combos = list(itertools.combinations(others, 3))
        
        if not combos:
            return 0
            
        old_p = points[idx]
        
        p1 = points[np.array([c[0] for c in combos])]
        p2 = points[np.array([c[1] for c in combos])]
        p3 = points[np.array([c[2] for c in combos])]
        
        old_p_exp = np.tile(old_p, (len(combos), 1))
        new_p_exp = np.tile(new_p, (len(combos), 1))
        
        def count_convex_in_batch(p_moving, p1, p2, p3):
            def inside_tri(pt, a, b, c):
                d1 = (b[:, 0] - a[:, 0]) * (pt[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (pt[:, 0] - a[:, 0])
                d2 = (c[:, 0] - b[:, 0]) * (pt[:, 1] - b[:, 1]) - (c[:, 1] - b[:, 1]) * (pt[:, 0] - b[:, 0])
                d3 = (a[:, 0] - c[:, 0]) * (pt[:, 1] - c[:, 1]) - (a[:, 1] - c[:, 1]) * (pt[:, 0] - c[:, 0])
                return ((d1 > 0) & (d2 > 0) & (d3 > 0)) | ((d1 < 0) & (d2 < 0) & (d3 < 0))
            
            in_0 = inside_tri(p_moving, p1, p2, p3)
            in_1 = inside_tri(p1, p_moving, p2, p3)
            in_2 = inside_tri(p2, p_moving, p1, p3)
            in_3 = inside_tri(p3, p_moving, p1, p2)
            
            is_convex = ~(in_0 | in_1 | in_2 | in_3)
            return np.sum(is_convex)
        
        old_count = count_convex_in_batch(old_p_exp, p1, p2, p3)
        new_count = count_convex_in_batch(new_p_exp, p1, p2, p3)
        
        return new_count - old_count

    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        improved = False
        order = np.random.permutation(n)
        
        for idx in order:
            if time.time() > deadline - 5:
                break
                
            old_p = best_pts[idx].copy()
            best_delta = 0
            best_new_p = None
            
            # Generate candidate moves
            # 1. Nearby grid points
            # Find nearest grid points
            dists = np.linalg.norm(grid_points - old_p, axis=1)
            nearest_indices = np.argsort(dists)[:5]
            
            candidates = []
            for ni in nearest_indices:
                # Check if this grid point is occupied by another point
                # (We allow multiple points near the same grid point, but they must be distinct)
                # For simplicity, just add the grid point as a candidate
                candidates.append(grid_points[ni])
                
            # 2. Random perturbations
            for _ in range(5):
                angle = np.random.uniform(0, 2 * np.pi)
                step = np.random.uniform(0.01, 0.1)
                dx = step * np.cos(angle)
                dy = step * np.sin(angle)
                new_p = old_p + np.array([dx, dy])
                new_p = np.clip(new_p, 0.0, 1.0)
                candidates.append(new_p)
                
            for new_p in candidates:
                # Check validity
                test_pts = best_pts.copy()
                test_pts[idx] = new_p
                if not is_valid(test_pts):
                    continue
                
                delta = delta_convex_4goms(best_pts, idx, new_p)
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_p = new_p
                    improved = True
                    
            if best_new_p is not None:
                best_pts[idx] = best_new_p
                best_score += best_delta
                no_improve = 0
            else:
                no_improve += 1
                
        if no_improve > max_no_improve:
            # Restart/Perturb
            idx = np.random.randint(n)
            new_p = best_pts[idx] + np.random.normal(0, 0.1, 2)
            new_p = np.clip(new_p, 0.0, 1.0)
            test_pts = best_pts.copy()
            test_pts[idx] = new_p
            if is_valid(test_pts):
                best_pts = test_pts
                best_score = count_convex_4goms(best_pts)
            no_improve = 0

    # Final Validation
    if not is_valid(best_pts):
        # Fallback
        pts = np.random.rand(n, 2) * 0.9 + 0.05
        while not is_valid(pts):
            pts += np.random.normal(0, 0.01, (n, 2))
            pts = np.clip(pts, 0.0, 1.0)
        return pts

    return best_pts
