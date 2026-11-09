import numpy as np
import time
import itertools
import random

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 630s limit
    
    n = 33
    
    # Precompute all 4-combinations for fast delta evaluation
    # C(33, 4) = 40920. This is the proxy for convexity.
    all_quads = list(itertools.combinations(range(n), 4))
    quad_indices = np.array(all_quads) # Shape (40920, 4)
    M = len(all_quads)
    
    def count_convex_quads(points):
        """
        Vectorized count of convex quadrilaterals.
        A set of 4 points is non-convex iff one point is inside the triangle of the other 3.
        """
        # Get points for all quads: (M, 4, 2)
        pts = points[quad_indices]
        
        p0 = pts[:, 0, :]
        p1 = pts[:, 1, :]
        p2 = pts[:, 2, :]
        p3 = pts[:, 3, :]
        
        # Check if p0 is inside p1-p2-p3
        # Cross products
        # v1 = p2 - p1, v2 = p3 - p1, v3 = p0 - p1
        # c1 = cross(v1, v2)
        # c2 = cross(v2, v3)
        # c3 = cross(v3, v1)
        # If all same sign, p0 is inside.
        
        def is_inside(pt, a, b, c):
            # pt, a, b, c are (M, 2)
            # Check if pt is inside triangle abc
            # Using cross products of edges with vector to point
            # e1 = b - a, e2 = c - b, e3 = a - c
            # v1 = pt - a, v2 = pt - b, v3 = pt - c
            # cross(e1, v1), cross(e2, v2), cross(e3, v3)
            
            e1 = b - a
            e2 = c - b
            e3 = a - c
            
            v1 = pt - a
            v2 = pt - b
            v3 = pt - c
            
            c1 = e1[:, 0] * v1[:, 1] - e1[:, 1] * v1[:, 0]
            c2 = e2[:, 0] * v2[:, 1] - e2[:, 1] * v2[:, 0]
            c3 = e3[:, 0] * v3[:, 1] - e3[:, 1] * v3[:, 0]
            
            # Inside if all > 0 or all < 0
            return ((c1 > 1e-12) & (c2 > 1e-12) & (c3 > 1e-12)) | \
                   ((c1 < -1e-12) & (c2 < -1e-12) & (c3 < -1e-12))
        
        in_0 = is_inside(p0, p1, p2, p3)
        in_1 = is_inside(p1, p0, p2, p3)
        in_2 = is_inside(p2, p0, p1, p3)
        in_3 = is_inside(p3, p0, p1, p2)
        
        is_convex = ~(in_0 | in_1 | in_2 | in_3)
        
        return np.sum(is_convex)

    def is_valid(points):
        if points.shape != (33, 2):
            return False
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return False
        if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
            return False
        
        # Distance check
        # We need to ensure no two points are too close.
        # Vectorized pairwise distance is O(N^2) = 1089 pairs. Very fast.
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, 1e18)
        if np.min(dist_sq) < 1e-12:
            return False
            
        # Collinearity check
        # O(N^3) = 33^3 = 35937 triples.
        # Can be vectorized or done in a loop. Loop is ~0.05s in Python.
        # Let's vectorize for speed if possible, or just loop since it's small.
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

    # Initialization: Double Parabola
    # Points on y = x^2 form a convex chain (cup).
    # Points on y = -x^2 form a convex chain (cap).
    # Combining them creates a shape where many points are interior to the convex hull of the whole set,
    # but more importantly, local subsets are less likely to be in convex position if they mix from both curves.
    
    # Let's try a "nested" structure.
    # Actually, the best known configurations for few convex k-gons are related to "cup-cap" structures.
    # But we want to MINIMIZE convex 7-gons.
    # Random points in a small square tend to have few points on the hull.
    # Points on a convex curve (parabola) have ALL points on the hull. This is BAD for convex 7-gons?
    # Wait. If all points are on the convex hull of the TOTAL set, does that mean all 7-subsets are convex?
    # No. A subset of points on a convex curve is in convex position.
    # So if we put all 33 points on a parabola, every 7-subset is a convex 7-gon.
    # C(33, 7) = 4.2M convex 7-gons. This is the WORST case.
    
    # We want the OPPOSITE. We want points to be "deep" inside.
    # Uniform random in a small square:
    # Expected hull size for N points in square is ~ N^{1/3} or N^{1/2}?
    # For uniform in square, E[hull] ~ c * N^{1/3} (actually it's N^{1/3} for disk, similar for square).
    # So only ~3-4 points are on the hull. The rest are interior.
    # Interior points cannot be part of a convex 7-gon if the 7-gon must contain them on its hull?
    # A convex 7-gon is a set of 7 points such that all 7 are on the convex hull of that set.
    # If a point is "deep" inside the global hull, it might still be on the hull of a small local subset.
    # However, if we cluster points, the "interior" points of the cluster are likely inside the hull of their neighbors.
    
    # Strategy: Cluster points into a small region.
    # To be safe, let's start with a random cluster and then OPTIMIZE to push points INWARD.
    
    np.random.seed(42)
    # Start with points in a small square in the center
    scale = 0.1
    center = 0.5
    pts = np.random.rand(n, 2) * scale + (center - scale/2)
    
    # Ensure valid start
    if not is_valid(pts):
        pts = np.random.rand(n, 2) * 0.2 + 0.4
        if not is_valid(pts):
             # Fallback
             pts = np.random.rand(n, 2)
             pts *= 0.5 + 0.25
             if not is_valid(pts):
                 return pts

    current_score = count_convex_quads(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    # Optimization:
    # We want to MINIMIZE convex quads.
    # Moving a point towards the centroid of its neighbors tends to make it more "interior".
    # Let's try a greedy hill-climbing where we move points towards the global centroid or local centroids.
    
    def delta_score(pts, idx, new_p):
        """
        Calculate change in convex quad count when moving point idx to new_p.
        Only quads containing idx are affected.
        """
        # Find quads containing idx
        # quad_indices is (M, 4).
        mask = np.any(quad_indices == idx, axis=1)
        affected_quads = quad_indices[mask]
        
        if len(affected_quads) == 0:
            return 0
        
        # Extract points for affected quads
        # We need to replace idx with new_p in these quads.
        
        old_pts = pts[affected_quads]
        new_pts = old_pts.copy()
        
        # Find where idx is in each quad
        # Since quad_indices are sorted combinations, idx is at a specific position?
        # No, it varies.
        # We can just check column by column.
        for col in range(4):
            positions = affected_quads[:, col] == idx
            if np.any(positions):
                new_pts[positions, col, :] = new_p
        
        # Count convex in old_pts
        def count_convex_batch(p_batch):
            # p_batch is (K, 4, 2)
            p0 = p_batch[:, 0, :]
            p1 = p_batch[:, 1, :]
            p2 = p_batch[:, 2, :]
            p3 = p_batch[:, 3, :]
            
            def is_inside(pt, a, b, c):
                e1 = b - a
                e2 = c - b
                e3 = a - c
                v1 = pt - a
                v2 = pt - b
                v3 = pt - c
                c1 = e1[:, 0] * v1[:, 1] - e1[:, 1] * v1[:, 0]
                c2 = e2[:, 0] * v2[:, 1] - e2[:, 1] * v2[:, 0]
                c3 = e3[:, 0] * v3[:, 1] - e3[:, 1] * v3[:, 0]
                return ((c1 > 1e-12) & (c2 > 1e-12) & (c3 > 1e-12)) | \
                       ((c1 < -1e-12) & (c2 < -1e-12) & (c3 < -1e-12))
            
            in_0 = is_inside(p0, p1, p2, p3)
            in_1 = is_inside(p1, p0, p2, p3)
            in_2 = is_inside(p2, p0, p1, p3)
            in_3 = is_inside(p3, p0, p1, p2)
            is_convex = ~(in_0 | in_1 | in_2 | in_3)
            return np.sum(is_convex)
        
        old_count = count_convex_batch(old_pts)
        new_count = count_convex_batch(new_pts)
        
        return new_count - old_count

    # Main Loop
    no_improve = 0
    max_no_improve = 50
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002]
    
    # Precompute centroid
    centroid = np.mean(pts, axis=0)
    
    while time.time() < deadline:
        improved = False
        # Shuffle
        order = np.random.permutation(n)
        
        for idx in order:
            if time.time() > deadline - 10:
                break
            
            old_p = best_pts[idx].copy()
            best_delta = 0
            best_new_p = None
            
            # Try moving towards centroid
            # And random perturbations
            candidates = []
            
            # Move towards centroid
            vec_to_centroid = centroid - old_p
            dist = np.linalg.norm(vec_to_centroid)
            if dist > 1e-6:
                dir_centroid = vec_to_centroid / dist
                for step in step_sizes:
                    new_p = old_p + dir_centroid * step
                    new_p = np.clip(new_p, 0.0, 1.0)
                    candidates.append(new_p)
            
            # Random moves
            for step in step_sizes:
                for _ in range(3):
                    angle = np.random.uniform(0, 2 * np.pi)
                    dx = step * np.cos(angle)
                    dy = step * np.sin(angle)
                    new_p = old_p + np.array([dx, dy])
                    new_p = np.clip(new_p, 0.0, 1.0)
                    candidates.append(new_p)
            
            for new_p in candidates:
                # Quick validity check (distance and bounds)
                # We can skip full collinearity check in inner loop for speed,
                # but must ensure we don't get too close to other points.
                # Distance check to other points:
                diffs = best_pts - new_p
                dists_sq = np.sum(diffs**2, axis=1)
                dists_sq[idx] = 1e18
                if np.min(dists_sq) < 1e-12:
                    continue
                
                # Delta calculation
                delta = delta_score(best_pts, idx, new_p)
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_p = new_p
                    improved = True
            
            if best_new_p is not None:
                best_pts[idx] = best_new_p
                best_score += best_delta
                no_improve = 0
                # Update centroid occasionally? Or just keep it static?
                # Static centroid is fine for "clumping" strategy.
            else:
                no_improve += 1
        
        if no_improve > max_no_improve:
            # Perturb to escape local minimum
            # Move a random point significantly
            idx = np.random.randint(n)
            # Move towards a random new center?
            new_center = np.random.rand(2) * 0.4 + 0.3
            new_p = best_pts[idx] + (new_center - best_pts[idx]) * 0.5
            new_p = np.clip(new_p, 0.0, 1.0)
            
            # Check validity
            diffs = best_pts - new_p
            dists_sq = np.sum(diffs**2, axis=1)
            dists_sq[idx] = 1e18
            if np.min(dists_sq) > 1e-12:
                best_pts[idx] = new_p
                best_score = count_convex_quads(best_pts)
            no_improve = 0

    # Final Validation
    if not is_valid(best_pts):
        # Try to fix validity
        for _ in range(100):
            idx = np.random.randint(n)
            best_pts[idx] += np.random.normal(0, 0.001, 2)
            best_pts[idx] = np.clip(best_pts[idx], 0.0, 1.0)
            if is_valid(best_pts):
                break
        
        if not is_valid(best_pts):
            # Fallback to a known valid configuration
            # Grid points with jitter
            pts = np.array([[i/10, j/10] for i in range(6) for j in range(6)])[:33]
            pts += np.random.normal(0, 0.01, (33, 2))
            pts = np.clip(pts, 0.01, 0.99)
            if not is_valid(pts):
                pts = np.random.rand(33, 2) * 0.8 + 0.1
            return pts

    return best_pts
