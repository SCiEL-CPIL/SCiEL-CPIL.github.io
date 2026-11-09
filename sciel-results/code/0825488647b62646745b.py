import numpy as np
import time
import math
import random
from itertools import combinations

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620  # Safety margin before 630s limit
    
    n = 33
    C_29_3 = 3654  # C(29, 3)
    
    # Precompute all 4-subsets indices for vectorized evaluation?
    # C(33, 4) = 40920. This is small enough to store indices.
    # But we need to update counts efficiently.
    
    # Strategy:
    # 1. Initialize with a "clustered" configuration (e.g., two dense clusters) 
    #    to ensure many points are inside hulls of others.
    # 2. Use Greedy Local Search: 
    #    - For each point, try small perturbations in 8 directions.
    #    - Calculate the change in the number of "non-convex" 4-subsets involving that point.
    #    - Accept the move if it increases the number of non-convex 4-subsets (decreases convex 7-gons).
    # 3. The count of convex 7-gons C is not simply sum of non-convex 4-subsets.
    #    Wait, the relationship is: A 7-set is convex iff ALL its C(7,4)=35 4-subsets are convex.
    #    So, C_convex_7 = Number of 7-sets S such that all 35 4-subsets of S are convex.
    #    This is hard to update incrementally with just 4-subset counts.
    #    
    #    Alternative Objective Proxy:
    #    Minimizing the number of convex 4-subsets is a strong heuristic.
    #    If we minimize convex 4-subsets, we likely minimize convex 7-gons.
    #    Let's optimize the number of convex 4-subsets directly.
    #    Let K be the number of convex 4-subsets.
    #    We want to minimize K.
    #    
    #    Delta calculation for moving point p_i:
    #    Only 4-subsets containing i change status.
    #    There are C(32, 3) = 4960 such subsets.
    #    For each subset {i, a, b, c}, we check if it was convex before and after.
    #    This is O(1) per subset with precomputed coordinates.
    #    4960 * 8 directions * 33 points = ~1.3M operations per full pass.
    #    This is fast enough for many passes in 600s.

    def is_convex_quad_coords(p0, p1, p2, p3):
        # p0, p1, p2, p3 are (x, y) tuples
        # Check if any point is inside the triangle of the other 3.
        # We can check if the area of the convex hull of the 4 points is equal to the area of the triangle formed by any 3?
        # No, simpler: 
        # Sort points by angle around centroid? No.
        # Check if p0 is inside p1p2p3, etc.
        
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            
        # Point in triangle check
        def point_in_triangle(pt, v1, v2, v3):
            # Returns True if pt is strictly inside v1v2v3
            # Using barycentric coordinates or sign checks
            c1 = cross(v1, v2, pt)
            c2 = cross(v2, v3, pt)
            c3 = cross(v3, v1, pt)
            has_neg = (c1 < 0) or (c2 < 0) or (c3 < 0)
            has_pos = (c1 > 0) or (c2 > 0) or (c3 > 0)
            return not (has_neg and has_pos)
            
        if point_in_triangle(p0, p1, p2, p3): return False
        if point_in_triangle(p1, p0, p2, p3): return False
        if point_in_triangle(p2, p0, p1, p3): return False
        if point_in_triangle(p3, p0, p1, p2): return False
        return True

    def count_convex_quads(points):
        """
        Count the number of convex 4-subsets.
        """
        count = 0
        for i in range(n-3):
            p0 = points[i]
            for j in range(i+1, n-2):
                p1 = points[j]
                for k in range(j+1, n-1):
                    p2 = points[k]
                    for l in range(k+1, n):
                        p3 = points[l]
                        if is_convex_quad_coords(p0, p1, p2, p3):
                            count += 1
        return count

    def delta_convex_quads(points, idx, new_p):
        """
        Calculate the change in the number of convex 4-subsets when moving point idx to new_p.
        Only subsets containing idx are affected.
        """
        old_p = points[idx]
        delta = 0
        others = [i for i in range(n) if i != idx]
        
        # Iterate over all triplets of others
        # C(32, 3) = 4960
        for i in range(len(others) - 2):
            p1 = points[others[i]]
            for j in range(i + 1, len(others) - 1):
                p2 = points[others[j]]
                for k in range(j + 1, len(others)):
                    p3 = points[others[k]]
                    
                    was_convex = is_convex_quad_coords(old_p, p1, p2, p3)
                    is_convex_now = is_convex_quad_coords(new_p, p1, p2, p3)
                    
                    if was_convex and not is_convex_now:
                        delta -= 1
                    elif not was_convex and is_convex_now:
                        delta += 1
        return delta

    def is_valid(points):
        if points.shape != (33, 2):
            return False
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return False
        if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
            return False
        # Distances
        for i in range(33):
            for j in range(i+1, 33):
                dx = points[i, 0] - points[j, 0]
                dy = points[i, 1] - points[j, 1]
                if dx*dx + dy*dy < 1e-12:
                    return False
        # Collinearity
        for i in range(33):
            for j in range(i+1, 33):
                for l in range(j+1, 33):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[l]
                    area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if area < 1e-9:
                        return False
        return True

    # Initialization: Two clusters
    # Cluster 1: Centered at (0.25, 0.25), radius 0.1
    # Cluster 2: Centered at (0.75, 0.75), radius 0.1
    # This ensures many points are inside the convex hull of the other cluster or mixed.
    # Actually, if points are in two tight clusters, a 4-subset with 2 from each might be convex (quad),
    # but 3 from one and 1 from the other might be non-convex (1 inside triangle of 3? No, if clusters are separate,
    # the 1 from the other cluster is far away, so the 3 from one cluster form a triangle, and the 4th is outside.
    # So 3-1 splits are likely convex if clusters are far apart.
    # We want points to be "interleaved" or "nested".
    
    # Better Initialization: A single dense cluster with some outliers?
    # Or a spiral?
    # Let's try a "nested" structure: Points on concentric circles?
    # Or just random points in a small region?
    
    # Let's start with a grid that is heavily jittered and compressed.
    random.seed(42)
    np.random.seed(42)
    
    # Start with points in a small square [0.4, 0.6] x [0.4, 0.6]
    # This maximizes the chance of points being inside triangles of others.
    pts = np.random.rand(33, 2) * 0.2 + 0.4
    
    # Ensure validity
    if not is_valid(pts):
        # Spread them out slightly if collision
        pts = np.random.rand(33, 2) * 0.9 + 0.05
        if not is_valid(pts):
             pts = np.random.rand(33, 2)
             
    current_count = count_convex_quads(pts)
    best_points = pts.copy()
    best_count = current_count
    
    # Greedy Local Search
    # Directions: 8 compass points + center?
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002]
    
    # We will iterate through points and try to improve.
    # To speed up, we can vectorize the delta calculation?
    # The inner loop `is_convex_quad_coords` is pure Python and slow.
    # 4960 calls per move * 8 dirs * 33 pts = 1.3M calls per pass.
    # 1.3M Python function calls might take ~5-10 seconds.
    # 600s / 10s = 60 passes. This is reasonable.
    
    # Optimization: Pre-compute the "others" indices and their coordinates once per point move?
    # No, points change.
    
    # Let's optimize `is_convex_quad_coords` by inlining and using local variables.
    def fast_is_convex(p0, p1, p2, p3):
        # Unrolled point-in-triangle check
        # p0 in p1p2p3?
        c1 = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        c2 = (p2[0] - p0[0]) * (p3[1] - p0[1]) - (p2[1] - p0[1]) * (p3[0] - p0[0])
        c3 = (p3[0] - p0[0]) * (p1[1] - p0[1]) - (p3[1] - p0[1]) * (p1[0] - p0[0])
        # Signs of c1, c2, c3 relative to triangle p1p2p3?
        # Actually, standard point in triangle for pt=p0, tri=p1,p2,p3:
        # d1 = cross(p1, p2, p0)
        # d2 = cross(p2, p3, p0)
        # d3 = cross(p3, p1, p0)
        # Inside if all same sign.
        
        d1 = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        d2 = (p2[0] - p0[0]) * (p3[1] - p0[1]) - (p2[1] - p0[1]) * (p3[0] - p0[0])
        d3 = (p3[0] - p0[0]) * (p1[1] - p0[1]) - (p3[1] - p0[1]) * (p1[0] - p0[0])
        if (d1 > 0 and d2 > 0 and d3 > 0) or (d1 < 0 and d2 < 0 and d3 < 0):
            return False # p0 inside 123
            
        d1 = (p0[0] - p1[0]) * (p2[1] - p1[1]) - (p0[1] - p1[1]) * (p2[0] - p1[0])
        d2 = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        d3 = (p3[0] - p1[0]) * (p0[1] - p1[1]) - (p3[1] - p1[1]) * (p0[0] - p1[0])
        if (d1 > 0 and d2 > 0 and d3 > 0) or (d1 < 0 and d2 < 0 and d3 < 0):
            return False # p1 inside 023
            
        d1 = (p0[0] - p2[0]) * (p1[1] - p2[1]) - (p0[1] - p2[1]) * (p1[0] - p2[0])
        d2 = (p1[0] - p2[0]) * (p3[1] - p2[1]) - (p1[1] - p2[1]) * (p3[0] - p2[0])
        d3 = (p3[0] - p2[0]) * (p0[1] - p2[1]) - (p3[1] - p2[1]) * (p0[0] - p2[0])
        if (d1 > 0 and d2 > 0 and d3 > 0) or (d1 < 0 and d2 < 0 and d3 < 0):
            return False # p2 inside 013
            
        d1 = (p0[0] - p3[0]) * (p1[1] - p3[1]) - (p0[1] - p3[1]) * (p1[0] - p3[0])
        d2 = (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])
        d3 = (p2[0] - p3[0]) * (p0[1] - p3[1]) - (p2[1] - p3[1]) * (p0[0] - p3[0])
        if (d1 > 0 and d2 > 0 and d3 > 0) or (d1 < 0 and d2 < 0 and d3 < 0):
            return False # p3 inside 012
            
        return True

    def fast_delta(points, idx, new_p):
        old_p = points[idx]
        delta = 0
        others = [i for i in range(n) if i != idx]
        # Use local vars for speed
        p0_old = old_p
        p0_new = new_p
        
        # Pre-fetch other points
        # This is still a loop.
        for i in range(len(others) - 2):
            p1 = points[others[i]]
            for j in range(i + 1, len(others) - 1):
                p2 = points[others[j]]
                for k in range(j + 1, len(others)):
                    p3 = points[others[k]]
                    
                    was_convex = fast_is_convex(p0_old, p1, p2, p3)
                    is_convex_now = fast_is_convex(p0_new, p1, p2, p3)
                    
                    if was_convex and not is_convex_now:
                        delta -= 1
                    elif not was_convex and is_convex_now:
                        delta += 1
        return delta

    # Main Loop
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        improved = False
        # Iterate through all points
        for idx in range(n):
            if time.time() > deadline - 10:
                break
                
            old_p = best_points[idx].copy()
            best_delta = 0
            best_new_p = None
            
            # Try multiple step sizes and directions
            for step in step_sizes:
                for dx, dy in directions:
                    new_p = old_p + np.array([dx * step, dy * step])
                    # Clip to [0, 1]
                    new_p = np.clip(new_p, 0.0, 1.0)
                    
                    # Quick validity check: distance to others
                    valid = True
                    for i in range(n):
                        if i == idx: continue
                        dx_d = new_p[0] - best_points[i, 0]
                        dy_d = new_p[1] - best_points[i, 1]
                        if dx_d*dx_d + dy_d*dy_d < 1e-12:
                            valid = False
                            break
                    if not valid:
                        continue
                        
                    # Collinearity check is expensive, skip for now, check at end of pass
                    delta = fast_delta(best_points, idx, new_p)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_new_p = new_p
                        improved = True # Tentative improvement found for this point
                    
            if best_new_p is not None:
                best_points[idx] = best_new_p
                current_count += best_delta
                if current_count < best_count:
                    best_count = current_count
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        if no_improve_count > max_no_improve:
            # Perturb randomly to escape local minimum
            idx = random.randint(0, n-1)
            new_p = best_points[idx] + np.random.normal(0, 0.05, 2)
            new_p = np.clip(new_p, 0.0, 1.0)
            if is_valid(np.concatenate([best_points[:idx], [new_p], best_points[idx+1:]])):
                best_points[idx] = new_p
                current_count = count_convex_quads(best_points)
                best_count = min(best_count, current_count)
            no_improve_count = 0

    # Final validation
    if not is_valid(best_points):
        # Try to fix validity by small jitter
        for _ in range(100):
            if is_valid(best_points): break
            idx = random.randint(0, n-1)
            best_points[idx] += np.random.normal(0, 0.01, 2)
            best_points[idx] = np.clip(best_points[idx], 0.0, 1.0)
            
    if not is_valid(best_points):
        # Fallback
        return np.random.rand(33, 2) * 0.9 + 0.05

    return best_points
