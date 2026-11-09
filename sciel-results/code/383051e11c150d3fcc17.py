import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull

def erdos_szekeres_problem7() -> np.ndarray:
    n = 33
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Initialize with random points
    np.random.seed(42)
    points = np.random.rand(n, 2)
    
    # Precompute indices for 4-gons containing each point
    all_combos4 = np.array(list(itertools.combinations(range(n), 4)))
    point_to_combos4 = [[] for _ in range(n)]
    for idx, combo in enumerate(all_combos4):
        for p in combo:
            point_to_combos4[p].append(idx)
    point_to_combos4_idx = [np.array(lst) for lst in point_to_combos4]

    def count_convex4(points):
        """Fast vectorized count of convex 4-gons."""
        combos = all_combos4
        pts = points[combos]
        x1, y1 = pts[:, 0, 0], pts[:, 0, 1]
        x2, y2 = pts[:, 1, 0], pts[:, 1, 1]
        x3, y3 = pts[:, 2, 0], pts[:, 2, 1]
        x4, y4 = pts[:, 3, 0], pts[:, 3, 1]
        
        cp1 = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
        cp2 = (x3 - x2) * (y4 - y3) - (y3 - y2) * (x4 - x3)
        cp3 = (x4 - x3) * (y1 - y4) - (y4 - y3) * (x1 - x4)
        cp4 = (x1 - x4) * (y2 - y1) - (y1 - y4) * (x2 - x4)
        
        eps = 1e-9
        all_pos = (cp1 > eps) & (cp2 > eps) & (cp3 > eps) & (cp4 > eps)
        all_neg = (cp1 < -eps) & (cp2 < -eps) & (cp3 < -eps) & (cp4 < -eps)
        
        return np.sum(all_pos | all_neg)

    def count_convex7_and_get_subsets(points):
        """
        Slow count of convex 7-gons.
        Returns (count, list_of_convex_7_tuples)
        """
        count = 0
        convex_subsets = []
        # To speed up, we can skip if C4 is high? No, C4 is low.
        # We iterate over all 7-combinations
        for combo in itertools.combinations(range(n), 7):
            pts = points[list(combo)]
            # Check convexity
            # Using ConvexHull is robust but slow in Python loop
            # Let's try the angle sort method, it's faster for n=7
            c = pts.mean(axis=0)
            angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
            sorted_pts = pts[np.argsort(angles)]
            
            # Check cross products
            # We can vectorize the cross product calculation for this single 7-set
            # p0, p1, ..., p6
            c1 = np.cross(sorted_pts[1] - sorted_pts[0], sorted_pts[2] - sorted_pts[1])
            c2 = np.cross(sorted_pts[2] - sorted_pts[1], sorted_pts[3] - sorted_pts[2])
            c3 = np.cross(sorted_pts[3] - sorted_pts[2], sorted_pts[4] - sorted_pts[3])
            c4 = np.cross(sorted_pts[4] - sorted_pts[3], sorted_pts[5] - sorted_pts[4])
            c5 = np.cross(sorted_pts[5] - sorted_pts[4], sorted_pts[6] - sorted_pts[5])
            c6 = np.cross(sorted_pts[6] - sorted_pts[5], sorted_pts[0] - sorted_pts[6])
            c7 = np.cross(sorted_pts[0] - sorted_pts[6], sorted_pts[1] - sorted_pts[0])
            
            eps = 1e-9
            if (c1 > eps and c2 > eps and c3 > eps and c4 > eps and c5 > eps and c6 > eps and c7 > eps) or \
               (c1 < -eps and c2 < -eps and c3 < -eps and c4 < -eps and c5 < -eps and c6 < -eps and c7 < -eps):
                count += 1
                convex_subsets.append(combo)
                
        return count, convex_subsets

    best_points = points.copy()
    best_c4 = count_convex4(points)
    
    # Initial full check
    c7, subsets = count_convex7_and_get_subsets(points)
    if c7 == 0:
        return points
        
    # Local Search Loop
    step_size = 0.05
    
    while time.time() < deadline:
        # Phase 1: Local search on C4
        for _ in range(50):
            if time.time() > deadline:
                break
            i = np.random.randint(n)
            orig_pos = points[i].copy()
            new_pos = orig_pos + np.random.normal(0, step_size, 2)
            
            if np.any(new_pos < 0) or np.any(new_pos > 1):
                continue
                
            points[i] = new_pos
            new_c4 = count_convex4(points)
            
            if new_c4 < best_c4:
                best_c4 = new_c4
                best_points = points.copy()
            else:
                points[i] = orig_pos
                
        # Phase 2: Check C7
        c7, subsets = count_convex7_and_get_subsets(points)
        if c7 == 0:
            return points
            
        # Phase 3: Directed Mutation to break convex 7-gons
        for subset in subsets:
            if time.time() > deadline:
                break
            # Pick a random point from the convex 7-gon
            idx = np.random.choice(subset)
            orig_pos = points[idx].copy()
            
            # Calculate centroid of the other 6 points
            other_indices = [i for i in subset if i != idx]
            centroid = points[other_indices].mean(axis=0)
            
            # Move towards centroid
            # Small step
            move_vec = centroid - orig_pos
            norm = np.linalg.norm(move_vec)
            if norm > 1e-6:
                new_pos = orig_pos + move_vec * 0.1 # 10% of the way to centroid
                
                if np.any(new_pos < 0) or np.any(new_pos > 1):
                    continue
                    
                points[idx] = new_pos
                new_c4 = count_convex4(points)
                
                # We don't necessarily want to accept if C4 increases, 
                # but we MUST break the C7.
                # So we accept if it breaks the C7, even if C4 increases slightly?
                # Actually, just check if C7 is reduced.
                # But checking C7 is slow.
                # Let's just accept the move if it's valid, and re-evaluate C7 in the next loop.
                # Or, accept if new_c4 <= best_c4 + 10 (small penalty)
                if new_c4 <= best_c4 + 50:
                    best_c4 = new_c4
                    best_points = points.copy()
                else:
                    points[idx] = orig_pos

    return best_points
