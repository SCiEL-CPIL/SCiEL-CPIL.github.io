# EVOLVE-BLOCK-START
import numpy as np
import time

def _cross2d(ax, ay, bx, by):
    """Cross product of vectors (ax,ay) and (bx,by)"""
    return ax * by - ay * bx

def _count_convex7_fast(points):
    """Count the number of convex 7-gons in the point set using the specified DP algorithm."""
    n = len(points)
    # Sort points lexicographically
    order = np.lexsort((points[:, 1], points[:, 0]))
    sorted_points = points[order]
    
    total_count = 0
    
    for i in range(n):
        pivot = sorted_points[i]
        # Get points with index > i in the sorted order
        later_indices = list(range(i + 1, n))
        if len(later_indices) < 6:
            continue
            
        later_points = sorted_points[later_indices]
        
        # Sort these points by angle around pivot
        dx = later_points[:, 0] - pivot[0]
        dy = later_points[:, 1] - pivot[1]
        angles = np.arctan2(dy, dx)
        angle_order = np.argsort(angles)
        
        sorted_later_points = later_points[angle_order]
        m = len(sorted_later_points)
        
        # Precompute turn matrices
        # T1[a, b] = 1 if turn(pivot, p_a, p_b) is convex (left turn)
        T1 = np.zeros((m, m), dtype=np.int8)
        for a in range(m):
            for b in range(m):
                if a != b:
                    # turn(pivot, sorted_later_points[a], sorted_later_points[b])
                    v1x = sorted_later_points[a, 0] - pivot[0]
                    v1y = sorted_later_points[a, 1] - pivot[1]
                    v2x = sorted_later_points[b, 0] - sorted_later_points[a, 0]
                    v2y = sorted_later_points[b, 1] - sorted_later_points[a, 1]
                    cross = v1x * v2y - v1y * v2x
                    if cross > 1e-9:
                        T1[a, b] = 1
        
        # T2[i, j, k] = 1 if turn(p_i, p_j, p_k) is convex
        T2 = np.zeros((m, m, m), dtype=np.int8)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                for k in range(m):
                    if j == k:
                        continue
                    v1x = sorted_later_points[j, 0] - sorted_later_points[i, 0]
                    v1y = sorted_later_points[j, 1] - sorted_later_points[i, 1]
                    v2x = sorted_later_points[k, 0] - sorted_later_points[j, 0]
                    v2y = sorted_later_points[k, 1] - sorted_later_points[j, 1]
                    cross = v1x * v2y - v1y * v2x
                    if cross > 1e-9:
                        T2[i, j, k] = 1
        
        # T3[i, j] = 1 if turn(p_i, p_j, pivot) is convex
        T3 = np.zeros((m, m), dtype=np.int8)
        for i in range(m):
            for j in range(m):
                if i != j:
                    v1x = sorted_later_points[j, 0] - sorted_later_points[i, 0]
                    v1y = sorted_later_points[j, 1] - sorted_later_points[i, 1]
                    v2x = pivot[0] - sorted_later_points[j, 0]
                    v2y = pivot[1] - sorted_later_points[j, 1]
                    cross = v1x * v2y - v1y * v2x
                    if cross > 1e-9:
                        T3[i, j] = 1
        
        # T4[i, j] = 1 if turn(p_j, pivot, p_i) is convex
        T4 = np.zeros((m, m), dtype=np.int8)
        for i in range(m):
            for j in range(m):
                if i != j:
                    # turn(p_j, pivot, p_i)
                    v1x = pivot[0] - sorted_later_points[j, 0]
                    v1y = pivot[1] - sorted_later_points[j, 1]
                    v2x = sorted_later_points[i, 0] - pivot[0]
                    v2y = sorted_later_points[i, 1] - pivot[1]
                    cross = v1x * v2y - v1y * v2x
                    if cross > 1e-9:
                        T4[i, j] = 1
        
        # DP
        # dp[first, prev, curr]
        # l=2: dp[a, a, b] = T1[a, b]
        dp = np.zeros((m, m, m), dtype=np.int64)
        for a in range(m):
            for b in range(m):
                if T1[a, b]:
                    dp[a, a, b] = 1
        
        # Iterate l=3 to 6
        for l in range(3, 7):
            # new_dp[first, prev, curr] = sum_{pp} dp[first, pp, prev] * T2[pp, prev, curr]
            new_dp = np.einsum('fpq,pqr->fqr', dp, T2, optimize=True)
            dp = new_dp
        
        # Check closing turns
        # valid[first, prev, curr] = dp[first, prev, curr] * T3[prev, curr] * T4[first, curr]
        valid_mask = T3[np.newaxis, :, :].astype(np.int64) * T4[:, np.newaxis, :].astype(np.int64)
        count = np.sum(dp * valid_mask)
        total_count += count
        
    return total_count

def _is_valid_fast(points):
    """Check validity constraints."""
    if points.shape != (33, 2):
        return False
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return False
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
    # Check distances
    for i in range(33):
        for j in range(i + 1, 33):
            dist_sq = np.sum((points[i] - points[j]) ** 2)
            if dist_sq < 1e-12:
                return False
    # Check collinearity
    for i in range(33):
        for j in range(i + 1, 33):
            for k in range(j + 1, 33):
                p1, p2, p3 = points[i], points[j], points[k]
                area = abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                if area < 1e-9:
                    return False
    return True

def _generate_initial_points():
    """Generate initial points using a structured approach to minimize convex 7-gons."""
    n = 33
    points = np.zeros((n, 2))
    
    # Strategy: Place points in dense clusters near the boundary of the unit square
    # This reduces the number of convex configurations because points in the same
    # cluster are unlikely to form large convex polygons together.
    
    # Cluster 1: 8 points near (0.1, 0.1) - bottom-left
    for i in range(8):
        angle = 2 * np.pi * i / 8
        r = 0.03
        points[i, 0] = 0.1 + r * np.cos(angle)
        points[i, 1] = 0.1 + r * np.sin(angle)
    
    # Cluster 2: 8 points near (0.9, 0.1) - bottom-right
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi / 8
        r = 0.03
        points[8 + i, 0] = 0.9 + r * np.cos(angle)
        points[8 + i, 1] = 0.1 + r * np.sin(angle)
    
    # Cluster 3: 8 points near (0.9, 0.9) - top-right
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi / 4
        r = 0.03
        points[16 + i, 0] = 0.9 + r * np.cos(angle)
        points[16 + i, 1] = 0.9 + r * np.sin(angle)
    
    # Cluster 4: 7 points near (0.1, 0.9) - top-left
    for i in range(7):
        angle = 2 * np.pi * i / 7 + np.pi / 6
        r = 0.03
        points[24 + i, 0] = 0.1 + r * np.cos(angle)
        points[24 + i, 1] = 0.9 + r * np.sin(angle)
    
    # Cluster 5: 2 points near (0.5, 0.5) - center
    points[31] = [0.48, 0.52]
    points[32] = [0.52, 0.48]
    
    # Add small random perturbations to break exact symmetries and ensure general position
    np.random.seed(12345)
    points += np.random.uniform(-0.005, 0.005, size=(n, 2))
    
    # Ensure points are within [0, 1]
    points = np.clip(points, 0.001, 0.999)
    
    return points

def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres "Happy Ending" problem. Place n = 33 points
    in general position inside the unit square so as to MINIMIZE the number of
    convex 7-gons whose vertices are among the points.
    
    Returns:
        points: np.ndarray of shape (33, 2) with the x,y coordinates of the points.
    """
    n = 33
    start_time = time.time()
    deadline = start_time + 590  # Safety margin (evaluator kills at 630s)
    
    # Generate initial points
    points = _generate_initial_points()
    
    # Ensure validity
    if not _is_valid_fast(points):
        # Fallback: add more perturbation
        np.random.seed(99999)
        points += np.random.uniform(-0.01, 0.01, size=(n, 2))
        points = np.clip(points, 0.001, 0.999)
        if not _is_valid_fast(points):
            # Last resort: random points
            points = np.random.rand(n, 2) * 0.998 + 0.001
    
    best_points = points.copy()
    best_count = _count_convex7_fast(points)
    
    # Hill climbing with multiple move strategies
    np.random.seed(777)
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = np.random.randint(n)
        old_point = points[idx].copy()
        
        # Choose move type
        move_type = np.random.random()
        
        if move_type < 0.4:
            # Small perturbation
            new_point = old_point + np.random.normal(0, 0.02, size=2)
            new_point = np.clip(new_point, 0.001, 0.999)
            points[idx] = new_point
        elif move_type < 0.7:
            # Move to a random position in the same "region" (nearby)
            region_size = 0.1
            new_point = old_point + np.random.uniform(-region_size, region_size, size=2)
            new_point = np.clip(new_point, 0.001, 0.999)
            points[idx] = new_point
        elif move_type < 0.85:
            # Move to a random position in the unit square
            new_point = np.random.uniform(0.001, 0.999, size=2)
            points[idx] = new_point
        else:
            # Move to one of the cluster centers
            centers = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9], [0.5, 0.5]]
            center = np.random.choice(centers)
            new_point = center + np.random.normal(0, 0.02, size=2)
            new_point = np.clip(new_point, 0.001, 0.999)
            points[idx] = new_point
        
        if not _is_valid_fast(points):
            points[idx] = old_point
            continue
            
        new_count = _count_convex7_fast(points)
        
        if new_count < best_count:
            best_count = new_count
            best_points = points.copy()
            if best_count <= 1:
                break
        else:
            # Revert
            points[idx] = old_point
    
    return best_points

# EVOLVE-BLOCK-END
