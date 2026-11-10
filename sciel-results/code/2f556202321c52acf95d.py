import numpy as np
import time
from itertools import combinations

def is_convex_7_fast(pts):
    centroid = np.mean(pts, axis=0)
    vecs = pts - centroid
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    order = np.argsort(angles)
    sorted_pts = pts[order]
    
    # Unroll loop for speed?
    # p0..p6
    p = sorted_pts
    # Check turns
    # Cross(p[i], p[i+1], p[i+2])
    # Must be all > 1e-9 or all < -1e-9
    
    c0 = (p[1][0]-p[0][0])*(p[2][1]-p[1][1]) - (p[1][1]-p[0][1])*(p[2][0]-p[1][0])
    c1 = (p[2][0]-p[1][0])*(p[3][1]-p[2][1]) - (p[2][1]-p[1][1])*(p[3][0]-p[2][0])
    c2 = (p[3][0]-p[2][0])*(p[4][1]-p[3][1]) - (p[3][1]-p[2][1])*(p[4][0]-p[3][0])
    c3 = (p[4][0]-p[3][0])*(p[5][1]-p[4][1]) - (p[4][1]-p[3][1])*(p[5][0]-p[4][0])
    c4 = (p[5][0]-p[4][0])*(p[6][1]-p[5][1]) - (p[5][1]-p[4][1])*(p[6][0]-p[5][0])
    c5 = (p[6][0]-p[5][0])*(p[0][1]-p[6][1]) - (p[6][1]-p[5][1])*(p[0][0]-p[6][0])
    c6 = (p[0][0]-p[6][0])*(p[1][1]-p[0][1]) - (p[0][1]-p[6][1])*(p[1][0]-p[0][0])
    
    crosses = np.array([c0, c1, c2, c3, c4, c5, c6])
    if np.all(crosses > 1e-9) or np.all(crosses < -1e-9):
        return True
    return False

def count_convex_7(points):
    n = len(points)
    count = 0
    # Pre-compute indices
    indices = range(n)
    for comb in combinations(indices, 7):
        pts = points[list(comb)]
        if is_convex_7_fast(pts):
            count += 1
    return count

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 500 # Leave 100s for safety/eval
    
    best_points = None
    best_count = float('inf')
    
    # Structure 1: Random
    np.random.seed(42)
    pts = np.random.rand(33, 2)
    # Jitter to ensure general position? Random is usually fine.
    # Check validity
    if is_valid(pts):
        c = count_convex_7(pts)
        if c < best_count:
            best_count = c
            best_points = pts.copy()
        if time.time() > deadline:
            return best_points

    # Structure 2: Grid
    # 6x6 grid = 36 points. Remove 3.
    x = np.linspace(0.1, 0.9, 6)
    y = np.linspace(0.1, 0.9, 6)
    grid_pts = np.array([(xi, yi) for xi in x for yi in y])
    # Remove 3 random
    remove_idx = np.random.choice(36, 3, replace=False)
    keep_idx = np.setdiff1d(np.arange(36), remove_idx)
    pts = grid_pts[keep_idx]
    # Jitter
    pts += np.random.normal(0, 0.01, (33, 2))
    pts = np.clip(pts, 0, 1)
    if is_valid(pts):
        c = count_convex_7(pts)
        if c < best_count:
            best_count = c
            best_points = pts.copy()
        if time.time() > deadline:
            return best_points

    # Structure 3: Two Lines (Strip)
    # 17 points on y=0.2, 16 on y=0.8
    x1 = np.linspace(0.1, 0.9, 17)
    y1 = np.full(17, 0.2)
    x2 = np.linspace(0.1, 0.9, 16)
    y2 = np.full(16, 0.8)
    pts = np.vstack([np.column_stack((x1, y1)), np.column_stack((x2, y2))])
    pts += np.random.normal(0, 0.01, (33, 2))
    pts = np.clip(pts, 0, 1)
    if is_valid(pts):
        c = count_convex_7(pts)
        if c < best_count:
            best_count = c
            best_points = pts.copy()
        if time.time() > deadline:
            return best_points

    # Structure 4: Concentric
    # 6, 6, 6, 15
    radii = [0.4, 0.3, 0.2, 0.1]
    counts = [6, 6, 6, 15]
    pts_list = []
    for r, cnt in zip(radii, counts):
        angles = np.linspace(0, 2*np.pi, cnt, endpoint=False)
        xs = r * np.cos(angles) + 0.5
        ys = r * np.sin(angles) + 0.5
        pts_list.append(np.column_stack((xs, ys)))
    pts = np.vstack(pts_list)
    pts += np.random.normal(0, 0.005, (33, 2))
    pts = np.clip(pts, 0, 1)
    if is_valid(pts):
        c = count_convex_7(pts)
        if c < best_count:
            best_count = c
            best_points = pts.copy()
        if time.time() > deadline:
            return best_points

    return best_points

def is_valid(points):
    # Check bounds
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
    # Check NaN/Inf
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return False
    # Check distance
    # O(N^2)
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            d2 = np.sum((points[i] - points[j])**2)
            if d2 < 1e-12:
                return False
    # Check collinearity
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                if area < 1e-9:
                    return False
    return True
