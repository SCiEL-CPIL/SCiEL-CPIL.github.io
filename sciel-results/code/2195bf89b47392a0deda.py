# EVOLVE-BLOCK-START
import numpy as np
import time


def _count_convex7(pts):
    """Count 7-subsets in convex position (all 7 on the convex hull).
    A 7-set is in convex position iff its convex hull has 7 vertices."""
    n = len(pts)
    if n < 7:
        return 0
    # Precompute for each point i, and each other point j, the angle around i.
    # For hull counting we use: a point is on the convex hull of the 7-set
    # iff it is not in the interior of the convex hull of the other 6.
    # Faster: use the fact that a 7-set is convex iff no point is inside the
    # convex hull of the other 6. We test each of the 7 points.
    # Even faster: compute convex hull of the 7 points; if hull size == 7, convex.
    # We'll use a vectorized cross-product approach:
    # For a 7-set S, it is in convex position iff for every point p in S,
    # p is on the convex hull of S.
    # Equivalently: sort the 7 points by angle around their centroid; if all
    # consecutive triples turn the same way (and the polygon is simple), it's convex.
    # But simplest robust test: count hull vertices via a standard O(7 log 7) or
    # O(7^2) convex hull.
    # We'll precompute nothing and just do the test per subset, but vectorize
    # over subsets where possible.
    # Given the time budget, we'll use a reasonably fast per-subset test.
    
    # Precompute: for each pair (i, j), the vector. Not directly helpful.
    # Let's just iterate subsets and do a quick hull test.
    from itertools import combinations
    count = 0
    # For each 7-subset, check if all 7 are hull vertices.
    # A point p is a hull vertex of set S iff there exists a direction d such that
    # p maximizes d . x over x in S. Equivalently, p is on the hull iff it is not
    # strictly inside the convex hull of S\{p}.
    # Quick test: for the 7 points, compute the convex hull using Andrew's monotone
    # chain (O(7 log 7)). If hull has 7 points, it's convex.
    
    def _hull_size(seven_pts):
        # seven_pts: (7,2) array
        p = np.array(seven_pts, dtype=np.float64)
        p = p[np.lexsort((p[:, 1], p[:, 0]))]
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        lower = []
        for pt in p:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
                lower.pop()
            lower.append(pt)
        upper = []
        for pt in p[::-1]:
            while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
                upper.pop()
            upper.append(pt)
        hull = lower[:-1] + upper[:-1]
        return len(hull)
    
    for subset in combinations(range(n), 7):
        if _hull_size(pts[list(subset)]) == 7:
            count += 1
    return count


def _is_valid(pts):
    """Check validity: shape, bounds, no near-duplicates, no near-collinear triples."""
    if pts.shape != (33, 2):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
        return False
    # Squared distance check
    from itertools import combinations
    for i, j in combinations(range(33), 2):
        d2 = np.sum((pts[i] - pts[j]) ** 2)
        if d2 < 1e-12:
            return False
    # Collinearity check
    for i, j, k in combinations(range(33), 3):
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        x3, y3 = pts[k]
        val = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
        if val < 1e-9:
            return False
    return True


def _make_initial():
    """Create a clustered initial configuration:
    27 interior points in a dense grid in the center,
    6 boundary points forming a hexagon."""
    np.random.seed(42)
    pts = np.zeros((33, 2))
    # 6 boundary points: hexagon vertices slightly inside the unit square
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    r = 0.45
    cx, cy = 0.5, 0.5
    for i in range(6):
        pts[i, 0] = cx + r * np.cos(angles[i]) + 0.005*np.random.randn()
        pts[i, 1] = cy + r * np.sin(angles[i]) + 0.005*np.random.randn()
    # 27 interior points: dense grid in center
    # 5x5 grid = 25, plus 2 more
    grid_size = 0.15
    cx, cy = 0.5, 0.5
    idx = 6
    for i in range(5):
        for j in range(5):
            if idx >= 33:
                break
            x = cx - grid_size + (2*grid_size/4)*i
            y = cy - grid_size + (2*grid_size/4)*j
            pts[idx, 0] = x + 0.003*np.random.randn()
            pts[idx, 1] = y + 0.003*np.random.randn()
            idx += 1
    # 2 more interior points
    pts[31, 0] = 0.45 + 0.02*np.random.randn()
    pts[31, 1] = 0.55 + 0.02*np.random.randn()
    pts[32, 0] = 0.55 + 0.02*np.random.randn()
    pts[32, 1] = 0.45 + 0.02*np.random.randn()
    return pts


def erdos_szekeres_problem7() -> np.ndarray:
    """
    Minimize convex 7-gons among 33 points in the unit square.
    Returns (33, 2) float array of points.
    """
    start_time = time.time()
    deadline = 620.0  # safety margin before 630s kill
    
    # Initial configuration
    pts = _make_initial()
    
    # Ensure validity of initial config
    if not _is_valid(pts):
        # Add small perturbations to fix any issues
        np.random.seed(123)
        pts = pts + 0.002 * np.random.randn(33, 2)
        pts = np.clip(pts, 0.01, 0.99)
    
    # Evaluate initial
    C0 = _count_convex7(pts)
    best_pts = pts.copy()
    best_C = C0
    best_val = 1.0 / (1.0 + best_C)
    
    # Local search
    np.random.seed(7)
    step_size = 0.05
    n_iter = 0
    no_improve = 0
    max_no_improve = 500
    
    while time.time() - start_time < deadline:
        n_iter += 1
        
        # Generate a perturbation
        if np.random.rand() < 0.1:
            # Multi-point move
            k = np.random.randint(2, 5)
            idxs = np.random.choice(33, k, replace=False)
            new_pts = pts.copy()
            new_pts[idxs] += step_size * np.random.randn(k, 2)
        else:
            # Single-point move
            idx = np.random.randint(33)
            new_pts = pts.copy()
            new_pts[idx] += step_size * np.random.randn(2)
        
        # Clip to valid range
        new_pts = np.clip(new_pts, 0.001, 0.999)
        
        # Quick validity check (skip full check for speed, do lightweight check)
        # Check for near-duplicates
        valid = True
        for i in range(33):
            for j in range(i+1, 33):
                d2 = np.sum((new_pts[i] - new_pts[j])**2)
                if d2 < 1e-12:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Evaluate
        C_new = _count_convex7(new_pts)
        
        if C_new < best_C:
            pts = new_pts
            best_pts = new_pts.copy()
            best_C = C_new
            best_val = 1.0 / (1.0 + best_C)
            no_improve = 0
            # Reduce step size slightly
            step_size *= 0.98
        else:
            # Accept with probability (simulated annealing-like)
            # Since we want to minimize C, accept worse with small prob
            if np.random.rand() < 0.01:
                pts = new_pts
            no_improve += 1
        
        # Occasionally re-center or restart if stuck
        if no_improve > max_no_improve:
            # Try a different strategy: move a random point toward the center
            idx = np.random.randint(33)
            pts[idx] = 0.5 + 0.1 * np.random.randn(2)
            pts = np.clip(pts, 0.001, 0.999)
            no_improve = 0
            step_size = 0.05
        
        # Periodically try a larger perturbation
        if n_iter % 500 == 0:
            step_size = max(0.01, step_size * 1.2)
        
        # Check if we've found 0 convex 7-gons
        if best_C == 0:
            break
    
    # Final validity check and fix
    if not _is_valid(best_pts):
        # Apply small perturbations to fix
        np.random.seed(999)
        best_pts = best_pts + 0.001 * np.random.randn(33, 2)
        best_pts = np.clip(best_pts, 0.001, 0.999)
        # Ensure no near-duplicates
        for i in range(33):
            for j in range(i+1, 33):
                d2 = np.sum((best_pts[i] - best_pts[j])**2)
                if d2 < 1e-12:
                    best_pts[j] += 0.01 * np.random.randn(2)
                    best_pts[j] = np.clip(best_pts[j], 0.001, 0.999)
    
    return best_pts


# EVOLVE-BLOCK-END
