# EVOLVE-BLOCK-START
import numpy as np
import time

def _count_convex7(points: np.ndarray) -> int:
    """
    Count the number of 7-point subsets in convex position using a dynamic program
    over convex chains, with vectorized transitions.
    """
    n = len(points)
    # Sort lexicographically
    order = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[order]

    total = 0

    for i in range(n - 6):
        later = pts[i + 1:]
        m = len(later)
        if m < 6:
            break

        # Angles from pivot i to each later point
        dx = later[:, 0] - pts[i, 0]
        dy = later[:, 1] - pts[i, 1]
        angles = np.arctan2(dy, dx)
        angle_order = np.argsort(angles)
        spts = later[angle_order]
        m = len(spts)

        # Precompute cross products for turns between consecutive-ish positions.
        # cross[a, b] = cross product of (spts[a] -> spts[b]) relative to pivot? No.
        # We need turn a->b->c: cross = (spts[b]-spts[a]) x (spts[c]-spts[b])
        # For DP, we only need turn from a to b given previous point a_prev? Actually the turn
        # at b depends on the previous point a and next point c. In a chain, when extending
        # a chain ending at a by adding b, we need the turn prev->a->b to be convex.
        # So we need cross(prev, a, b) > 1e-9.
        
        # Let's precompute cross3[a, b, c] for a < b < c? That's O(m^3) which is too much.
        # Instead, for each pair (a, b) with a < b, we can compute the cross product of
        # (spts[a] -> spts[b]) with ... wait.
        
        # The turn at point b in the sequence ... -> a -> b -> c is:
        # (spts[b] - spts[a]) x (spts[c] - spts[b])
        # This depends on a, b, c.
        
        # Alternative: For each pivot i, and for each pair (a, b) in the angular-sorted list,
        # we can define a "compatibility" that depends on the previous point.
        
        # Let's use the standard DP:
        # dp[k][b] = number of convex chains of length k (ending at position b in the angular-sorted list)
        # where the chain starts at some position a < b (the first point after pivot).
        # Actually, the chain is: i -> q1 -> q2 -> ... -> qk
        # where q1, q2, ..., qk are from spts in increasing angular order.
        # The turns that must be convex:
        # 1. i -> q1 -> q2
        # 2. q1 -> q2 -> q3
        # ...
        # k. q_{k-1} -> qk -> i  (closing)
        
        # For the DP, when we extend a chain ending at q_j (j < k) by adding q_k,
        # we need the turn q_{j-1} -> q_j -> q_k to be convex.
        # So dp[k][q_k] = sum over j < k of dp[k-1][q_j] * [turn q_{j-1} -> q_j -> q_k is convex]
        # But dp[k-1][q_j] doesn't tell us what q_{j-1} was.
        
        # We need a 2D DP: dp[k][a][b] where a is the second-to-last point and b is the last.
        # dp[k][a][b] = number of convex chains of length k (k points from spts, not counting i)
        # ending with ... -> a -> b, where a < b in angular order.
        # Base: dp[2][a][b] = 1 if turn i -> spts[a] -> spts[b] is convex, else 0.
        # Transition: dp[k][b][c] = sum over a < b of dp[k-1][a][b] * [turn spts[a] -> spts[b] -> spts[c] is convex]
        # where a < b < c in angular order.
        
        # Final: for each a < b, the number of 7-gons with pivot i and last two points a, b is
        # dp[6][a][b] * [turn spts[a] -> spts[b] -> i is convex]
        
        # m is at most 32. The number of pairs (a, b) with a < b is C(m, 2) <= 496.
        # For each k from 3 to 6, for each pair (b, c) with b < c, we sum over a < b.
        # This is O(m^3) per pivot, which is ~32^3 = 32768 per pivot, times 33 pivots = ~1M operations.
        # This is very fast.
        
        # Let's implement this.
        # dp[a, b] for current k. We'll iterate k from 2 to 6.
        
        # First, precompute turn_convex[a, b, c] for a < b < c? No, we need it on the fly.
        # turn_convex(a, b, c) = cross(spts[a], spts[b], spts[c]) > 1e-9
        # where cross(p, q, r) = (q[0]-p[0])*(r[1]-q[1]) - (q[1]-p[1])*(r[0]-q[0])
        
        # Also need turn from i to a to b: cross(i, spts[a], spts[b]) > 1e-9
        # And closing turn: cross(spts[a], spts[b], i) > 1e-9
        
        # Precompute cross_i[a, b] = cross(pts[i], spts[a], spts[b])
        # Precompute cross_close[a, b] = cross(spts[a], spts[b], pts[i])
        
        pi_pt = pts[i]
        
        # cross_i[a, b] for a < b
        # cross_i[a, b] = (spts[a][0]-pi_pt[0])*(spts[b][1]-spts[a][1]) - (spts[a][1]-pi_pt[1])*(spts[b][0]-spts[a][0])
        
        # Let's compute these as matrices.
        # For efficiency, we can vectorize.
        
        # cross_i[a, b]:
        # Let's compute for all a < b.
        # We can use broadcasting.
        # spts shape (m, 2)
        # diff_ab = spts[None, :, :] - spts[:, None, :]  # shape (m, m, 2)
        # This gives spts[b] - spts[a] for (a, b).
        
        # Actually, let's just use loops for clarity since m is small.
        
        # dp[a, b] = number of chains of length k ending with a, b (a < b)
        # Start with k=2:
        dp = np.zeros((m, m), dtype=np.int64)
        
        # For all a < b, check if turn i -> spts[a] -> spts[b] is convex
        for a in range(m - 1):
            for b in range(a + 1, m):
                cross = (spts[a, 0] - pi_pt[0]) * (spts[b, 1] - spts[a, 1]) - \
                        (spts[a, 1] - pi_pt[1]) * (spts[b, 0] - spts[a, 0])
                if cross > 1e-9:
                    dp[a, b] = 1
        
        # Now extend to length 3, 4, 5, 6
        for k in range(3, 7):
            new_dp = np.zeros((m, m), dtype=np.int64)
            for b in range(m - 1):
                for c in range(b + 1, m):
                    # Sum over a < b of dp[a, b] * [turn spts[a] -> spts[b] -> spts[c] is convex]
                    s = 0
                    for a in range(b):
                        if dp[a, b] > 0:
                            cross = (spts[b, 0] - spts[a, 0]) * (spts[c, 1] - spts[b, 1]) - \
                                    (spts[b, 1] - spts[a, 1]) * (spts[c, 0] - spts[b, 0])
                            if cross > 1e-9:
                                s += dp[a, b]
                    new_dp[b, c] = s
            dp = new_dp
        
        # Now dp[a, b] for a < b gives the number of chains of length 6 ending with a, b.
        # Check closing turn: spts[a] -> spts[b] -> i
        for a in range(m - 1):
            for b in range(a + 1, m):
                if dp[a, b] > 0:
                    cross = (spts[b, 0] - spts[a, 0]) * (pi_pt[1] - spts[b, 1]) - \
                            (spts[b, 1] - spts[a, 1]) * (pi_pt[0] - spts[b, 0])
                    if cross > 1e-9:
                        total += dp[a, b]
    
    return total

def _is_valid(points: np.ndarray) -> bool:
    """Check if the point set is valid."""
    if points.shape != (33, 2):
        return False
    if not np.all(np.isfinite(points)):
        return False
    if not np.all(points >= -1e-6) or not np.all(points <= 1 + 1e-6):
        return False
    
    # Check no two points too close
    for i in range(33):
        for j in range(i + 1, 33):
            d2 = (points[i, 0] - points[j, 0]) ** 2 + (points[i, 1] - points[j, 1]) ** 2
            if d2 < 1e-12:
                return False
    
    # Check no three collinear
    for i in range(33):
        for j in range(i + 1, 33):
            for k in range(j + 1, 33):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                cross = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                if abs(cross) < 1e-9:
                    return False
    
    return True

def _make_valid(points: np.ndarray) -> np.ndarray:
    """Ensure points are valid by adding small perturbations if needed."""
    points = points.copy()
    for _ in range(100):
        if _is_valid(points):
            return points
        # Add small random perturbations
        perturb = np.random.uniform(-0.001, 0.001, size=points.shape)
        points = points + perturb
        points = np.clip(points, 0.001, 0.999)
    # If still invalid, spread points more
    np.random.shuffle(points)
    points = np.clip(points, 0.001, 0.999)
    return points

def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres "Happy Ending" problem. Place n = 33 points
    in general position inside the unit square so as to MINIMIZE the number of
    convex 7-gons whose vertices are among the points.
    """
    n = 33
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    best_points = None
    best_count = float('inf')
    
    # Try multiple initial configurations
    configs = []
    
    # Config 1: Corner clusters
    np.random.seed(42)
    corner_centers = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    sigma = 0.02
    pts1 = np.zeros((n, 2))
    for i in range(n):
        corner = corner_centers[i % 4]
        pts1[i, 0] = corner[0] + np.random.normal(0, sigma)
        pts1[i, 1] = corner[1] + np.random.normal(0, sigma)
    pts1 = np.clip(pts1, 0.01, 0.99)
    configs.append(pts1)
    
    # Config 2: More spread corner clusters
    np.random.seed(123)
    corner_centers2 = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    sigma2 = 0.03
    pts2 = np.zeros((n, 2))
    for i in range(n):
        corner = corner_centers2[i % 4]
        pts2[i, 0] = corner[0] + np.random.normal(0, sigma2)
        pts2[i, 1] = corner[1] + np.random.normal(0, sigma2)
    pts2 = np.clip(pts2, 0.01, 0.99)
    configs.append(pts2)
    
    # Config 3: Random uniform
    np.random.seed(7)
    pts3 = np.random.uniform(0.05, 0.95, size=(n, 2))
    configs.append(pts3)
    
    # Config 4: Grid-like with perturbations
    np.random.seed(99)
    grid_x = np.linspace(0.1, 0.9, 6)
    grid_y = np.linspace(0.1, 0.9, 6)
    grid_pts = []
    for gx in grid_x:
        for gy in grid_y:
            grid_pts.append([gx, gy])
    grid_pts = np.array(grid_pts[:n])
    perturb = np.random.uniform(-0.02, 0.02, size=grid_pts.shape)
    pts4 = grid_pts + perturb
    pts4 = np.clip(pts4, 0.01, 0.99)
    configs.append(pts4)
    
    for cfg in configs:
        if time.time() > deadline:
            break
        
        pts = _make_valid(cfg)
        count = _count_convex7(pts)
        
        if count < best_count:
            best_count = count
            best_points = pts.copy()
        
        # Hill climbing from this configuration
        step_size = 0.01
        no_improve = 0
        max_no_improve = 50
        
        while time.time() < deadline and no_improve < max_no_improve:
            idx = np.random.randint(n)
            dx = np.random.normal(0, step_size)
            dy = np.random.normal(0, step_size)
            
            new_pts = pts.copy()
            new_pts[idx, 0] += dx
            new_pts[idx, 1] += dy
            new_pts = np.clip(new_pts, 0.001, 0.999)
            
            if not _is_valid(new_pts):
                no_improve += 1
                continue
            
            new_count = _count_convex7(new_pts)
            
            if new_count < count:
                count = new_count
                pts = new_pts.copy()
                no_improve = 0
                step_size *= 0.95
            else:
                no_improve += 1
                if no_improve % 30 == 0:
                    step_size = min(step_size * 1.5, 0.05)
            
            # Update global best
            if count < best_count:
                best_count = count
                best_points = pts.copy()
    
    if best_points is None:
        # Fallback: return a valid configuration
        best_points = _make_valid(np.random.uniform(0.05, 0.95, size=(n, 2)))
    
    return best_points

# EVOLVE-BLOCK-END
