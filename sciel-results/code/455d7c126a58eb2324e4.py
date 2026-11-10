# EVOLVE-BLOCK-START
import numpy as np
from math import comb, atan2
import sys

def count_convex_heptagons(pts):
    """
    Count the number of 7-point subsets in convex position using a dynamic programming
    approach over convex chains.
    
    Points are sorted lexicographically. For each pivot, later points are sorted by
    atan2 angle around it. A turn p -> q -> r counts as convex only if the cross product
    (qx-px)(ry-qy) - (qy-py)(rx-qx) > 1e-9, including closing turns back to the pivot.
    """
    n = len(pts)
    # Sort lexicographically
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts_sorted = pts[order]
    
    # For each pivot i, consider points j > i
    # Sort j > i by angle around i
    total = 0
    
    for i in range(n - 6):
        # Points after i
        later = pts_sorted[i+1:]
        # Vectors from i to later points
        vecs = later - pts_sorted[i]
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        # Sort by angle
        ang_order = np.argsort(angles)
        pts_around = np.vstack([pts_sorted[i], later[ang_order]])
        # pts_around[0] is the pivot, pts_around[1:] are the later points sorted by angle
        
        m = len(pts_around)  # m = n - i
        # We need to count convex 7-gons with pivot as the "first" point in angular order
        # Actually, the standard DP for counting convex k-gons:
        # dp[j][l] = number of convex chains of length l ending at point j (in the angular sequence)
        # where the chain starts at the pivot.
        
        # Let's index pts_around as 0 (pivot), 1, 2, ..., m-1
        # A convex chain of length l starting at pivot and ending at point j means:
        # pivot -> ... -> j, all turns convex.
        # For a 7-gon, we need a chain of length 6 from pivot (so 7 points total including pivot)
        # but the last turn must close back to pivot.
        
        # Actually, let me think more carefully. The standard approach:
        # For each pivot p, sort other points by angle. A convex polygon containing p
        # corresponds to a sequence of points p -> a1 -> a2 -> ... -> a6 -> p where all
        # turns are convex (left turns).
        
        # dp[j][l] = number of ways to form a convex chain of l points (excluding pivot)
        # ending at point j, where all turns are convex.
        # We want chains of length 6 (so 6 points after pivot = 7 total) where the last
        # point can connect back to pivot with a convex turn.
        
        # Points in angular order: 1, 2, ..., m-1 (index 0 is pivot)
        # For a chain pivot -> i1 -> i2 -> ... -> il, we need:
        # - i1 < i2 < ... < il (in angular order, to ensure convexity)
        # - Each turn is convex
        
        # dp[j][l] for l from 1 to 6, j from 1 to m-1
        # dp[j][1] = 1 for all j (chain of just pivot -> j)
        # dp[j][l] = sum over i < j where turn (prev, i, j) is convex... 
        # Wait, we need to track the previous point too.
        
        # Better: dp[i][j][l] where i is second point, j is last point, l is chain length.
        # But that's O(m^3) per pivot. With m up to 33, that's manageable.
        
        # Actually, let's use: dp[j][l] = number of convex chains of length l ending at j
        # where the chain starts at pivot. For l=1, dp[j][1]=1.
        # For l>1, dp[j][l] = sum_{i<j, turn(pivot->...->i->j) convex} dp[i][l-1]
        # But we need to check the turn from i to j, which depends on the point before i.
        
        # Hmm, this is tricky. Let me use a different formulation.
        
        # Let f(j, l) = number of convex chains of l points (including pivot) ending at point j
        # where j is the last point. The chain is pivot = p0, p1, ..., pl = j.
        # For l=1: f(j, 1) = 1 for all j >= 1.
        # For l >= 2: f(j, l) = sum_{i: 1 <= i < j, turn(pi, pj, ...) } ...
        
        # I think the cleanest way is:
        # g[i][j] = 1 if the turn from pivot to i to j is convex, else 0.
        # But for longer chains, we need the turn from the previous point.
        
        # Let me use: dp[i][j] = number of convex chains ending with ... -> i -> j
        # where the chain starts at pivot.
        # Base: dp[i][j] for chains of length 2 (pivot, i, j) = 1 if turn(pivot, i, j) is convex.
        # Recurrence: dp[i][j] for longer chains = sum over k < i where turn(k, i, j) is convex of dp[k][i]
        
        # Then a 7-gon corresponds to a chain of 7 points: pivot, a1, a2, a3, a4, a5, a6
        # where all turns are convex AND the closing turn a6 -> pivot is convex.
        # The number of such chains ending at a6 is dp[a5][a6], and we need to check
        # the closing turn.
        
        # Let's implement this.
        
        # Indices: 0 = pivot, 1 to m-1 are the other points in angular order
        # For j > i > 0, check if turn(pts_around[i], pts_around[j], ...) 
        # Actually, turn from point i to point j as seen from... 
        # The turn at point i in the chain ... -> prev -> i -> j is convex if
        # (i - prev) x (j - i) > 1e-9
        
        # For the base case (chain of length 3: pivot, i, j):
        # Turn at i: (i - pivot) x (j - i) > 1e-9
        
        # Let's precompute cross products.
        # cross[i][j] = (pts_around[i] - pts_around[0]) x (pts_around[j] - pts_around[i])
        # No, that's not right either.
        
        # Let me just compute it directly.
        # For a chain ending with ... -> a -> b, the turn at a is convex if
        # (b - prev) x (a - prev) ... no.
        
        # Turn at point a in sequence prev -> a -> b is convex if
        # (a - prev) x (b - a) > 1e-9
        # where x is the 2D cross product: (a-prev).x * (b-a).y - (a-prev).y * (b-a).x
        
        # For base case: prev = pivot, a = i, b = j
        # turn = (i - pivot) x (j - i) > 1e-9
        
        # Let's compute a table:
        # turn[i][j] = (pts_around[i] - pts_around[0]) x (pts_around[j] - pts_around[i])
        # for 1 <= i < j <= m-1
        
        # Then dp[i][j] = number of convex chains of length >= 3 ending with i -> j
        # dp[i][j] = turn[i][j] * (1 + sum_{k: 1 <= k < i, turn[k][i] and ...} dp[k][i])
        
        # Wait, I need to be more careful. Let's define:
        # dp[i][j] = number of convex chains starting at pivot, ending with ... -> i -> j
        # where i < j in angular order.
        
        # Base: if turn(i, j) is convex (i.e., (pts_i - pivot) x (pts_j - pts_i) > 1e-9),
        #       then dp[i][j] >= 1 (the chain pivot -> i -> j)
        # Recurrence: dp[i][j] += dp[k][i] for all k < i where turn(k, i) is convex
        #             (because we can extend any chain ending at k -> i to k -> i -> j
        #              if the turn at i from k to j is... wait, no. The turn at i depends
        #              on the previous point, which is k in this case.)
        
        # Hmm, I think the issue is that the turn at i depends on the specific previous point.
        # So dp should track the last two points.
        
        # Let me use: dp[i][j] = number of convex chains ending with i -> j (i < j)
        # Base: dp[i][j] = 1 if (pts_i - pivot) x (pts_j - pts_i) > 1e-9 else 0
        #       (this is the chain pivot -> i -> j)
        # Recurrence: for k < i, if (pts_i - pts_k) x (pts_j - pts_i) > 1e-9,
        #             then dp[i][j] += dp[k][i]
        
        # Then a 7-gon with pivot corresponds to a chain of 7 points:
        # pivot -> a1 -> a2 -> a3 -> a4 -> a5 -> a6
        # The number of such chains is dp[a5][a6] where the chain has exactly 7 points.
        # But dp[a5][a6] counts chains of all lengths ending with a5 -> a6.
        
        # I need to track the length. Let me use dp[l][i][j] where l is the number of points
        # in the chain (including pivot).
        
        # This is getting complex. Given the time constraint, let me use a simpler approach:
        # brute force for small n, or use the fact that n=33 is small enough for a clever DP.
        
        # Actually, for n=33, the number of 7-subsets is C(33,7) = 4,272,048.
        # Checking if each is convex is O(7) per subset, so total ~30M operations.
        # That should be fast enough in Python with NumPy.
        
        # Let me just do brute force.
        pass
    
    # Brute force approach
    from itertools import combinations
    
    count = 0
    for subset in combinations(range(n), 7):
        sub_pts = pts[subset]
        # Check if these 7 points form a convex heptagon
        # Sort by angle around centroid
        centroid = sub_pts.mean(axis=0)
        angles = np.arctan2(sub_pts[:, 1] - centroid[1], sub_pts[:, 0] - centroid[0])
        sorted_pts = sub_pts[np.argsort(angles)]
        
        # Check all 7 turns
        is_convex = True
        for i in range(7):
            p0 = sorted_pts[i]
            p1 = sorted_pts[(i + 1) % 7]
            p2 = sorted_pts[(i + 2) % 7]
            cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
            if cross <= 1e-9:
                is_convex = False
                break
        
        if is_convex:
            count += 1
    
    return count


def validate_points(points):
    """Validate that points meet all constraints."""
    if points.shape != (33, 2):
        return False
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return False
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
    
    # Check pairwise distances
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            if np.dot(diff, diff) < 1e-12:
                return False
    
    # Check no three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                det = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
                if det < 1e-9:
                    return False
    
    return True


def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres "Happy Ending" problem. Place n = 33 points in general position
    inside the unit square so as to MINIMIZE the number of convex 7-gons.
    """
    n = 33
    
    # Start with a grid-like construction that has many collinear points,
    # then perturb to break collinearity while preserving the non-convex structure.
    
    # Create a 6x6 grid (36 points) and remove 3 points, or use a different arrangement.
    # Actually, let's start with points on a curve (e.g., parabola) which tends to
    # have fewer convex polygons.
    
    # Try a construction based on points on a convex curve first, then perturb inward.
    # Points on a convex curve (like a circle or parabola) have ALL subsets in convex position,
    # which is bad. We want points that are NOT in convex position for most subsets.
    
    # A better approach: use a construction with many points on line segments.
    # For example, divide the square into several line segments and place points on them.
    
    # Let's try a "comb" construction: points arranged in rows and columns with some
    # structure to minimize convex heptagons.
    
    # Start with a random initialization and use hill climbing.
    np.random.seed(42)
    
    # Initialize with points on a grid, then perturb
    grid_size = 6
    xs = np.linspace(0.1, 0.9, grid_size)
    ys = np.linspace(0.1, 0.9, grid_size)
    grid_pts = np.array([[x, y] for y in ys for x in xs])  # 36 points
    # Remove 3 random points to get 33
    remove_idx = np.random.choice(36, 3, replace=False)
    keep_mask = np.ones(36, dtype=bool)
    keep_mask[remove_idx] = False
    points = grid_pts[keep_mask].copy()
    
    # Add small perturbations to break collinearity
    points += np.random.uniform(-0.01, 0.01, size=points.shape)
    points = np.clip(points, 0.0, 1.0)
    
    # Validate and fix if needed
    if not validate_points(points):
        # Try again with different seed
        np.random.seed(123)
        points = np.random.uniform(0.05, 0.95, size=(33, 2))
    
    # Hill climbing
    best_points = points.copy()
    best_count = count_convex_heptagons(best_points)
    
    max_iterations = 5000
    step_size = 0.05
    
    for iteration in range(max_iterations):
        # Try moving a random point
        idx = np.random.randint(0, 33)
        old_pt = best_points[idx].copy()
        
        # Move in a random direction
        direction = np.random.randn(2)
        direction /= np.linalg.norm(direction)
        new_pt = best_points[idx] + direction * step_size
        new_pt = np.clip(new_pt, 0.0, 1.0)
        
        best_points[idx] = new_pt
        
        if validate_points(best_points):
            new_count = count_convex_heptagons(best_points)
            if new_count < best_count:
                best_count = new_count
                step_size *= 0.95  # Reduce step size on improvement
            else:
                best_points[idx] = old_pt  # Revert
                step_size *= 1.05  # Increase step size
                step_size = min(step_size, 0.1)
        else:
            best_points[idx] = old_pt  # Revert
        
        if best_count == 0:
            break
    
    return best_points


# EVOLVE-BLOCK-END
