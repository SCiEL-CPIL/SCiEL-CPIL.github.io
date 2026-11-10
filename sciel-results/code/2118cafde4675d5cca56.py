import numpy as np
import time

def count_convex_7_fast(points):
    n = len(points)
    if n < 7:
        return 0

    # Sort lexicographically
    idx = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[idx]
    
    total = 0

    for i in range(n - 6):
        # Pivot is pts[i]
        later = pts[i+1:]
        m = len(later)
        
        if m < 6:
            continue

        # Vector from pivot to later points
        vecs = later - pts[i]
        
        # Compute angles
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        
        # Sort by angle
        order = np.argsort(angles)
        later_sorted = later[order]
        
        # DP: dp[j][l] = number of convex chains of length l ending at j (index in later_sorted)
        # l=1: dp[j][1] = 1
        # l=2: dp[j][2] = sum_{k < j} dp[k][1] * I(cross(i, k, j) > 1e-9)
        # ...
        # l=6: dp[j][6]
        
        # We need to store the "previous point" to check the turn?
        # No, if the chain is convex, the turn condition is local.
        # But the DP state $dp[j][l]$ aggregates over all previous points.
        # The transition $k \to j$ requires the turn at $k$ to be valid?
        # The turn at $k$ involves the point BEFORE $k$.
        # So $dp[j][l]$ is NOT sufficient.
        
        # We MUST use $dp[j][l][p]$ or equivalent.
        # Given $m \approx 30$, $m^3 = 27000$.
        # $33 \times 27000 \times 6 \approx 5$ million operations.
        # This is fast enough.
        
        # dp[j][l][p] : chains of length l ending at j, with previous point p.
        # j, p are indices in 0..m-1.
        # l ranges 1..6.
        
        # Initialize
        # l=1: No previous point. Just 1.
        # We can treat "previous point" as -1 for l=1.
        
        # Let's use a 3D array: dp[l][j][p]
        # l=1: dp[1][j][0] = 1 (dummy prev)
        # l=2: dp[2][j][p] = 1 if cross(i, p, j) > 1e-9, else 0.
        # l=3: dp[3][j][p] = sum_{q < p} dp[2][p][q] * I(cross(p, j, q?) No.
        # Turn at p is (q, p, j). So we need cross(q, p, j) > 1e-9.
        # So dp[3][j][p] = sum_{q < p} dp[2][p][q] * I(cross(q, p, j) > 1e-9).
        
        # This is O(m^3) per l.
        
        # Precompute cross products for all triples (a, b, c) in later_sorted?
        # And cross(i, a, b)?
        
        # Let's compute cross(i, a, b) for all a, b.
        # And cross(a, b, c) for all a, b, c.
        
        # m is small.
        
        # cross_ip_j[i, j] = cross(pts[i], later_sorted[i], later_sorted[j])
        # cross_abc[i, j, k] = cross(later_sorted[i], later_sorted[j], later_sorted[k])
        
        # This is getting complex to vectorize perfectly.
        # Let's just do the loops. m=30.
        
        # Precompute cross(i, a, b)
        cross_i = np.zeros((m, m))
        for a in range(m):
            for b in range(m):
                if a == b: continue
                v1 = later_sorted[a] - pts[i]
                v2 = later_sorted[b] - pts[i]
                # cross(v1, v2)
                cross_i[a, b] = v1[0]*v2[1] - v1[1]*v2[0]
                
        # Precompute cross(a, b, c)
        # cross(later_sorted[a], later_sorted[b], later_sorted[c])
        # This is cross(b-a, c-b)
        cross_abc = np.zeros((m, m, m))
        for a in range(m):
            for b in range(m):
                if a == b: continue
                v1 = later_sorted[b] - later_sorted[a]
                for c in range(m):
                    if b == c: continue
                    v2 = later_sorted[c] - later_sorted[b]
                    cross_abc[a, b, c] = v1[0]*v2[1] - v1[1]*v2[0]
                    
        # DP
        # dp[l][j][p]
        # l=1
        dp = np.zeros((7, m, m))
        for j in range(m):
            dp[1][j][0] = 1 # p=0 is dummy
            
        # l=2
        for j in range(m):
            for p in range(j): # p must be before j in angle order? 
                # The chain is i -> p -> j.
                # Turn i, p, j must be left.
                if cross_i[p, j] > 1e-9:
                    dp[2][j][p] = 1
                    
        # l=3..6
        for l in range(3, 7):
            for j in range(m):
                for p in range(j): # p is the point before j
                    # We sum over q < p
                    # dp[l-1][p][q] * I(cross(q, p, j) > 1e-9)
                    # Note: q must be before p.
                    s = 0
                    for q in range(p):
                        if cross_abc[q, p, j] > 1e-9:
                            s += dp[l-1][p][q]
                    dp[l][j][p] = s
                    
        # Now, for each chain of length 6 ending at j with prev p:
        # We have i, ..., p, j.
        # We need to check:
        # 1. Turn(p, j, i) > 1e-9 (Closing turn at j? No, turn at j is (prev_j, j, i). Here prev_j is p? No, the chain is i..p, j. The last point is j. The turn at j in the polygon is (p, j, i).)
        # 2. Turn(j, i, v1) > 1e-9. v1 is the first point in the chain.
        #    We don't store v1.
        
        # This DP doesn't track v1.
        # So we can't check the turn at i.
        
        # However, if the points are sorted by angle, and the chain is convex, 
        # is the turn at i automatically valid?
        # The angle of v1 is the smallest, angle of j is the largest.
        # The turn at i is (j, i, v1).
        # This depends on the specific v1.
        
        # So this DP is insufficient.
        
        # Given the time, I will revert to the subset enumeration but optimize it with a C-extension-like approach?
        # No, I'll just use the subset enumeration with a fast convexity check.
        
        # I'll implement the subset enumeration in the main function and hope it's fast enough.
        pass

def erdos_szekeres_problem7():
    start_time = time.time()
    deadline = start_time + 580
    
    np.random.seed(42)
    best_points = np.random.rand(33, 2)
    best_score = 1.0 / (1.0 + count_convex_7_subset(best_points))
    
    current_points = best_points.copy()
    current_score = best_score
    step_size = 0.05
    
    while time.time() < deadline:
        neighbor = current_points.copy()
        idx = np.random.randint(33)
        neighbor[idx, 0] += np.random.normal(0, step_size)
        neighbor[idx, 1] += np.random.normal(0, step_size)
        neighbor[idx, 0] = np.clip(neighbor[idx, 0], 0, 1)
        neighbor[idx, 1] = np.clip(neighbor[idx, 1], 0, 1)
        
        if not is_valid(neighbor):
            continue
            
        score = 1.0 / (1.0 + count_convex_7_subset(neighbor))
        
        if score > current_score:
            current_score = score
            current_points = neighbor
            if score > best_score:
                best_score = score
                best_points = neighbor.copy()
                
    return best_points

def count_convex_7_subset(points):
    n = len(points)
    if n < 7: return 0
    from itertools import combinations
    count = 0
    for combo in combinations(range(n), 7):
        pts_sub = points[list(combo)]
        cx = np.mean(pts_sub[:, 0])
        cy = np.mean(pts_sub[:, 1])
        dx = pts_sub[:, 0] - cx
        dy = pts_sub[:, 1] - cy
        angles = np.arctan2(dy, dx)
        order = np.argsort(angles)
        sorted_pts = pts_sub[order]
        
        is_convex = True
        for k in range(7):
            p0 = sorted_pts[k]
            p1 = sorted_pts[(k+1)%7]
            p2 = sorted_pts[(k+2)%7]
            v1 = p1 - p0
            v2 = p2 - p1
            if v1[0]*v2[1] - v1[1]*v2[0] <= 1e-9:
                is_convex = False
                break
        if is_convex:
            count += 1
    return count

def is_valid(points):
    n = len(points)
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
    if not np.all(np.isfinite(points)):
        return False
    for i in range(n):
        for j in range(i+1, n):
            d2 = np.sum((points[i] - points[j])**2)
            if d2 < 1e-12:
                return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                val = p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])
                if abs(val) < 1e-9:
                    return False
    return True
