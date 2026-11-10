# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by maximizing the complement S (excluded points).

    A point x is excluded (in S) if there exists a line L through x such that all
    other points on L are in N = F_49^2 \\ S. Equivalently, S is a set of points
    such that for every s in S, there is a line L with s in L and (L \ {s}) ∩ S = ∅.

    We use a greedy algorithm: start with S = all 2401 points, and iteratively remove
    points from S that are "bad" (i.e., for every line through them, there's another
    S point on that line). We keep points that have at least one "clean" line.

    Multiple restarts with different random orders help find larger valid S.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # 580s safety margin

    p = 7
    q = p * p  # 49
    alpha_sq = 3  # alpha^2 = 3 in F_49

    # Precompute field operations
    # Element (a, b) represents a + b*alpha
    # Multiplication: (a1+b1*alpha)*(a2+b2*alpha) = (a1*a2 + b1*b2*alpha_sq) + (a1*b2 + a2*b1)*alpha
    
    def field_mul(a1, b1, a2, b2):
        ra = (a1 * a2 + b1 * b2 * alpha_sq) % p
        rb = (a1 * b2 + a2 * b1) % p
        return ra, rb

    def field_add(a1, b1, a2, b2):
        return (a1 + a2) % p, (b1 + b2) % p

    def field_sub(a1, b1, a2, b2):
        return (a1 - a2) % p, (b1 - b2) % p

    # Total points: 49 * 49 = 2401
    # Directions: 49 slopes + 1 vertical = 50 directions
    # Total lines: 2401 * 50 / 49 = 2450 lines (each line has 49 points)

    # Enumerate all lines
    # A line is defined by (point, direction). Direction v = (vx, vy) where (vx,vy) != (0,0).
    # Two directions are equivalent if they are scalar multiples.
    # We can normalize: for non-vertical lines, use v = (1, m) for m in F_49.
    # For vertical lines, use v = (0, 1).
    
    # Represent directions as (vx_a, vx_b, vy_a, vy_b)
    directions = []
    # Non-vertical: v = (1, m) where m = (ma, mb)
    for ma in range(p):
        for mb in range(p):
            directions.append((1, 0, ma, mb))  # vx=(1,0), vy=(ma,mb)
    # Vertical: v = (0, 1)
    directions.append((0, 1, 0, 0))  # vx=(0,1), vy=(0,0) -- wait, v=(0,1) means vx=0, vy=1
    # Actually v = (vx, vy) = (0, 1) means vx=(0,0), vy=(1,0)
    directions = []
    for ma in range(p):
        for mb in range(p):
            # v = (1, m) where 1=(1,0), m=(ma,mb)
            directions.append(((1, 0), (ma, mb)))
    # v = (0, 1) where 0=(0,0), 1=(1,0)
    directions.append(((0, 0), (1, 0)))
    
    num_dirs = len(directions)  # 50

    # For each line (identified by a representative point and direction), 
    # we need to know which points are on it.
    # Total lines = 2401 * 50 / 49 = 2450
    
    # Build point index: point (xa, xb, ya, yb) -> index = ((xa*p + xb)*p + ya)*p + yb
    def point_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
    
    def idx_to_point(idx):
        yb = idx % p
        idx //= p
        ya = idx % p
        idx //= p
        xb = idx % p
        idx //= p
        xa = idx
        return xa, xb, ya, yb

    total_points = q * q  # 2401

    # Build all lines: for each direction and each "base" point, generate the line
    # To avoid duplicates, we use a canonical representative.
    # A line with direction v can be represented by its "first" point in some ordering.
    # Simpler: for each direction d and each point p0, the line is {p0 + t*v : t in F_49}.
    # Two (p0, d) pairs give the same line if p0' = p0 + t*v for some t.
    # We can canonicalize: for a given direction d, pick the lexicographically smallest point on the line.

    # Precompute: for each direction, for each point, compute the line's canonical representative
    # and the set of points on the line.

    # Actually, let's build lines differently:
    # For each direction index di, and for each point p, compute the line.
    # Use a dict: line_key -> set of points
    # line_key = (di, canonical_point)

    # More efficient: for each direction, partition points into lines.
    # For direction v, two points p, p' are on the same line iff p - p' is a scalar multiple of v.
    
    # Let's precompute for each direction the equivalence classes.
    # This might be expensive. Alternative: just enumerate lines by (base_point, direction) 
    # and deduplicate.

    # Let's use a different approach: 
    # For the greedy algorithm, we need:
    # 1. For each point s, check if there exists a line L through s with no other S points.
    # 2. If yes, s can stay in S; otherwise, s must be removed.
    
    # To check (1), for point s and direction d, the line through s with direction d is:
    # {s + t*v_d : t in F_49}
    # We need to check if all points on this line except s are not in S.
    
    # Let's precompute for each point and direction, the list of other points on the line.
    # That's 2401 * 50 * 48 = 5,762,400 entries. Too much memory maybe.
    
    # Alternative: for each direction, precompute the lines as arrays of point indices.
    # Number of lines per direction = 2401 / 49 = 49.
    # Total lines = 50 * 49 = 2450. Each line has 49 points.
    # Total storage: 2450 * 49 = 120,050 point indices. That's fine.

    lines = []  # list of arrays of point indices
    line_of_point = []  # for each point, list of line indices it belongs to

    # For each direction, find the 49 lines
    for di, (vx, vy) in enumerate(directions):
        # vx = (vxa, vxb), vy = (vya, vyb)
        vxa, vxb = vx
        vya, vyb = vy
        
        # Generate all points on lines with this direction
        # A point (xa, xb, ya, yb) is on a line with direction v.
        # Two points are on the same line iff their difference is a multiple of v.
        
        # Let's find canonical representatives.
        # For a given direction, the line through (xa, xb, ya, yb) consists of:
        # (xa + t*vxa, xb + t*vxb, ya + t*vya, yb + t*vyb) for t in F_49.
        
        # To find the 49 lines, we can iterate over points and group them.
        # Use a visited set.
        
        visited = set()
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        idx = point_idx(xa, xb, ya, yb)
                        if idx in visited:
                            continue
                        # Generate the line
                        line_pts = []
                        for ta in range(p):
                            for tb in range(p):
                                # t = ta + tb*alpha
                                # p_t = p0 + t*v
                                pxa = (xa + field_mul(ta, tb, vxa, vxb)[0]) % p
                                pxb = (xb + field_mul(ta, tb, vxa, vxb)[1]) % p
                                pya = (ya + field_mul(ta, tb, vya, vyb)[0]) % p
                                pyb = (yb + field_mul(ta, tb, vya, vyb)[1]) % p
                                pt_idx = point_idx(pxa, pxb, pya, pyb)
                                line_pts.append(pt_idx)
                                visited.add(pt_idx)
                        lines.append(np.array(line_pts, dtype=np.int32))
        
        # Should have added 49 lines for this direction
    
    # Now build line_of_point
    for i in range(total_points):
        line_of_point.append([])
    
    for li, line_pts in enumerate(lines):
        for pt in line_pts:
            line_of_point[pt].append(li)
    
    # Convert to numpy for efficiency
    line_of_point = [np.array(lp, dtype=np.int32) for lp in line_of_point]
    
    # Greedy algorithm to maximize S
    def greedy_s(max_iter=None):
        """
        Start with S = all points, iteratively remove points that don't have a clean line.
        A point s has a clean line if there exists a line L through s such that
        all other points on L are not in S.
        """
        in_s = np.ones(total_points, dtype=bool)
        s_count = total_points
        
        if max_iter is None:
            max_iter = 100
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # For each point in S, check if it has a clean line
            # A point s is "removable" if for ALL lines through s, there exists another S point on that line.
            # A point s is "keepable" if there exists a line through s with no other S points.
            
            # We want to keep keepable points and remove others.
            # But this is interdependent. Let's do:
            # - Mark all points as tentatively keepable
            # - For each point s in S, if no line through s is clean, mark s as removable
            # - Remove all removable points
            # - Repeat until no change
            
            changed = False
            to_remove = []
            
            for s in range(total_points):
                if not in_s[s]:
                    continue
                
                # Check if s has a clean line
                has_clean = False
                for li in line_of_point[s]:
                    line_pts = lines[li]
                    # Check if all other points on this line are not in S
                    clean = True
                    for pt in line_pts:
                        if pt != s and in_s[pt]:
                            clean = False
                            break
                    if clean:
                        has_clean = True
                        break
                
                if not has_clean:
                    to_remove.append(s)
            
            if to_remove:
                changed = True
                for s in to_remove:
                    in_s[s] = False
                    s_count -= 1
        
        return in_s, s_count
    
    # Run multiple restarts with different random orderings
    # Actually, the greedy above is deterministic. Let's add randomness.
    
    def greedy_s_random(seed, max_iter=50):
        rng = np.random.RandomState(seed)
        in_s = np.ones(total_points, dtype=bool)
        s_count = total_points
        
        # Randomize the order of processing
        order = rng.permutation(total_points)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            changed = False
            to_remove = []
            
            for s in order:
                if not in_s[s]:
                    continue
                
                # Check if s has a clean line
                has_clean = False
                for li in line_of_point[s]:
                    line_pts = lines[li]
                    clean = True
                    for pt in line_pts:
                        if pt != s and in_s[pt]:
                            clean = False
                            break
                    if clean:
                        has_clean = True
                        break
                
                if not has_clean:
                    to_remove.append(s)
            
            if to_remove:
                changed = True
                for s in to_remove:
                    in_s[s] = False
                    s_count -= 1
            else:
                break
        
        return in_s, s_count
    
    # Try the deterministic version first
    best_in_s, best_count = greedy_s(max_iter=20)
    
    # Try random restarts
    num_restarts = 20
    for seed in range(num_restarts):
        if time.time() > deadline:
            break
        in_s, count = greedy_s_random(seed, max_iter=30)
        if count > best_count:
            best_in_s = in_s
            best_count = count
    
    # Construct the Nikodym set N = F_49^2 \\ S
    points = []
    for i in range(total_points):
        if not best_in_s[i]:
            xa, xb, ya, yb = idx_to_point(i)
            points.append([[xa, xb], [ya, yb]])
    
    if not points:
        # Fallback: return all points (valid but not optimal)
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
