# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # 10s safety margin
    
    p = 7
    q = p * p  # 49
    
    # alpha^2 = 3 (quadratic non-residue mod 7)
    alpha_sq = 3
    
    # Precompute field operations
    # Element is (a, b) representing a + b*alpha
    # Add: (a1+b1*alpha) + (a2+b2*alpha) = (a1+a2) + (b1+b2)*alpha
    # Mul: (a1+b1*alpha)*(a2+b2*alpha) = (a1*a2 + b1*b2*alpha_sq) + (a1*b2 + a2*b1)*alpha
    # Mod p reduction for coefficients
    
    # Directions: v = (1, m) for m in F_49, or v = (0, 1)
    # Total 50 directions
    
    # Build list of all points: 2401 points
    # Index: point (xa, xb, ya, yb) -> index = ((xa*p + xb)*p + ya)*p + yb
    def point_index(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
    
    def index_to_point(idx):
        yb = idx % p
        idx //= p
        ya = idx % p
        idx //= p
        xb = idx % p
        idx //= p
        xa = idx
        return (xa, xb, ya, yb)
    
    # Precompute for each direction and each point, which other points are on the line
    # For direction v=(1, m) where m=(ma, mb):
    #   Line through x=(xa,xb,ya,yb): points are (xa + t, xb, ya + t*ma, yb + t*mb) for t in F_49
    #   Wait, x-coordinate: x + t*1 = x + t, so xa + t_a, xb + t_b where t = (t_a, t_b)
    #   y-coordinate: y + t*m
    #
    # For direction v=(0, 1):
    #   Line through x: (xa, xb, ya + t_a, yb + t_b) for t in F_49
    #   x stays constant, y varies by all field elements
    
    # Let's precompute lines for each point and each direction
    # lines[point_idx][dir_idx] = list of other point indices on that line
    
    # Number of directions: 49 (for v=(1,m)) + 1 (for v=(0,1)) = 50
    num_dirs = q + 1
    
    # For each point, for each direction, compute the set of other points on the line
    # This is expensive: 2401 * 50 * 49 operations = ~5.9 million, should be fast
    
    # Let's use a different approach: for each direction, group points into lines
    # Then for each point on a line, the line's other points are the complement
    
    # Actually, let's think about the Nikodym condition more carefully:
    # For every point x, there exists a direction v such that all other points on line x+tv are in N.
    # This means: for point x, we need at least one direction where all 48 other points on that line are in N.
    
    # Strategy: Start with all points in N, then try to remove points.
    # A point can be removed if for every other point y that shares a line with this point in some direction,
    # y still has some other direction where all other points are in N.
    
    # Better: For each point x, track which directions "cover" x (i.e., all other points on that line are in N).
    # x is "safe" to remove if removing it doesn't break coverage for any other point.
    
    # Let's precompute: for each point x and direction d, what are the other 48 points on line x+td?
    
    # For v = (1, m) where m = (ma, mb):
    # Point on line: x + t*v = (xa + t_a, xb + t_b, ya + t_a*ma + t_b*mb*alpha_sq, yb + t_a*mb + t_b*ma)
    # Wait, let me redo: y + t*m where y=(ya,yb), m=(ma,mb)
    # (ya + b*alpha) * (ma + mb*alpha) = ya*ma + ya*mb*alpha + b*ma*alpha + b*mb*alpha^2
    # = (ya*ma + b*mb*alpha_sq) + (ya*mb + b*ma)*alpha
    # So y + t*m = (ya + t_a*ma + t_b*mb*alpha_sq, yb + t_a*mb + t_b*ma) mod p
    # And x + t = (xa + t_a, xb + t_b) mod p
    
    # For v = (0, 1):
    # x + t*(0,1) = (xa, xb, ya + t_a, yb + t_b)
    
    # Let's precompute the line membership
    # lines[point_idx][dir_idx] = frozenset of other point indices
    
    # This uses too much memory. Let's compute on the fly or use a different approach.
    
    # Alternative: For each direction, partition points into lines.
    # For direction v=(1,m): lines are parameterized by (y - x*m) or similar
    # Actually, for v=(1,m), two points (x1,y1) and (x2,y2) are on the same line iff y1 - x1*m = y2 - x2*m
    # So the line is determined by c = y - x*m
    # For v=(0,1), lines are vertical: determined by x
    
    # Let's precompute for each direction the line ID of each point
    # line_id[point_idx][dir_idx] = integer ID of the line
    
    # For v=(1,m) where m=(ma,mb):
    # c = y - x*m
    # c_a = ya - xa*ma - xb*mb*alpha_sq mod p
    # c_b = yb - xa*mb - xb*ma mod p
    # Line ID = (c_a, c_b) -> index = c_a*p + c_b
    
    # For v=(0,1):
    # Line ID = (xa, xb) -> index = xa*p + xb
    
    # Let's build this
    # dir_to_line_id[point_idx][dir_idx] = line_id
    
    # And for each direction and line, the set of points on that line
    # dir_lines[dir_idx][line_id] = list of point indices
    
    # Memory: 50 * 49 * 49 = 120050 entries for dir_lines, each with ~49 points on average
    # Actually for v=(1,m): 49 lines, each with 49 points
    # For v=(0,1): 49 lines, each with 49 points
    # Total: 50 * 49 * 49 = 120050 point-direction-line memberships
    
    # Let's build it
    dir_lines = []  # dir_lines[d][l] = list of point indices
    
    # Directions 0..48: v = (1, m) where m = (ma, mb), index = ma*p + mb
    for ma in range(p):
        for mb in range(p):
            d = ma * p + mb
            lines = {}
            for xa in range(p):
                for xb in range(p):
                    for ya in range(p):
                        for yb in range(p):
                            # c = y - x*m
                            ca = (ya - xa * ma - xb * mb * alpha_sq) % p
                            cb = (yb - xa * mb - xb * ma) % p
                            lid = ca * p + cb
                            if lid not in lines:
                                lines[lid] = []
                            lines[lid].append(point_index(xa, xb, ya, yb))
            dir_lines.append(lines)
    
    # Direction 49: v = (0, 1)
    lines = {}
    for xa in range(p):
        for xb in range(p):
            lid = xa * p + xb
            lines[lid] = []
            for ya in range(p):
                for yb in range(p):
                    lines[lid].append(point_index(xa, xb, ya, yb))
    dir_lines.append(lines)
    
    # Now, for each point and direction, we can quickly find the other points on the line
    # Let's build: point_line_members[point_idx][dir_idx] = list of other point indices on the same line
    
    # Actually, let's build a lookup: for each point, for each direction, the line ID
    point_line_id = np.zeros((q * q, num_dirs), dtype=np.int32)
    
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    pidx = point_index(xa, xb, ya, yb)
                    for ma in range(p):
                        for mb in range(p):
                            d = ma * p + mb
                            ca = (ya - xa * ma - xb * mb * alpha_sq) % p
                            cb = (yb - xa * mb - xb * ma) % p
                            point_line_id[pidx, d] = ca * p + cb
                    point_line_id[pidx, q] = xa * p + xb  # vertical direction
    
    # Now build: for each point and direction, the set of other points on the line
    # other_points[pidx][d] = set of point indices (excluding pidx)
    
    # This is 2401 * 50 * 48 entries... let's use a list of lists
    other_points = [[None] * num_dirs for _ in range(q * q)]
    
    for d in range(num_dirs):
        for lid, plist in dir_lines[d].items():
            for pidx in plist:
                others = [p for p in plist if p != pidx]
                other_points[pidx][d] = others
    
    # Now we have the structure. Let's implement the local search.
    
    # State: in_N[pidx] = True/False
    # coverage[pidx] = number of directions d where all other points on line pidx+d are in N
    
    in_N = np.ones(q * q, dtype=bool)
    
    # Compute initial coverage
    def compute_coverage(in_N):
        coverage = np.zeros(q * q, dtype=np.int32)
        for pidx in range(q * q):
            for d in range(num_dirs):
                all_in = True
                for opidx in other_points[pidx][d]:
                    if not in_N[opidx]:
                        all_in = False
                        break
                if all_in:
                    coverage[pidx] += 1
        return coverage
    
    coverage = compute_coverage(in_N)
    
    # A point can be removed if:
    # 1. It is currently in N
    # 2. After removal, every other point still has coverage >= 1
    #    i.e., for every point y that shares a line with this point in direction d,
    #    y must have some direction d' (possibly d) where all other points are in N
    #    But if we remove x, then for y on the same line as x in direction d,
    #    the line y+d now has one fewer point (x is removed), so it's still all in N
    #    Wait no: the condition is that ALL OTHER points on the line are in N.
    #    If x is removed from N, then for y on the same line, x is no longer in N,
    #    so the line y+d is NOT fully covered (since x is on it and not in N).
    #    So y loses coverage in direction d.
    #    y can still be covered by other directions.
    
    # So: removing x is safe iff for every y such that x is on the line y+d for some d,
    #     y has coverage >= 2 (so it still has at least one covering direction after losing d)
    #     OR y has coverage == 1 but the only covering direction for y is NOT affected by removing x
    #     Actually, if x is on line y+d, then removing x from N means line y+d is no longer fully in N,
    #     so y loses coverage in direction d.
    #     So y needs coverage >= 2, OR coverage == 1 and the covering direction is not d.
    
    # Let's track: for each point y, which directions cover it?
    # cover_dirs[y] = set of directions d where line y+d is fully in N
    
    def compute_cover_dirs(in_N):
        cover_dirs = [set() for _ in range(q * q)]
        for pidx in range(q * q):
            for d in range(num_dirs):
                all_in = True
                for opidx in other_points[pidx][d]:
                    if not in_N[opidx]:
                        all_in = False
                        break
                if all_in:
                    cover_dirs[pidx].add(d)
        return cover_dirs
    
    cover_dirs = compute_cover_dirs(in_N)
    
    # For each point x, find all (y, d) pairs where x is on line y+d
    # affected[x] = list of (y, d) where removing x from N would cause y to lose coverage in direction d
    
    affected = [[] for _ in range(q * q)]
    for d in range(num_dirs):
        for lid, plist in dir_lines[d].items():
            for pidx in plist:
                for opidx in plist:
                    if pidx != opidx:
                        affected[pidx].append((opidx, d))
    
    # Now implement greedy removal
    # Try to remove points one by one
    # For efficiency, we can try all points and pick the one that allows the most future removals
    # But that's expensive. Let's just try each point in order and remove if safe.
    
    # Actually, let's be smarter: try to remove points that are "least critical"
    # A point x is critical if many other points rely on x for coverage
    
    # Let's just do a simple greedy: iterate through points, try to remove each one
    # Repeat until no more removals possible
    
    removed_any = True
    while removed_any and time.time() < deadline - 10:
        removed_any = False
        for pidx in range(q * q):
            if not in_N[pidx]:
                continue
            # Check if removing pidx is safe
            safe = True
            for (y, d) in affected[pidx]:
                if not in_N[y]:
                    continue
                # y loses direction d if it was covered by d
                if d in cover_dirs[y]:
                    # y needs another covering direction
                    if len(cover_dirs[y]) <= 1:
                        safe = False
                        break
            if safe:
                # Remove pidx
                in_N[pidx] = False
                removed_any = True
                # Update cover_dirs for affected points
                for (y, d) in affected[pidx]:
                    if d in cover_dirs[y] and in_N[y]:
                        # Check if y still has this direction covered
                        # Since we removed pidx, line y+d is no longer fully in N
                        cover_dirs[y].discard(d)
                # Recompute coverage for all points? No, just update affected ones
                # But other points might gain coverage? No, removing points only decreases coverage.
                # Wait, actually removing a point from N can only decrease coverage, never increase it.
                # Because a line is covered iff all its other points are in N.
                # Removing a point from N can only make lines less covered.
                # So we only need to update points that lose coverage.
    
    # Now try simultaneous removals of 2 points
    # This is more expensive but might escape local minima
    # For pairs (x, y), check if removing both is safe
    
    # This is O(n^2) per iteration which is 2401^2 = ~5.7 million, times the check cost
    # Let's limit this to a few iterations
    
    if time.time() < deadline - 30:
        # Try pair removals
        pair_removed = True
        while pair_removed and time.time() < deadline - 15:
            pair_removed = False
            # Try random pairs or systematic pairs
            # Systematic is better for reproducibility
            for i in range(q * q):
                if not in_N[i]:
                    continue
                for j in range(i + 1, q * q):
                    if not in_N[j]:
                        continue
                    # Check if removing both i and j is safe
                    safe = True
                    # For each point affected by i or j, check coverage
                    # Affected points: those on lines with i or j
                    affected_set = set()
                    for (y, d) in affected[i]:
                        if in_N[y]:
                            affected_set.add((y, d))
                    for (y, d) in affected[j]:
                        if in_N[y]:
                            affected_set.add((y, d))
                    for (y, d) in affected_set:
                        if not in_N[y]:
                            continue
                        # y loses direction d if it was covered
                        if d in cover_dirs[y]:
                            # Check if y has another covering direction that is not affected
                            # Actually, we need to check: after removing i and j, is line y+d still fully in N?
                            # It's not if i or j is on line y+d and was in N
                            # Since i and j are being removed, if either is on line y+d, the line is not fully in N
                            # So y loses direction d
                            if len(cover_dirs[y]) <= 1:
                                # y might have other covering directions
                                # But we need to check if any other direction is still covered
                                # This is getting complex. Let's just check: does y have any direction d' 
                                # where line y+d' is fully in N after removing i and j?
                                found = False
                                for d2 in cover_dirs[y]:
                                    # Check if line y+d2 is still fully in N after removing i and j
                                    ok = True
                                    for opidx in other_points[y][d2]:
                                        if opidx == i or opidx == j:
                                            # This point is being removed, so it's not in N
                                            # But wait, if opidx is i or j, and we're removing them,
                                            # then the line is not fully in N
                                            ok = False
                                            break
                                        if not in_N[opidx]:
                                            ok = False
                                            break
                                    if ok:
                                        found = True
                                        break
                                if not found:
                                    safe = False
                                    break
                    if safe:
                        # Remove both
                        in_N[i] = False
                        in_N[j] = False
                        pair_removed = True
                        # Update cover_dirs
                        for (y, d) in affected[i]:
                            if d in cover_dirs[y] and in_N[y]:
                                cover_dirs[y].discard(d)
                        for (y, d) in affected[j]:
                            if d in cover_dirs[y] and in_N[y]:
                                cover_dirs[y].discard(d)
                        break  # Restart the loop after a removal
    
    # Convert to output format
    points = []
    for pidx in range(q * q):
        if in_N[pidx]:
            xa, xb, ya, yb = index_to_point(pidx)
            points.append([[xa, xb], [ya, yb]])
    
    if not points:
        # Should not happen, but just in case
        points = [[[0, 0], [0, 0]]]
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
