# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 using greedy set cover + local search.
    
    For each point x, we assign a line through x. N is the union of all assigned lines.
    A point y is in the complement (not in N) iff no assigned line contains y.
    We minimize |N| to maximize the complement.
    """
    start_time = time.time()
    deadline = start_time + 600  # safety margin
    
    p = 7
    k = 3  # alpha^2 = 3 (quadratic non-residue mod 7: 3^3 = 27 = 6 = -1 mod 7)
    
    # Precompute field operations
    # F_49 = F_7[alpha], alpha^2 = 3
    # Element: (a, b) representing a + b*alpha
    
    def add_f(a1, b1, a2, b2):
        return ((a1 + a2) % p, (b1 + b2) % p)
    
    def sub_f(a1, b1, a2, b2):
        return ((a1 - a2) % p, (b1 - b2) % p)
    
    def mul_f(a1, b1, a2, b2):
        # (a1 + b1*alpha)(a2 + b2*alpha) = a1*a2 + b1*b2*alpha^2 + (a1*b2 + a2*b1)*alpha
        # = a1*a2 + b1*b2*3 + (a1*b2 + a2*b1)*alpha
        real = (a1 * a2 + b1 * b2 * k) % p
        imag = (a1 * b2 + a2 * b1) % p
        return (real, imag)
    
    def neg_f(a, b):
        return ((-a) % p, (-b) % p)
    
    # Enumerate all points: index = xa*49 + xb*7 + ya*1 + yb? Let's use a flat index
    # Point (xa, xb, ya, yb) -> index = ((xa*p + xb)*p + ya)*p + yb
    q = p * p  # 49
    total_points = q * q  # 2401
    
    def point_to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
    
    def idx_to_point(idx):
        yb = idx % p
        idx //= p
        ya = idx % p
        idx //= p
        xb = idx % p
        idx //= p
        xa = idx % p
        return (xa, xb, ya, yb)
    
    # Enumerate all lines. A line is defined by a point (x0, y0) and a direction v.
    # Direction v is either (1, m) for m in F_49, or (0, 1) (vertical).
    # There are q + 1 = 50 directions.
    # For each direction, there are q parallel lines (one through each "base" point).
    # Total lines: (q+1) * q = 50 * 49 = 2450
    
    # Precompute all lines: for each line, store the set of point indices on it.
    # Direction types:
    #   Type 0: vertical, v = (0, 1). Line: x = x0, y varies.
    #   Type 1+: slope m = (ma, mb). v = (1, m). Line: y = y0 + m*(x - x0).
    
    lines = []  # list of sets of point indices
    line_id_for_point = defaultdict(list)  # point_idx -> list of line_ids
    
    # Vertical lines: x = (xa, xb), y varies
    for xa in range(p):
        for xb in range(p):
            pts = []
            for ya in range(p):
                for yb in range(p):
                    pts.append(point_to_idx(xa, xb, ya, yb))
            lid = len(lines)
            lines.append(frozenset(pts))
            for pt in pts:
                line_id_for_point[pt].append(lid)
    
    # Slanted lines: direction (1, m) where m = (ma, mb)
    for ma in range(p):
        for mb in range(p):
            # For each intercept (y0 = ya0 + yb0*alpha), line: y = y0 + m*(x - x0)
            # We can parametrize by the y-value at x=0, i.e., y0.
            for ya0 in range(p):
                for yb0 in range(p):
                    pts = []
                    for xa in range(p):
                        for xb in range(p):
                            # x = xa + xb*alpha
                            # y = y0 + m * x
                            # m*x = (ma+mb*alpha)(xa+xb*alpha) = (ma*xa + mb*xb*k, ma*xb + mb*xa)
                            mx_real = (ma * xa + mb * xb * k) % p
                            mx_imag = (ma * xb + mb * xa) % p
                            ya = (ya0 + mx_real) % p
                            yb = (yb0 + mx_imag) % p
                            pts.append(point_to_idx(xa, xb, ya, yb))
                    lid = len(lines)
                    lines.append(frozenset(pts))
                    for pt in pts:
                        line_id_for_point[pt].append(lid)
    
    total_lines = len(lines)
    
    # For each point, the candidate lines are those passing through it.
    # A line L through point x "satisfies" x if all points on L other than x are in N.
    # N is the union of all selected lines.
    
    # Greedy set cover approach:
    # We need to assign one line to each point.
    # N = union of assigned lines.
    # We want to minimize |N|.
    
    # Strategy: 
    # 1. Start with an initial assignment (e.g., for each point, pick the line that minimizes new points added).
    # 2. Local search: try reassigning a point to a different line if it reduces |N|.
    
    # For efficiency, track:
    # - For each line, how many points are assigned to it.
    # - N is the set of points that are on at least one assigned line.
    
    # Initial assignment: for each point, assign the line through it with the fewest "extra" points
    # (i.e., the line that adds the fewest new points to N). But this is greedy and order-dependent.
    
    # Better: randomized greedy with multiple restarts.
    
    def greedy_assign(seed):
        rng = random.Random(seed)
        # Shuffle points
        points_order = list(range(total_points))
        rng.shuffle(points_order)
        
        # assignment[point] = line_id
        assignment = [-1] * total_points
        # line_count[line_id] = number of points assigned to this line
        line_count = [0] * total_lines
        # N_set: set of points in N
        N_set = set()
        
        for pt in points_order:
            best_line = -1
            best_new = total_points + 1
            for lid in line_id_for_point[pt]:
                # New points added by assigning pt to lid
                new_pts = 0
                for other_pt in lines[lid]:
                    if other_pt == pt:
                        continue
                    if other_pt not in N_set:
                        new_pts += 1
                if new_pts < best_new:
                    best_new = new_pts
                    best_line = lid
            assignment[pt] = best_line
            line_count[best_line] += 1
            for other_pt in lines[best_line]:
                if other_pt != pt:
                    N_set.add(other_pt)
            N_set.add(pt)  # pt is in N since it's on its assigned line
        
        return assignment, line_count, N_set
    
    # Local search: try reassigning a point to a different line if it reduces |N|.
    def local_search(assignment, line_count, N_set, max_iterations=10000, time_limit=30):
        start_ls = time.time()
        improved = True
        while improved and (time.time() - start_ls) < time_limit:
            improved = False
            # Randomly sample some points to try reassigning
            for _ in range(200):
                pt = random.randint(0, total_points - 1)
                old_line = assignment[pt]
                # Try other lines through pt
                candidates = line_id_for_point[pt]
                random.shuffle(candidates)
                for new_line in candidates:
                    if new_line == old_line:
                        continue
                    # Compute change in N
                    # Remove pt from old_line's contribution
                    # Add pt to new_line's contribution
                    
                    # Points that will be removed from N: those on old_line that are NOT on any other assigned line
                    # Points that will be added to N: those on new_line that are NOT on any other assigned line
                    
                    # This is expensive to compute exactly. Approximate:
                    # For each point on old_line (except pt), check if it's still covered by another line.
                    # For each point on new_line (except pt), check if it's newly covered.
                    
                    # Quick check: if new_line == old_line, skip.
                    # Compute new N size
                    new_N_size = len(N_set)
                    
                    # Points to potentially remove: on old_line, not pt, and not on any other line with count > 0
                    to_remove = []
                    for other_pt in lines[old_line]:
                        if other_pt == pt:
                            continue
                        # Is other_pt covered by any other line?
                        covered = False
                        for other_lid in line_id_for_point[other_pt]:
                            if other_lid != old_line and line_count[other_lid] > 0:
                                covered = True
                                break
                        if not covered:
                            to_remove.append(other_pt)
                    
                    # Points to potentially add: on new_line, not pt, and not already in N
                    to_add = []
                    for other_pt in lines[new_line]:
                        if other_pt == pt:
                            continue
                        if other_pt not in N_set:
                            to_add.append(other_pt)
                    
                    delta = len(to_add) - len(to_remove)
                    if delta < 0:
                        # Apply the change
                        for r in to_remove:
                            N_set.discard(r)
                        for a in to_add:
                            N_set.add(a)
                        line_count[old_line] -= 1
                        line_count[new_line] += 1
                        assignment[pt] = new_line
                        improved = True
                        break  # Move to next point
                    
        return assignment, line_count, N_set
    
    # Run multiple restarts
    best_N_size = total_points
    best_result = None
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        assignment, line_count, N_set = greedy_assign(r)
        N_size = len(N_set)
        
        if N_size < best_N_size:
            best_N_size = N_size
            best_result = (assignment, line_count, N_set.copy())
        
        # Local search
        assignment, line_count, N_set = local_search(assignment, line_count, N_set, max_iterations=5000, time_limit=10)
        N_size = len(N_set)
        
        if N_size < best_N_size:
            best_N_size = N_size
            best_result = (assignment, line_count, N_set.copy())
    
    # Extract the Nikodym set N from the best result
    if best_result is None:
        # Fallback: return all points
        points = []
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)
    
    _, _, N_set = best_result
    
    # Convert N_set to array of points
    points = []
    for pt_idx in sorted(N_set):
        xa, xb, ya, yb = idx_to_point(pt_idx)
        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
