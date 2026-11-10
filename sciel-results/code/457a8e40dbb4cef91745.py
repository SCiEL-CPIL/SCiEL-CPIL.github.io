# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = 49, q = p^2, p = 7.
    A set N is a Nikodym set if for every point x in F_q^2 there exists a direction
    v (either (1, m) for some m in F_q, or (0,1)) such that the punctured line
    {x + t*v : t in F_q, t != 0} is fully contained in N.
    We want to maximize the complement size 2401 - |N|.

    Strategy: Start with the full space as N (complement empty). Then greedily add
    points to the complement S (removing from N) while maintaining the Nikodym property.
    A point x can be added to S if, for every direction v, there is at least one point
    on the line L(x, v) other than x that is NOT in S (i.e., still in N).
    
    Actually, the condition is: for each x in F_q^2, there must exist a direction v such
    that ALL points on L(x,v) except x are in N. So x can be in S only if for every
    direction v, at least one point on L(x,v)\{x} is in S.
    
    Wait, that's the opposite. Let me re-read.
    
    "every point x of F_49^2 (in N or not) must lie on a line {x + t*v : t in F_49},
    with v = (1, m) for some m in F_49 or v = (0, 1), whose points other than x all
    belong to N."
    
    So for each x, there must be SOME direction v such that all points on the line
    through x in direction v, excluding x, are in N.
    
    So if x is in S (not in N), we need a direction v where all other points on the
    line are in N. If x is in N, same condition.
    
    So the constraint is: for each x, there exists v such that L(x,v) \ {x} ⊆ N.
    Equivalently, L(x,v) \ {x} ∩ S = ∅ for some v.
    
    So x can be in S only if there is no direction v where L(x,v)\{x} ⊆ N... wait no.
    x can be in S as long as there EXISTS a v with L(x,v)\{x} ⊆ N.
    x CANNOT be in S if for ALL v, L(x,v)\{x} ∩ S ≠ ∅ (i.e., every line through x
    in a valid direction has at least one other point in S).
    
    So the greedy approach: try to add points to S one by one. A point x can be added
    to S if after adding, for every point y (including x itself), the condition still
    holds. But that's expensive to check globally.
    
    Actually, adding x to S can only violate the condition for points y that share a
    line with x in some direction. Specifically, for a point y to still satisfy the
    condition, y must have at least one direction v where L(y,v)\{y} ⊆ N. Adding x
    to S removes x from N, so it can only break the condition for y if x was the ONLY
    point keeping some line through y valid... no, x being in S means x is NOT in N,
    so if x is on L(y,v)\{y}, then L(y,v)\{y} is no longer fully in N (since x ∉ N).
    
    So adding x to S breaks the validity of direction v for point y if x ∈ L(y,v)\{y}
    and v was the ONLY direction that worked for y, or more precisely, if for y, the
    set of valid directions becomes empty after x is removed from N.
    
    This is complex. Let me use a different approach: use the known construction from
    the current program but try to improve it by using a smarter selection of the
    excluded set S.
    
    The current program uses:
    - Unital U = {(x,y) : Trace(x) + Norm(y) = 0}
    - Y = {(u^2 mod p, v) : u,v in F_p}
    - S = U \ B where B = U ∩ ({all x} × Y)
    
    Size of S: |U| - |B|. |U| = q + 1 = 50? No, for F_q^2 with q=49, a Hermitian unital
    has q + 1 points? Actually a Hermitian curve in F_q^2 has q + 1 points. Wait, the
    equation Trace(x) + Norm(y) = 0... let me think.
    
    Actually, let me just try to improve the heuristic for selecting S. The key insight
    is that we want S to be as large as possible while maintaining the Nikodym property.
    
    Let me try a greedy algorithm with a deadline guard.
    """
    start_time = time.time()
    deadline = 600.0
    safety_margin = 30.0
    
    p = 7
    q = p * p  # 49
    
    # Field F_q = F_p[alpha] where alpha^2 = 3 (smallest QR non-residue mod 7)
    # Check: 3^3 = 27 = 6 mod 7 = -1, so 3 is a non-residue. Good.
    k = 3
    
    def field_add(a, b):
        """a, b are tuples (a0, a1) representing a0 + a1*alpha"""
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
    
    def field_mul(a, b):
        # (a0 + a1*alpha)(b0 + b1*alpha) = a0*b0 + a1*b1*alpha^2 + (a0*b1 + a1*b0)*alpha
        # = a0*b0 + a1*b1*k + (a0*b1 + a1*b0)*alpha
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    
    def field_trace(a):
        # Trace(a + b*alpha) = (a + b*alpha) + (a + b*alpha)^p = 2a for p odd
        return (2 * a[0]) % p
    
    def field_norm(a):
        # Norm(a + b*alpha) = (a+b*alpha)(a+b*alpha)^p = (a+b*alpha)(a-b*alpha) = a^2 - b^2*k
        return (a[0]*a[0] - a[1]*a[1]*k) % p
    
    # Enumerate all points in F_q^2
    # Point is (x, y) where x = (xa, xb), y = (ya, yb)
    # Total 49*49 = 2401 points
    
    # Directions: (1, m) for m in F_q (49 directions) and (0, 1) (1 direction) = 50 directions
    # For each direction v, and each point x, the line is {x + t*v : t in F_q}
    # This line has 49 points.
    
    # Let's precompute for each point x and each direction v, the set of points on the line
    # L(x, v) = {x + t*v : t in F_q, t != 0} (excluding x itself)
    
    # Represent points as indices 0..2400
    # Point index = ((xa*7 + xb)*7 + ya)*7 + yb
    
    def point_to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
    
    def idx_to_point(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p*p)) % p
        xa = (idx // (p*p*p)) % p
        return (xa, xb, ya, yb)
    
    n_points = q * q  # 2401
    
    # Precompute lines: for each point x and direction v, store the list of other points
    # on the line through x in direction v
    # There are 50 directions. For each direction, the lines partition the space.
    
    # Directions as (dx, dy) where dx, dy are in F_q
    # v = (1, m) means dx = (1,0), dy = (m0, m1)
    # v = (0, 1) means dx = (0,0), dy = (0,1)
    
    directions = []
    # (1, m) for all m in F_q
    for m0 in range(p):
        for m1 in range(p):
            directions.append(((1, 0), (m0, m1)))
    # (0, 1)
    directions.append(((0, 0), (0, 1)))
    
    n_dirs = len(directions)  # 50
    
    # For each point x and direction v, compute the line points (excluding x)
    # This is expensive: 2401 * 50 * 49 = ~5.9 million operations, should be fine
    
    # Actually, for a given direction v, the lines are parallel classes. Each line has 49 points.
    # For point x on line L, L(x,v)\{x} has 48 points.
    
    # Let's build: lines[x][v] = list of 48 point indices on the line through x in direction v, excluding x
    
    # For efficiency, for each direction v, we can compute the "line index" of each point,
    # and then group points by line.
    
    # For direction v = (dx, dy):
    # If dx != 0, we can write x = x0 + t*dx, so t = (x - x0)/dx. The line is determined by
    # the "offset" in the dy direction. Specifically, if dx = (1,0), then the line is
    # {x0 + t*(1, m) : t in F_q} where x0 is the point with x-coordinate 0... 
    # Actually, for v = (1, m), the line through x is {x + t*(1,m) : t in F_q}.
    # Two points x, x' are on the same line iff x' - x = t*(1,m) for some t,
    # i.e., x'_x - x_x = t and x'_y - x_y = t*m, so x'_y - x_y = m*(x'_x - x_x).
    # So the line is determined by x_y - m*x_x (in some sense).
    
    # Let me just precompute the line membership for each direction.
    
    # For each direction v, create an array line_id of length 2401, where line_id[x] is
    # the ID of the line through x in direction v.
    # Then for each line, collect the 49 points.
    
    # For v = (1, m): line through x has equation y = m*x + c where c = y - m*x.
    # c is in F_q, so there are 49 lines per direction.
    
    # For v = (0, 1): line through x is {x + t*(0,1) : t} = {x0 + t, y0}... wait,
    # x + t*(0,1) = (x_x, x_y + t). So the line is vertical: x-coordinate fixed.
    # There are 49 vertical lines.
    
    # Let's precompute for each direction the mapping from point to line ID,
    # and for each line the list of points.
    
    # This is getting complex. Let me just do the naive precomputation.
    
    # lines[x][v] = frozenset of point indices on line through x in direction v, excluding x
    
    # Memory: 2401 * 50 * 48 integers. That's 2401*50*48 = 5,762,400 integers.
    # Each integer is 4-8 bytes, so ~23-46 MB. Should be fine.
    
    # Actually, let's store as lists for easier manipulation.
    
    # For each direction v (indexed 0..49), compute line assignments
    # line_assign[v][x] = line_id
    # line_points[v][line_id] = list of points on that line
    
    line_assign = []  # list of arrays, one per direction
    line_points = []  # list of lists of lists, one per direction
    
    for v_idx, (dx, dy) in enumerate(directions):
        # For each point, compute which line it's on
        # For v = (1, m): line ID determined by y - m*x
        # For v = (0, 1): line ID determined by x
        
        assign = np.zeros(n_points, dtype=np.int32)
        points_by_line = {}
        
        for idx in range(n_points):
            xa, xb, ya, yb = idx_to_point(idx)
            x = (xa, xb)
            y = (ya, yb)
            
            if v_idx < 49:
                # v = (1, m)
                m = dy
                # c = y - m*x
                mx = field_mul(m, x)
                c = field_sub(y, mx)
                line_id = c[0] * p + c[1]
            else:
                # v = (0, 1)
                line_id = x[0] * p + x[1]
            
            assign[idx] = line_id
            if line_id not in points_by_line:
                points_by_line[line_id] = []
            points_by_line[line_id].append(idx)
        
        line_assign.append(assign)
        
        # Convert to list of lists indexed by line_id
        max_line_id = max(points_by_line.keys()) if points_by_line else 0
        lp = [[] for _ in range(max_line_id + 1)]
        for lid, pts in points_by_line.items():
            lp[lid] = pts
        line_points.append(lp)
    
    # Now, for each point x and direction v, the other points on the line are
    # line_points[v][line_assign[v][x]] minus x.
    
    # Greedy algorithm: start with S = empty set. Try to add points to S in some order.
    # A point x can be added to S if, after adding, for every point y, there exists
    # a direction v such that L(y,v)\{y} ⊆ N (i.e., L(y,v)\{y} ∩ S = ∅).
    
    # This is expensive to check globally. Instead, note that adding x to S can only
    # affect the validity of directions for points y that are on a line with x.
    # Specifically, for a point y and direction v, if x ∈ L(y,v)\{y}, then adding x
    # to S means L(y,v)\{y} is no longer fully in N.
    
    # So for each y, let valid_dirs(y) be the set of directions v such that L(y,v)\{y} ⊆ N.
    # Initially (S empty), valid_dirs(y) = all 50 directions for all y.
    # When we add x to S, for each y and each v where x ∈ L(y,v)\{y}, we remove v
    # from valid_dirs(y). If valid_dirs(y) becomes empty, the Nikodym property is violated.
    
    # So the constraint is: when adding x to S, for no y should valid_dirs(y) become empty.
    
    # We can maintain valid_dirs as a bitset or count.
    
    # Let's maintain: for each point y, valid_count[y] = number of valid directions for y.
    # And for each point y and direction v, a flag is_valid[y][v] = whether L(y,v)\{y} ⊆ N.
    
    # When we add x to S:
    # For each direction v, find all y such that x ∈ L(y,v)\{y}.
    # For those y, if is_valid[y][v] was True, set it to False and decrement valid_count[y].
    # If any valid_count[y] becomes 0, we cannot add x (or we need to revert).
    
    # How to find all y such that x ∈ L(y,v)\{y}?
    # x ∈ L(y,v)\{y} means x = y + t*v for some t != 0, i.e., y = x - t*v for t != 0.
    # So y is on the line through x in direction v, and y != x.
    # So y ∈ L(x,v)\{x}.
    
    # So when adding x to S, for each direction v, for each y in L(x,v)\{x}:
    #   if is_valid[y][v]: set to False, valid_count[y] -= 1
    #   if valid_count[y] == 0: violation!
    
    # This is symmetric: the pairs (y, v) affected are exactly (y, v) where y ∈ L(x,v)\{x}.
    
    # Precompute: for each point x and direction v, the list of points L(x,v)\{x}.
    # We already have line_points. L(x,v)\{x} = line_points[v][line_assign[v][x]] - {x}.
    
    # Let's precompute other_points[x][v] = list of points on line through x in direction v, excluding x.
    
    # Actually, we can compute this on the fly from line_points and line_assign.
    
    # Initialize
    valid_count = np.full(n_points, n_dirs, dtype=np.int32)
    is_valid = np.ones((n_points, n_dirs), dtype=bool)
    
    # S is the set of excluded points
    in_S = np.zeros(n_points, dtype=bool)
    
    # Order to try adding points: let's try all points in order, or use a heuristic.
    # A good heuristic: try to add points that are "easy" to add, i.e., points that
    # affect few (y, v) pairs. Or just try in order.
    
    # Let's try a greedy approach: iterate through points in order, try to add each one.
    # If adding would cause a violation, skip it.
    
    # To check if adding x is safe:
    # For each direction v, for each y in L(x,v)\{x}:
    #   if is_valid[y][v]:
    #     tentatively set is_valid[y][v] = False, valid_count[y] -= 1
    #     if valid_count[y] == 0: violation
    # If no violation, commit the changes.
    
    # This is O(50 * 48) = 2400 operations per point, times 2401 points = ~5.8 million ops.
    # Should be fast.
    
    # But we need to be careful: when we add x, we modify is_valid and valid_count.
    # Later, when trying to add another point x', the state is updated.
    
    # Let's implement this.
    
    for x_idx in range(n_points):
        if time.time() > deadline - safety_margin:
            break
        
        if in_S[x_idx]:
            continue
        
        # Check if adding x_idx to S is safe
        safe = True
        changes = []  # list of (y, v) that would be changed
        
        for v in range(n_dirs):
            lid = line_assign[v][x_idx]
            line_pts = line_points[v][lid]
            for y in line_pts:
                if y == x_idx:
                    continue
                if is_valid[y, v]:
                    changes.append((y, v))
                    # Check if this would make valid_count[y] zero
                    if valid_count[y] == 1:
                        safe = False
                        break
            if not safe:
                break
        
        if safe:
            # Commit the changes
            in_S[x_idx] = True
            for y, v in changes:
                is_valid[y, v] = False
                valid_count[y] -= 1
    
    # Build the result: N = all points not in S
    points = []
    for idx in range(n_points):
        if not in_S[idx]:
            xa, xb, ya, yb = idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
    
    result = np.array(points, dtype=np.int64)
    
    # Verify: result should have shape (k, 2, 2) with k >= 1
    if len(result) == 0:
        # Fallback: return all points
        all_points = []
        for idx in range(n_points):
            xa, xb, ya, yb = idx_to_point(idx)
            all_points.append([[xa, xb], [ya, yb]])
        result = np.array(all_points, dtype=np.int64)
    
    return result


# EVOLVE-BLOCK-END
