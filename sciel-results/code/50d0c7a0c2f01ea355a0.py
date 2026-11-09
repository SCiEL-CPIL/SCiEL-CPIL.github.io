# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a greedy approach to maximize the
    complement size (points excluded from N).
    
    Field: F_49 = F_7[alpha] with alpha^2 = 3.
    A point is [[x_a, x_b], [y_a, y_b]] representing (x_a + x_b*alpha, y_a + y_b*alpha).
    
    A Nikodym set N must satisfy: for every point x in F_49^2, there exists a
    direction v (either (1,m) for some m in F_49, or (0,1)) such that all other
    points on the line through x in direction v belong to N.
    
    Equivalently, the complement S = F_49^2 \ N must be such that for every point x,
    there exists a direction v where NO OTHER point on the line {x + t*v : t in F_49}
    is in S (i.e., the punctured line contains no points of S except possibly x itself,
    but since x might be in S, we need the line minus {x} to have no S points... 
    wait, let me re-read the condition).
    
    Actually: "every point x of F_49^2 (in N or not) must lie on a line {x + t*v : t in F_49},
    with v = (1, m) for some m in F_49 or v = (0, 1), whose points other than x all belong to N."
    
    So for each point x, there must exist a direction v such that for all t != 0, 
    x + t*v is in N. Equivalently, for each x, there exists a direction v such that 
    the line through x in direction v contains no points of S other than possibly x itself.
    
    Wait: "points other than x all belong to N" means that if x is in S, then the other 
    48 points on that line must be in N. If x is in N, the other 48 points must be in N.
    So either way, the line through x in direction v must have at most one point from S, 
    and that point can be x itself.
    
    So S must be a set where no line (in the specified directions) contains more than 
    one point of S.
    
    The specified directions are: v = (1, m) for all m in F_49, plus v = (0, 1).
    That's 49 + 1 = 50 directions.
    
    So S must be a set with at most one point on each line in these 50 directions.
    We want to maximize |S|, i.e., make S as large as possible subject to this constraint.
    
    This is equivalent to finding a large set S where no two points are "collinear" 
    in any of the 50 directions.
    
    Two points p1, p2 are collinear in direction v if p2 - p1 = t*v for some t in F_49.
    
    Let me think about this differently. There are 2401 points and 50 directions.
    Each direction has 49 parallel lines, each with 49 points.
    Total line-point incidences: 50 * 49 * 49 = 50 * 2401.
    
    We want a set S with at most one point on each of the 50*49 = 2450 lines.
    So |S| <= 2450, but also |S| <= 2401 obviously.
    
    A greedy approach: start with S empty, add points one by one. A point can be added 
    to S if it doesn't create a line with more than one S-point in any of the 50 directions.
    Equivalently, a point p can be added if for each direction v, the line through p in 
    direction v doesn't already contain an S-point.
    
    Let me implement this greedy approach.
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    alpha_sq = 3  # alpha^2 = 3 mod 7
    
    deadline = time.time() + 580  # safety margin
    
    # Precompute field operations
    # Represent element as (a, b) = a + b*alpha
    # Addition: component-wise mod p
    # Multiplication: (a,b)*(c,d) = (ac - bd*alpha_sq, ad + bc) mod p
    
    def mul_field(a, b, c, d):
        """Multiply (a + b*alpha) * (c + d*alpha)"""
        ra = (a * c - b * d * alpha_sq) % p
        rb = (a * d + b * c) % p
        return ra, rb
    
    # Compute all 49 elements of F_49 as indices
    # Element index i: a = i // p, b = i % p
    field_elems = [(i // p, i % p) for i in range(q)]
    
    # Precompute multiplication table
    mul_table = np.zeros((q, q, 2), dtype=np.int64)
    for i in range(q):
        ai, bi = field_elems[i]
        for j in range(q):
            aj, bj = field_elems[j]
            ri, rj = mul_field(ai, bi, aj, bj)
            mul_table[i, j, 0] = ri
            mul_table[i, j, 1] = rj
    
    # Precompute addition table
    add_table = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        ai, bi = field_elems[i]
        for j in range(q):
            aj, bj = field_elems[j]
            add_table[i, j] = (ai + aj) * p + (bi + bj)
    
    # Precompute negation
    neg_table = np.zeros(q, dtype=np.int64)
    for i in range(q):
        ai, bi = field_elems[i]
        ni = ((p - ai) % p) * p + ((p - bi) % p)
        neg_table[i] = ni
    
    # Direction vectors: 49 of the form (1, m) and 1 of the form (0, 1)
    # Represent direction as (vx, vy) in F_49
    directions = []
    for m in range(q):
        directions.append((1, m))
    directions.append((0, 1))
    num_dirs = len(directions)  # 50
    
    # For each direction, precompute the line ID for each point
    # Line through point i in direction (vx, vy):
    # Points are i + t*(vx, vy) for t in F_49
    # We need to identify which line each point is on for each direction.
    
    # For direction (1, m): lines are parameterized by y-intercept c where y = x*m + c
    # Actually, let me think about this more carefully.
    
    # For direction (1, m), the line through (x0, y0) is:
    # (x0 + t, y0 + t*m) for t in F_49
    # Two points (x1, y1) and (x2, y2) are on the same line in direction (1,m) iff
    # y2 - y1 = (x2 - x1) * m, i.e., y2 - y1 - (x2-x1)*m = 0
    
    # For direction (0, 1), lines are vertical: x = const.
    
    # Let me precompute for each direction and each point, which line it's on.
    # Line ID for direction (1, m) through (x0, y0): the value c = y0 - x0*m
    # Two points are on the same line iff they have the same c.
    
    # For direction (0, 1), line ID is just x.
    
    # Actually, for the greedy algorithm, I need to check: can I add point i to S?
    # Point i can be added if for each direction d, the line through i in direction d
    # doesn't already contain an S-point.
    
    # Let me use a different representation. For each direction, I'll maintain a count
    # of S-points on each line. A point can be added if all 50 lines through it have count 0.
    
    # Precompute line IDs for each direction and each point.
    # line_id[dir][point] = line identifier
    
    line_id = np.zeros((num_dirs, total_points), dtype=np.int64)
    
    for d_idx, (vx, vy) in enumerate(directions):
        for pt_idx in range(total_points):
            xa = pt_idx // (p * p * p)
            xb = (pt_idx // (p * p)) % p
            ya = (pt_idx // p) % p
            yb = pt_idx % p
            
            # Point is (x, y) where x = (xa, xb), y = (ya, yb)
            if vx == 0:
                # Direction (0, 1): vertical line, line ID is x
                line_id[d_idx, pt_idx] = xa * p + xb
            else:
                # Direction (1, m) where m = (vx, vy) but vx=1
                # Wait, vx is always 1 for the first 49 directions.
                # m = (0, vy) since vx=1 means the element is 1*1 + vy*alpha? No.
                # Actually, the direction is (1, m) where m is an element of F_49.
                # So m = field_elems[vy]? No, vy is the index.
                # m = field_elems[vy] = (vy // p, vy % p)
                # Line: y - x*m = c
                # c = y0 - x0 * m
                # x0 = (xa, xb), m = (vy // p, vy % p)
                mx, my = field_elems[vy]
                # x0 * m
                xm_x, xm_y = mul_field(xa, xb, mx, my)
                # c = y0 - x0*m
                cx = (ya - xm_x) % p
                cy = (yb - xm_y) % p
                line_id[d_idx, pt_idx] = cx * p + cy
    
    # Now implement greedy: try to add points to S in some order.
    # A point can be added if for all directions d, the line line_id[d, pt] has count 0.
    
    def greedy_s():
        """Greedy construction of S, return the set of excluded points."""
        line_counts = np.zeros((num_dirs, q), dtype=np.int64)  # q=49 lines per direction
        excluded = np.zeros(total_points, dtype=bool)
        count = 0
        
        # Try points in order 0, 1, 2, ..., 2400
        for pt in range(total_points):
            can_add = True
            for d in range(num_dirs):
                lid = line_id[d, pt]
                if line_counts[d, lid] > 0:
                    can_add = False
                    break
            
            if can_add:
                excluded[pt] = True
                count += 1
                for d in range(num_dirs):
                    line_counts[d, line_id[d, pt]] += 1
        
        return excluded, count
    
    # First try natural order
    excluded, count = greedy_s()
    best_count = count
    best_excluded = excluded.copy()
    
    # Try random permutations
    rng = np.random.default_rng(42)
    n_tries = 200
    for i in range(n_tries):
        if time.time() > deadline:
            break
        order = rng.permutation(total_points)
        
        line_counts = np.zeros((num_dirs, q), dtype=np.int64)
        excluded_rand = np.zeros(total_points, dtype=bool)
        count_rand = 0
        
        for pt in order:
            can_add = True
            for d in range(num_dirs):
                lid = line_id[d, pt]
                if line_counts[d, lid] > 0:
                    can_add = False
                    break
            
            if can_add:
                excluded_rand[pt] = True
                count_rand += 1
                for d in range(num_dirs):
                    line_counts[d, line_id[d, pt]] += 1
        
        if count_rand > best_count:
            best_count = count_rand
            best_excluded = excluded_rand.copy()
    
    # The Nikodym set N is the complement of S
    nikodym_indices = np.where(~best_excluded)[0]
    
    if len(nikodym_indices) == 0:
        # Fallback: return all points
        points = []
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        points.append([[xa, xb], [ya, yb]])
        result = np.array(points, dtype=np.int64)
    else:
        points = []
        for idx in nikodym_indices:
            yb = idx % p
            ya = (idx // p) % p
            xb = (idx // (p * p)) % p
            xa = (idx // (p * p * p)) % p
            points.append([[xa, xb], [ya, yb]])
        result = np.array(points, dtype=np.int64)
    
    return result


# EVOLVE-BLOCK-END
