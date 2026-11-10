# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 using local search on the complement set S.

    A set N is a Nikodym set if for every point x in F_q^2, there exists a direction v
    (either (1, m) for some m in F_q, or (0, 1)) such that all other points on the line
    {x + t*v : t in F_q} are in N. Equivalently, for every x and direction v, the line
    through x in direction v must contain at least one point in S = F_q^2 \\ N other than x.

    We formulate this as a Max-SAT-like problem: each point has a boolean variable s_i
    indicating membership in S. The constraint is that for every (x, v) pair, the line
    through x in direction v (excluding x itself) must have at least one point with s=1.
    We minimize |S| (maximize |N|) subject to these constraints.

    Strategy: Start from a valid solution (the baseline construction), then use simulated
    annealing / WalkSAT-style local search to flip points from S to N (removing from S)
    while maintaining validity. We track which constraints are violated incrementally.
    """
    p = 7
    q = p * p  # 49
    n_points = q * q  # 2401

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 30.0  # seconds

    # Field F_49 = F_7[alpha], alpha^2 = 3
    # Element is (a, b) representing a + b*alpha
    # Multiplication: (a, b) * (c, d) = (ac + 2bd, ad + bc) since alpha^2 = 3
    # Wait: alpha^2 = 3, so (a+b*alpha)(c+d*alpha) = ac + ad*alpha + bc*alpha + bd*alpha^2
    # = (ac + 3bd) + (ad+bc)*alpha
    k_alpha = 3  # alpha^2 = 3

    def mul_field(xa, xb, ya, yb):
        """Multiply two elements of F_49."""
        return ((xa * ya + k_alpha * xb * yb) % p, (xa * yb + xb * ya) % p)

    def add_field(xa, xb, ya, yb):
        """Add two elements of F_49."""
        return ((xa + ya) % p, (xb + yb) % p)

    def sub_field(xa, xb, ya, yb):
        """Subtract two elements of F_49."""
        return ((xa - ya) % p, (xb - yb) % p)

    def neg_field(xa, xb):
        """Negate an element of F_49."""
        return ((-xa) % p, (-xb) % p)

    # Precompute all field elements as indices 0..48
    # Element i = (i // p, i % p)
    def idx_to_coords(i):
        return (i // p, i % p)

    def coords_to_idx(a, b):
        return a * p + b

    # Precompute field addition and multiplication tables
    add_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        ia, ib = idx_to_coords(i)
        for j in range(q):
            ja, jb = idx_to_coords(j)
            add_table[i, j] = coords_to_idx((ia + ja) % p, (ib + jb) % p)
            ma, mb = mul_field(ia, ib, ja, jb)
            mul_table[i, j] = coords_to_idx(ma, mb)

    # All 2401 points in F_49^2, indexed 0..2400
    # Point index = x * q + y where x, y in 0..48
    def point_idx(x, y):
        return x * q + y

    def point_coords(i):
        x = i // q
        y = i % q
        xa, xb = idx_to_coords(x)
        ya, yb = idx_to_coords(y)
        return (xa, xb, ya, yb)

    # Directions: (1, m) for m in F_49 (49 directions) and (0, 1) (1 direction) = 50 directions
    # Direction is a vector (vx, vy) in F_49^2
    # For (1, m): vx = (1, 0), vy = m = (m_a, m_b)
    # For (0, 1): vx = (0, 0), vy = (1, 0)

    directions = []
    # 49 directions of form (1, m)
    for m in range(q):
        ma, mb = idx_to_coords(m)
        directions.append(((1, 0), (ma, mb)))
    # 1 direction of form (0, 1)
    directions.append(((0, 0), (1, 0)))

    n_dirs = len(directions)  # 50

    # For each point x and direction v, compute the set of other points on the line
    # Line: x + t*v for t in F_49, t != 0
    # Point on line: (x + t*vx, y + t*vy)
    # We need to precompute for each (x, v) the list of point indices (excluding x)

    # Precompute line members: for each x (0..2400) and v (0..49), list of 48 other points
    # This is 2401 * 50 * 48 ~ 5.7 million entries, manageable

    # Actually, let's think about this differently.
    # A point x is "covered" by direction v if at least one other point on the line x + t*v is in S.
    # For validity, EVERY point x must be covered by AT LEAST ONE direction v.
    # Wait, re-reading the problem: "every point x of F_49^2 must lie on a line {x + t*v : t in F_49}, with v = (1, m) for some m in F_49 or v = (0, 1), whose points other than x all belong to N."
    
    # This means: for every x, there EXISTS a direction v such that ALL other points on that line are in N (i.e., NOT in S).
    # So the constraint is: for each x, there exists v such that no other point on line x+t*v is in S.
    # Equivalently: NOT (for ALL v, there exists another point on line x+t*v that is in S).

    # So we want to maximize |N| = 2401 - |S|, i.e., minimize |S|,
    # subject to: for each x, exists v such that all other points on line(x, v) are not in S.

    # Let's define: for each x and v, let L(x,v) = set of other points on the line.
    # Constraint: for each x, OR over v of (AND over y in L(x,v) of (y not in S))
    # i.e., for each x, OR over v of (no point in L(x,v) is in S)

    # This is a complex constraint. Let me think about the baseline.

    # The baseline uses a Hermitian unital and excludes certain points.
    # Let me just implement the baseline first, then try to improve with local search.

    # Baseline: Unital U = {(x, y) : Trace(x) + Norm(y) = 0}
    # Trace((a,b)) = 2a (since the conjugate of a+b*alpha is a-b*alpha, trace = 2a)
    # Norm((a,b)) = a^2 - b^2*3 (since (a+b*alpha)(a-b*alpha) = a^2 - 3b^2)
    
    # Wait, the current code uses:
    # in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
    # where k is the quadratic non-residue. For p=7, the non-residues are 3,5,6.
    # Smallest is 3. So k=3.
    # So unital: 2*xa + ya^2 - 3*yb^2 = 0 mod 7

    # Special y set: Y = {(u^2 mod p, v) : u, v in F_p}
    # y_special: ya must be a square mod 7. Squares mod 7: 0,1,2,4.
    # So ya in {0,1,2,4}

    # Excluded set S = {points in U with y NOT in Y}
    # Nikodym set N = all points - S

    # Let me compute the baseline |S| first.
    
    # Actually, let me just implement the baseline and return it, then add local search.
    
    # For local search, I'll represent S as a boolean array of size 2401.
    # A point is in S if s[i] = True.
    
    # Validity check: for each point x, exists direction v such that no other point on line(x,v) is in S.
    
    # To make this efficient, for each x, I'll track how many directions v have NO points in S on the line.
    # Let count[x] = number of directions v for which all other points on line(x,v) are NOT in S.
    # Valid iff count[x] > 0 for all x.
    
    # When I flip a point y from S to not-S (remove from S), it can only help validity
    # (more lines become "clear"). When I flip from not-S to S, it can hurt.
    
    # Strategy: Start with baseline S. Try to remove points from S (flip to not-S) while maintaining validity.
    # A point y can be removed from S if after removal, all x still have count[x] > 0.
    
    # When y is in S and we remove it:
    # For each x and direction v such that y is on line(x,v) (y != x), the line(x,v) gains one more non-S point.
    # But we need ALL other points to be non-S. So removing y from S helps if y was the only S point on some line.
    
    # Actually, let me define: for each (x, v), let s_count(x,v) = number of points in S on line(x,v) excluding x.
    # Then x is valid for direction v iff s_count(x,v) == 0.
    # x is valid overall iff exists v with s_count(x,v) == 0.
    
    # When we remove y from S: for each (x,v) where y is on line(x,v), y != x: s_count(x,v) decreases by 1.
    # When we add y to S: s_count(x,v) increases by 1 for each such (x,v).
    
    # This is still expensive. Let me think of a simpler approach.
    
    # Given the time constraint, let me just implement the baseline and try a simple greedy removal.

    # Build baseline S
    s_in = np.zeros(n_points, dtype=bool)  # True means in S (excluded from N)
    
    for xa in range(p):
        for xb in range(p):
            x = coords_to_idx(xa, xb)
            for ya in range(p):
                for yb in range(p):
                    y = coords_to_idx(ya, yb)
                    pi = point_idx(x, y)
                    # Check if in unital
                    in_unital = (2 * xa + ya * ya - 3 * yb * yb) % p == 0
                    # Check if y is special (ya is a square mod 7)
                    ya_sq = (ya * ya) % p
                    is_special = (ya == 0 or ya == 1 or ya == 2 or ya == 4)
                    if in_unital and not is_special:
                        s_in[pi] = True
    
    # Baseline |S|
    base_s_count = s_in.sum()
    # print(f"Baseline |S| = {base_s_count}")

    # Now try to remove points from S while maintaining validity.
    # For efficiency, precompute for each point y, the list of (x, v) pairs where y is on line(x,v), y != x.
    
    # For a direction v = (vx, vy), the line through x in direction v is:
    # x + t*v for t in F_49
    # A point y is on this line iff y = x + t*v for some t != 0
    # i.e., y - x = t*v for some t != 0
    # i.e., (y_x - x_x, y_y - x_y) = t * (vx, vy)
    
    # For v = (1, m): y_x - x_x = t, y_y - x_y = t*m
    # So t = y_x - x_x, and we need y_y - x_y = (y_x - x_x) * m
    # For v = (0, 1): y_x - x_x = 0, y_y - x_y = t
    # So x_x = y_x, and t = y_y - x_y (any t != 0, so x_y != y_y)
    
    # Let me precompute for each y, the set of (x, v) such that y is on line(x, v), y != x.
    # This is 2401 * 50 * 48 ~ 5.7M entries. Might be too much memory.
    
    # Alternative: for each y in S, try to remove it and check validity incrementally.
    # But checking validity is O(2401 * 50 * 48) which is ~5.7M per removal, too slow.
    
    # Let me try a different approach: random local search.
    # Start with baseline S. Randomly pick a point in S, try to remove it.
    # Check if the result is still valid. If yes, keep it. If no, revert.
    
    # For validity check, I need to check all 2401 points. For each point, check if any direction has all other points not in S.
    # This is 2401 * 50 * 48 ~ 5.7M operations per check. With 600s budget, I can do maybe 1000-5000 checks.
    
    # Let me implement a fast validity check using precomputed line data.

    # Precompute: for each point x (0..2400) and direction v (0..49), the list of 48 other points on the line.
    # Store as a 3D array: line_points[x, v, 0..47] = point indices
    
    # Actually, let me store it more compactly. For each (x, v), store the 48 point indices.
    # Total: 2401 * 50 * 48 = 5,762,400 integers. ~23 MB. OK.
    
    line_points = np.zeros((n_points, n_dirs, q - 1), dtype=np.int32)
    
    for x in range(n_points):
        xa, xb, ya, yb = point_coords(x)
        for vi, (vx, vy) in enumerate(directions):
            vxa, vxb = vx
            vya, vyb = vy
            # Line: (xa + t*vxa, xb + t*vxb, ya + t*vya, yb + t*vyb) for t in F_49
            # t = (ta, tb)
            count = 0
            for ta in range(p):
                for tb in range(p):
                    if ta == 0 and tb == 0:
                        continue
                    # t = (ta, tb)
                    # x' = x + t*vx
                    # x'_a = xa + (ta*vxa + k_alpha*tb*vxb) % p
                    # x'_b = xb + (ta*vxb + tb*vxa) % p
                    # y' = y + t*vy
                    # y'_a = ya + (ta*vya + k_alpha*tb*vyb) % p
                    # y'_b = yb + (ta*vyb + tb*vya) % p
                    txa = (ta * vxa + k_alpha * tb * vxb) % p
                    txb = (ta * vxb + tb * vxa) % p
                    tya = (ta * vya + k_alpha * tb * vyb) % p
                    tyb = (ta * vyb + tb * vya) % p
                    nxa = (xa + txa) % p
                    nxb = (xb + txb) % p
                    nya = (ya + tya) % p
                    nyb = (yb + tyb) % p
                    nx = coords_to_idx(nxa, nxb)
                    ny = coords_to_idx(nya, nyb)
                    pi = point_idx(nx, ny)
                    line_points[x, vi, count] = pi
                    count += 1
    
    # Now validity check: for each x, check if exists v such that no point in line_points[x, v] is in S.
    def is_valid(s):
        for x in range(n_points):
            found = False
            for vi in range(n_dirs):
                # Check if any point in line_points[x, vi] is in S
                line = line_points[x, vi]
                if not np.any(s[line]):
                    found = True
                    break
            if not found:
                return False
        return True
    
    # Check baseline validity
    # print("Checking baseline validity...")
    # valid = is_valid(s_in)
    # print(f"Baseline valid: {valid}")

    # Now try to improve: greedily remove points from S.
    # Try each point in S, see if removing it keeps validity.
    # This is O(|S| * 2401 * 50 * 48) which is too slow for full check.
    
    # Instead, use a heuristic: for each point y in S, compute how many (x, v) pairs have y as the ONLY S point on the line.
    # If removing y doesn't invalidate any x, we can remove it.
    
    # For each y in S, and for each (x, v) where y is on line(x,v):
    #   Check if s_count(x,v) == 1 (y is the only S point on that line)
    #   If yes, then x is currently valid only via direction v (or other directions).
    #   If we remove y, x might lose validity if v was its only valid direction.
    
    # This is complex. Let me just do random sampling with full validity checks.
    
    # Random local search
    rng = np.random.RandomState(42)
    current_s = s_in.copy()
    current_valid = is_valid(current_s)
    # print(f"Initial valid: {current_valid}, |S| = {current_s.sum()}")
    
    if not current_valid:
        # Return the Nikodym set (complement of S)
        n_points_list = [i for i in range(n_points) if not current_s[i]]
        return np.array([[point_coords(i)[0], point_coords(i)[1], point_coords(i)[2], point_coords(i)[3]] for i in n_points_list], dtype=np.int64).reshape(-1, 2, 2)
    
    best_s = current_s.copy()
    best_s_count = current_s.sum()
    
    # Try random removals
    s_indices = np.where(current_s)[0]
    num_tries = 0
    max_tries = 20000
    
    while num_tries < max_tries and time.time() < deadline - safety_margin:
        # Pick a random point in S
        idx = rng.choice(s_indices)
        if not current_s[idx]:
            continue
        
        # Try to remove it
        current_s[idx] = False
        num_tries += 1
        
        if num_tries % 10 == 0:
            # Check validity periodically
            if is_valid(current_s):
                best_s = current_s.copy()
                best_s_count = current_s.sum()
            else:
                current_s[idx] = True
                # Update s_indices
                s_indices = np.where(current_s)[0]
        
        # Update s_indices
        if not current_s[idx]:
            s_indices = np.where(current_s)[0]
    
    # Final check
    if is_valid(current_s):
        best_s = current_s.copy()
    
    # Return the Nikodym set N = complement of best_s
    n_indices = np.where(~best_s)[0]
    result = np.zeros((len(n_indices), 2, 2), dtype=np.int64)
    for i, pi in enumerate(n_indices):
        xa, xb, ya, yb = point_coords(pi)
        result[i, 0, 0] = xa
        result[i, 0, 1] = xb
        result[i, 1, 0] = ya
        result[i, 1, 1] = yb
    
    return result


# EVOLVE-BLOCK-END
