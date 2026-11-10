import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = 580.0  # safety margin before 630s kill

    p = 7
    k = 3  # alpha^2 = 3, non-residue mod 7

    # Field operations on F_49 = F_7[alpha] where alpha^2 = 3
    # Element is (a, b) representing a + b*alpha

    def f_add(e1, e2):
        return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)

    def f_mul(e1, e2):
        a, b = e1
        c, d = e2
        return ((a * c + b * d * k) % p, (a * d + b * c) % p)

    def f_sub(e1, e2):
        return ((e1[0] - e2[0]) % p, (e1[1] - e2[1]) % p)

    def f_inv(e):
        a, b = e
        norm = (a * a - b * b * k) % p
        if norm == 0:
            raise ValueError("zero")
        norm_inv = pow(norm, p - 2, p)
        return ((a * norm_inv) % p, (-b * norm_inv) % p)

    def f_div(e1, e2):
        return f_mul(e1, f_inv(e2))

    # Precompute all field elements
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    field_set = set(field_elements)

    # Precompute multiplication table
    mul_table = {}
    for e1 in field_elements:
        for e2 in field_elements:
            mul_table[(e1, e2)] = f_mul(e1, e2)

    # Precompute inverse table
    inv_table = {}
    for e in field_elements:
        if e != (0, 0):
            inv_table[e] = f_inv(e)

    # Total points in F_49^2
    total_points = p * p * p * p  # 2401

    # Build all points
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append((xa, xb, ya, yb))

    # For each point, precompute all lines through it
    # A line through point x with direction v has points {x + t*v : t in F_49}
    # Directions: v = (1, m) for m in F_49, or v = (0, 1)

    # Precompute for each point and each direction, the set of other points on that line
    # This is expensive, so we'll compute on the fly but cache

    def get_line_points(x, v):
        """Get all points on the line through x with direction v, excluding x itself."""
        xa, xb, ya, yb = x
        vx_a, vx_b = v[0]
        vy_a, vy_b = v[1]
        points = []
        for ta in range(p):
            for tb in range(p):
                if ta == 0 and tb == 0:
                    continue
                t = (ta, tb)
                # x + t*v
                tx_a, tx_b = f_mul(t, (vx_a, vx_b))
                ty_a, ty_b = f_mul(t, (vy_a, vy_b))
                nx_a = (xa + tx_a) % p
                nx_b = (xb + tx_b) % p
                ny_a = (ya + ty_a) % p
                ny_b = (yb + ty_b) % p
                points.append((nx_a, nx_b, ny_a, ny_b))
        return points

    # Directions
    directions = []
    # v = (0, 1)
    directions.append(((0, 0), (1, 0)))
    # v = (1, m) for m in F_49
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))

    def verify_nikodym_fast(excluded_set):
        """Check if F_49^2 \ excluded_set is a valid Nikodym set.
        For every point x, there must exist a direction v such that
        all points on the line through x with direction v (except x) are in N."""
        if time.time() - start_time > deadline:
            return False

        excluded = set(excluded_set)
        N_set = set(all_points) - excluded

        # For each point, check if there's a valid direction
        for x in all_points:
            found_valid_direction = False
            for v in directions:
                line_points = get_line_points(x, v)
                all_in_N = True
                for lp in line_points:
                    if lp not in N_set:
                        all_in_N = False
                        break
                if all_in_N:
                    found_valid_direction = True
                    break
            if not found_valid_direction:
                return False

        return True

    # Strategy: Use the Hermitian unital as a base, then try to expand the complement
    # by greedily adding points that maintain the Nikodym property

    # The Hermitian unital U = {(x, y) : Tr(x) + y^2 - k*y^2 = 0}
    # Actually, for F_49, the unital is defined by:
    # x + x^7 + y^2 + y^7 - k*(y^2 + y^7) = 0
    # But in F_49, x^7 = x (Frobenius), so x + x^7 = 2x
    # y^2 + y^7 = y^2 + y (since y^7 = y)
    # So the equation is: 2x + y^2 + y - k*(y^2 + y) = 0
    # = 2x + (1-k)y^2 + (1-k)y = 0
    # = 2x + (1-3)y^2 + (1-3)y = 0
    # = 2x - 2y^2 - 2y = 0
    # = 2(x - y^2 - y) = 0
    # Since 2 != 0 mod 7, this gives x = y^2 + y

    # Let me use a different unital definition. The standard Hermitian unital in F_q^2
    # for q = p^2 is: x^q + x + y^{q+1} = 0
    # For q = 49, p = 7: x^49 + x + y^50 = 0
    # In F_49, x^49 = x (since x^49 = x^(7^2) = x for x in F_49)
    # y^50 = y^(2*25) = (y^2)^25. Since y in F_49, y^49 = y, so y^50 = y
    # Wait, y^49 = y for y in F_49, so y^50 = y^49 * y = y*y = y^2
    # So the equation is: x + x + y^2 = 0, i.e., 2x + y^2 = 0, i.e., x = -y^2/2 = -y^2 * 4 (since 1/2 = 4 mod 7)

    # Let me just use a known valid construction. The simplest Nikodym set is
    # N = F_49^2 \ {a single point}, but that's not valid because the point not in N
    # has no line through it where all other points are in N (unless the line has only
    # that one point, which is impossible since lines have 49 points).

    # Actually, for a Nikodym set, every point x (in or out of N) must lie on a line
    # where all other points are in N. So if x is not in N, we need a line through x
    # where the other 48 points are in N.

    # A simple valid Nikodym set: take N = F_49^2 \ L where L is a single line.
    # Then for any point x on L, the line L itself has all other points in N.
    # For any point x not on L, we need a line through x where all other points are in N.
    # This is not guaranteed.

    # Let me use the known result: the complement of a blocking set is a Nikodym set
    # if and only if the blocking set is an "orthogonal blocking set" (every line contains
    # at least one point from the blocking set).

    # So we need to find a maximum set S such that every line in F_49^2 contains at least
    # one point from S. Then N = F_49^2 \ S is a Nikodym set.

    # This is equivalent to finding a minimum blocking set.

    # A blocking set in F_49^2 is a set of points that intersects every line.
    # The minimum size of a blocking set in F_q^2 is q+1 (a line).

    # So the maximum complement is 2401 - 49 = 2352, achieved by taking S to be any line.

    # Let's verify: if S is a line L, then N = F_49^2 \ L.
    # For x in L: the line L has all other points in N. ✓
    # For x not in L: we need a line through x where all other points are in N.
    # Take any line through x that is not L. This line intersects L in exactly one point.
    # So this line has 49 points, one of which is in L (hence not in N), and the other 48
    # are in N. But x is not in L, so x is in N. The line through x has x (in N) and one
    # point in L (not in N) and 47 other points in N. So the line has 48 points in N and
    # 1 point not in N. For x to be valid, we need a line through x where ALL other points
    # are in N. The line through x and the intersection point with L has that intersection
    # point not in N, so it doesn't work.

    # So a single line is NOT a valid blocking set for the Nikodym property.

    # Let me reconsider. The Nikodym property requires: for every point x, there exists
    # a line through x such that all points on that line except x are in N.

    # If S is the complement of N, then for every x, there exists a line through x such
    # that all points on that line except x are NOT in S, i.e., the line contains at most
    # one point from S (namely x itself, if x is in S).

    # So S must be such that every line contains at most one point from S... no wait.
    # If x is not in S (i.e., x is in N), then we need a line through x with no points
    # from S other than possibly x (but x is not in S, so no points from S at all).
    # If x is in S, then we need a line through x with no other points from S.

    # So S must be a set where every line contains at most one point from S. That means
    # S is a set of points no two of which are on the same line. But in F_49^2, any two
    # distinct points determine a unique line. So S can have at most one point!

    # Wait, that can't be right. Let me re-read the definition.

    # "every point x of F_49^2 (in N or not) must lie on a line {x + t*v : t in F_49},
    # with v = (1, m) for some m in F_49 or v = (0, 1), whose points other than x all
    # belong to N."

    # So for each x, there exists a direction v (either (1,m) for some m, or (0,1)) such
    # that for all t != 0, x + t*v is in N.

    # This means: for each x, there's a line through x (with a specific direction from
    # the allowed set) where all other points are in N.

    # The allowed directions are: (0,1) and (1,m) for all m in F_49. That's 49 + 1 = 50
    # directions.

    # So we need: for each x, at least one of the 50 lines through x (with allowed
    # directions) has all its other 48 points in N.

    # Equivalently, for each x, at least one of the 50 lines through x has no points
    # from S (the complement) other than possibly x.

    # If x is in S: the line must have no other points from S.
    # If x is not in S: the line must have no points from S at all.

    # So we need to find a maximum set S such that for every x, at least one of the 50
    # lines through x (with allowed directions) contains no points from S other than x
    # (if x is in S) or no points from S at all (if x is not in S).

    # This is a complex constraint. Let me try a different approach.

    # Greedy approach: start with S = empty, and try to add points to S one by one,
    # maintaining the property that for every x, at least one of the 50 lines through x
    # is "clean" (no points from S other than possibly x).

    # Actually, let me try the opposite: start with N = F_49^2 (S = empty), and try to
    # remove points from N (add to S) while maintaining the Nikodym property.

    # Initially, N = F_49^2 is valid (every line is fully in N).
    # We want to maximize |S| = 2401 - |N|.

    # Let me try a simple heuristic: for each point, check if we can remove it from N
    # (add to S) while maintaining validity.

    # This is expensive to verify each time, so let me use a smarter approach.

    # Key insight: if we remove a point x from N, then for x to still be valid, there
    # must be a line through x where all other points are in N. Since we're removing x,
    # x is now in S. For x to be valid, there must be a line through x where no other
    # points are in S.

    # For other points y, removing x from N might invalidate y if the only clean line
    # through y was the one containing x.

    # This is getting complex. Let me try a different strategy.

    # Strategy: Use a random greedy approach with restarts.
    # 1. Start with a random subset S0.
    # 2. Check if N = F_49^2 \ S0 is valid.
    # 3. If valid, try to add more points to S (remove from N) greedily.
    # 4. Repeat with different random seeds.

    # But verification is expensive. Let me optimize.

    # Precompute for each point x and each direction v, the set of other points on that line.
    # Store as a list of point indices.

    point_index = {pt: i for i, pt in enumerate(all_points)}

    # line_points[x][v] = list of point indices on the line through x with direction v, excluding x
    # This is 2401 * 50 * 48 entries, which is ~5.7 million. Too much memory.

    # Instead, let me compute on the fly but use numpy for speed.

    # Alternative: represent points as integers 0..2400
    # point = xa * 343 + xb * 49 + ya * 7 + yb

    def point_to_idx(xa, xb, ya, yb):
        return xa * 343 + xb * 49 + ya * 7 + yb

    def idx_to_point(idx):
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        return (xa, xb, ya, yb)

    # Precompute line memberships
    # For each point index and each direction index, store the set of other point indices on that line
    # This is 2401 * 50 sets of 48 elements each = ~5.7 million elements. Feasible.

    # Actually, let me just do a simple greedy approach with limited verification.

    # Start with the known valid construction: S = a line (49 points), N = 2352 points.
    # But we showed this might not be valid. Let me verify.

    # Take S = {(0,0, ya, yb) : ya, yb in F_7} (the line x = 0)
    # For x = (0, 0, ya, yb) in S: need a line through x with no other S points.
    # The line x = 0 has 49 points, all in S. So this doesn't work.
    # Try a different direction. The line through (0,0,ya,yb) with direction (1,0) is
    # {(t, 0, ya, yb) : t in F_49}. Points are (ta, tb, ya, yb) for ta, tb in F_7.
    # S contains only points with xa=0, xb=0. So on this line, only (0,0,ya,yb) is in S.
    # So all other 48 points are in N. ✓

    # For x not in S (xa != 0 or xb != 0): need a line through x with no S points.
    # Try direction (0,1): line is {(xa, xb, ya+t_a, yb+t_b) : t in F_49}
    # = {(xa, xb, ya+ta, yb+tb) : ta, tb in F_7}
    # S contains points with xa=0, xb=0. Since xa != 0 or xb != 0, no point on this line is in S. ✓

    # So S = {(0,0,ya,yb)} IS a valid blocking set! |S| = 49, |N| = 2352.

    # Let me verify more carefully. S = {(0, 0, ya, yb) : ya, yb in F_7}.
    # This is the set of points where x = 0 (as a field element).

    # For x = (0, 0, ya, yb) in S:
    #   Use direction v = (1, 0) (i.e., m = 0).
    #   Line: {(t, 0, ya, yb) : t in F_49} = {(ta, tb, ya, yb) : ta, tb in F_7}
    #   Points in S on this line: only (0, 0, ya, yb) since S requires xa=0, xb=0.
    #   So all other 48 points are in N. ✓

    # For x = (xa, xb, ya, yb) not in S (i.e., (xa,xb) != (0,0)):
    #   Use direction v = (0, 1).
    #   Line: {(xa, xb, ya+t_a, yb+t_b) : t in F_49} = {(xa, xb, ya+ta, yb+tb) : ta, tb in F_7}
    #   Points in S on this line: S requires xa=0, xb=0. Since (xa,xb) != (0,0), no points in S.
    #   So all 49 points on this line are in N, including x. ✓

    # So S = {(0,0,ya,yb)} is valid! |S| = 49, |N| = 2352.

    # Can we do better? Can we find a larger S?

    # Let's try S = {(xa, xb, ya, yb) : xa = 0} (all points with xa = 0).
    # |S| = 7 * 7 * 7 = 343.
    # For x = (0, xb, ya, yb) in S:
    #   Need a line through x with no other S points.
    #   Try direction (1, 0): line is {(t, xb, ya, yb) : t in F_49} = {(ta, tb, ya, yb) : ta, tb in F_7}
    #   Wait, that's not right. x + t*v where v = (1,0):
    #   x = (0, xb, ya, yb), v = ((1,0), (0,0))
    #   x + t*v = (0 + t*1, xb + t*0, ya + t*0, yb + t*0) = (t, xb, ya, yb)
    #   t = (ta, tb), so t as a field element is ta + tb*alpha.
    #   x + t*v = (ta, tb, xb, ya, yb)... no, this is getting confusing.

    # Let me be more careful. A point is (x, y) where x = (xa, xb) in F_49 and y = (ya, yb) in F_49.
    # Direction v = (vx, vy) where vx, vy in F_49.
    # Line through x0 = (x0x, x0y) with direction v: {x0 + t*v : t in F_49}
    # = {(x0x + t*vx, x0y + t*vy) : t in F_49}

    # For v = (1, 0) where 1 = (1,0) and 0 = (0,0) in F_49:
    # x0 + t*v = (x0x + t*1, x0y + t*0) = (x0x + t, x0y)
    # If x0x = 0, then x0 + t*v = (t, x0y)
    # t = (ta, tb), so the x-coordinate is (ta, tb) and y-coordinate is x0y = (ya, yb)
    # Points: {(ta, tb, ya, yb) : ta, tb in F_7}
    # S = {(0, xb, ya, yb) : xb, ya, yb in F_7} = all points with xa = 0.
    # On this line, points in S are those with ta = 0, i.e., {(0, tb, ya, yb) : tb in F_7} = 7 points.
    # So there are 7 points in S on this line, not just 1. This doesn't work for the Nikodym property.

    # So S = {xa = 0} is NOT valid.

    # Back to S = {(0,0,ya,yb)} which we verified is valid with |S| = 49.

    # Let me try to extend this. Can we add more points to S while maintaining validity?

    # Current S0 = {(0,0,ya,yb) : ya, yb in F_7}, |S0| = 49.
    # N0 = F_49^2 \ S0, |N0| = 2352.

    # Try to add a point p = (pa, pb, qc, qd) to S, where (pa,pb) != (0,0).
    # After adding, S = S0 ∪ {p}.
    # Need to check:
    # 1. For p: there exists a line through p with no other S points.
    # 2. For all x in S0: there still exists a line through x with no other S points.
    # 3. For all x in N0 \ {p}: there still exists a line through x with no S points.

    # This is complex. Let me just go with |S| = 49 for now and return N = F_49^2 \ S0.

    # Actually, let me try a slightly different S.
    # S = {(0, 0, ya, yb) : ya, yb in F_7} works.
    # What about S = {(xa, xb, 0, 0) : xa, xb in F_7}? (y = 0)
    # By symmetry, this should also work.

    # Let me try to find a larger S using a greedy approach.

    # Start with S = empty, N = all points.
    # Greedily add points to S while maintaining validity.

    # For efficiency, I'll use a simplified check.

    # Actually, let me just return the known valid set with |S| = 49.

    # S = {(0, 0, ya, yb) : ya, yb in F_7}
    excluded = set()
    for ya in range(p):
        for yb in range(p):
            excluded.add((0, 0, ya, yb))

    # Verify this is valid
    valid = verify_nikodym_fast(excluded)
    if not valid:
        # Fallback: use a single point
        excluded = {(0, 0, 0, 0)}

    # Build N = F_49^2 \ excluded
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (xa, xb, ya, yb) not in excluded:
                        points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)
