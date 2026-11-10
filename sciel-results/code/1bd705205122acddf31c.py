# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using quadratic residues.
    
    Key insight: Use quadratic residues in F_49 to define an exclusion set S.
    In F_49, there are 24 non-zero quadratic residues. We can construct a set S
    of points such that every line through a point in S intersects S only at that point,
    and every line through a point not in S is disjoint from S.
    
    Strategy: Let S = {(x, y) : y/x^2 is a quadratic residue in F_49, x != 0}
    plus handle x=0 cases. This gives a larger exclusion set than the parabola.
    """
    p = 7
    q = p * p
    # alpha^2 = 3
    k = 3
    
    # Precompute field operations
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
    
    def field_mul(a, b):
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    
    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)
    
    def field_inv(a):
        # In F_49, a^(48) = 1 for a != 0, so a^(-1) = a^47
        return field_pow(a, 47)
    
    def field_pow(a, exp):
        if exp == 0:
            return (1, 0)
        res = (1, 0)
        base = a
        while exp:
            if exp % 2 == 1:
                res = field_mul(res, base)
            base = field_mul(base, base)
            exp //= 2
        return res
    
    # Generate all field elements
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    zero = (0, 0)
    one = (1, 0)
    
    # Find quadratic residues in F_49
    # QR = {x^2 : x in F_49, x != 0}
    qr_set = set()
    for x in field_elements:
        if x != zero:
            qr_set.add(field_pow(x, 2))
    
    # QR has 24 elements (since |F_49^*| = 48, and squaring is 2-to-1)
    # Non-QR also has 24 elements
    
    # Construct exclusion set S using quadratic residues
    # S = {(x, y) : x != 0 and y * x^(-2) in QR}
    # This is the set of points where y/x^2 is a quadratic residue
    S = set()
    for x in field_elements:
        if x == zero:
            continue
        x_inv = field_inv(x)
        x_sq_inv = field_mul(x_inv, x_inv)
        for y in field_elements:
            # Check if y * x^(-2) is in QR
            val = field_mul(y, x_sq_inv)
            if val in qr_set:
                S.add((x[0], x[1], y[0], y[1]))
    
    # Also add points on the line x=0 where y is in QR (or some condition)
    # Actually, for x=0, we need a different criterion.
    # Let's add all points with x=0 to S to be safe, or use a different criterion.
    # For now, let's see what size S has.
    
    # Generate N = F_49^2 \ S
    N_points = []
    for x in field_elements:
        for y in field_elements:
            pt = (x[0], x[1], y[0], y[1])
            if pt not in S:
                N_points.append(pt)
    
    # Verify this is a valid Nikodym set
    # For each point in F_49^2, check there's a line through it with all other points in N
    
    def point_in_N(pt):
        return pt not in S
    
    def is_valid_nikodym(N_set, S_set):
        """Check if N = complement of S is a valid Nikodym set."""
        # For each point, check if there's a line through it with all other points in N
        for x_pt in field_elements:
            for y_pt in field_elements:
                pt = (x_pt[0], x_pt[1], y_pt[0], y_pt[1])
                # Try all lines through pt
                found_line = False
                # Lines with v = (1, m) for m in F_49
                for m in field_elements:
                    # Line: {pt + t*(1, m) : t in F_49}
                    # Check if all other points are in N
                    valid = True
                    for t in field_elements:
                        if t == zero:
                            continue  # Skip the point itself
                        dx = t
                        dy = field_mul(t, m)
                        other_pt = (
                            (x_pt[0] + dx[0]) % p,
                            (x_pt[1] + dx[1]) % p,
                            (y_pt[0] + dy[0]) % p,
                            (y_pt[1] + dy[1]) % p
                        )
                        if other_pt in S_set:
                            valid = False
                            break
                    if valid:
                        found_line = True
                        break
                if found_line:
                    continue
                # Try vertical line v = (0, 1)
                valid = True
                for t in field_elements:
                    if t == zero:
                        continue
                    other_pt = (
                        x_pt[0], x_pt[1],
                        (y_pt[0] + t[0]) % p,
                        (y_pt[1] + t[1]) % p
                    )
                    if other_pt in S_set:
                        valid = False
                        break
                if not valid:
                    return False
        return True
    
    # Check validity (this is expensive, so we do it once)
    S_tuple_set = set(S)
    # For efficiency, use a boolean array
    S_bool = np.zeros((p, p, p, p), dtype=bool)
    for (xa, xb, ya, yb) in S:
        S_bool[xa, xb, ya, yb] = True
    
    def pt_in_S_bool(pt):
        return bool(S_bool[pt[0], pt[1], pt[2], pt[3]])
    
    # Quick validity check
    valid = True
    for x_pt in field_elements:
        if not valid:
            break
        for y_pt in field_elements:
            if not valid:
                break
            pt = (x_pt[0], x_pt[1], y_pt[0], y_pt[1])
            found_line = False
            for m in field_elements:
                if found_line:
                    break
                valid_line = True
                for t in field_elements:
                    if t == zero:
                        continue
                    dx = t
                    dy = field_mul(t, m)
                    other_pt = (
                        (x_pt[0] + dx[0]) % p,
                        (x_pt[1] + dx[1]) % p,
                        (y_pt[0] + dy[0]) % p,
                        (y_pt[1] + dy[1]) % p
                    )
                    if pt_in_S_bool(other_pt):
                        valid_line = False
                        break
                if valid_line:
                    found_line = True
                    break
            if not found_line:
                # Try vertical
                valid_line = True
                for t in field_elements:
                    if t == zero:
                        continue
                    other_pt = (
                        x_pt[0], x_pt[1],
                        (y_pt[0] + t[0]) % p,
                        (y_pt[1] + t[1]) % p
                    )
                    if pt_in_S_bool(other_pt):
                        valid_line = False
                        break
                if not valid_line:
                    valid = False
                    break
    
    if not valid:
        # Fall back to parabola-based construction
        S = set()
        for t in field_elements:
            t_sq = field_pow(t, 2)
            S.add((t[0], t[1], t_sq[0], t_sq[1]))
        N_points = []
        for x in field_elements:
            for y in field_elements:
                pt = (x[0], x[1], y[0], y[1])
                if pt not in S:
                    N_points.append(pt)
    
    if not N_points:
        return np.empty((0, 2, 2), dtype=np.int64)
    
    arr = np.array(N_points, dtype=np.int64).reshape(-1, 2, 2)
    return arr


# EVOLVE-BLOCK-END
