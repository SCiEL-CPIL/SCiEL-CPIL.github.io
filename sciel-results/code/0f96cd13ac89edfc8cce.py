# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = 49 (p=7) using a greedy approach.
    We start with all points in N and try to remove points while maintaining
    the Nikodym property. A point x can be removed from N (i.e., added to complement C)
    only if for every direction v, there exists t != 0 such that x + t*v is already
    in C. This ensures x remains 'covered'.
    """
    p = 7
    q = p * p
    deadline = time.time() + 580  # Safety margin
    
    # Precompute field operations
    # alpha^2 = 3 (smallest QR mod 7)
    k_const = 3
    
    def add_f(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def mul_f(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = (a0*b0 + a1*b1*k) + (a0*b1 + a1*b0)*alpha
        return ((a[0]*b[0] + a[1]*b[1]*k_const) % p, 
                (a[0]*b[1] + a[1]*b[0]) % p)
    
    # Represent points as (xa, xb, ya, yb)
    # All points in F_49^2
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append((xa, xb, ya, yb))
    
    n_points = len(all_points)
    # Index each point
    point_to_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # Generate all lines with direction v = (1, m) for m in F_49, and v = (0, 1)
    # A line through point x with direction v: {x + t*v : t in F_49}
    
    # Precompute lines: for each point and each direction, store the other points on the line
    # Directions: (1, m) for m in F_49 (49 directions) + (0, 1) (1 direction) = 50 directions
    
    def line_points(x, v):
        """Return list of points on line through x with direction v, excluding x."""
        pts = []
        # t from 1 to q-1 (skip t=0 which gives x)
        for ta in range(p):
            for tb in range(p):
                if ta == 0 and tb == 0:
                    continue
                t = (ta, tb)
                # tv = t * v
                tv = mul_f(t, v)
                # x + tv
                new_x = add_f(x, tv)
                pt = (new_x[0], new_x[1], x[2], x[3])  # Wait, x is (xa,xb,ya,yb)
                # Actually x is a point (xa, xb, ya, yb), v is a direction (va, vb, vc, vd)
                # Let me fix this
        return pts
    
    # Let me redo this more carefully
    # A point is (x, y) where x = (xa, xb), y = (ya, yb)
    # A direction v = (vx, vy) where vx = (vxa, vxb), vy = (vya, vyb)
    # Line: {(x, y) + t*(vx, vy) : t in F_49}
    
    def add_point(p1, p2):
        return (p1[0]+p2[0], p1[1]+p1[1], p1[2]+p2[2], p1[3]+p2[3])
    
    # Actually let me be very explicit
    # Point: (xa, xb, ya, yb) representing (xa+xb*alpha, ya+yb*alpha)
    # Direction v: (vxa, vxb, vya, vyb)
    # t*v: t=(ta,tb), so t*v = (ta*vxa + tb*vxb*k, ta*vxb + tb*vxa, ta*vya + tb*vyb*k, ta*vyb + tb*vya)
    
    def mul_t_v(t, v):
        ta, tb = t
        vxa, vxb, vya, vyb = v
        return ((ta*vxa + tb*vxb*k_const) % p,
                (ta*vxb + tb*vxa) % p,
                (ta*vya + tb*vyb*k_const) % p,
                (ta*vyb + tb*vya) % p)
    
    def add_pt(p1, p2):
        return ((p1[0]+p2[0])%p, (p1[1]+p2[1])%p, (p1[2]+p2[2])%p, (p1[3]+p2[3])%p)
    
    # Generate directions
    directions = []
    # v = (1, m) for m in F_49: v = (1, 0, m_a, m_b)
    for ma in range(p):
        for mb in range(p):
            directions.append((1, 0, ma, mb))
    # v = (0, 1): v = (0, 1, 0, 0)
    directions.append((0, 1, 0, 0))
    
    n_dirs = len(directions)
    
    # For each point and each direction, precompute the set of other points on the line
    # line_mask[point_idx][dir_idx] = set of point indices on the line (excluding the point itself)
    
    # This might be too memory intensive. Let's compute on the fly or use a different approach.
    
    # Alternative: Start with C = empty, N = all points.
    # Try to add points to C greedily.
    # A point x can be added to C if for every direction v, there exists t != 0 
    # such that x + t*v is already in C.
    
    # Initially C is empty, so no point can be added. We need a different approach.
    
    # Let's use the dual: Start with N = all points. Try to remove points from N.
    # A point x can be removed from N (added to C) if for every direction v,
    # there exists t != 0 such that x + t*v is NOT in N (i.e., already in C).
    
    # But initially C is empty, so we can't remove any point.
    
    # Better approach: Use the known construction as a starting point, then try to improve.
    # Or use a completely different strategy.
    
    # Let me try: Start with the baseline construction, then try to add more points to C
    # (remove from N) while maintaining validity.
    
    # First, build the baseline N
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    C = set()  # Complement
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k_const) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        C.add((xa, xb, ya, yb))
    
    # Now try to add more points to C
    # A point x can be added to C if for every direction v, there exists t != 0
    # such that x + t*v is in C.
    
    def can_add_to_C(x, C):
        for v in directions:
            found = False
            for ta in range(p):
                for tb in range(p):
                    if ta == 0 and tb == 0:
                        continue
                    t = (ta, tb)
                    tv = mul_t_v(t, v)
                    y = add_pt(x, tv)
                    if y in C:
                        found = True
                        break
                if found:
                    break
            if not found:
                return False
        return True
    
    # Greedily add points to C
    changed = True
    while changed:
        changed = False
        for pt in all_points:
            if pt in C:
                continue
            if time.time() > deadline:
                break
            if can_add_to_C(pt, C):
                C.add(pt)
                changed = True
        if time.time() > deadline:
            break
    
    # Build N = all_points - C
    N = [pt for pt in all_points if pt not in C]
    
    # Convert to array
    points = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in N], dtype=np.int64)
    
    return points

# EVOLVE-BLOCK-END
