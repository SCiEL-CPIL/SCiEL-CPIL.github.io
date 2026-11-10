# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by maximizing the size of the complement S.
    
    A Nikodym set N is a subset of F_q^2 such that for every point x in F_q^2,
    there exists a direction v = (1, m) (with m in F_q) or v = (0, 1) such that
    the punctured line {x + t*v : t in F_q, t != 0} is fully contained in N.
    
    Equivalently, the complement S = F_q^2 \ N must satisfy:
    For every point x in F_q^2, there exists a direction v such that
    (L(x, v) \ {x}) ∩ S = ∅, where L(x, v) = {x + t*v : t in F_q}.
    
    This means x has a "free" direction if the line through x in direction v
    contains no other points of S (other than possibly x itself).
    
    We start with S from the algebraic construction and use local search to grow S.
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    # Deadline guard
    start_time = time.time()
    timeout = 590  # seconds, leaving 10s margin before 600s limit
    
    # Field arithmetic helpers for F_49 = F_7[alpha], alpha^2 = 3
    # Element represented as (a, b) meaning a + b*alpha
    def add_f(x, y):
        return ((x[0] + y[0]) % p, (x[1] + y[1]) % p)
    
    def sub_f(x, y):
        return ((x[0] - y[0]) % p, (x[1] - y[1]) % p)
    
    def neg_f(x):
        return ((-x[0]) % p, (-x[1]) % p)
    
    def mul_f(x, y):
        # (a + b*alpha)(c + d*alpha) = (ac + bd*alpha^2) + (ad + bc)*alpha
        # alpha^2 = 3
        return ((x[0]*y[0] + 3*x[1]*y[1]) % p, (x[0]*y[1] + x[1]*y[0]) % p)
    
    def inv_f(x):
        # Inverse of a + b*alpha = (a - b*alpha)/(a^2 - 3b^2)
        norm = (x[0]*x[0] - 3*x[1]*x[1]) % p
        if norm == 0:
            return (0, 0)
        norm_inv = pow(norm, p-2, p)
        return mul_f((x[0], (-x[1]) % p), (norm_inv, 0))
    
    def div_f(x, y):
        return mul_f(x, inv_f(y))
    
    # Generate all elements of F_49
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    
    # Generate all 50 directions: 49 directions v = (1, m) for m in F_49, plus v = (0, 1)
    directions = []
    for m in field_elements:
        directions.append(((1, 0), m))  # v = (1, m)
    directions.append(((0, 1), (0, 0)))  # v = (0, 1)
    
    # For each point x and direction v, compute the set of other points on the line
    # We'll precompute the line membership for efficiency
    # line_members[x_idx][d_idx] = set of point indices on the line through x in direction d, excluding x
    
    # Map point (xa, xb, ya, yb) to index
    def point_to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * q) + (ya * p + yb)
    
    def idx_to_point(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // q) % p
        xa = (idx // (q * p)) % p
        return (xa, xb, ya, yb)
    
    # Precompute all lines
    # For each direction d and each point x, store the list of other points on the line
    # lines[x_idx][d_idx] = list of other point indices
    lines = {}
    for d_idx, (vx, vy) in enumerate(directions):
        # For each point, find the line it's on in this direction
        # A line in direction v is determined by its "intercept"
        # For v = (1, m): lines are y = m*x + c
        # For v = (0, 1): lines are x = const
        
        if vx == (0, 1):
            # Vertical lines: x = const
            for xa in range(p):
                for xb in range(p):
                    # All points with this x-coordinate
                    line_pts = []
                    for ya in range(p):
                        for yb in range(p):
                            idx = point_to_idx(xa, xb, ya, yb)
                            line_pts.append(idx)
                    for pt_idx in line_pts:
                        if pt_idx not in lines:
                            lines[pt_idx] = {}
                        lines[pt_idx][d_idx] = [other for other in line_pts if other != pt_idx]
        else:
            # v = (1, m) where m = vy
            # Line: (y - y0) = m * (x - x0)
            # For each intercept, collect points
            # We can iterate over all points and group by intercept
            
            # For point (x, y), the intercept c satisfies y = m*x + c, so c = y - m*x
            groups = {}
            for xa in range(p):
                for xb in range(p):
                    for ya in range(p):
                        for yb in range(p):
                            x = (xa, xb)
                            y = (ya, yb)
                            # c = y - m*x
                            mx = mul_f(vy, x)
                            c = sub_f(y, mx)
                            if c not in groups:
                                groups[c] = []
                            idx = point_to_idx(xa, xb, ya, yb)
                            groups[c].append(idx)
            
            for c, line_pts in groups.items():
                for pt_idx in line_pts:
                    if pt_idx not in lines:
                        lines[pt_idx] = {}
                    lines[pt_idx][d_idx] = [other for other in line_pts if other != pt_idx]
    
    # Initial S from algebraic construction (the seed)
    # S = { (x,y) in U : y not in Y }
    # U = {(x,y) : Trace(x) + Norm(y) = 0}
    # Trace((a,b)) = 2a, Norm((a,b)) = a^2 - 3b^2
    
    S = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x = (xa, xb)
                    y = (ya, yb)
                    trace_x = (2 * xa) % p
                    norm_y = (ya * ya - 3 * yb * yb) % p
                    in_unital = (trace_x + norm_y) % p == 0
                    # Y = {(u^2 mod p, v) : u, v in F_p}
                    in_Y = (ya in [u*u % p for u in range(p)])
                    if in_unital and not in_Y:
                        idx = point_to_idx(xa, xb, ya, yb)
                        S.add(idx)
    
    best_S = set(S)
    best_size = len(S)
    
    # Check validity of S
    def is_valid(S_set):
        """Check if S_set is a valid complement of a Nikodym set."""
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        idx = point_to_idx(xa, xb, ya, yb)
                        # Check if there exists a direction v such that
                        # (L(x,v) \ {x}) ∩ S = ∅
                        found_free = False
                        for d_idx in range(len(directions)):
                            other_pts = lines[idx][d_idx]
                            conflict = False
                            for op in other_pts:
                                if op in S_set:
                                    conflict = True
                                    break
                            if not conflict:
                                found_free = True
                                break
                        if not found_free:
                            return False
        return True
    
    # Local search: try to add points to S
    all_points = list(range(total_points))
    random.shuffle(all_points)
    
    current_S = set(S)
    
    while time.time() - start_time < timeout:
        added_any = False
        for pt_idx in all_points:
            if pt_idx in current_S:
                continue
            # Try adding pt_idx to current_S
            new_S = set(current_S)
            new_S.add(pt_idx)
            
            # Check validity: only need to check points whose free directions might be affected
            # A point y loses a free direction d if pt_idx is on the line through y in direction d
            # and that line had no other S points before
            valid = True
            for xa in range(p):
                for xb in range(p):
                    for ya in range(p):
                        for yb in range(p):
                            y_idx = point_to_idx(xa, xb, ya, yb)
                            # If pt_idx == y_idx, skip (we're checking y's free directions)
                            # Check if y still has a free direction
                            found_free = False
                            for d_idx in range(len(directions)):
                                other_pts = lines[y_idx][d_idx]
                                conflict = False
                                for op in other_pts:
                                    if op in new_S:
                                        conflict = True
                                        break
                                if not conflict:
                                    found_free = True
                                    break
                            if not found_free:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Also need to check that pt_idx itself has a free direction in new_S
            # pt_idx is now in S, so for pt_idx, we need a direction d such that
            # (L(pt_idx, d) \ {pt_idx}) ∩ new_S = ∅
            # Since pt_idx was not in current_S, this is equivalent to checking
            # (L(pt_idx, d) \ {pt_idx}) ∩ current_S = ∅ (since pt_idx not in current_S)
            # Wait, we need to check against new_S which includes pt_idx, but pt_idx is excluded
            # from the punctured line, so we check against current_S
            found_free_for_pt = False
            for d_idx in range(len(directions)):
                other_pts = lines[pt_idx][d_idx]
                conflict = False
                for op in other_pts:
                    if op in current_S:
                        conflict = True
                        break
                if not conflict:
                    found_free_for_pt = True
                    break
            if not found_free_for_pt:
                continue
            
            # Valid addition
            current_S = new_S
            added_any = True
            if len(current_S) > best_size:
                best_S = set(current_S)
                best_size = len(current_S)
        
        if not added_any:
            # Try removing some points and re-adding? Or just stop
            # Try a random restart with different order
            random.shuffle(all_points)
            # If we can't add anything, we're at a local optimum
            # Try to swap: remove one point, add another
            # For simplicity, just break and try again with different shuffle
            if time.time() - start_time > timeout - 60:
                break
    
    # Construct N = F_49^2 \ best_S
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = point_to_idx(xa, xb, ya, yb)
                    if idx not in best_S:
                        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
