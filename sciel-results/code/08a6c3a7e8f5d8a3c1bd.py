import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = p^2 and p = 7.
    Uses a Hermitian unital based construction with a flipped exclusion condition
    to maximize the complement size, followed by a local repair step.
    """
    p = 7
    q = p * p
    
    # Smallest quadratic non-residue mod p (the alpha^2 = k constant).
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break
            
    # Precompute squares mod p
    squares = set((u * u) % p for u in range(p))

    # Define the unital U: Trace(x) + Norm(y) = 0
    # Trace((xa, xb)) = xa + x(-a) ... wait.
    # In F_{p^2} = F_p[alpha], alpha^2 = k.
    # Conjugate of a + b*alpha is a - b*alpha.
    # Trace(a + b*alpha) = (a + b*alpha) + (a - b*alpha) = 2a.
    # Norm(a + b*alpha) = (a + b*alpha)(a - b*alpha) = a^2 - b^2*alpha^2 = a^2 - b^2*k.
    
    # The unital equation is Trace(x) + Norm(y) = 0.
    # 2*xa + ya^2 - yb^2*k = 0 mod p.
    
    # We want to construct S (the complement of N).
    # Original: Exclude if in U AND y NOT in Y_special (where Y_special was y_a in squares).
    # New: Exclude if in U AND y IN Y_special (where Y_special is y_a in squares).
    
    # Let's represent points as indices 0..2400 for fast lookup.
    # index = ((xa * p + xb) * p + ya) * p + yb
    
    # 1. Identify candidate points for S
    # S starts as empty. We will add points that are "excluded" from N.
    # The algebraic rule says: if point is in U and y_a is a square, it is in S.
    # Otherwise, it is in N (initially).
    
    s_set = set()
    
    for xa in range(p):
        for xb in range(p):
            # xb does not affect Trace or Norm, so for a fixed xa, ya, yb,
            # the condition is independent of xb.
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    y_is_square = ya in squares
                    
                    if in_unital and y_is_square:
                        # This point is in S
                        for xb_val in range(p):
                            idx = ((xa * p + xb_val) * p + ya) * p + yb
                            s_set.add(idx)
                    # Else, point is in N (not in S)

    # 2. Local Repair / Optimization
    # We want to maximize |S| subject to:
    # For every x in S, there exists a line L containing x such that L \cap S = {x}.
    
    # Helper to get lines through a point
    # Lines are defined by direction v=(1, m) or v=(0,1).
    # There are q + 1 = 50 lines through each point.
    
    # To check if x is isolated in S:
    # For each of the 50 lines L through x, check if (L \ {x}) \cap S is empty.
    # If yes for any L, x is isolated.
    
    # We will iterate and remove points that are not isolated.
    # This is a greedy approach.
    
    # Precompute lines for all points?
    # 2401 points * 50 lines * 49 points/line = ~6M operations. Feasible.
    
    # Let's create a list of points in S.
    s_list = list(s_set)
    s_bool = np.zeros(q*q, dtype=bool)
    for idx in s_list:
        s_bool[idx] = True
        
    # Function to check if a point index is isolated in S
    def is_isolated(idx):
        xa = (idx // (p*p)) % p
        xb = (idx // p) % p
        ya = (idx // 1) % p # Wait, index calculation:
        # idx = ((xa * p + xb) * p + ya) * p + yb
        # yb = idx % p
        # ya = (idx // p) % p
        # xb = (idx // (p*p)) % p
        # xa = (idx // (p*p*p)) % p
        
        xa = idx // (p**3)
        xb = (idx // (p**2)) % p
        ya = (idx // p) % p
        yb = idx % p
        
        # Check vertical line v=(0,1): points (xa, xb, ya, yb) with varying yb?
        # No, line is {x + t*v}.
        # v=(0,1) means (0, 0, 0, 1)? No.
        # Point is [[xa, xb], [ya, yb]].
        # v = (vx_a, vx_b, vy_a, vy_b).
        # v=(0,1) in F_q^2 usually means the vector (0, 1) in the field coordinates.
        # So vx = 0, vy = 1.
        # Points: (xa, xb), (ya + t, yb).
        # So xa, xb are constant. ya, yb vary.
        
        # Line 1: Vertical (vx=0, vy=1)
        # Points: (xa, xb, ya+t, yb) for t in F_q.
        # We need to check if any other point in S is on this line.
        # A point (xa', xb', ya', yb') is on this line if xa'=xa, xb'=xb, and (ya', yb') = (ya, yb) + t*(0, 1).
        # This implies xa'=xa, xb'=xb. And (ya', yb') is in the affine line defined by (ya, yb) with direction (0,1).
        # Since direction is (0,1) in F_q, this means ya' = ya + t*0 = ya, and yb' = yb + t*1.
        # So ya' must be ya. And yb' can be anything?
        # Wait. t is in F_q.
        # (ya, yb) + t*(0, 1) = (ya + t*0, yb + t*1) = (ya, yb+t).
        # So the line consists of all points (xa, xb, ya, yb') for yb' in F_q.
        # So we just need to check if there is any yb' != yb such that (xa, xb, ya, yb') is in S.
        
        # If no other point in S has same xa, xb, ya, then vertical line isolates x.
        
        # Check vertical line
        isolated = True
        for yb_check in range(p):
            if yb_check == yb: continue
            idx_check = ((xa * p + xb) * p + ya) * p + yb_check
            if s_bool[idx_check]:
                isolated = False
                break
        
        if isolated:
            return True
            
        # Check other lines: v = (1, m) for m in F_q.
        # Points: (xa, xb) + t*(1, m) = (xa + t, xb + t*m_a), (ya + t*m_b, yb + t*m_b*alpha)?
        # Let's be precise with field arithmetic.
        # x = (xa, xb), y = (ya, yb).
        # v = (1, m) where m = (ma, mb).
        # x + t*v = (xa + t*1, xb + t*ma), (ya + t*ma, yb + t*mb)?
        # No. v is a vector in F_q^2. v = (v1, v2).
        # v1 = 1 = (1, 0). v2 = m = (ma, mb).
        # t = (ta, tb).
        # t * v1 = t * 1 = t = (ta, tb).
        # t * v2 = t * m.
        # So x_new = x + t*v1 = (xa + ta, xb + tb).
        # y_new = y + t*v2 = (ya + ta*ma - tb*mb*k, yb + ta*mb + tb*ma).
        
        # This is complex to iterate.
        # Alternative: Precompute for each point, the 50 lines it belongs to, and for each line, the set of points on it.
        # Then for a point x, check each line L containing x.
        # If L \cap S == {x}, then x is isolated.
        
        # Given the small size (2401 points), we can afford to be a bit slow in the check if we do it smartly.
        # But we might need to check this for many points.
        
        # Let's precompute lines.
        # There are q*(q+1) lines.
        # Each line has q points.
        # Total incidences: q^2(q+1) = 2401 * 50 = 120050.
        
        # We'll create a dictionary: line_id -> set of point indices.
        # And for each point, list of line_ids.
        
        pass # Placeholder for logic, will implement below

    # Precomputation of lines
    point_lines = [[] for _ in range(q*q)]
    line_points = {}
    line_id = 0
    
    # Vertical lines: v=(0,1)
    # Defined by xa, xb, ya. yb varies.
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                pts = []
                for yb in range(p):
                    idx = ((xa * p + xb) * p + ya) * p + yb
                    pts.append(idx)
                line_points[line_id] = pts
                for idx in pts:
                    point_lines[idx].append(line_id)
                line_id += 1
                
    # Sloped lines: v=(1, m)
    # m = (ma, mb)
    for ma in range(p):
        for mb in range(p):
            # Line defined by point (x0, y0) and direction (1, m).
            # To avoid duplicates, we fix a "base" point.
            # A line with direction (1, m) intersects the "x=0" line (xa=0, xb=0) at a unique point?
            # If v1=1, then as t varies, xa changes.
            # We can identify the line by its intersection with the line xa=0, xb=0.
            # Let the base point be (0, 0, yb0).
            # Then the line is { (t, t*m, yb0 + t*m) }?
            # Let's just iterate over all "base" points on the line xa=0, xb=0.
            # There are p^2 such points (ya, yb).
            for ya0 in range(p):
                for yb0 in range(p):
                    # Base point: xa=0, xb=0, ya=ya0, yb=yb0
                    pts = []
                    for ta in range(p):
                        for tb in range(p):
                            # t = ta + tb*alpha
                            # x = (0,0) + t*(1,0) = t = (ta, tb)
                            xa_new = ta
                            xb_new = tb
                            
                            # y = (ya0, yb0) + t*(ma, mb)
                            # t*m = (ta + tb*alpha)(ma + mb*alpha)
                            # = ta*ma + ta*mb*alpha + tb*ma*alpha + tb*mb*alpha^2
                            # = ta*ma + tb*mb*k + (ta*mb + tb*ma)*alpha
                            ya_new = (ya0 + ta*ma + tb*mb*k) % p
                            yb_new = (yb0 + ta*mb + tb*ma) % p
                            
                            idx = ((xa_new * p + xb_new) * p + ya_new) * p + yb_new
                            pts.append(idx)
                    
                    line_points[line_id] = pts
                    for idx in pts:
                        point_lines[idx].append(line_id)
                    line_id += 1

    # Now, repair S
    # We maintain s_bool.
    # Iterate through points in S. If not isolated, remove it.
    # To be efficient, we can just do a few passes.
    
    changed = True
    while changed:
        changed = False
        for idx in list(s_set):
            if not s_bool[idx]:
                continue
            
            # Check if idx is isolated
            isolated = False
            for lid in point_lines[idx]:
                # Check if any other point in line_points[lid] is in S
                has_other = False
                for p_idx in line_points[lid]:
                    if p_idx == idx:
                        continue
                    if s_bool[p_idx]:
                        has_other = True
                        break
                if not has_other:
                    isolated = True
                    break
            
            if not isolated:
                # Remove idx from S
                s_set.remove(idx)
                s_bool[idx] = False
                changed = True

    # Optional: Try to add points back?
    # Given time constraints, we stop here.
    
    # Convert s_set to array
    if not s_set:
        # Should not happen, but just in case
        return np.zeros((1, 2, 2), dtype=np.int64)
        
    points = []
    for idx in s_set:
        xa = idx // (p**3)
        xb = (idx // (p**2)) % p
        ya = (idx // p) % p
        yb = idx % p
        points.append([[xa, xb], [ya, yb]])
        
    return np.array(points, dtype=np.int64)
