import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    # k = smallest quadratic non-residue
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break
            
    # Precompute field elements as indices 0..48
    # (a, b) -> a + b*p
    # Inverse map: index -> (a, b)
    
    # Construct U
    # Equation: 2*xa + ya*ya - yb*yb*k = 0 mod p
    # Points in U
    U_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2*xa + ya*ya - yb*yb*k) % p == 0:
                        U_points.append((xa, xb, ya, yb))
                        
    # Map point to index
    # Total points 2401.
    # Index = ((xa*p + xb)*p + ya)*p + yb
    
    # Lines
    # Directions: (1, m) for m in F_q, and (0, 1)
    # m is in 0..48.
    # Vertical direction is separate.
    
    lines = []
    # For each slope
    # m is an element of F_49.
    # We need to handle m as (ma, mb).
    # v = (1, m).
    # Line: x = x0 + t, y = y0 + t*m.
    # t in F_49.
    
    # Precompute addition and multiplication in F_49
    def add(a, b):
        # a, b are (ua, ub), (va, vb)
        return ((a[0]+b[0])%p, (a[1]+b[1])%p)
        
    def mul(a, b):
        # a, b are (ua, ub), (va, vb)
        # (ua + ub alpha)(va + vb alpha) = ua*va + (ua*vb + ub*va) alpha + ub*vb alpha^2
        # alpha^2 = k
        ra = (a[0]*b[0] + a[1]*b[1]*k) % p
        rb = (a[0]*b[1] + a[1]*b[0]) % p
        return (ra, rb)
        
    def to_idx(pt):
        xa, xb, ya, yb = pt
        return ((xa*p + xb)*p + ya)*p + yb
        
    def from_idx(idx):
        yb = idx % p
        idx //= p
        ya = idx % p
        idx //= p
        xb = idx % p
        idx //= p
        xa = idx % p
        return (xa, xb, ya, yb)

    # Generate lines
    # Direction vectors
    # v = (1, m) where m in F_49.
    # m is (ma, mb).
    # There are 49 such directions.
    # Plus vertical (0, 1).
    
    all_dirs = []
    for ma in range(p):
        for mb in range(p):
            all_dirs.append(((1,0), (ma, mb))) # v_x = (1,0), v_y = (ma, mb)
    all_dirs.append(((0,0), (1,0))) # v_x = (0,0), v_y = (1,0) i.e. (0,1)

    # For each direction, there are 49 lines?
    # No, lines are parallel.
    # For a fixed direction, lines are distinguished by intercept.
    # It's easier to iterate over starting points and directions, but that duplicates lines.
    # Number of lines is 2450.
    
    # Let's generate all unique lines.
    # A line is defined by a point on it and a direction.
    # To avoid duplicates, we can pick the "first" point on the line in some ordering.
    # Or just store set of points for each line.
    
    line_sets = []
    seen_lines = set()
    
    # Iterate over all points as potential "anchors"?
    # Too slow.
    
    # Better:
    # For each direction v:
    #   Find all lines with direction v.
    #   How?
    #   Project points onto normal?
    #   Or just:
    #   For a fixed direction, the lines are disjoint.
    #   Pick one point not yet covered by a line of this direction.
    
    # We have 50 directions.
    # For each direction, we have 49 lines.
    
    # Let's precompute points in a 2D array of indices.
    # P[xa, xb, ya, yb]
    
    # For each direction:
    for v in all_dirs:
        vx, vy = v
        # Find all lines
        visited = [False] * 2401
        for i in range(2401):
            if not visited[i]:
                # Start line at i
                pt_i = from_idx(i)
                line_pts = []
                t = 0
                while t < p*p: # 49 steps
                    # pt = pt_i + t * v
                    # t is (ta, tb)
                    # Iterate t
                    pass
                # This while is messy.
                break
        break
        
    # Simpler:
    # Generate lines by iterating slope and intercept.
    # Affine lines: y = m x + c.
    # m, c in F_49.
    # Vertical: x = c.
    
    lines_pts = []
    
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    pts = []
                    # x in F_49
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            y = add(mul(m, x), c)
                            pts.append(to_idx((x[0], x[1], y[0], y[1])))
                    lines_pts.append(pts)
                    
    # Vertical lines
    for xa in range(p):
        for xb in range(p):
            pts = []
            for ya in range(p):
                for yb in range(p):
                    pts.append(to_idx((xa, xb, ya, yb)))
            lines_pts.append(pts)
            
    # Now we have 2450 lines.
    # Precompute U membership for each line
    U_mask = np.zeros(2401, dtype=bool)
    for pt in U_points:
        U_mask[to_idx(pt)] = True
        
    # line_U_intersects[i] = list of points in line i that are in U
    line_U_pts = []
    for l in lines_pts:
        pts_in_U = [idx for idx in l if U_mask[idx]]
        line_U_pts.append(pts_in_U)
        
    # Precompute points on each line for each point?
    # point_lines[i] = list of line indices containing point i
    point_lines = [[] for _ in range(2401)]
    for i, l in enumerate(lines_pts):
        for idx in l:
            point_lines[idx].append(i)
            
    # Search for best S
    # S is defined by A = set of ya values to exclude.
    # ya is 0..6.
    # 128 subsets.
    
    best_S_size = -1
    best_A = None
    
    start_time = time.time()
    time_limit = 600 # seconds
    
    # Sort subsets by size descending
    subsets = []
    for mask in range(128):
        A = [i for i in range(7) if (mask >> i) & 1]
        subsets.append((len(A), mask, A))
    subsets.sort(key=lambda x: -x[0])
    
    for size, mask, A in subsets:
        if time.time() - start_time > time_limit - 30:
            break
            
        # Construct S mask
        S_mask = np.zeros(2401, dtype=bool)
        for idx in range(2401):
            if U_mask[idx]:
                pt = from_idx(idx)
                ya = pt[2]
                if ya in A:
                    S_mask[idx] = True
                    
        # Check validity
        # witness_exists = [False]*2401
        witness_exists = np.zeros(2401, dtype=bool)
        
        # For each line
        for i, l in enumerate(lines_pts):
            # Count S in line
            # l is list of indices
            s_count = 0
            s_pt = -1
            for idx in l:
                if S_mask[idx]:
                    s_count += 1
                    s_pt = idx
            if s_count == 0:
                # All x not in S get witness
                for idx in l:
                    if not S_mask[idx]:
                        witness_exists[idx] = True
            elif s_count == 1:
                # s_pt gets witness (if s_pt in S, which it is)
                witness_exists[s_pt] = True
            # else: no witness from this line
        
        # Check all
        if np.all(witness_exists):
            if size > best_S_size:
                best_S_size = size
                best_A = A
                # We can break if size is 7?
                if size == 7:
                    break
                # Actually, we want max size, so we go down.
                # But we sorted desc, so first valid is best?
                # Yes.
                break
            
    if best_A is None:
        # Fallback to original
        best_A = [3, 5, 6]
        
    # Construct final S
    S_mask = np.zeros(2401, dtype=bool)
    for idx in range(2401):
        if U_mask[idx]:
            pt = from_idx(idx)
            ya = pt[2]
            if ya in best_A:
                S_mask[idx] = True
                
    # N is complement of S
    N_mask = ~S_mask
    
    # Return points in N
    points = []
    for idx in range(2401):
        if N_mask[idx]:
            pt = from_idx(idx)
            points.append([[pt[0], pt[1]], [pt[2], pt[3]]])
            
    return np.array(points, dtype=np.int64)

