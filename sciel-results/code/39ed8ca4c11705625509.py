# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Returns the complement of a subset S of the Hermitian unital U.
    The metric is the size of the complement, so we want to maximize |S|.
    
    Strategy:
    1. Start with a base S (points in U with y_a in {0, 3, 5, 6}).
    2. Use a greedy algorithm to add more points to S from U, verifying validity at each step.
    3. Validity check is optimized using line coverage counts.
    """
    start_time = time.time()
    deadline = 600  # seconds
    
    p = 7
    q = p * p
    k = 3  # alpha^2 = 3

    # Field operations
    def add_f49(u, v):
        return ((u[0] + v[0]) % p, (u[1] + v[1]) % p)
    
    def mul_f49(u, v):
        a, b = u
        c, d = v
        return ((a*c + b*d*k) % p, (a*d + b*c) % p)

    # Generate points
    points = []
    point_to_idx = {}
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    pt = ((xa, xb), (ya, yb))
                    idx = len(points)
                    points.append(pt)
                    point_to_idx[pt] = idx

    n_points = len(points)

    # Directions: (1, m) for m in F_49, and (0, 1)
    directions = []
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 1), (0, 0)))
    
    n_dirs = len(directions)

    # Precompute lines
    # lines: list of sets of point indices
    # point_to_lines: list of list of line indices
    lines = [set() for _ in range(n_points * n_dirs)]
    point_to_lines = [[] for _ in range(n_points)]
    
    # We need to map (point, dir) to a unique line ID.
    # A line is defined by point and direction. But (p, d) and (p+d, d) define the same line.
    # We can normalize: for each line, pick the "first" point in some order?
    # Or just use a dictionary: line_key -> line_id
    # line_key can be (point_idx, dir_idx). But this creates duplicates.
    
    # Better: Iterate over all points and directions, generate the set of points on the line.
    # Use a frozenset of point indices as the key? That's expensive.
    
    # Alternative:
    # For each direction d, the lines are parallel classes.
    # There are q lines in each class.
    # Total lines = q * (q+1) = 49 * 50 = 2450.
    
    # Let's restructure.
    # For each direction index di:
    #   For each "offset" (representing the line in the parallel class):
    #     Find the points on this line.
    
    # How to find points on a line with direction d = (dx, dy)?
    # If dx != 0 (i.e., direction (1, m)), we can parametrize by x.
    # But x is in F_49.
    # Actually, it's easier to just generate the line from a representative point.
    
    # Let's stick to the dictionary approach but optimize it.
    # line_id_map: frozenset of point indices -> line_id? No, too slow to hash sets of 49 ints repeatedly.
    
    # Let's use the fact that a line is determined by a point and a direction.
    # We can assign an ID to each (point, direction) pair, but then merge duplicates.
    
    # Efficient way:
    # For each direction d:
    #   For each point P:
    #     Generate line L = {P + t*d}.
    #     Find the "canonical" point on L (e.g., min index).
    #     Use (canonical_point, d) as key.
    
    lines = []
    line_id_map = {}
    point_to_lines = [[] for _ in range(n_points)]
    
    for di, d in enumerate(directions):
        dx, dy = d
        for p_idx in range(n_points):
            p = points[p_idx]
            # Generate line points
            line_pts = []
            for t in range(q):
                # t is an element of F_49. Represented as (ta, tb).
                # We need to iterate t over F_49.
                # t = (ta, tb).
                # This is O(q) per line, O(q^2) per direction, O(q^3) total.
                # 49^3 = 117k. This is fine.
                pass
            
            # Optimization:
            # We can just compute the set of points.
            # But we need to do this efficiently.
            
            # Let's precompute the list of t values in F_49.
            # t_values = [(ta, tb) for ta in range(p) for tb in range(p)]
            
            # Actually, we can just iterate ta, tb.
            
            # For a given direction (dx, dy) and point (px, py):
            # Line points: (px + t*dx, py + t*dy)
            
            # Let's just generate the set.
            line_set = set()
            for ta in range(p):
                for tb in range(p):
                    t = (ta, tb)
                    # t * d
                    # t * (dx, dy) = (t*dx, t*dy)
                    # t*dx = (ta, tb) * (dx_a, dx_b)
                    # Note: d is ((1,0) or (0,1), m)
                    # If d = (1, m):
                    #   x' = px + t*1 = px + t
                    #   y' = py + t*m
                    # If d = (0, 1):
                    #   x' = px
                    #   y' = py + t
            
            # This is getting complicated to inline.
            # Let's use a helper to get line points.
            
            # For speed, let's precompute for each direction and point, the line points?
            # No, just compute on the fly for the canonical check.
            
            # To find canonical point:
            # Compute all points on the line.
            # Find min index.
            
            # This is O(q) per line.
            # Total lines ~ 2450.
            # 2450 * 49 = 120k operations.
            
            # Let's implement it.
            
            line_pts_indices = []
            for ta in range(p):
                for tb in range(p):
                    t = (ta, tb)
                    # Add t*d to p
                    # d is (dx, dy)
                    # t*d = (t*dx, t*dy)
                    # t*dx:
                    #   if dx = (1,0): t*(1,0) = (ta, tb)
                    #   if dx = (0,1): t*(0,1) = (0,0)
                    # t*dy:
                    #   dy = (ma, mb)
                    #   t*dy = (ta*ma + tb*mb*k, ta*mb + tb*ma)
                    
                    if dx == (1, 0):
                        # t*dx = t
                        tx_a, tx_b = ta, tb
                    else:
                        # t*dx = 0
                        tx_a, tx_b = 0, 0
                    
                    # t*dy
                    ma, mb = dy
                    ty_a = (ta*ma + tb*mb*k) % p
                    ty_b = (ta*mb + tb*ma) % p
                    
                    # New point: p + (tx, ty)
                    px, py = p
                    nx_a = (px[0] + tx_a + ty_a) % p  # Wait, t*d is vector.
                    # Vector addition: (px, py) + (t*dx, t*dy)
                    # x_new = px + t*dx
                    # y_new = py + t*dy
                    # x_new = (px_a + tx_a, px_b + tx_b)
                    # y_new = (py_a + ty_a, py_b + ty_b)
                    
                    nx = ((px[0] + tx_a) % p, (px[1] + tx_b) % p)
                    ny = ((py[0] + ty_a) % p, (py[1] + ty_b) % p)
                    
                    npoint = (nx, ny)
                    nidx = point_to_idx[npoint]
                    line_pts_indices.append(nidx)
            
            # Find canonical index
            can_idx = min(line_pts_indices)
            key = (can_idx, di)
            
            if key not in line_id_map:
                line_id_map[key] = len(lines)
                lines.append(set(line_pts_indices))
            
            lid = line_id_map[key]
            point_to_lines[p_idx].append(lid)
            
            # Check time
            if time.time() - start_time > deadline - 60:
                # Fallback to simple construction
                return _fallback_nikodym_set()

    n_lines = len(lines)
    
    # Initialize S
    # Base S: points in U with y_a in {0, 3, 5, 6}
    S = set()
    in_U = [False] * n_points
    
    for idx, pt in enumerate(points):
        xa, xb = pt[0]
        ya, yb = pt[1]
        val = (2 * xa + ya * ya - yb * yb * k) % p
        if val == 0:
            in_U[idx] = True
            if ya in {0, 3, 5, 6}:
                S.add(idx)
    
    # Initialize line counts
    # line_s_count[l] = number of points in S on line l
    line_s_count = [0] * n_lines
    
    for s_idx in S:
        for lid in point_to_lines[s_idx]:
            line_s_count[lid] += 1
    
    # Check validity
    def is_valid(S, line_s_count, point_to_lines, in_S):
        # in_S is a boolean array or set
        # For each point x:
        #   If x in S: need line L through x with line_s_count[L] == 1
        #   If x not in S: need line L through x with line_s_count[L] == 0
        
        for x_idx in range(n_points):
            found = False
            for lid in point_to_lines[x_idx]:
                if x_idx in S:
                    if line_s_count[lid] == 1:
                        found = True
                        break
                else:
                    if line_s_count[lid] == 0:
                        found = True
                        break
            if not found:
                return False
        return True

    # Convert S to set for fast lookup in is_valid
    S_set = S
    
    # Verify base S
    if not is_valid(S_set, line_s_count, point_to_lines, S_set):
        # Base is invalid? Should not happen.
        # Try to fix or fallback.
        pass

    # Candidates: points in U not in S
    candidates = []
    for idx in range(n_points):
        if in_U[idx] and idx not in S:
            candidates.append(idx)
    
    # Sort candidates? Maybe by y_a or something.
    # Let's just iterate.
    
    for c_idx in candidates:
        # Check time
        if time.time() - start_time > deadline - 60:
            break
            
        # Add c_idx to S
        S.add(c_idx)
        
        # Update line counts
        for lid in point_to_lines[c_idx]:
            line_s_count[lid] += 1
            
        # Check validity
        if is_valid(S, line_s_count, point_to_lines, S):
            # Keep
            pass
        else:
            # Revert
            S.remove(c_idx)
            for lid in point_to_lines[c_idx]:
                line_s_count[lid] -= 1
    
    # Construct result N = F_49^2 \ S
    points_N = []
    for idx in range(n_points):
        if idx not in S:
            pt = points[idx]
            xa, xb = pt[0]
            ya, yb = pt[1]
            points_N.append([[xa, xb], [ya, yb]])
    
    return np.array(points_N, dtype=np.int64)

def _fallback_nikodym_set():
    # Simple construction: N = complement of S where S = points in U with y_a in {3, 5, 6}
    p = 7
    k = 3
    points_N = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    val = (2 * xa + ya * ya - yb * yb * k) % p
                    if val == 0:
                        if ya in {3, 5, 6}:
                            continue
                    points_N.append([[xa, xb], [ya, yb]])
    return np.array(points_N, dtype=np.int64)

# EVOLVE-BLOCK-END
