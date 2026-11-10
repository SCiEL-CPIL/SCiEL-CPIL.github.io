import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by greedily maximizing the complement set S.
    Uses integer-based field representation for speed.
    """
    p = 7
    q = p * p  # 49
    n_points = q * q  # 2401
    n_lines = q * q + q  # 2450 (49 vertical + 2401 sloped)
    
    # Field operations for F_49 = F_7[alpha], alpha^2 = 3
    # Element is integer 0..48 representing a + b*alpha where a = idx // 7, b = idx % 7
    
    # Precompute multiplication table
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ia, ib = i // p, i % p
        for j in range(q):
            ja, jb = j // p, j % p
            # (ia + ib*alpha) * (ja + jb*alpha) = (ia*ja + 3*ib*jb) + (ia*jb + ib*ja)*alpha
            ra = (ia * ja + 3 * ib * jb) % p
            rb = (ia * jb + ib * ja) % p
            mul_table[i, j] = ra * p + rb
    
    # Addition is just (i + j) % q since we're adding a+b*alpha
    # Actually, (a1 + b1*alpha) + (a2 + b2*alpha) = (a1+a2) + (b1+b2)*alpha
    # So in integer form: ((i//7 + j//7) % 7) * 7 + ((i%7 + j%7) % 7)
    # But since i, j < 49, and a, b < 7, we can just do (i + j) % 49? No.
    # i = a1*7 + b1, j = a2*7 + b2
    # i + j = (a1+a2)*7 + (b1+b2)
    # If b1+b2 >= 7, we need to carry. But (a1+a2) and (b1+b2) are each < 14.
    # So (i+j) % 49 is NOT the same as field addition.
    # Let's precompute add table too.
    add_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ia, ib = i // p, i % p
        for j in range(q):
            ja, jb = j // p, j % p
            ra = (ia + ja) % p
            rb = (ib + jb) % p
            add_table[i, j] = ra * p + rb
    
    # Precompute lines
    # Line index: 0..48 are vertical (x = fixed value)
    # Line index: 49..2448 are sloped (y = m*x + c)
    # For vertical line i (0..48): points are (i, y) for all y in 0..48
    # For sloped line with m, c: points are (x, m*x + c) for all x in 0..48
    
    # Point index: x * q + y where x, y in 0..48
    # So point (x, y) has index x * 49 + y
    
    # Build lines as arrays of point indices
    lines = []
    
    # Vertical lines
    for x_val in range(q):
        line_pts = np.array([x_val * q + y_val for y_val in range(q)], dtype=np.int32)
        lines.append(line_pts)
    
    # Sloped lines
    for m in range(q):
        for c in range(q):
            line_pts = np.empty(q, dtype=np.int32)
            for x_val in range(q):
                y_val = add_table[mul_table[m, x_val], c]
                line_pts[x_val] = x_val * q + y_val
            lines.append(line_pts)
    
    # Precompute point_lines: for each point, list of line indices passing through it
    point_lines = [[] for _ in range(n_points)]
    for li, line_pts in enumerate(lines):
        for pt in line_pts:
            point_lines[pt].append(li)
    
    # Convert to numpy arrays for faster access
    point_lines_arrays = []
    for pl in point_lines:
        point_lines_arrays.append(np.array(pl, dtype=np.int32))
    
    # Initialize
    S = np.zeros(n_points, dtype=bool)
    # witness_count[y] = number of lines L through y such that L \ {y} is disjoint from S
    # Initially S is empty, so every line is a witness for every point on it.
    # Each point is on q+1 = 50 lines.
    witness_count = np.full(n_points, q + 1, dtype=np.int32)
    
    # line_s_count[li] = number of points in S that are on line li
    line_s_count = np.zeros(n_lines, dtype=np.int32)
    # line_s_point[li] = one point in S on line li (if count == 1), else -1
    line_s_point = np.full(n_lines, -1, dtype=np.int32)
    
    start_time = time.time()
    time_limit = 590.0  # Safety margin before 600s limit
    
    # Greedy loop: try to add points to S
    # A point x can be added to S if:
    # 1. x not in S
    # 2. x will have at least one witness line after addition
    #    i.e., there exists a line L through x with line_s_count[L] == 0
    #    (after adding x, L \cap S = {x}, so L is a witness for x)
    # 3. No other point y loses its last witness
    #    For each line L through x:
    #      If line_s_count[L] == 0: all y in L (y != x) lose 1 witness
    #      If line_s_count[L] == 1: the single S point u on L loses 1 witness
    #      If line_s_count[L] >= 2: no one loses a witness
    #    We need witness_count[y] >= 2 for all affected y
    
    # Shuffle points for diversity
    order = np.arange(n_points)
    np.random.shuffle(order)
    
    added = 0
    for x in order:
        if time.time() - start_time > time_limit:
            break
        if S[x]:
            continue
        
        # Check condition 2: does x have a clean line?
        has_clean = False
        x_lines = point_lines_arrays[x]
        for li in x_lines:
            if line_s_count[li] == 0:
                has_clean = True
                break
        
        if not has_clean:
            continue
        
        # Check condition 3: does any point lose its last witness?
        reject = False
        for li in x_lines:
            sc = line_s_count[li]
            if sc == 0:
                # All y in line (y != x) lose 1 witness
                line_pts = lines[li]
                for y in line_pts:
                    if y != x and witness_count[y] < 2:
                        reject = True
                        break
            elif sc == 1:
                u = line_s_point[li]
                if witness_count[u] < 2:
                    reject = True
                    break
            # sc >= 2: no one loses a witness
        
        if reject:
            continue
        
        # Safe to add x
        S[x] = True
        added += 1
        
        # Update state
        for li in x_lines:
            sc = line_s_count[li]
            if sc == 0:
                # x becomes the only S point on this line
                line_s_count[li] = 1
                line_s_point[li] = x
                # All y in line (y != x) lose 1 witness
                line_pts = lines[li]
                for y in line_pts:
                    if y != x:
                        witness_count[y] -= 1
            elif sc == 1:
                u = line_s_point[li]
                line_s_count[li] = 2
                line_s_point[li] = -1
                # u loses 1 witness
                witness_count[u] -= 1
            else:
                line_s_count[li] += 1
                line_s_point[li] = -1
        
        # x itself: its witness count is no longer relevant, but let's set it to 0
        witness_count[x] = 0
    
    # Construct N = complement of S
    N_indices = np.where(~S)[0]
    
    # Convert to array of shape (k, 2, 2)
    k = len(N_indices)
    result = np.zeros((k, 2, 2), dtype=np.int64)
    for i, idx in enumerate(N_indices):
        x_val = idx // q
        y_val = idx % q
        xa, xb = x_val // p, x_val % p
        ya, yb = y_val // p, y_val % p
        result[i, 0, 0] = xa
        result[i, 0, 1] = xb
        result[i, 1, 0] = ya
        result[i, 1, 1] = yb
    
    return result
