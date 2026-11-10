# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 with q = 49 (p=7) using a vectorized greedy approach.
    We start with a baseline N (small) and try to remove points from N (add to C) 
    while maintaining the Nikodym property: every point x (in N or C) must lie on 
    a line L such that L \ {x} subset N.
    Equivalently, for x in C, for every direction v, there must be t != 0 such that 
    x + t*v is in C.
    """
    p = 7
    q = p * p
    n_points = q * q
    deadline = time.time() + 580  # Safety margin
    
    # Field F_49 = F_7[alpha], alpha^2 = 3
    k_const = 3

    # Precompute field operations using vectorized NumPy where possible, 
    # but for small q=49, a lookup table for multiplication is fast and simple.
    # Elements are 0..48.
    # Map (a, b) to a + b*7
    def to_field(a, b):
        return a + b * p

    def from_field(z):
        return z % p, z // p

    # Precompute multiplication table for F_49
    # mul_table[i, j] = i * j in F_49 (represented as 0..48)
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ia, ib = from_field(i)
        for j in range(q):
            ja, jb = from_field(j)
            # (ia + ib*alpha) * (ja + jb*alpha) = (ia*ja + ib*jb*3) + (ia*jb + ib*ja)*alpha
            ra = (ia * ja + ib * jb * k_const) % p
            rb = (ia * jb + ib * ja) % p
            mul_table[i, j] = to_field(ra, rb)

    # Precompute addition table (or just use mod q? No, field add is not mod q)
    # add_table[i, j] = i + j in F_49
    add_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ia, ib = from_field(i)
        for j in range(q):
            ja, jb = from_field(j)
            ra = (ia + ja) % p
            rb = (ib + jb) % p
            add_table[i, j] = to_field(ra, rb)

    # Points in F_49^2 are pairs (x, y) where x, y in 0..48.
    # Total points: 2401.
    # Index: x * q + y
    
    # Directions:
    # v = (1, m) for m in F_49 (49 dirs)
    # v = (0, 1) (1 dir)
    # Total 50 directions.
    
    # For each direction d, and each line offset c, define the line.
    # Line L_{d, c}: { (x, y) : y - m*x = c } for v=(1,m)
    # Line L_{inf, c}: { (x, y) : x = c } for v=(0,1)
    
    # We need to map each point to the lines it belongs to.
    # For a point (x, y):
    # For each m in F_49, c = y - m*x. This point is on line (m, c).
    # For vertical, c = x. This point is on line (inf, c).
    
    # Let's generate the list of points on each line.
    # There are 50 * 49 = 2450 lines.
    # Each line has 49 points.
    
    # lines[dir_idx, offset_idx] = array of 49 point indices
    # dir_idx: 0..48 for m=0..48, 49 for vertical
    # offset_idx: 0..48
    
    lines = np.zeros((50, q, q), dtype=np.int32) # [dir, offset, pos] -> point_idx
    
    # We also need for each point, the list of (dir, offset, pos) it is part of.
    # point_lines[point_idx] = list of (dir, offset, pos)
    # Since each point is on exactly 50 lines (one for each dir), 
    # point_lines can be a 2D array of shape (n_points, 50, 3) or similar.
    # Let's use point_dir_offset_pos[point_idx, dir_idx] = (offset, pos)
    # Actually, let's just store the line ID and position.
    # line_id = dir * q + offset.
    
    point_line_ids = np.zeros((n_points, 50), dtype=np.int32)
    point_line_pos = np.zeros((n_points, 50), dtype=np.int32)
    
    # Fill lines and point mappings
    # Iterate through points
    for x in range(q):
        for y in range(q):
            pt_idx = x * q + y
            
            # Vertical line: dir=49, offset=x
            # We need to know the position 'pos' of this point on the line.
            # The line consists of points (x, y') for y' in F_49.
            # Let's order points by y'.
            # pos = y
            
            lines[49, x, y] = pt_idx
            point_line_ids[pt_idx, 49] = 49 * q + x
            point_line_pos[pt_idx, 49] = y
            
            # Sloped lines: dir=0..48 (m=0..48)
            for m in range(q):
                # c = y - m*x
                c = (y - mul_table[m, x]) % q # Wait, field subtraction is y + (-m*x).
                # In F_49, -1 is 48 (since 1+48=49=0 mod 7? No, 1+6=0 mod 7, so -1=6 in F7. 
                # In F49, -1 is (6,0) = 6.
                # Let's compute c properly using add_table.
                # c = y + (-m)*x. 
                # (-m) = q - m if m!=0 else 0.
                neg_m = (q - m) % q
                c = add_table[y, mul_table[neg_m, x]]
                
                # The line is { (x', y') : y' = m*x' + c }
                # Order points by x'.
                # pos = x' = x
                lines[m, c, x] = pt_idx
                point_line_ids[pt_idx, m] = m * q + c
                point_line_pos[pt_idx, m] = x

    # Now we have lines[dir, offset, pos] -> pt_idx
    # And for each pt_idx and dir, we know the line_id and pos.
    
    # Baseline Construction:
    # Start with a valid Nikodym set N.
    # A simple valid N is the set of all points.
    # But we want to maximize C (complement).
    # So start with N = all points, C = empty.
    # Or start with a known small N and expand C.
    # The prompt suggests starting with a baseline and improving.
    # Let's start with C = empty.
    
    in_C = np.zeros(n_points, dtype=bool)
    
    # Greedy Expansion of C:
    # A point p can be added to C if for every direction d, there exists a point q on the line L(p, d) 
    # such that q != p and q is in C.
    
    # To vectorize this:
    # For each point p, we need to check 50 conditions.
    # Condition d: exists q in Line(p, d) \ {p} such that in_C[q] is True.
    
    # We can precompute for each line, the mask of points in C.
    # line_in_C[dir, offset, :] is a boolean array of length q.
    
    # Iteration:
    # 1. Compute line_in_C from in_C.
    #    line_in_C[dir, offset, pos] = in_C[ lines[dir, offset, pos] ]
    # 2. For each point p and direction d:
    #    line_id = point_line_ids[p, d]
    #    pos = point_line_pos[p, d]
    #    dir = line_id // q
    #    offset = line_id % q
    #    line_mask = line_in_C[dir, offset, :]
    #    # Exclude p itself
    #    line_mask_no_p = line_mask.copy()
    #    line_mask_no_p[pos] = False
    #    if np.any(line_mask_no_p): condition met.
    
    # This is still O(N * D * Q) per iteration if done naively in Python.
    # We can vectorize the "exists" check.
    
    # For a specific direction d:
    # For all points p, we want to know if their line has any other point in C.
    # Let's process direction by direction.
    
    # can_add = np.ones(n_points, dtype=bool)
    # for d in range(50):
    #    # For each line in direction d, check if any point in C is on it.
    #    # If a line has NO points in C, then NO point on that line can be added to C (for this direction).
    #    # If a line HAS points in C, then any point on that line (except the ones in C themselves? No, 
    #    # if p is in C, it's already added. We are looking for p NOT in C).
    #    # Wait, if p is NOT in C, and the line has SOME point q in C (q != p), then condition is met.
    #    # If the line has NO points in C, condition fails.
    #    # If the line has ONLY p in C? But p is not in C. So if line has any point in C, it's fine.
    #    
    #    line_has_C = np.any(line_in_C[d, :, :], axis=2) # Shape (q,)
    #    # For each point p with direction d:
    #    # line_id = point_line_ids[p, d]
    #    # if not line_has_C[line_id % q]: can_add[p] = False
    #    
    #    # Vectorized:
    #    # line_ids = point_line_ids[:, d] # Shape (N,)
    #    # offsets = line_ids % q
    #    # valid_offsets = line_has_C[offsets]
    #    # can_add &= valid_offsets
    
    # This logic works!
    # If a line has at least one point in C, then for ANY point p on that line (not in C), 
    # there exists q in C on the line.
    # Note: If p is in C, we don't care about adding it.
    # So:
    # 1. Compute line_has_C for all lines.
    # 2. For each direction d, check if the line of each point p has C.
    # 3. Intersection over all d.
    # 4. Also, p must not be in C.
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
            
        # 1. Compute line_has_C
        # line_in_C is boolean array (50, q, q)
        # line_in_C[d, o, pos] = in_C[lines[d, o, pos]]
        line_in_C = in_C[lines] # Broadcasting: in_C[lines] where lines is (50, q, q) int array.
        # This creates a boolean array (50, q, q).
        
        line_has_C = np.any(line_in_C, axis=2) # Shape (50, q)
        
        # 2. Determine which points can be added
        # can_add[p] = True if for all d, line_has_C[d, offset(p, d)] is True.
        
        # Initialize can_add as True for all points not in C.
        can_add = ~in_C
        
        # Check each direction
        for d in range(50):
            # Get the offset for each point for this direction
            line_ids = point_line_ids[:, d] # Shape (N,)
            offsets = line_ids % q # Shape (N,)
            
            # Check if the line for each point has any point in C
            # line_has_C[d] is shape (q,)
            # We need line_has_C[d, offsets]
            valid = line_has_C[d, offsets] # Shape (N,)
            
            can_add &= valid
            
        # 3. Add all points in can_add to C
        if np.any(can_add):
            in_C[can_add] = True
            changed = True
        else:
            changed = False

    # Build N = all points - C
    # N indices are where in_C is False
    N_indices = np.where(~in_C)[0]
    
    # Convert indices to coordinates
    # idx = x * q + y
    # x = idx // q
    # y = idx % q
    # x = xa + xb * 7
    # y = ya + yb * 7
    
    x_coords = N_indices // q
    y_coords = N_indices % q
    
    xa = x_coords % p
    xb = x_coords // p
    ya = y_coords % p
    yb = y_coords // p
    
    # Construct array (k, 2, 2)
    k = len(N_indices)
    points = np.zeros((k, 2, 2), dtype=np.int64)
    points[:, 0, 0] = xa
    points[:, 0, 1] = xb
    points[:, 1, 0] = ya
    points[:, 1, 1] = yb
    
    return points

# EVOLVE-BLOCK-END
