# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a greedy local search strategy.
    
    1. Start with a known valid Nikodym set based on the Hermitian unital construction.
    2. Represent the set N as a boolean array of size 2401 (True if point is in N).
    3. Iteratively try to remove points from N (add to complement S) if the Nikodym 
       property is preserved.
    4. Return the resulting set N as an array of coordinates.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, keep safety margin
    
    p = 7
    q = p * p  # 49
    
    # Field F_49 = F_7[alpha] with alpha^2 = 3
    # Element is (a, b) representing a + b*alpha
    # Multiplication: (a, b) * (c, d) = (ac + bd*3, ad + bc) mod 7
    
    # Precompute field operations for speed
    # Indexing: idx = a * p + b
    # We need to check lines. A line is defined by a point (x0, y0) and a slope m.
    # Points on line: (x0 + t, y0 + m*t) for t in F_49.
    # Vertical line: x = x0, points (x0, y) for y in F_49.
    
    # Total points = 2401
    total_points = q * q
    
    # Map (xa, xb, ya, yb) to index 0..2400
    # idx = ((xa * p + xb) * q) + (ya * p + yb)
    
    # 1. Generate the base Nikodym set (Unital construction)
    # Unital U: {(x, y) : Tr(x) + N(y) = 0}
    # Tr(x) = x + x^7 = 2*xa (since x in F_49, x^7 = x_bar = xa - xb*alpha, so Tr = 2*xa)
    # N(y) = y * y^7 = y * y_bar = ya^2 - yb^2 * 3
    # Condition: 2*xa + ya^2 - 3*yb^2 == 0 (mod 7)
    
    # Blocking set B: points in U where y is in Y = {(u^2, v) : u, v in F_7}
    # Y is the set of y-coordinates whose real part is a quadratic residue mod 7.
    # Squares mod 7: 0, 1, 2, 4
    squares = {0, 1, 2, 4}
    
    # in_N[i] is True if point i is in the Nikodym set N
    in_N = np.ones(total_points, dtype=bool)
    
    # Iterate through all points to initialize in_N based on the unital construction
    # A point is EXCLUDED (not in N) if it is in U and its y is NOT in Y
    # So it is IN N if: (Not in U) OR (In U AND y in Y)
    
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Check if in Unital
                    val = (2 * xa + ya * ya - 3 * yb * yb) % p
                    in_unital = (val == 0)
                    
                    # Check if y is in Y (ya is a square)
                    in_y = (ya in squares)
                    
                    if in_unital and not in_y:
                        # This point is excluded from N
                        idx = ((xa * p + xb) * q) + (ya * p + yb)
                        in_N[idx] = False
                    else:
                        in_N[idx] = True
                        
    # Helper function to check if a specific point x (index) has a valid line
    # For point x, we need to find a direction v such that all other points on the line are in N.
    # Directions: (1, m) for m in F_49, and (0, 1) for vertical.
    # If we remove a point p from N, we only need to re-check lines that contain p.
    # However, a simpler greedy approach:
    # Try to remove each point from N. If valid, keep it removed.
    # To make this efficient, we can precompute for each point which lines it belongs to.
    # But 2401 points * 50 lines/point = 120k lines. Checking validity of a line is O(49).
    # Total work per removal attempt: ~50 * 49 = 2450 ops.
    # 2401 points * 2450 = ~6M ops per full pass. Very fast.
    
    # Let's precompute the lines.
    # Line defined by (x0, y0) and slope m (or vertical).
    # For a point to be "covered", there must exist at least one line through it where all *other* points are in N.
    # Note: The point itself is NOT required to be in N for the line to be valid?
    # "whose points other than x all belong to N". Yes, x itself can be outside N.
    
    # We need to check validity for ALL points in F_49^2, not just those in N.
    # So if we remove a point from N, we must ensure that for EVERY point z in F_49^2, 
    # there is still a valid line through z.
    # This is expensive to check from scratch.
    
    # Alternative: Only remove points that are "redundant".
    # A point p in N is redundant if removing it doesn't invalidate any line for any point.
    # Actually, removing p from N only affects the validity of lines that contain p.
    # For a point z, if its "witness" line was the one containing p, and p was the only point 
    # making that line valid? No, the condition is "points OTHER than x".
    # So if p is on the line through z, p is one of the "other points". 
    # If p is removed from N, the line through z containing p is no longer valid 
    # (unless there's another valid line for z).
    
    # So: To check if we can remove point p:
    # 1. Mark p as not in N.
    # 2. For every point z that shares a line with p (i.e., for every line L containing p, 
    #    for every z on L), check if z still has at least one valid line.
    #    Note: z can be any point in F_49^2.
    #    Lines containing p: 50 lines (49 slopes + vertical).
    #    Points on each line: 49 points.
    #    So we need to check validity for 50 * 49 = 2450 points z.
    #    For each z, we check its 50 lines.
    #    Total checks: 2450 * 50 = 122,500 line checks.
    #    Each line check is O(49) to verify all points are in N.
    #    Total ops: 122,500 * 49 = ~6M ops.
    #    This is feasible per point removal attempt.
    #    2401 points * 6M = 14.4B ops. Too slow in Python.
    
    # Optimization: 
    # We don't need to check ALL z. We only need to check z for which p was the "critical" point?
    # No, p is just one point on the line.
    # Let's use a faster validation method.
    # Precompute for each point z, the number of valid lines through it.
    # If count[z] > 0, z is covered.
    # When we remove p from N:
    # - For each line L containing p:
    #   - For each point z on L:
    #     - If L was valid for z, it might no longer be valid.
    #     - We need to decrement count[z] if L was valid.
    #     - But L is valid for z iff all points on L (except z) are in N.
    #     - Before removal, L was valid for z iff all points on L (except z) were in N.
    #     - After removal of p, L is valid for z iff all points on L (except z, p) are in N.
    #     - So if p != z, and L was valid before, it is invalid now IF p was the only point 
    #       that made it valid? No, the condition is "ALL other points in N".
    #       If p is in N, and we remove it, the condition "all other points in N" fails 
    #       because p is not in N.
    #       So ANY line L containing p (and z != p) that was valid for z BEFORE 
    #       becomes INVALID for z AFTER, because p is now missing from N.
    #       Wait, is that true?
    #       Line L through z. Points on L: {z, p, ...}.
    #       Condition for L to be a witness for z: {w in L | w != z} subset N.
    #       If p in L and p != z, then p is in {w | w != z}.
    #       If p is removed from N, then p is not in N.
    #       So the subset condition fails.
    #       Therefore, removing p from N INVALIDATES every line L that contains p and some other point z.
    #       This means for every z on any line through p (z != p), we lose one potential witness line.
    #       If z had only that one witness line, z becomes uncovered.
    
    # So the algorithm:
    # 1. Maintain `valid_lines_count[z]` = number of lines through z that are currently valid.
    # 2. Initially compute this for all z.
    # 3. To try removing p:
    #    a. For each of the 50 lines L through p:
    #       - Check if L is currently valid for any z on L.
    #       - Actually, if L is valid for z, it means all points on L except z are in N.
    #       - If we remove p, L is no longer valid for ANY z on L (z != p).
    #       - So for each z on L (z != p), decrement valid_lines_count[z].
    #       - If any valid_lines_count[z] becomes 0, removal is invalid.
    #    b. If no count becomes 0, commit the removal.
    
    # This is much faster!
    # Complexity per removal attempt:
    # 50 lines * 49 points/line = 2450 decrements.
    # Check if any became 0.
    # Total: 2401 points * 2450 = ~6M ops. Very fast.
    
    # Step 1: Precompute lines and initial counts.
    # Line representation: (x0, y0, m) where m is slope or None for vertical.
    # We need to map a point to the lines passing through it.
    # And for each line, the set of points on it.
    
    # Let's precompute `lines_through_point[idx]` = list of line IDs.
    # And `points_on_line[line_id]` = list of point indices.
    # And `is_valid_line[line_id]` = boolean? No, validity depends on N.
    # But we can check if a line is valid for a specific z quickly.
    
    # Actually, we don't need to store `is_valid_line` globally.
    # We just need to know, for a line L and a point z on L, if L is a valid witness for z.
    # L is valid for z iff all points w in L (w != z) are in N.
    
    # Let's precompute:
    # `lines_by_point`: dict mapping point idx -> list of line ids.
    # `line_points`: dict mapping line id -> list of point idxs.
    # `line_slopes`: not strictly needed if we have the points.
    
    # Total lines: 2401 * 50 = 120,050 lines? No.
    # Number of lines in F_q^2 is q(q+1) = 49 * 50 = 2450.
    # Each line has 49 points.
    # Each point is on 50 lines.
    # 2450 * 49 = 120,050. 2401 * 50 = 120,050. Matches.
    
    # Let's generate all lines.
    # Lines are determined by (x0, y0, m) for m in F_49, and (x0, None) for vertical.
    # But (x0, y0, m) and (x0', y0', m) define the same line if they are collinear.
    # Better to enumerate lines by their equation.
    # Non-vertical: y = m*x + c. m, c in F_49.
    # Vertical: x = c. c in F_49.
    
    # Let's map (xa, xb) to an integer 0..48 for x-coord.
    # x_val = xa * p + xb.
    
    # Precompute addition and multiplication tables for F_49?
    # Or just use the pair representation and mod arithmetic.
    
    # To speed up, let's use integer indices for field elements.
    # field_idx = a * p + b.
    
    # Multiplication table: mul[i, j] = i * j in F_49.
    mul_table = np.zeros((q, q), dtype=int)
    for i in range(q):
        ia = i // p
        ib = i % p
        for j in range(q):
            ja = j // p
            jb = j % p
            # (ia + ib*alpha) * (ja + jb*alpha)
            # = ia*ja + ia*jb*alpha + ib*ja*alpha + ib*jb*alpha^2
            # = (ia*ja + 3*ib*jb) + (ia*jb + ib*ja)*alpha
            ra = (ia * ja + 3 * ib * jb) % p
            rb = (ia * jb + ib * ja) % p
            mul_table[i, j] = ra * p + rb
            
    # Addition table: add[i, j] = i + j in F_49.
    add_table = np.zeros((q, q), dtype=int)
    for i in range(q):
        ia = i // p
        ib = i % p
        for j in range(q):
            ja = j // p
            jb = j % p
            ra = (ia + ja) % p
            rb = (ib + jb) % p
            add_table[i, j] = ra * p + rb
            
    # Generate lines
    # Line ID 0..2449
    # 0..2400: Non-vertical lines y = m*x + c
    # 2401..2449: Vertical lines x = c
    
    line_points = []
    point_lines = [[] for _ in range(total_points)]
    
    # Helper to get point index from x, y field elements
    def get_idx(x, y):
        return x * q + y
        
    # Non-vertical lines
    line_id = 0
    for m in range(q):
        for c in range(q):
            pts = []
            for x in range(q):
                # y = m*x + c
                y = add_table[mul_table[m, x], c]
                idx = get_idx(x, y)
                pts.append(idx)
            line_points.append(pts)
            for idx in pts:
                point_lines[idx].append(line_id)
            line_id += 1
            
    # Vertical lines
    for c in range(q):
        pts = []
        for y in range(q):
            idx = get_idx(c, y)
            pts.append(idx)
        line_points.append(pts)
        for idx in pts:
            point_lines[idx].append(line_id)
        line_id += 1
        
    # Now, compute initial valid_lines_count for each point.
    # valid_lines_count[z] = number of lines L through z such that all w in L (w != z) are in N.
    
    valid_lines_count = np.zeros(total_points, dtype=int)
    
    for z_idx in range(total_points):
        for l_id in point_lines[z_idx]:
            pts = line_points[l_id]
            valid = True
            for w_idx in pts:
                if w_idx == z_idx:
                    continue
                if not in_N[w_idx]:
                    valid = False
                    break
            if valid:
                valid_lines_count[z_idx] += 1
                
    # Greedy removal
    # We want to remove points from N.
    # Iterate through points. If in_N[p] is True, try to remove it.
    # To make it more effective, we can prioritize points that are "less critical" 
    # or just iterate in a fixed order. Randomization might help escape local minima, 
    # but let's start with a simple pass.
    
    # To improve, we can do multiple passes.
    
    removed_count = 0
    for _pass in range(10):  # Multiple passes to allow cascading removals
        if time.time() > deadline - 10:
            break
            
        any_removed = False
        for p_idx in range(total_points):
            if not in_N[p_idx]:
                continue
                
            # Try to remove p_idx
            # Check if any z on lines through p_idx will lose its last valid line.
            # For each line L through p_idx:
            #   For each z on L (z != p_idx):
            #     If L is currently a valid witness for z, then removing p_idx makes L invalid for z.
            #     So valid_lines_count[z] would decrease by 1.
            #     If valid_lines_count[z] was 1, it becomes 0 -> Invalid.
            
            # Optimization: We only need to check z for which L is a valid witness.
            # How to know if L is a valid witness for z?
            # L is valid for z iff all w in L (w != z) are in N.
            # Since p_idx is in N (we are checking removal), and p_idx is on L,
            # if L is valid for z, then p_idx is in N (true) and all others are in N.
            
            # We can precompute for each line L, the set of z for which L is valid?
            # That's expensive.
            
            # Alternative: 
            # When we try to remove p, we simulate the change.
            # For each line L through p:
            #   For each z on L (z != p):
            #     Check if L is valid for z.
            #     If yes, check if valid_lines_count[z] == 1.
            #     If yes, removal fails.
            
            # To check if L is valid for z:
            #   All w in L (w != z) must be in N.
            #   This is O(49).
            
            # Total cost: 50 lines * 49 z * 49 w = 120k ops per point.
            # 2401 * 120k = 288M ops per pass.
            # 10 passes = 2.8B ops. Might be too slow in Python (10-30s per pass?).
            
            # Let's optimize the check.
            # We can precompute `line_valid_for_z`? No, it changes.
            
            # Let's just try it and see if it fits in time.
            # If too slow, reduce passes or points.
            
            can_remove = True
            for l_id in point_lines[p_idx]:
                pts = line_points[l_id]
                for z_idx in pts:
                    if z_idx == p_idx:
                        continue
                    
                    # Check if L is a valid witness for z
                    # i.e., all w in pts (w != z_idx) are in N
                    is_valid_witness = True
                    for w_idx in pts:
                        if w_idx == z_idx:
                            continue
                        if not in_N[w_idx]:
                            is_valid_witness = False
                            break
                            
                    if is_valid_witness:
                        # L is currently a valid witness for z.
                        # If we remove p_idx, L is no longer valid for z.
                        # So valid_lines_count[z_idx] decreases by 1.
                        if valid_lines_count[z_idx] == 1:
                            can_remove = False
                            break
                if not can_remove:
                    break
                    
            if can_remove:
                # Commit removal
                in_N[p_idx] = False
                removed_count += 1
                any_removed = True
                
                # Update valid_lines_count for affected z's
                # For each line L through p_idx:
                #   For each z on L (z != p_idx):
                #     If L was valid for z, decrement count.
                for l_id in point_lines[p_idx]:
                    pts = line_points[l_id]
                    for z_idx in pts:
                        if z_idx == p_idx:
                            continue
                        # Check if L is valid for z (before removal, but p_idx is still in N in our logic? 
                        # No, we set in_N[p_idx] = False. 
                        # But the check "L is valid for z" requires p_idx to be in N if p_idx != z.
                        # Since we just set it to False, L is NOT valid for z anymore.
                        # So we only decrement if it WAS valid before.
                        # How to know if it was valid before?
                        # It was valid before iff all w in L (w != z) were in N.
                        # Since p_idx is one of those w, and it was in N before removal,
                        # we need to check the other points.
                        
                        # This is getting complicated to update efficiently.
                        # Let's just recompute valid_lines_count for affected z's?
                        # Or simply: if we remove p, any line L through p is no longer valid 
                        # for any z on L (z != p).
                        # So for each such z, if L was a valid witness, count[z] -= 1.
                        
                        # To know if L was a valid witness, we check if all w in L (w != z, w != p) are in N.
                        # (p was in N, now it's not, so we ignore p's status for the "was valid" check? 
                        # No, "was valid" means p WAS in N. So we check the rest.)
                        
                        is_was_valid = True
                        for w_idx in pts:
                            if w_idx == z_idx or w_idx == p_idx:
                                continue
                            if not in_N[w_idx]:
                                is_was_valid = False
                                break
                                
                        if is_was_valid:
                            valid_lines_count[z_idx] -= 1
                            
        if not any_removed:
            break
            
    # Construct output array
    points = []
    for idx in range(total_points):
        if in_N[idx]:
            y = idx % q
            x = idx // q
            ya = y // p
            yb = y % p
            xa = x // p
            xb = x % p
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
