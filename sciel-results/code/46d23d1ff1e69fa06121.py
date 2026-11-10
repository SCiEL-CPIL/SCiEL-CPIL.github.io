import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    n_points = p ** 4  # 2401
    n_dirs = p * p + 1  # 50
    
    # Field operations for F_49 = F_7[alpha] with alpha^2 = 3
    k = 3
    
    def field_mul(a, b):
        a0, a1 = a
        b0, b1 = b
        c0 = (a0 * b0 + a1 * b1 * k) % p
        c1 = (a0 * b1 + a1 * b0) % p
        return (c0, c1)
        
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)

    # Encode point (xa, xb, ya, yb) to index 0..2400
    def encode(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
        
    # Decode index to (xa, xb, ya, yb)
    def decode(idx):
        yb = idx % p
        idx //= p
        ya = idx % p
        idx //= p
        xb = idx % p
        xa = idx // p
        return (xa, xb, ya, yb)

    # Encode field element (a, b) to int 0..48
    def field_to_int(f):
        return f[0] * p + f[1]
        
    def int_to_field(i):
        return (i // p, i % p)

    # Precompute line_id[point_idx, dir_idx]
    # dir_idx 0..48: m = (ma, mb)
    # dir_idx 49: vertical (0, 1)
    line_id = np.zeros((n_points, n_dirs), dtype=np.int32)
    
    # Generate all points and their coordinates
    coords = np.zeros((n_points, 4), dtype=np.int32)
    for idx in range(n_points):
        xa, xb, ya, yb = decode(idx)
        coords[idx] = [xa, xb, ya, yb]
        
    # Fill line_id
    for d in range(n_dirs):
        if d < n_dirs - 1:
            ma = d // p
            mb = d % p
            m = (ma, mb)
            for idx in range(n_points):
                xa, xb, ya, yb = coords[idx]
                x = (xa, xb)
                y = (ya, yb)
                mx = field_mul(m, x)
                c = field_add(y, field_neg(mx))
                line_id[idx, d] = field_to_int(c)
        else:
            # Vertical direction (0,1), line is x = const
            for idx in range(n_points):
                xa, xb, ya, yb = coords[idx]
                line_id[idx, d] = field_to_int((xa, xb))

    # Precompute line_members[d][lid] = list of points on that line
    line_members = [None] * n_dirs
    for d in range(n_dirs):
        line_members[d] = [None] * q
        for lid in range(q):
            mask = line_id[:, d] == lid
            line_members[d][lid] = np.where(mask)[0]

    # State:
    # S: bool array, S[i]=True means point i is in the "excluded" set (not in N)
    # We want to maximize |S| (which minimizes |N|, wait. Metric is 2401 - |N| = |S|).
    # So we want to MAXIMIZE |S|.
    # Validity: Every point y is satisfied.
    # Satisfied(y): Exists direction d such that:
    #   if y in S: count_S_on_line(y, d) == 1
    #   if y not in S: count_S_on_line(y, d) == 0
    
    # We maintain:
    # counts[d][lid]: number of S points on line (d, lid)
    # sat[y]: is y satisfied?
    
    def initialize_state(S):
        counts = np.zeros((n_dirs, q), dtype=np.int32)
        for d in range(n_dirs):
            lids = line_id[S, d]
            counts[d] = np.bincount(lids, minlength=q)
        
        sat = np.zeros(n_points, dtype=bool)
        # Initialize sat
        for y in range(n_points):
            is_sat = False
            for d in range(n_dirs):
                lid = line_id[y, d]
                c = counts[d, lid]
                if S[y]:
                    if c == 1:
                        is_sat = True
                        break
                else:
                    if c == 0:
                        is_sat = True
                        break
            sat[y] = is_sat
        return counts, sat

    def get_unsatisfied(S, sat):
        return np.where(~sat)[0]

    def try_add_point(S, counts, sat, x):
        """Tries to add x to S. Returns True if successful (and state updated), False otherwise (and state reverted)."""
        if S[x]:
            return True # Already there, no change needed, but let's assume caller checks
        
        # Identify affected points
        affected = set()
        for d in range(n_dirs):
            lid = line_id[x, d]
            members = line_members[d][lid]
            affected.update(members.tolist())
        
        # Save old counts for affected lines
        old_counts = {}
        for d in range(n_dirs):
            lid = line_id[x, d]
            old_counts[(d, lid)] = counts[d, lid]
        
        # Apply addition
        S[x] = True
        for d in range(n_dirs):
            lid = line_id[x, d]
            counts[d, lid] += 1
            
        # Update sat for affected points
        # Note: Points not in affected set have unchanged counts for all lines through them?
        # No, counts only changed for lines passing through x.
        # A point y is affected if it shares a line with x in some direction.
        # Since any two points in F_q^2 share exactly one line in some direction?
        # Actually, for any y != x, there is a unique direction d such that y and x are on the same line in direction d?
        # No, in a projective plane, any two points determine a unique line.
        # Here we have affine lines. Two points determine a unique affine line unless they are vertical?
        # Yes, any two distinct points in F_q^2 lie on exactly one affine line (either vertical or slope m).
        # So EVERY point y != x is on a line with x in exactly one direction d_xy.
        # Thus, adding x affects the count for the line (d_xy, lid_xy) containing both x and y.
        # So EVERY point y != x is "affected" in the sense that one of its line counts changed.
        # Therefore, we must re-evaluate sat for ALL points y != x.
        # This is O(N * D) per move. 2401 * 50 = 120k ops.
        # In Python, this is ~1-2 ms. 600s / 2ms = 300,000 moves.
        # This is feasible.
        
        # Optimization: We can just re-evaluate sat for all points.
        # It's simpler and less error-prone than tracking dependencies.
        
        for y in range(n_points):
            if y == x:
                continue # x is now in S, must be satisfied by some line with count 1.
            is_sat = False
            for d in range(n_dirs):
                lid = line_id[y, d]
                c = counts[d, lid]
                if S[y]:
                    if c == 1:
                        is_sat = True
                        break
                else:
                    if c == 0:
                        is_sat = True
                        break
            sat[y] = is_sat
        
        # Check if all satisfied
        if np.all(sat):
            return True
        else:
            # Revert
            S[x] = False
            for d in range(n_dirs):
                lid = line_id[x, d]
                counts[d, lid] = old_counts[(d, lid)]
            
            # Revert sat
            for y in range(n_points):
                if y == x:
                    continue
                is_sat = False
                for d in range(n_dirs):
                    lid = line_id[y, d]
                    c = counts[d, lid]
                    if S[y]:
                        if c == 1:
                            is_sat = True
                            break
                    else:
                        if c == 0:
                            is_sat = True
                            break
                sat[y] = is_sat
            return False

    def try_remove_point(S, counts, sat, x):
        """Tries to remove x from S. Returns True if successful."""
        if not S[x]:
            return False
            
        old_counts = {}
        for d in range(n_dirs):
            lid = line_id[x, d]
            old_counts[(d, lid)] = counts[d, lid]
            
        S[x] = False
        for d in range(n_dirs):
            lid = line_id[x, d]
            counts[d, lid] -= 1
            
        # Re-evaluate sat for all points
        for y in range(n_points):
            is_sat = False
            for d in range(n_dirs):
                lid = line_id[y, d]
                c = counts[d, lid]
                if S[y]:
                    if c == 1:
                        is_sat = True
                        break
                else:
                    if c == 0:
                        is_sat = True
                        break
            sat[y] = is_sat
            
        # Removal always maintains validity if it was valid before?
        # If S was valid, removing a point might make some points unsatisfied?
        # If y in S was satisfied by line L with count 1, and we remove x from L, count becomes 0.
        # Then y is NOT satisfied by L anymore.
        # So removal can break validity.
        if np.all(sat):
            return True
        else:
            # Revert
            S[x] = True
            for d in range(n_dirs):
                lid = line_id[x, d]
                counts[d, lid] = old_counts[(d, lid)]
            for y in range(n_points):
                is_sat = False
                for d in range(n_dirs):
                    lid = line_id[y, d]
                    c = counts[d, lid]
                    if S[y]:
                        if c == 1:
                            is_sat = True
                            break
                    else:
                        if c == 0:
                            is_sat = True
                            break
                sat[y] = is_sat
            return False

    start_time = time.time()
    deadline = start_time + 580 # 20s margin
    
    best_S = None
    best_size = 0
    
    # Simulated Annealing
    # Start with empty S
    S = np.zeros(n_points, dtype=bool)
    counts, sat = initialize_state(S)
    current_size = 0
    current_valid = np.all(sat)
    
    T = 100.0
    T_min = 0.01
    alpha = 0.999
    
    all_points = list(range(n_points))
    
    while time.time() < deadline:
        if not current_valid:
            # Repair: Remove a random unsatisfied point from S
            unsat_in_S = np.where(S & ~sat)[0]
            if len(unsat_in_S) > 0:
                x = unsat_in_S[random.randint(0, len(unsat_in_S)-1)]
                if try_remove_point(S, counts, sat, x):
                    current_size -= 1
                    current_valid = np.all(sat)
                else:
                    # Should not happen if we just removed the point causing conflict?
                    # Actually, try_remove_point reverts if invalid.
                    # If it reverts, we are stuck. Force remove?
                    # No, we must maintain validity for the "current state" to be a valid solution candidate.
                    # If we can't repair, restart?
                    break
            else:
                # All unsatisfied points are NOT in S.
                # This means we have non-S points with count > 0 on all lines.
                # We need to remove S points from their lines.
                unsat_not_in_S = np.where(~S & ~sat)[0]
                if len(unsat_not_in_S) > 0:
                    y = unsat_not_in_S[random.randint(0, len(unsat_not_in_S)-1)]
                    # Pick a direction d for y
                    d = random.randint(0, n_dirs-1)
                    lid = line_id[y, d]
                    # Pick a random S point on this line
                    s_points_on_line = line_members[d][lid][S[line_members[d][lid]]]
                    if len(s_points_on_line) > 0:
                        x = s_points_on_line[random.randint(0, len(s_points_on_line)-1)]
                        if try_remove_point(S, counts, sat, x):
                            current_size -= 1
                            current_valid = np.all(sat)
                        else:
                            break
                    else:
                        break
                else:
                    break
            continue
            
        # Current state is valid. Update best.
        if current_size > best_size:
            best_size = current_size
            best_S = S.copy()
            
        # Move
        move_type = random.random()
        
        if move_type < 0.7:
            # Add a random point
            x = random.choice(all_points)
            if not S[x]:
                if try_add_point(S, counts, sat, x):
                    current_size += 1
                    current_valid = True
                else:
                    # Metropolis: accept invalid state with prob exp(-delta/T)?
                    # But we can't hold an invalid state in this simple tracker easily without complex rollback.
                    # So we just reject.
                    pass
        elif move_type < 0.9:
            # Remove a random point from S
            s_points = np.where(S)[0]
            if len(s_points) > 0:
                x = s_points[random.randint(0, len(s_points)-1)]
                if try_remove_point(S, counts, sat, x):
                    current_size -= 1
                    current_valid = True
        else:
            # Swap: Remove one, Add one
            s_points = np.where(S)[0]
            if len(s_points) > 0:
                x_out = s_points[random.randint(0, len(s_points)-1)]
                x_in = random.choice(all_points)
                if not S[x_in]:
                    # Try swap
                    # We need to check if (S - x_out + x_in) is valid.
                    # This is expensive.
                    # Let's just do sequential: remove x_out, then add x_in.
                    # But if remove breaks validity, we can't add.
                    # So this is just two moves.
                    if try_remove_point(S, counts, sat, x_out):
                        current_size -= 1
                        current_valid = True
                        # Now try to add x_in
                        if try_add_point(S, counts, sat, x_in):
                            current_size += 1
                            current_valid = True
                        else:
                            # Revert add? No, add failed so state is unchanged from after remove.
                            pass

        T *= alpha
        if T < T_min:
            T = T_min

    if best_S is None:
        best_S = np.zeros(n_points, dtype=bool)

    # Convert best_S to N
    N_points = []
    for idx in range(n_points):
        if not best_S[idx]:
            xa, xb, ya, yb = decode(idx)
            N_points.append([[xa, xb], [ya, yb]])
            
    if not N_points:
        for idx in range(n_points):
            xa, xb, ya, yb = decode(idx)
            N_points.append([[xa, xb], [ya, yb]])

    return np.array(N_points, dtype=np.int64)
