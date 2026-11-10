# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
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

    # Generate directions
    directions = []
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 1), (0, 0)))
    n_dirs = len(directions)

    # Precompute lines
    # line_id -> list of point indices
    # point_idx -> list of line_ids
    line_map = {}
    lines = []
    point_to_lines = [[] for _ in range(n_points)]
    
    for x_idx, x in enumerate(points):
        for dx, dy in directions:
            line_pts = []
            for ta in range(p):
                for tb in range(p):
                    t = (ta, tb)
                    px = add_f49(x[0], mul_f49(t, dx))
                    py = add_f49(x[1], mul_f49(t, dy))
                    pt = (px, py)
                    line_pts.append(point_to_idx[pt])
            
            line_key = tuple(sorted(line_pts))
            if line_key not in line_map:
                line_map[line_key] = len(lines)
                lines.append(line_key)
            
            lid = line_map[line_key]
            point_to_lines[x_idx].append(lid)
            
    n_lines = len(lines)
    lines_arr = [np.array(l) for l in lines] # List of arrays of point indices

    # Initialize S
    # Start with the known good set: Points in U with y_a in {0, 3, 5, 6}
    S = np.zeros(n_points, dtype=bool)
    for pt, idx in point_to_idx.items():
        xa, xb = pt[0]
        ya, yb = pt[1]
        if (2 * xa + ya * ya - yb * yb * k) % p == 0:
            if ya in {0, 3, 5, 6}:
                S[idx] = True

    # State arrays
    # c_L: count of points in S on line L
    c_L = np.zeros(n_lines, dtype=int)
    for idx in np.where(S)[0]:
        for lid in point_to_lines[idx]:
            c_L[lid] += 1

    # Z_x: count of lines through x with c_L == 0
    # O_x: count of lines through x with c_L == 1
    Z_x = np.zeros(n_points, dtype=int)
    O_x = np.zeros(n_points, dtype=int)
    
    for lid in range(n_lines):
        if c_L[lid] == 0:
            for idx in lines_arr[lid]:
                Z_x[idx] += 1
        elif c_L[lid] == 1:
            for idx in lines_arr[lid]:
                O_x[idx] += 1

    def is_valid_state():
        # Check all points
        # If S[idx]: O_x[idx] > 0
        # If not S[idx]: Z_x[idx] > 0
        for idx in range(n_points):
            if S[idx]:
                if O_x[idx] == 0:
                    return False
            else:
                if Z_x[idx] == 0:
                    return False
        return True

    # Initial validity check
    if not is_valid_state():
        # Fallback to empty S? Or just return current N?
        # The seed is assumed valid.
        pass

    # Greedy Expansion
    # Candidates: points not in S
    candidates = np.where(~S)[0]
    
    # We will try to add candidates.
    # To improve, we can shuffle and try multiple passes.
    best_S_size = np.sum(S)
    best_S = S.copy()
    
    start_time = time.time()
    deadline = start_time + 580 # 20s margin
    
    # Shuffle candidates to avoid bias
    random.shuffle(candidates)
    
    added_any = True
    while added_any and time.time() < deadline:
        added_any = False
        # Iterate through candidates
        # We need to re-evaluate candidates because state changes
        # But checking all 2400 every time is 2400 * 2401 ops ~ 5M.
        # 5M ops in Python is ~0.5s. So we can do ~1000 iterations?
        # No, 5M simple ops is fast. Let's do it.
        
        for idx in candidates:
            if S[idx]:
                continue
            if time.time() > deadline:
                break
                
            # Tentatively add idx
            S[idx] = True
            
            # Update c_L for lines through idx
            affected_lines = []
            for lid in point_to_lines[idx]:
                c_L[lid] += 1
                affected_lines.append(lid)
                
            # Update Z_x and O_x for points on affected lines
            # We need to know the old counts to update Z/O correctly?
            # c_L changed from old to new.
            # If old == 0 and new == 1:
            #   Line was in Z_x for all pts on line. Now it is not.
            #   Line is now in O_x for all pts on line.
            #   So for all pts on line: Z_x -= 1, O_x += 1.
            # If old == 1 and new == 2:
            #   Line was in O_x. Now it is not.
            #   So for all pts on line: O_x -= 1.
            
            # We didn't store old c_L. Let's store it temporarily or recompute?
            # We can just recompute Z_x and O_x from scratch?
            # n_lines = 2450. n_points = 2401.
            # Recomputing Z/O takes 2450 * 49 ~ 120k ops.
            # This is very fast.
            
            # Let's just recompute Z and O if a point was added.
            # This simplifies logic.
            
            # But wait, if we recompute, we don't need to track affected lines for Z/O update.
            # We just recompute Z_x and O_x arrays.
            
            # Optimization: Only recompute if we actually add the point?
            # We need to check validity BEFORE committing.
            # So we need the state AFTER adding.
            # So we update c_L, then recompute Z/O, then check validity.
            
            Z_x_new = np.zeros(n_points, dtype=int)
            O_x_new = np.zeros(n_points, dtype=int)
            for lid in range(n_lines):
                if c_L[lid] == 0:
                    for p_idx in lines_arr[lid]:
                        Z_x_new[p_idx] += 1
                elif c_L[lid] == 1:
                    for p_idx in lines_arr[lid]:
                        O_x_new[p_idx] += 1
            
            # Check validity with new Z/O
            valid = True
            for i in range(n_points):
                if S[i]:
                    if O_x_new[i] == 0:
                        valid = False
                        break
                else:
                    if Z_x_new[i] == 0:
                        valid = False
                        break
            
            if valid:
                # Commit
                Z_x = Z_x_new
                O_x = O_x_new
                added_any = True
                best_S_size = np.sum(S)
                best_S = S.copy()
            else:
                # Revert
                S[idx] = False
                for lid in affected_lines:
                    c_L[lid] -= 1
                # Z_x and O_x are not changed because we used Z_x_new for check
                # and didn't assign it to Z_x/O_x.
                # So original Z_x/O_x are still valid for the reverted state.
                # Wait, we modified c_L. We reverted c_L.
                # So state is consistent.
                
    # Construct N
    N_points = []
    for pt, idx in point_to_idx.items():
        if not best_S[idx]:
            xa, xb = pt[0]
            ya, yb = pt[1]
            N_points.append([[xa, xb], [ya, yb]])
            
    return np.array(N_points, dtype=np.int64)

# EVOLVE-BLOCK-END
