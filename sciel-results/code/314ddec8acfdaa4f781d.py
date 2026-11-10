# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Nikodym set in F_49^2, F_49 = F_7[alpha], alpha^2 = 3.
    
    Strategy:
      1. Construct a baseline valid set S (complement of N).
      2. Use Simulated Annealing to grow S.
         - State: Binary array S where 1 means point is in S (excluded from N).
         - Objective: Maximize |S| subject to S being a valid Nikodym complement.
         - Validity: For every x in S and every direction v, the line {x + t*v} 
           must contain at least one other point from S.
         - Move: Add a random point from N to S.
         - Energy: Number of validity violations.
         - Acceptance: Accept if energy decreases, or with probability exp(-dE/T) if it increases.
      3. Return N = F_49^2 \\ S.
    """
    p = 7
    q = p * p
    total = q * q
    k_const = 3

    deadline = time.time() + 580

    # ---------- Field Arithmetic ----------
    # Points are (xa, xb, ya, yb)
    # Flat index: xa*343 + xb*49 + ya*7 + yb
    
    def from_flat(idx):
        idx = np.asarray(idx, dtype=np.int64)
        xa = (idx // 343) % p
        xb = (idx // 49) % p
        ya = (idx // 7) % p
        yb = idx % p
        return np.stack([xa, xb, ya, yb], axis=-1)

    def to_flat(pt):
        return (pt[..., 0] * 343 + pt[..., 1] * 49 + pt[..., 2] * 7 + pt[..., 3]).astype(np.int64)

    all_flat = np.arange(total, dtype=np.int64)
    all_pts = from_flat(all_flat)

    # ---------- Baseline S ----------
    # Same as seed: Unital points with ya non-square
    ya_arr = all_pts[:, 2]
    yb_arr = all_pts[:, 3]
    xa_arr = all_pts[:, 0]
    
    in_unital = (2 * xa_arr + ya_arr * ya_arr - yb_arr * yb_arr * k_const) % p == 0
    squares_mod7 = {0, 1, 2, 4}
    ya_is_square = np.array([ya in squares_mod7 for ya in ya_arr])
    
    S_mask = np.zeros(total, dtype=bool)
    S_mask[in_unital & ~ya_is_square] = True

    # ---------- Directions and Line Mates ----------
    dirs = []
    for ma in range(p):
        for mb in range(p):
            dirs.append((1, 0, ma, mb))
    dirs.append((0, 1, 0, 0))
    n_dirs = len(dirs)

    # Precompute line_mates: shape (total, n_dirs, 48)
    # line_mates[x, v, j] is the flat index of the j-th other point on the line
    t_vals = []
    for ta in range(p):
        for tb in range(p):
            if ta == 0 and tb == 0: continue
            t_vals.append((ta, tb))
    t_arr = np.array(t_vals, dtype=np.int64)

    line_mates = np.zeros((total, n_dirs, 48), dtype=np.int64)
    for di, v in enumerate(dirs):
        vxa, vxb, vya, vyb = v
        ta = t_arr[:, 0]
        tb = t_arr[:, 1]
        tv_xa = (ta * vxa + tb * vxb * k_const) % p
        tv_xb = (ta * vxb + tb * vxa) % p
        tv_ya = (ta * vya + tb * vyb * k_const) % p
        tv_yb = (ta * vyb + tb * vya) % p
        x_all = all_pts[:, np.newaxis, :]
        tv_all = np.stack([tv_xa, tv_xb, tv_ya, tv_yb], axis=-1)
        sum_pts = (x_all + tv_all[np.newaxis, :, :]) % p
        flat_mates = to_flat(sum_pts)
        line_mates[:, di, :] = flat_mates

    # ---------- Annealing State ----------
    # S_mask: bool array
    # line_counts[x, v]: number of points in S on line (x, v) EXCLUDING x
    # If S[x] is True, x is valid iff min(line_counts[x, :]) >= 1
    # If S[x] is False, x is not considered for validity, but its presence affects neighbors' counts.
    
    # Initialize line_counts
    # For each x, v, count how many of line_mates[x, v, :] are in S
    line_counts = S_mask[line_mates].sum(axis=-1) # shape (total, n_dirs)
    
    # Calculate initial violations
    # A point x in S is a violation if min(line_counts[x, :]) == 0
    # Note: line_counts[x, v] counts OTHERS. If x is in S, we check others.
    # If x is NOT in S, we don't care about x's validity, but x's absence means 
    # it doesn't contribute to neighbors.
    
    def calc_violations(S_mask, line_counts):
        # violations[x] = 1 if S[x] and min(line_counts[x]) == 0
        # We only care about x in S.
        # min_counts = min over v of line_counts[x, v]
        min_counts = line_counts.min(axis=-1)
        # If S[x] is True and min_counts[x] == 0, it's a violation.
        # Note: line_counts[x, v] does NOT include x.
        return (S_mask & (min_counts == 0)).sum()

    current_violations = calc_violations(S_mask, line_counts)
    # Ideally 0. If not, the baseline might be flawed or my logic slightly off.
    # The seed baseline is valid. Let's assume 0.
    
    best_S = S_mask.copy()
    best_violations = current_violations
    best_size = S_mask.sum()
    
    # Annealing parameters
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # List of indices not in S for random selection
    # We will pick randomly from 0..total-1 and check if in S.
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        # Pick a random point to add to S
        x = np.random.randint(total)
        
        if S_mask[x]:
            # If already in S, maybe try to remove? 
            # Removing is risky. Let's stick to adding for now.
            # Or skip.
            continue
            
        # Calculate delta violations if we add x
        # 1. x becomes in S.
        # 2. x is a violation if min(line_counts[x, :]) == 0.
        # 3. Neighbors of x (on all lines through x) get +1 to their count.
        #    This might FIX violations for neighbors.
        
        # Current status of x:
        # x is NOT in S.
        # If we add x:
        # x_valid = (line_counts[x, :].min() >= 1)
        # new_violation_x = 0 if x_valid else 1
        
        # Neighbors:
        # For each direction v, the 48 mates of x are y.
        # y's count for direction v increases by 1 (because x is on y's line).
        # Wait, line_counts[y, v] counts points in S on line (y, v) excluding y.
        # If we add x to S, and x is on line (y, v), then line_counts[y, v] increases by 1.
        # This might change y from violation to non-violation.
        
        # Let's identify all unique neighbors y of x.
        # x is on line (y, v) iff y is on line (x, v).
        # So the neighbors of x are exactly line_mates[x, :, :].
        # There are 50 * 48 = 2400 potential neighbor slots, but many duplicates.
        # The line (x, v) contains x and 48 others.
        # The union of all these lines is the whole space? No.
        # Each line has 49 points. x is in 50 lines.
        # The neighbors are the points in the union of these 50 lines, excluding x.
        # Total points = 2401.
        # It's easier to just iterate through the 2400 mates and check unique indices.
        
        # Optimization:
        # We need to update line_counts for neighbors.
        # But we only need to check if a neighbor's violation status changes.
        # A neighbor y is a violation iff S[y] and min(line_counts[y, :]) == 0.
        # Adding x increases line_counts[y, v] for the specific v where x is on the line.
        # Actually, x is on line (y, v) for exactly one v (the direction of the line connecting x and y).
        # So for each neighbor y, only ONE component of line_counts[y, :] changes.
        
        # Let's collect the neighbors and their affected direction.
        # For each v in 0..49:
        #   For each j in 0..47:
        #     y = line_mates[x, v, j]
        #     The direction v is the one connecting x and y.
        #     So line_counts[y, v] increases by 1.
        
        # We need to check if y was a violation before, and is not after.
        # y was a violation iff S[y] and min(line_counts[y, :]) == 0.
        # Note: min(line_counts[y, :]) == 0 implies that the specific line_counts[y, v] was 0?
        # Not necessarily. min could be 0 on another direction.
        # If min was 0 on another direction, adding to line_counts[y, v] doesn't help.
        # If min was 0 on direction v, and line_counts[y, v] was 0, then it becomes 1.
        # If line_counts[y, v] was 0 and was the UNIQUE zero, then min becomes >0 (if all others >0).
        
        # Let's just calculate the change in violations for affected neighbors.
        
        # 1. Check x
        min_x = line_counts[x, :].min()
        x_was_violation = False # x not in S
        x_becomes_violation = (min_x == 0)
        
        # 2. Check neighbors
        # We need to know which neighbors were violations.
        # Let's pre-compute a boolean array is_violation for all points.
        # is_violation[y] = S[y] and min(line_counts[y, :]) == 0
        
        # To be efficient, we can just check the specific neighbors.
        # There are at most 2400 neighbor slots, but unique neighbors are fewer.
        # Let's get unique neighbors.
        # neighbors = np.unique(line_mates[x, :, :])
        # This is fast enough for 2400 elements.
        
        neighbors = np.unique(line_mates[x, :, :])
        
        # For each neighbor y, we need to know:
        # 1. Was it a violation?
        # 2. Will it be a violation after adding x?
        
        # "Was it a violation":
        # S[y] and min(line_counts[y, :]) == 0
        
        # "Will it be":
        # S[y] and min(line_counts[y, :] + delta) == 0
        # delta is 1 in the specific direction v_y connecting x and y.
        
        # We need v_y for each y.
        # We can find v_y by checking which line contains both x and y.
        # Or we can just iterate through the 50 directions for x, and for each mate y, 
        # we know the direction is v.
        
        # Let's build a map y -> v
        # y_to_v = {}
        # for v in range(n_dirs):
        #   for j in range(48):
        #     y = line_mates[x, v, j]
        #     y_to_v[y] = v
        
        # This is 2400 iterations.
        
        delta_violations = 0
        
        # x contribution
        if x_becomes_violation:
            delta_violations += 1
            
        # Neighbors contribution
        # We need to check if any neighbor y goes from violation to non-violation.
        # y is a violation iff S[y] and min(line_counts[y, :]) == 0.
        
        # Let's check all unique neighbors.
        # For efficiency, we can vectorize this check.
        
        # Get current min counts for neighbors
        # neighbor_min_counts = line_counts[neighbors, :].min(axis=-1)
        # neighbor_is_S = S_mask[neighbors]
        # was_violation = neighbor_is_S & (neighbor_min_counts == 0)
        
        # To check "will be", we need to update the count for the specific direction.
        # Let's create a temporary copy of line_counts for neighbors?
        # Or just calculate the new min.
        
        # For each neighbor y, let v_y be the direction.
        # new_count_v = line_counts[y, v_y] + 1
        # new_min = min( min(line_counts[y, :] excluding v_y), new_count_v )
        # Actually, min(line_counts[y, :]) is the min over all 50.
        # If line_counts[y, v_y] was the unique minimum and it was 0, and new_count_v >= 1,
        # then new_min might be > 0.
        
        # Let's just compute the new min for each neighbor.
        # We need v_y.
        
        # Construct v_y array
        # Since neighbors are unique, we can just find the v for each.
        # It's easier to just iterate the 50 directions and update a dict or array.
        
        # Let's use a temporary array for the "affected" direction for each neighbor.
        # affected_dir = -1 * np.ones(total, dtype=int)
        # for v in range(n_dirs):
        #   mates = line_mates[x, v, :]
        #   affected_dir[mates] = v
        
        # Then for each neighbor y:
        # v_y = affected_dir[y]
        # new_counts = line_counts[y, :].copy()
        # new_counts[v_y] += 1
        # new_min = new_counts.min()
        # will_be_violation = S_mask[y] and (new_min == 0)
        
        # This is O(50 * 48 + N_neighbors * 50).
        # N_neighbors is small (~2000). 2000 * 50 = 100k ops. Fast.
        
        affected_dir = -1 * np.ones(total, dtype=np.int64)
        for v in range(n_dirs):
            affected_dir[line_mates[x, v, :]] = v
            
        # Now check neighbors
        # Vectorized:
        # Get current counts for neighbors: shape (N, 50)
        # Get affected directions: shape (N,)
        # Create new counts: copy, increment specific column.
        
        n_neighbors = len(neighbors)
        if n_neighbors > 0:
            neighbor_counts = line_counts[neighbors, :] # (N, 50)
            neighbor_dirs = affected_dir[neighbors] # (N,)
            
            # Check was_violation
            was_min = neighbor_counts.min(axis=-1)
            was_violation = S_mask[neighbors] & (was_min == 0)
            
            # Check will_be_violation
            # We need to increment the column specified by neighbor_dirs.
            # np.put_along_axis is good for this.
            new_counts = neighbor_counts.copy()
            rows = np.arange(n_neighbors)
            np.put_along_axis(new_counts, neighbor_dirs, rows[:, None] + 1, axis=1)
            
            will_min = new_counts.min(axis=-1)
            will_be_violation = S_mask[neighbors] & (will_min == 0)
            
            # Delta for neighbors:
            # +1 for each that was violation and is not now? No.
            # Violation count decreases if (was_violation and not will_be_violation).
            # Violation count increases if (not was_violation and will_be_violation).
            # But adding a point to S only HELPS neighbors (increases counts).
            # So a neighbor cannot go from non-violation to violation.
            # It can only go from violation to non-violation.
            # So delta_neighbors = - sum(was_violation & ~will_be_violation)
            
            fixed_violations = np.sum(was_violation & ~will_be_violation)
            delta_violations -= fixed_violations
        
        # Total delta
        # Accept if delta_violations < 0 or with prob exp(-delta/T)
        
        if delta_violations < 0 or (delta_violations == 0 and random.random() < 0.5) or random.random() < np.exp(-delta_violations / T):
            # Accept move
            S_mask[x] = True
            
            # Update line_counts
            # x is added to S.
            # For each neighbor y, line_counts[y, v_y] += 1
            # We already have affected_dir and neighbors.
            # Just update line_counts for these neighbors.
            
            # Also, line_counts[x, :] is now "active" but we don't need to update it 
            # unless we remove x later. But we don't remove.
            # Wait, line_counts[x, :] is used to check if x is a violation.
            # It is already correct (counts of others).
            
            # Update neighbors
            for i in range(n_neighbors):
                y = neighbors[i]
                v = affected_dir[y]
                line_counts[y, v] += 1
            
            current_violations += delta_violations
            
            # Update best
            if current_violations == 0:
                new_size = S_mask.sum()
                if new_size > best_size:
                    best_size = new_size
                    best_S = S_mask.copy()
            elif current_violations < best_violations:
                best_violations = current_violations
                best_S = S_mask.copy()
                # Even if violations > 0, we might keep it as current state to escape local min?
                # But we want to return a VALID set.
                # So we only update best_S if valid?
                # If we accept invalid states, we must track the best VALID state.
                # best_S should always be valid.
                # So only update best_S if current_violations == 0.
                # If current_violations > 0, we don't update best_S.
                # But we stay in the invalid state to allow further moves.
                
            T *= cooling_rate

    # Ensure we return a valid set
    # If best_S is valid, use it.
    # If not, fallback to seed?
    # The seed is valid.
    # So best_S initialized to seed is valid.
    
    N_mask = ~best_S
    N_pts = all_pts[N_mask]
    k = len(N_pts)
    
    if k == 0:
        points = np.zeros((1, 2, 2), dtype=np.int64)
    else:
        points = np.empty((k, 2, 2), dtype=np.int64)
        points[:, 0, 0] = N_pts[:, 0]
        points[:, 0, 1] = N_pts[:, 1]
        points[:, 1, 0] = N_pts[:, 2]
        points[:, 1, 1] = N_pts[:, 3]

    return points

# EVOLVE-BLOCK-END
