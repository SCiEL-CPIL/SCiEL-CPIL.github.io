import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    n_points = p ** 4  # 2401
    n_dirs = p * p + 1  # 50
    k = 3  # alpha^2 = 3

    # Field operations for F_49 = F_7[alpha] with alpha^2 = 3
    # Represent field elements as integers 0..48 where val = a*7 + b
    # (a, b) corresponds to a + b*alpha
    
    # Precompute addition and multiplication tables for speed
    # add_tab[x, y] = x + y in F_49
    # mul_tab[x, y] = x * y in F_49
    add_tab = np.zeros((q, q), dtype=np.int32)
    mul_tab = np.zeros((q, q), dtype=np.int32)
    
    def field_from_int(val):
        return (val // p, val % p)
        
    def int_from_field(a, b):
        return a * p + b
        
    for i in range(q):
        a1, b1 = field_from_int(i)
        for j in range(q):
            a2, b2 = field_from_int(j)
            # Add
            res_a = (a1 + a2) % p
            res_b = (b1 + b2) % p
            add_tab[i, j] = int_from_field(res_a, res_b)
            
            # Mul
            # (a1 + b1*a) * (a2 + b2*a) = a1*a2 + (a1*b2 + b1*a2)*a + b1*b2*a^2
            # = (a1*a2 + 3*b1*b2) + (a1*b2 + b1*a2)*a
            res_mul_a = (a1 * a2 + k * b1 * b2) % p
            res_mul_b = (a1 * b2 + b1 * a2) % p
            mul_tab[i, j] = int_from_field(res_mul_a, res_mul_b)

    # Precompute line_id[point_idx, dir_idx]
    # point_idx: 0..2400
    # dir_idx: 0..48 (slope m), 49 (vertical)
    line_id = np.zeros((n_points, n_dirs), dtype=np.int32)
    
    # Decode point index to coordinates
    # idx = ((xa * p + xb) * p + ya) * p + yb
    coords = np.zeros((n_points, 4), dtype=np.int32)
    for idx in range(n_points):
        yb = idx % p
        idx //= p
        ya = idx % p
        idx //= p
        xb = idx % p
        xa = idx // p
        coords[idx] = [xa, xb, ya, yb]
        
    # Vectorized computation of line_id
    # x = (xa, xb) -> x_int
    # y = (ya, yb) -> y_int
    x_ints = coords[:, 0] * p + coords[:, 1]
    y_ints = coords[:, 2] * p + coords[:, 3]
    
    for d in range(n_dirs):
        if d < n_dirs - 1:
            m = d
            # line_id = y - m*x = y + (-m)*x
            # -m in F_49 is (p - m) % p? No, field negation.
            # neg(m) = ( -ma, -mb )
            ma, mb = field_from_int(m)
            neg_ma = (-ma) % p
            neg_mb = (-mb) % p
            neg_m = int_from_field(neg_ma, neg_mb)
            
            # mx = m * x
            # We need to compute m*x for all x.
            # x is an array of 2401 ints. m is a scalar int.
            # mul_tab[m, :] gives m * 0, m * 1, ..., m * 48
            # So mx_ints = mul_tab[m, x_ints]
            mx_ints = mul_tab[m, x_ints]
            # line_id = y + (-m)*x
            # add_tab[y, mx_neg] ? No, y + (-m*x)
            # Let's compute neg_mx first or just use add_tab[y, neg(mx)]
            # Actually, add_tab is commutative.
            # We need y + (-m*x).
            # Let's precompute neg_mx_ints = -mx_ints in field.
            # Or just: line_id = add_tab[y_ints, neg_mx_ints]
            # How to get neg_mx_ints? 
            # We can precompute neg_tab[i] = -i in F_49
            neg_tab = np.zeros(q, dtype=np.int32)
            for i in range(q):
                a, b = field_from_int(i)
                neg_tab[i] = int_from_field((-a)%p, (-b)%p)
            
            neg_mx_ints = neg_tab[mx_ints]
            line_id[:, d] = add_tab[y_ints, neg_mx_ints]
        else:
            # Vertical: x = const
            line_id[:, d] = x_ints

    # Precompute line_members for fast iteration
    # line_members[d, lid] = list of points
    # Since we need to update counts, we also need to know which points are on which lines.
    # But for local search, we just need to update counts for lines through the moved point.
    
    # State:
    # S: bool array (n_points)
    # counts: int array (n_dirs, q)
    
    def initialize_state(S):
        counts = np.zeros((n_dirs, q), dtype=np.int32)
        for d in range(n_dirs):
            lids = line_id[S, d]
            counts[d] = np.bincount(lids, minlength=q)
        return counts

    def calc_cost(S, counts):
        # Cost = number of unsatisfied points
        # Point y is satisfied if:
        #   S[y] and exists d: counts[d, line_id[y,d]] == 1
        #   not S[y] and exists d: counts[d, line_id[y,d]] == 0
        
        # Vectorized check
        # We need to check for each y if ANY direction satisfies it.
        # sat[y] = OR_d (condition(y, d))
        
        satisfied = np.zeros(n_points, dtype=bool)
        
        for d in range(n_dirs):
            lids = line_id[:, d]
            c = counts[d, lids]
            
            # Condition for S points: c == 1
            # Condition for non-S points: c == 0
            
            valid_d = (S & (c == 1)) | (~S & (c == 0))
            satisfied |= valid_d
            
            if np.all(satisfied):
                return 0
                
        return np.sum(~satisfied)

    def get_unsatisfied(S, counts):
        satisfied = np.zeros(n_points, dtype=bool)
        for d in range(n_dirs):
            lids = line_id[:, d]
            c = counts[d, lids]
            valid_d = (S & (c == 1)) | (~S & (c == 0))
            satisfied |= valid_d
            if np.all(satisfied):
                return []
        return np.where(~satisfied)[0]

    # Local Search
    start_time = time.time()
    deadline = start_time + 590
    
    best_S = None
    best_cost = n_points # Max cost is 2401
    best_size = 0
    
    # Start with a random valid set? 
    # Empty set is valid (cost 0, size 0).
    # We want to maximize size.
    # Let's start with a small random set and try to grow it, or start empty and add.
    # Actually, starting with a random set of size ~500 might have high cost.
    # Let's start with S = empty (valid, cost 0, size 0).
    # Then try to add points. If cost increases, we can still keep it if we can repair it later?
    # No, standard local search minimizes cost. 
    # If we start at cost 0 (valid), adding a point usually increases cost to > 0.
    # We need to accept cost increases to grow S.
    # This is a "Maximize Size subject to Cost == 0" problem.
    # We can use a penalty method: Minimize Cost - lambda * Size.
    # Or simply: Random Restarts.
    # 1. Generate random S.
    # 2. Repair S to be valid (remove points that cause violations).
    # 3. Improve S by adding points (greedy or random).
    
    # Let's try the "Repair then Grow" strategy.
    
    def repair(S, counts):
        # Remove points until valid
        # This is expensive if done naively.
        # Instead, we identify unsatisfied points.
        # If a point y is unsatisfied:
        #   If y in S: we need to reduce counts on lines through y.
        #   If y not in S: we need to increase counts (impossible) or... wait.
        #   If y not in S, it's unsatisfied if for ALL d, counts[d, lid] > 0.
        #   To satisfy y, we must remove all S points from at least one line through y.
        
        # Simple repair: randomly remove points from S until valid.
        # This shrinks S significantly.
        
        # Better repair: 
        # While not valid:
        #   Pick a random unsatisfied point y.
        #   If y in S: 
        #       Find a line L through y where count > 1. 
        #       Remove a random other point z from S on L.
        #   If y not in S:
        #       Find a line L through y where count > 0.
        #       Remove a random point z from S on L.
        
        # This is complex to implement efficiently in Python.
        
        # Alternative: 
        # Just use the local search on the cost function directly.
        pass

    # Let's stick to the Local Search on Cost.
    # Objective: Minimize Cost. Tie-breaker: Maximize Size.
    # Score = -Cost * 1000 + Size.
    
    # Start with a random S of size ~100-500.
    # Calculate cost.
    # Loop:
    #   Pick a move (Add, Remove, Swap).
    #   Calculate new cost (incremental or full).
    #   If new_score > old_score, accept.
    #   Else, accept with prob exp((new_score - old_score) / T).
    
    # Incremental cost calculation is hard because "satisfied" is an OR over directions.
    # Changing one point affects counts on 50 lines.
    # This affects satisfaction of ~2450 points.
    # Recalculating satisfaction for 2450 points is fast if vectorized.
    
    def get_satisfaction(S, counts):
        satisfied = np.zeros(n_points, dtype=bool)
        for d in range(n_dirs):
            lids = line_id[:, d]
            c = counts[d, lids]
            valid_d = (S & (c == 1)) | (~S & (c == 0))
            satisfied |= valid_d
        return satisfied

    # We will maintain `satisfied` array.
    # When we move a point x (add or remove):
    #   Update counts.
    #   Identify affected points (on lines through x).
    #   Recalculate satisfied for affected points.
    
    # Precompute affected_points for each x?
    # affected[x] = set of points on lines through x.
    # Size ~2450. 2401 * 2450 ~ 6M entries. Too much memory? 6M ints is 24MB. OK.
    # But building it takes time.
    # Let's compute it on the fly or precompute line_members.
    
    # line_members[d, lid] is already available implicitly via line_id.
    # We can precompute `points_on_lines_through_x` as a list of arrays?
    # No, just use the line_id array.
    
    # For a point x, the affected points are union of line_members[d, line_id[x,d]] for all d.
    # We can compute this mask quickly.
    
    def get_affected_mask(x):
        mask = np.zeros(n_points, dtype=bool)
        for d in range(n_dirs):
            lid = line_id[x, d]
            mask[line_id[:, d] == lid] = True
        return mask

    # This is O(n_dirs * n_points) per move. 50 * 2401 = 120k ops.
    # 120k ops in numpy is ~1ms.
    # 600s / 1ms = 600k moves. This is plenty.
    
    # Let's implement the local search.
    
    for trial in range(20):
        if time.time() > deadline:
            break
            
        # Initialize random S
        # Start with small size to be close to valid?
        # Or random size?
        # Let's try random size 500.
        S = np.zeros(n_points, dtype=bool)
        random_indices = np.random.choice(n_points, size=500, replace=False)
        S[random_indices] = True
        
        counts = initialize_state(S)
        satisfied = get_satisfaction(S, counts)
        cost = np.sum(~satisfied)
        size = np.sum(S)
        score = -cost * 1000 + size
        
        if cost == 0 and size > best_size:
            best_size = size
            best_S = S.copy()
            
        T = 100.0 # Temperature
        # Cooling
        for step in range(5000):
            if time.time() > deadline:
                break
                
            # Pick a move
            move_type = random.random()
            if move_type < 0.4:
                # Add a random point not in S
                non_S = np.where(~S)[0]
                if len(non_S) == 0: continue
                x = random.choice(non_S)
                S[x] = True
                # Update counts
                for d in range(n_dirs):
                    lid = line_id[x, d]
                    counts[d, lid] += 1
                # Update satisfied
                mask = get_affected_mask(x)
                # Recalculate satisfied for mask
                # We need to check all directions for these points
                # To be safe, recalculate for all points? No, just affected.
                # But satisfied is a global array.
                # Let's just recalculate satisfied for the whole array? 
                # 2401 points * 50 dirs = 120k ops.
                # It might be faster to just call get_satisfaction again.
                # Let's benchmark mentally: 
                # get_satisfaction: 50 loops, each with bincount-like ops (actually just indexing).
                # It's fast. Let's just call it.
                satisfied = get_satisfaction(S, counts)
                new_cost = np.sum(~satisfied)
                new_size = np.sum(S)
                new_score = -new_cost * 1000 + new_size
                
                delta = new_score - score
                if delta > 0 or random.random() < np.exp(delta / T):
                    score = new_score
                    if new_cost == 0 and new_size > best_size:
                        best_size = new_size
                        best_S = S.copy()
                else:
                    # Revert
                    S[x] = False
                    for d in range(n_dirs):
                        lid = line_id[x, d]
                        counts[d, lid] -= 1
                    satisfied = get_satisfaction(S, counts) # Revert satisfied too? 
                    # Since we reverted S and counts, satisfied should be same as before.
                    # We can just keep the old satisfied array.
            
            elif move_type < 0.8:
                # Remove a random point in S
                S_indices = np.where(S)[0]
                if len(S_indices) == 0: continue
                x = random.choice(S_indices)
                S[x] = False
                for d in range(n_dirs):
                    lid = line_id[x, d]
                    counts[d, lid] -= 1
                satisfied = get_satisfaction(S, counts)
                new_cost = np.sum(~satisfied)
                new_size = np.sum(S)
                new_score = -new_cost * 1000 + new_size
                
                delta = new_score - score
                if delta > 0 or random.random() < np.exp(delta / T):
                    score = new_score
                    if new_cost == 0 and new_size > best_size:
                        best_size = new_size
                        best_S = S.copy()
                else:
                    # Revert
                    S[x] = True
                    for d in range(n_dirs):
                        lid = line_id[x, d]
                        counts[d, lid] += 1
                    satisfied = get_satisfaction(S, counts)
            
            else:
                # Swap: Remove x, Add y
                S_indices = np.where(S)[0]
                non_S_indices = np.where(~S)[0]
                if len(S_indices) == 0 or len(non_S_indices) == 0: continue
                x = random.choice(S_indices)
                y = random.choice(non_S_indices)
                
                S[x] = False
                S[y] = True
                for d in range(n_dirs):
                    lid_x = line_id[x, d]
                    lid_y = line_id[y, d]
                    counts[d, lid_x] -= 1
                    counts[d, lid_y] += 1
                
                satisfied = get_satisfaction(S, counts)
                new_cost = np.sum(~satisfied)
                new_size = np.sum(S) # Size unchanged
                new_score = -new_cost * 1000 + new_size
                
                delta = new_score - score
                if delta > 0 or random.random() < np.exp(delta / T):
                    score = new_score
                    if new_cost == 0 and new_size > best_size:
                        best_size = new_size
                        best_S = S.copy()
                else:
                    # Revert
                    S[x] = True
                    S[y] = False
                    for d in range(n_dirs):
                        lid_x = line_id[x, d]
                        lid_y = line_id[y, d]
                        counts[d, lid_x] += 1
                        counts[d, lid_y] -= 1
                    satisfied = get_satisfaction(S, counts)
            
            # Cool down
            T *= 0.999
            
            if T < 1.0:
                break

    if best_S is None:
        # Fallback: Empty S (N = all points)
        best_S = np.zeros(n_points, dtype=bool)
    else:
        # Final check: ensure best_S is valid
        counts = initialize_state(best_S)
        satisfied = get_satisfaction(best_S, counts)
        if not np.all(satisfied):
            # It's invalid. We must repair or fallback.
            # Quick repair: remove points until valid.
            # This might shrink it a lot.
            # Better to fallback to a known valid set (empty) if repair fails fast.
            # Let's try a simple repair: remove random points from unsatisfied set.
            unsat = np.where(~satisfied)[0]
            while len(unsat) > 0:
                if time.time() > deadline - 5:
                    break
                y = random.choice(unsat)
                if best_S[y]:
                    # Remove y
                    best_S[y] = False
                    for d in range(n_dirs):
                        lid = line_id[y, d]
                        counts[d, lid] -= 1
                else:
                    # y not in S, but unsatisfied.
                    # Need to remove S points from lines through y.
                    # Pick a random line d through y.
                    d = random.randint(0, n_dirs - 1)
                    lid = line_id[y, d]
                    # Find S points on this line
                    # points on line: np.where(line_id[:, d] == lid)
                    # Among them, find S points
                    line_pts = np.where(line_id[:, d] == lid)[0]
                    s_pts = line_pts[best_S[line_pts]]
                    if len(s_pts) > 0:
                        z = random.choice(s_pts)
                        best_S[z] = False
                        for d2 in range(n_dirs):
                            lid2 = line_id[z, d2]
                            counts[d2, lid2] -= 1
            
            # Recheck
            counts = initialize_state(best_S)
            satisfied = get_satisfaction(best_S, counts)
            if not np.all(satisfied):
                # Still invalid. Fallback to empty.
                best_S = np.zeros(n_points, dtype=bool)

    # Convert best_S to N (complement)
    N_points = []
    for idx in range(n_points):
        if not best_S[idx]:
            xa, xb, ya, yb = coords[idx]
            N_points.append([[xa, xb], [ya, yb]])
            
    if not N_points:
        for idx in range(n_points):
            xa, xb, ya, yb = coords[idx]
            N_points.append([[xa, xb], [ya, yb]])

    return np.array(N_points, dtype=np.int64)
