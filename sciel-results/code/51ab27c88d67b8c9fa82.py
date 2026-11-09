import numpy as np
import time

# Precompute field constants
P = 7
K = 3  # alpha^2 = 3, since 3 is a non-residue mod 7 (3^3 = 27 = 6 = -1)

# Total points in F_49^2
N_TOTAL = P**4

# Precompute all points as indices 0..2400
# Index = ((xa * P + xb) * P + ya) * P + yb
# xa, xb, ya, yb in 0..6

# We represent the excluded set S as a boolean array of length 2401.
# A point is "bad" if there is NO line through it (direction (1,m) or (0,1)) 
# such that all other points on that line are in S.
# Wait, the condition is:
# For every point x in F_q^2, there exists a direction v such that 
# {x + t*v : t in F_q, t != 0} subset S.
# This means x itself does NOT need to be in S.
# Actually, the Nikodym set N is the complement of S.
# The condition is: For every x in F_q^2, there is a line L through x 
# such that L \ {x} subset S.
# Equivalently: For every x, x is NOT a "bad" point.
# A point x is "bad" if for ALL directions v, there exists some t != 0 
# such that x + t*v is NOT in S.
# So we want to maximize |S| such that there are NO bad points.

# Let's precompute the lines.
# There are 2 directions types: (1, m) for m in F_49 (49 dirs) and (0, 1) (1 dir).
# Total 50 directions.
# For each direction, the field is partitioned into parallel lines.
# For direction (1, m): lines are determined by y - m*x = c.
# For direction (0, 1): lines are determined by x = c.

# We will precompute for each point x, the list of "other points" on the line 
# for each of the 50 directions.
# Actually, it's more efficient to precompute the lines themselves.
# For each direction d, we have a list of lines. Each line is a set of points.
# A point x is "covered" by direction d if (Line_d(x) \ {x}) subset S.
# x is "bad" if it is not covered by ANY direction.

# Let's structure the data:
# directions: list of 50 line-partitions.
# Each partition is a list of arrays of point indices.
# For a given S (boolean array), we can compute "covered" counts.
# But checking if (Line \ {x}) subset S is fast if we precompute sums?
# No, S changes.
# However, we can maintain a count for each point x of how many directions cover it.
# If count[x] > 0, x is good. If count[x] == 0, x is bad.
# We want to keep count[x] > 0 for all x.

# When we add a point p to S:
# For each direction d:
#   Let L be the line in direction d containing p.
#   For every other point x in L:
#     Check if L \ {x} subset S.
#     Note that p is now in S.
#     Previously, was L \ {x} subset S?
#     If yes, then x was already covered by d.
#     If no, let's see if adding p makes it true.
#     L \ {x} contains p. So L \ {x} subset S iff (L \ {x, p}) subset S AND p in S.
#     Since p in S is true now, we just need (L \ {x, p}) subset S.
#     This is equivalent to: (L \ {x}) subset S (with p potentially missing).
#     Actually, let's look at it this way:
#     A point x is covered by d if ALL points in L_d(x) except x are in S.
#     Let's precompute for each line L and each point x in L, the set of other points.
#     That's expensive to store.
#     
# Alternative approach for efficiency:
# For each direction d, and each line L in d:
#   Let points be p_1, ..., p_q (q=49).
#   For each p_i in L, the condition is: all p_j (j!=i) are in S.
#   This is equivalent to: sum_{j!=i} S[p_j] == q-1.
#   Or: (sum_{j in L} S[p_j]) - S[p_i] == q-1.
#   Let Sum_L = sum_{j in L} S[p_j].
#   Then p_i is covered by d iff Sum_L - S[p_i] == q-1.
#   Since S[p_i] is 0 or 1:
#   If S[p_i] == 1: condition is Sum_L - 1 == q-1 => Sum_L == q. (All points in L are in S)
#   If S[p_i] == 0: condition is Sum_L == q-1. (All other points in L are in S)
#   
# So, for a fixed S, we can compute for each direction d:
#   For each line L:
#     Sum_L = sum(S[L])
#     For each p in L:
#       if S[p] == 1 and Sum_L == q: p is covered by d.
#       if S[p] == 0 and Sum_L == q-1: p is covered by d.
#
# We can maintain an array `cover_count` of length 2401.
# `cover_count[x]` is the number of directions that cover x.
# Initially, we can compute this from scratch.
# When we toggle a point p (add or remove):
#   For each direction d:
#     Let L be the line containing p.
#     Old Sum_L, New Sum_L.
#     For each point x in L:
#       Determine if x was covered by d before.
#       Determine if x is covered by d after.
#       Update cover_count[x] accordingly.
#
# This update is O(50 * 49) = 2450 operations per move.
# With 600s budget, we can do many moves.
# Even 10000 moves * 2450 ops = 24.5 million ops. Python might be slow, but NumPy helps?
# Actually, the inner loop "for each point x in L" is sequential logic.
# We can vectorize the line update if we precompute the line indices.

# Precomputation:
# lines[d][l] is an array of point indices for line l in direction d.
# Also need mapping from point index to (d, l, position) ? No, just (d, l) is enough to find the line.
# point_to_line[d][p] = l

# Let's build this structure.

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # 1. Precompute lines
    # Directions: 0..48 for (1, m), 49 for (0, 1)
    # m is represented as (ma, mb).
    # (1, m) = (1, 0) + (0, 1)*m ? No.
    # Vector v = (vx, vy).
    # For (1, m): vx = 1, vy = m.
    # m in F_49. Let m = (ma, mb).
    # Line equation: y - m*x = c.
    # (ya + yb*alpha) - (ma + mb*alpha)*(xa + xb*alpha) = c
    # This is complex. Let's just generate lines by stepping.
    
    # Field elements: 0..48
    # idx = a + b*P
    # val = a + b*alpha
    
    # Precompute field ops? 
    # For line generation, we just need to add vector v to point p.
    # p = (xa, xb, ya, yb)
    # v = (1, m) -> vx = (1,0), vy = (ma, mb)
    # p + t*v:
    # xa' = xa + t*1
    # xb' = xb + t*0
    # ya' = ya + t*ma
    # yb' = yb + t*mb
    # Wait, t is in F_49. t = (ta, tb).
    # t*1 = (ta, tb).
    # xa' = xa + ta
    # xb' = xb
    # t*m = (ta, tb)*(ma, mb) = (ta*ma - tb*mb*K, ta*mb + tb*ma)
    # ya' = ya + (ta*ma - tb*mb*K)
    # yb' = yb + (ta*mb + tb*ma)
    
    # This is getting complicated to vectorize the line generation for all m.
    # But we only need to do it once.
    
    # Let's create the 50 direction partitions.
    # directions = []
    # for d in range(50):
    #    lines = []
    #    # Generate all lines for this direction
    #    # A line is determined by a representative point.
    #    # We can iterate over all points, and if not visited, form a line.
    #    visited = np.zeros(N_TOTAL, dtype=bool)
    #    for p in range(N_TOTAL):
    #       if visited[p]: continue
    #       line = []
    #       cur = p
    #       for t in range(P*P): # t in F_49
    #           # Calculate point cur + t*v
    #           # ...
    #           # Mark visited
    #       lines.append(line)
    #    directions.append(lines)
    
    # This precomputation is O(50 * 49 * 49) ~ 120k ops. Fast.
    
    # Let's implement the field addition and multiplication for line generation.
    # We represent points as flat indices 0..2400.
    # idx -> (xa, xb, ya, yb)
    # xa = idx // (P*P*P)
    # xb = (idx // (P*P)) % P
    # ya = (idx // P) % P
    # yb = idx % P
    
    def idx_to_coords(idx):
        xa = idx // (P*P*P)
        xb = (idx // (P*P)) % P
        ya = (idx // P) % P
        yb = idx % P
        return xa, xb, ya, yb
    
    def coords_to_idx(xa, xb, ya, yb):
        return ((xa * P + xb) * P + ya) * P + yb
    
    # Precompute lines for each direction
    # Direction d corresponds to vector v_d.
    # d=0..48: v = (1, m_d) where m_d is the d-th element of F_49.
    # d=49: v = (0, 1).
    
    # List of m values (as coords)
    m_vals = []
    for ma in range(P):
        for mb in range(P):
            m_vals.append((ma, mb))
            
    all_lines = [] # all_lines[d] = list of arrays of point indices
    
    for d in range(50):
        lines = []
        visited = np.zeros(N_TOTAL, dtype=bool)
        
        # Determine vector v
        if d < 49:
            ma, mb = m_vals[d]
            # v = (1, m)
            # vx = (1, 0)
            # vy = (ma, mb)
            vx_a, vx_b = 1, 0
            vy_a, vy_b = ma, mb
        else:
            # v = (0, 1)
            vx_a, vx_b = 0, 1
            vy_a, vy_b = 0, 0 # Wait, (0,1) means x doesn't change, y increases by 1.
            # v = (0, 1) in F_49^2.
            # vx = 0, vy = 1.
            # 1 in F_49 is (1, 0).
            # So vy_a = 1, vy_b = 0.
            vx_a, vx_b = 0, 0
            vy_a, vy_b = 1, 0
            
        for p_start in range(N_TOTAL):
            if visited[p_start]:
                continue
            
            line = []
            xa, xb, ya, yb = idx_to_coords(p_start)
            
            for t_idx in range(P*P):
                ta = t_idx // P
                tb = t_idx % P
                
                # p + t*v
                # xa' = xa + t*vx_a
                # xb' = xb + t*vx_b
                # ya' = ya + t*vy_a
                # yb' = yb + t*vy_b
                
                # t * vx
                # vx is (vx_a, vx_b). t is (ta, tb).
                # t*vx = (ta*vx_a - tb*vx_b*K, ta*vx_b + tb*vx_a)
                tx_a = (ta * vx_a - tb * vx_b * K) % P
                tx_b = (ta * vx_b + tb * vx_a) % P
                
                # t * vy
                ty_a = (ta * vy_a - tb * vy_b * K) % P
                ty_b = (ta * vy_b + tb * vy_a) % P
                
                nxa = (xa + tx_a) % P
                nxb = (xb + tx_b) % P
                nya = (ya + ty_a) % P
                nyb = (yb + ty_b) % P
                
                n_idx = coords_to_idx(nxa, nxb, nya, nyb)
                line.append(n_idx)
                visited[n_idx] = True
            
            lines.append(np.array(line, dtype=np.int32))
        
        all_lines.append(lines)
        
    # Map point index to line index for each direction
    # point_line[d][p] = l
    point_line = []
    for d in range(50):
        mapping = np.zeros(N_TOTAL, dtype=np.int32)
        for l, line in enumerate(all_lines[d]):
            for p in line:
                mapping[p] = l
        point_line.append(mapping)
    
    # 2. Initialize S (excluded set)
    # Start with the seed construction from the prompt?
    # Or start empty?
    # If S is empty, cover_count is 0 for all. Bad points = all.
    # We want to ADD points to S to increase cover counts.
    # But adding a point p to S might DECREASE cover count for others?
    # Let's check.
    # Condition for x covered by d: Sum_L - S[x] == q-1.
    # If we add p to S:
    #   For lines not containing p: no change.
    #   For line L containing p:
    #     Sum_L increases by 1.
    #     For x in L:
    #       New condition: (Sum_L + 1) - S_new[x] == q-1.
    #       If x != p: S_new[x] = S_old[x].
    #         Old: Sum_L - S_old[x] == q-1 ?
    #         New: Sum_L + 1 - S_old[x] == q-1 => Sum_L - S_old[x] == q-2.
    #         So if x was covered (val q-1), it becomes NOT covered (val q-2 != q-1).
    #         If x was not covered (val < q-1), it might become covered if val was q-2.
    #       If x == p: S_new[p] = 1.
    #         New: Sum_L + 1 - 1 == q-1 => Sum_L == q-1.
    #         Old: Sum_L - 0 == q-1 => Sum_L == q-1.
    #         So p's coverage status doesn't change?
    #         Wait.
    #         Old state: p not in S. S[p]=0.
    #         Covered if Sum_L == q-1.
    #         New state: p in S. S[p]=1.
    #         Covered if Sum_L + 1 - 1 == q-1 => Sum_L == q-1.
    #         So p remains covered iff it was covered.
    #         
    #     But for other x in L:
    #     Old covered if Sum_L - S[x] == q-1.
    #     New covered if Sum_L + 1 - S[x] == q-1 => Sum_L - S[x] == q-2.
    #     So if x was covered (Sum_L - S[x] == q-1), it becomes UNCOVERED.
    #     If x was uncovered with Sum_L - S[x] == q-2, it becomes COVERED.
    #     
    # So adding a point can UNCOVER other points on the same lines!
    # This makes the problem harder. We are not just adding coverage, we are shifting it.
    # This is why simple greedy "add if doesn't violate" works:
    # We only add p if it doesn't uncover any point that was ONLY covered by this line/direction combo?
    # No, "doesn't violate" means no point becomes bad (cover_count 0).
    # If a point x had cover_count 1, and we add p to a line L_d(x) such that x loses coverage from d,
    # then x becomes bad.
    # So we must ensure that for any x that loses coverage from d, it still has cover_count > 1 
    # (i.e., covered by another direction) OR it was not covered by d to begin with (but then it didn't lose it).
    # Actually, if x was covered by d, and we add p to L_d(x), x loses coverage from d.
    # So we need cover_count[x] > 1 for all x in L_d(p) that were covered by d.
    
    # Let's implement the update logic carefully.
    
    # State:
    # S: boolean array (2401,)
    # cover_count: int array (2401,)
    # line_sums: 2D array (50, num_lines) -> sum of S in each line
    
    # Initialize S to empty?
    # If S is empty, line_sums are 0.
    # For x, covered by d if 0 - 0 == 48? No.
    # So cover_count is 0 for all. All points are bad.
    # We need to build S.
    
    # Better initialization: Use the seed construction from the prompt.
    # Seed: Exclude S_seed = { (x,y) in U and y not in Y }.
    # N_seed = F_q^2 \ S_seed.
    # This is a valid Nikodym set.
    # So S_seed is a valid excluded set.
    # We can start with S = S_seed and try to ADD more points to S.
    
    # Generate S_seed
    S = np.zeros(N_TOTAL, dtype=bool)
    y_special = set()
    for u in range(P):
        for v in range(P):
            y_special.add(((u * u) % P, v))
            
    for xa in range(P):
        for xb in range(P):
            for ya in range(P):
                for yb in range(P):
                    in_unital = (2 * xa + ya * ya - yb * yb * K) % P == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        idx = coords_to_idx(xa, xb, ya, yb)
                        S[idx] = True
                        
    # Compute initial line_sums and cover_count
    line_sums = np.zeros((50, 0), dtype=np.int32) # Placeholder, will resize
    # Actually, let's just compute cover_count directly first.
    cover_count = np.zeros(N_TOTAL, dtype=np.int32)
    
    # Compute line_sums
    line_sums_list = []
    for d in range(50):
        sums = np.zeros(len(all_lines[d]), dtype=np.int32)
        for l, line in enumerate(all_lines[d]):
            sums[l] = np.sum(S[line])
        line_sums_list.append(sums)
        
    # Compute cover_count
    for d in range(50):
        for l, line in enumerate(all_lines[d]):
            sum_l = line_sums_list[d][l]
            for x in line:
                s_x = 1 if S[x] else 0
                if sum_l - s_x == P*P - 1:
                    cover_count[x] += 1
                    
    # Now we have a valid state.
    # We want to ADD points to S.
    # Candidate points: those not in S.
    # We pick a candidate p.
    # Check if adding p keeps all cover_count > 0.
    # If yes, add it.
    # If no, maybe try removing a point? (Simulated Annealing allows this)
    # But removing a point from S decreases |S|, which is bad for the metric.
    # However, it might allow adding MORE points later.
    # Let's try a greedy "add if possible" first, with random order.
    # If stuck, try "swap": remove p, add q.
    
    # To speed up checking "add p":
    # We need to know which points x will lose coverage.
    # For each direction d:
    #   Line L = all_lines[d][point_line[d][p]]
    #   Old sum = line_sums[d][l]
    #   New sum = Old sum + 1
    #   For each x in L:
    #     Old covered? (Old sum - S[x] == q-1)
    #     New covered? (New sum - S_new[x] == q-1)
    #     Note S_new[p] = 1, S_new[x] = S[x] for x!=p.
    #     
    #     If x == p:
    #       Old covered if Old sum == q-1.
    #       New covered if New sum - 1 == q-1 => Old sum == q-1.
    #       Status unchanged.
    #     
    #     If x != p:
    #       Old covered if Old sum - S[x] == q-1.
    #       New covered if New sum - S[x] == q-1 => Old sum + 1 - S[x] == q-1 => Old sum - S[x] == q-2.
    #       
    #       Case 1: x was covered (Old sum - S[x] == q-1).
    #         Then New sum - S[x] == q. Not covered.
    #         x LOSES coverage from d.
    #       Case 2: x was not covered, but Old sum - S[x] == q-2.
    #         Then New sum - S[x] == q-1. Covered.
    #         x GAINS coverage from d.
    #       Case 3: Other. No change.
    #     
    # So, to check validity of adding p:
    #   For each d:
    #     L = line containing p.
    #     Identify x in L (x!=p) that were covered by d.
    #     For each such x, check if cover_count[x] > 1.
    #     If any x has cover_count[x] == 1, adding p makes x bad. INVALID.
    #     
    # We don't need to check gains, only losses.
    
    # Precompute for each (d, l) the list of points? We have all_lines.
    # To check quickly, we can just iterate x in L.
    # Cost: 50 * 49 = 2450 checks.
    
    # Greedy loop
    candidates = np.where(~S)[0]
    np.random.shuffle(candidates)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add each candidate
        for p in candidates:
            if S[p]:
                continue
                
            valid = True
            # Check all directions
            for d in range(50):
                l = point_line[d][p]
                line = all_lines[d][l]
                sum_l = line_sums_list[d][l]
                
                # Find x in line that are covered by d
                # Covered if sum_l - S[x] == q-1
                # Since we are checking BEFORE adding p, S[p] is 0.
                # For x in line:
                #   if x == p: S[x]=0. Covered if sum_l == q-1.
                #   if x != p: S[x] is 0 or 1. Covered if sum_l - S[x] == q-1.
                
                # We only care about x that LOSE coverage.
                # x loses coverage if:
                #   Old covered AND New not covered.
                #   New not covered for x!=p means New sum - S[x] != q-1.
                #   New sum = sum_l + 1.
                #   So sum_l + 1 - S[x] != q-1 => sum_l - S[x] != q-2.
                #   Old covered means sum_l - S[x] == q-1.
                #   So if sum_l - S[x] == q-1, it is covered.
                #   And sum_l - S[x] != q-2 is true (since q-1 != q-2).
                #   So ALL x in line (x!=p) that are currently covered by d will lose coverage.
                
                # So we just need to check:
                # For all x in line (x!=p) such that (sum_l - S[x] == q-1),
                # is cover_count[x] > 1?
                
                for x in line:
                    if x == p:
                        continue
                    s_x = 1 if S[x] else 0
                    if sum_l - s_x == P*P - 1:
                        # x is currently covered by d
                        if cover_count[x] <= 1:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                # Add p to S
                S[p] = True
                # Update line_sums and cover_count
                for d in range(50):
                    l = point_line[d][p]
                    line = all_lines[d][l]
                    old_sum = line_sums_list[d][l]
                    new_sum = old_sum + 1
                    line_sums_list[d][l] = new_sum
                    
                    for x in line:
                        s_x_old = 1 if (S[x] and x != p) else 0 # S[x] is updated for p, so careful
                        # Actually S[p] is now True.
                        # For x != p, S[x] is unchanged.
                        # For x == p, S[x] is now True.
                        
                        s_x = 1 if S[x] else 0
                        
                        # Determine old covered status
                        # Old sum was old_sum.
                        # Old s_x: if x==p, it was 0. Else S[x].
                        if x == p:
                            old_s_x = 0
                        else:
                            old_s_x = s_x # S[x] unchanged
                        
                        old_covered = (old_sum - old_s_x == P*P - 1)
                        new_covered = (new_sum - s_x == P*P - 1)
                        
                        if old_covered and not new_covered:
                            cover_count[x] -= 1
                        elif not old_covered and new_covered:
                            cover_count[x] += 1
                # Remove p from candidates? Or just mark S[p]=True.
                improved = True
                # Restart candidate loop? Or continue?
                # If we continue, we might add more.
                # But adding p might make other candidates invalid.
                # It's safer to restart or just check next candidate against new state.
                # Since we check validity against current state, continuing is fine.
                
    # Convert S to N (complement)
    # N is the set of points NOT in S.
    # Return array of points in N.
    
    # Wait, the task asks for the Nikodym set N.
    # "returning an integer array ... listing the points of a set N"
    # "N must be a Nikodym set"
    # Metric: complement size 2401 - |N|.
    # So we want to MINIMIZE |N|, i.e., MAXIMIZE |S|.
    # My logic above maximizes |S|.
    # So N = F_q^2 \ S.
    
    n_points = []
    for i in range(N_TOTAL):
        if not S[i]:
            xa, xb, ya, yb = idx_to_coords(i)
            n_points.append([[xa, xb], [ya, yb]])
            
    return np.array(n_points, dtype=np.int64)

# EVOLVE-BLOCK-END
