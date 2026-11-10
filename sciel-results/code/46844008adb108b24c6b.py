import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    n_points = p ** 4  # 2401
    n_dirs = p * p + 1  # 50
    k = 3

    start_time = time.time()
    deadline = start_time + 570  # 60s safety margin

    # Field operations
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

    def field_to_int(f):
        return f[0] * p + f[1]
        
    def int_to_field(i):
        return (i // p, i % p)

    # Encode/Decode
    def encode(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
        
    def decode(idx):
        yb = idx % p
        idx //= p
        ya = idx % p
        idx //= p
        xb = idx % p
        xa = idx // p
        return (xa, xb, ya, yb)

    # Precompute line_id and line_members
    # line_id[idx, d] is the ID of the line in direction d passing through point idx
    # line_members[d][lid] is a list of point indices on that line
    line_id = np.zeros((n_points, n_dirs), dtype=np.int32)
    coords = np.zeros((n_points, 4), dtype=np.int32)
    
    for idx in range(n_points):
        xa, xb, ya, yb = decode(idx)
        coords[idx] = [xa, xb, ya, yb]
        
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
            for idx in range(n_points):
                xa, xb, ya, yb = coords[idx]
                line_id[idx, d] = field_to_int((xa, xb))

    line_members = [None] * n_dirs
    for d in range(n_dirs):
        line_members[d] = [None] * q
        for lid in range(q):
            mask = line_id[:, d] == lid
            line_members[d][lid] = np.where(mask)[0]

    # Strategy: Build R (complement of N) greedily.
    # N = All \ R.
    # Condition for N to be valid:
    # For every point y:
    #   If y in N (y not in R): exists d such that line(d, y) \ {y} subset N.
    #     This means: line(d, y) contains NO points from R, except possibly y? 
    #     Wait. "points other than y all belong to N".
    #     So if y in N, we need a line L through y such that L \ {y} subset N.
    #     Since y in N, this is equivalent to L subset N.
    #     So L contains NO points from R.
    #   If y not in N (y in R): exists d such that line(d, y) \ {y} subset N.
    #     This means: line(d, y) contains NO points from R, EXCEPT y.
    #     So the number of points in R on line(d, y) is exactly 1 (which is y).
    
    # Let's track for each point y, how many "valid" lines it has.
    # A line L is "valid" for y if:
    #   If y in N: count(R on L) == 0
    #   If y in R: count(R on L) == 1 (and that point is y)
    
    # Initially R is empty. N is all points.
    # For every y, any line L through y has count(R on L) == 0.
    # So initially, all 50 lines are valid for every y.
    # valid_count[y] = 50 for all y.
    
    # We want to add points to R.
    # When we add x to R:
    #   1. For x itself:
    #      x is now in R.
    #      Previously, x was in N. Valid lines for x were those with count(R)=0.
    #      Now, x is in R. A line L through x is valid for x if count(R on L) == 1 (and it's x).
    #      Since we just added x, any line L through x that had count(R)=0 now has count(R)=1 (just x).
    #      So all lines that were valid for x (count 0) become valid for x (count 1).
    #      Lines that were not valid for x (count > 0) are still not valid (count > 1).
    #      So valid_count[x] stays the same? 
    #      Wait. If a line had count(R)=0 before, it's valid for x (in N).
    #      After adding x, count(R)=1. It's valid for x (in R).
    #      So the set of valid lines for x doesn't change in size?
    #      Let's check:
    #      Before: L valid iff count_R(L) == 0.
    #      After: L valid iff count_R(L) == 1.
    #      Since we added x to R, for any L through x, count_R(L) increases by 1.
    #      If count_R(L) was 0, it becomes 1. (Valid -> Valid)
    #      If count_R(L) was 1, it becomes 2. (Invalid -> Invalid)
    #      So valid_count[x] is unchanged.
    #      
    #   2. For any other point y (y != x):
    #      y is either in N or R.
    #      If y in N: L is valid iff count_R(L) == 0.
    #      If y in R: L is valid iff count_R(L) == 1.
    #      
    #      Consider a line L through y.
    #      If x is NOT on L, count_R(L) is unchanged. Validity unchanged.
    #      If x IS on L, count_R(L) increases by 1.
    #      
    #      Case A: y in N.
    #      Before: L valid iff count_R(L) == 0.
    #      After: L valid iff count_R(L) == 0.
    #      If x is on L, count_R(L) increases by 1.
    #      If it was 0, it becomes 1. So L was valid, now invalid.
    #      If it was >0, it was invalid, still invalid.
    #      So if y in N and x is on L, and L was valid (count 0), it becomes invalid.
    #      valid_count[y] decreases by 1 for each such L.
    #      
    #      Case B: y in R.
    #      Before: L valid iff count_R(L) == 1.
    #      After: L valid iff count_R(L) == 1.
    #      If x is on L, count_R(L) increases by 1.
    #      If it was 0, it becomes 1. So L was invalid, now valid.
    #      If it was 1, it becomes 2. So L was valid, now invalid.
    #      If it was >1, it was invalid, still invalid.
    #      So if y in R and x is on L:
    #      - If count_R(L) was 0 (before adding x), L becomes valid. valid_count[y] increases by 1.
    #      - If count_R(L) was 1 (before adding x), L becomes invalid. valid_count[y] decreases by 1.
    #      - If count_R(L) > 1, no change.
        
    # We need to efficiently track count_R(L) for lines L.
    # Let line_count[d][lid] = number of points in R on line(d, lid).
    # Initially all 0.
    
    # valid_count[y] = number of lines d such that:
    #   if y in N: line_count[d][line_id[y,d]] == 0
    #   if y in R: line_count[d][line_id[y,d]] == 1
    
    # We maintain:
    # R: boolean array of size n_points
    # line_count: 2D array (n_dirs, q) of integers
    # valid_count: 1D array of size n_points
    
    # Initialization:
    # R is all False.
    # line_count is all 0.
    # For all y, y in N. Condition: line_count == 0.
    # All 50 lines have count 0. So valid_count[y] = 50 for all y.
    
    R = np.zeros(n_points, dtype=bool)
    line_count = np.zeros((n_dirs, q), dtype=np.int32)
    valid_count = np.full(n_points, n_dirs, dtype=np.int32)
    
    # We want to maximize |R|.
    # Greedy: iterate through points in some order. Try to add x to R.
    # When adding x:
    #   1. Update line_count for all lines through x.
    #   2. Update valid_count for all y affected.
    #   3. If any valid_count[y] becomes 0, the move is invalid (unless y==x and we handle it? No, if valid_count[y]==0, y has no valid line).
    #      Wait, if y==x, we established valid_count[x] doesn't change.
    #      But for other y, if valid_count[y] drops to 0, we must revert.
    
    # To make this efficient, we need to quickly identify which y are affected by adding x.
    # y is affected if x is on some line L through y.
    # This means y and x are collinear in some direction d.
    # For each direction d, x is on a specific line L_x_d.
    # All points y on L_x_d (except x) are affected.
    # So for each d, we look at line_members[d][line_id[x,d]].
    # This set has size 49. One is x. 48 others.
    # Total affected points: 50 * 48 = 2400 (all other points).
    # This is too slow to update valid_count for all 2400 points for every x in a greedy loop of 2401 iterations.
    # 2401 * 2400 ~ 5.7 million operations. This is fine in Python if optimized?
    # Actually, 5.7M simple ops is very fast (< 0.1s).
    # But we need to check validity (valid_count > 0) for all affected points.
    # And if invalid, revert.
    
    # Let's refine the update logic.
    # When adding x to R:
    #   For each d in 0..49:
    #       lid = line_id[x, d]
    #       old_count = line_count[d, lid]
    #       new_count = old_count + 1
    #       line_count[d, lid] = new_count
    #       
    #       For each y in line_members[d, lid]:
    #           if y == x: continue
    #           
    #           # y is on this line.
    #           # Determine if y's validity for this specific line d changed.
    #           
    #           if R[y]: # y in R
    #               # Valid if count == 1
    #               if old_count == 0:
    #                   # Was invalid (0 != 1), now valid (1 == 1)
    #                   valid_count[y] += 1
    #               elif old_count == 1:
    #                   # Was valid (1 == 1), now invalid (2 != 1)
    #                   valid_count[y] -= 1
    #           else: # y in N
    #               # Valid if count == 0
    #               if old_count == 0:
    #                   # Was valid (0 == 0), now invalid (1 != 0)
    #                   valid_count[y] -= 1
    #               # else: was invalid, still invalid
    #
    #   After updating all, check if any valid_count[y] == 0 for y != x.
    #   Also, check x itself?
    #   For x:
    #   Before: x in N. Valid if count == 0.
    #   After: x in R. Valid if count == 1.
    #   We know valid_count[x] doesn't change.
    #   But we should verify it's > 0.
    #   Initially 50. It never drops below 0?
    #   Let's check if valid_count[x] can become 0.
    #   If valid_count[x] == 0, x has no valid line.
    #   But we just added x.
    #   Is it possible that all 50 lines through x become invalid?
    #   For x in R, line d is valid if count(d, x) == 1.
    #   count(d, x) includes x.
    #   So count(d, x) >= 1.
    #   It is valid if count(d, x) == 1. i.e., no other points in R on that line.
    #   If we add x, and there are already points in R on all 50 lines through x, then all lines have count >= 2.
    #   So valid_count[x] would become 0.
    #   So we MUST check x as well.
    
    # Algorithm for adding x:
    # 1. Store old state for affected lines and points.
    # 2. Perform updates.
    # 3. Check if min(valid_count) == 0.
    # 4. If yes, revert. If no, keep x in R.
    
    # To speed up, we can precompute the list of affected points for each x.
    # affected[x] = list of (y, d) pairs? Or just list of y?
    # Actually, for each x, the affected y's are all other points.
    # But the update depends on d.
    # We can just loop d, then loop y in line_members[d][lid].
    
    # Optimization:
    # Instead of checking ALL valid_count for min, we can track the minimum?
    # Or just check the affected ones.
    # Only affected y's change valid_count.
    # So we only need to check if any affected y has valid_count == 0.
    # And x.
    
    # Let's implement this.
    
    # Precompute affected points for each x?
    # affected_by_x[x] = list of y such that x is on a line through y.
    # This is just all y != x.
    # But we need to know which d caused the change to update valid_count correctly?
    # No, we just decrement/increment valid_count[y].
    # We don't need to know d for the check, only for the update logic.
    
    # Let's just do the double loop.
    
    # Order of addition:
    # Random order? Or some heuristic?
    # Let's try random order first.
    
    best_R_size = 0
    best_R = None
    
    # Multiple restarts to avoid local minima?
    # Greedy is deterministic if order is fixed. Random order helps.
    
    max_restarts = 10
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        R = np.zeros(n_points, dtype=bool)
        line_count = np.zeros((n_dirs, q), dtype=np.int32)
        valid_count = np.full(n_points, n_dirs, dtype=np.int32)
        
        order = list(range(n_points))
        random.shuffle(order)
        
        current_size = 0
        for x in order:
            if time.time() > deadline - 5:
                break
                
            # Try to add x to R
            # Save state
            old_line_counts = []
            old_valid_counts = {}
            
            affected_ys = set()
            
            # Perform tentative update
            for d in range(n_dirs):
                lid = line_id[x, d]
                old_count = line_count[d, lid]
                new_count = old_count + 1
                line_count[d, lid] = new_count
                old_line_counts.append((d, lid, old_count))
                
                members = line_members[d, lid]
                for y in members:
                    if y == x:
                        continue
                    affected_ys.add(y)
                    
                    if R[y]:
                        if old_count == 0:
                            valid_count[y] += 1
                            old_valid_counts[y] = valid_count[y] - 1
                        elif old_count == 1:
                            valid_count[y] -= 1
                            old_valid_counts[y] = valid_count[y] + 1
                    else:
                        if old_count == 0:
                            valid_count[y] -= 1
                            old_valid_counts[y] = valid_count[y] + 1
            
            # Update x's state
            R[x] = True
            # valid_count[x] logic:
            # Before: x in N. Valid if count == 0.
            # After: x in R. Valid if count == 1.
            # We don't change valid_count[x] in the loop above.
            # But we need to ensure it's valid.
            # Actually, we should re-evaluate valid_count[x] from scratch?
            # Or derive it.
            # The change in valid_count[x] is:
            # Lines that were valid (count 0) are now valid (count 1).
            # Lines that were invalid (count > 0) are now invalid (count > 1).
            # So valid_count[x] is UNCHANGED.
            # So we don't need to update it.
            # But we must check if it is > 0.
            
            # Check validity
            is_valid = True
            if valid_count[x] == 0:
                is_valid = False
            else:
                for y in affected_ys:
                    if valid_count[y] == 0:
                        is_valid = False
                        break
            
            if is_valid:
                current_size += 1
            else:
                # Revert
                R[x] = False
                for d, lid, old_count in old_line_counts:
                    line_count[d, lid] = old_count
                for y, old_val in old_valid_counts.items():
                    valid_count[y] = old_val
        
        if current_size > best_R_size:
            best_R_size = current_size
            best_R = R.copy()
            # Early stop if we reach a good size?
            # Max possible R size?
            # If N is a line, R is 2401 - 49 = 2352.
            # But N must be a Nikodym set.
            # Smallest Nikodym set in F_q^2 is q^2 - q + 1? No.
            # In F_q^2, a Nikodym set has size at least q^2 - q + 1?
            # Actually, the complement R can be at most q^2 - 1?
            # If R is a line, N is everything else.
            # Is N a Nikodym set?
            # For x in N (not on the line), we need a line L through x such that L \ {x} subset N.
            # If L is not the excluded line, L intersects the excluded line at one point.
            # That point is in R. So L \ {x} contains a point in R.
            # So L \ {x} is NOT subset N.
            # So this fails.
            # So R cannot be a line.
            
            # What is the max size of R?
            # If R is small, N is large.
            # We want to maximize R.
            # Known results: In F_q^2, max size of R (complement of Nikodym set) is q^2 - q?
            # No, the "Nikodym set" usually refers to the set of points that *do not* have a line.
            # Wait, the definition here: "every point x ... must lie on a line ... whose points other than x all belong to N".
            # This is the standard definition of a Nikodym set (or rather, the set N is the "Nikodym set" in the sense that it's the set of points that are "covered" in this specific way? No, N is the set of points that are *allowed* to be on the lines).
            # Actually, usually a Nikodym set is a set of lines.
            # Here N is a set of points.
            # The condition is: For every x, there is a line L_x such that L_x \ {x} subset N.
            # This means N is a "Nikodym set" in the sense of a set of points that supports such lines.
            # The complement R is the set of points that are "excluded" from the lines (except for the line's own center).
            
            # If R is a single point, N is all others.
            # For x in N, if x is not the excluded point, can we find a line?
            # Yes, take a line not passing through the excluded point.
            # For x = excluded point, take any line through x. The other points are in N.
            # So R size 1 is valid.
            
            # Can R be larger?
            # If R has 2 points, x1, x2.
            # For x1, need line L1 such that L1 \ {x1} subset N.
            # This means L1 cannot contain x2.
            # So there must be a line through x1 not containing x2.
            # This is always true (only 1 line contains both).
            # So R size 2 is valid?
            # Check x3 in N. Need line L3 through x3 such that L3 \ {x3} subset N.
            # This means L3 cannot contain x1 or x2.
            # Is there a line through x3 avoiding x1 and x2?
            # In F_q^2, through any point, there are q+1 lines.
            # At most 2 lines pass through x1 or x2 (one through x1, one through x2, unless collinear).
            # If x1, x2, x3 are collinear, the line through x3 is the one containing x1, x2.
            # Then L3 \ {x3} contains x1, x2 which are in R.
            # So L3 is not valid.
            # Are there other lines through x3?
            # Yes, q lines not containing x1, x2.
            # So for x3, we can pick one of those.
            # So R size 2 is valid.
            
            # It seems R can be quite large.
            # The constraint is tight when points in R are collinear?
            # If R is a line, we saw it fails.
            # So max R size is less than 49.
            
            # Let's just run the greedy and see.
            
    if best_R is None:
        best_R = np.zeros(n_points, dtype=bool)
        
    # Construct N from R
    N = ~best_R
    
    # Convert to array
    N_points = []
    for idx in range(n_points):
        if N[idx]:
            xa, xb, ya, yb = decode(idx)
            N_points.append([[xa, xb], [ya, yb]])
            
    if not N_points:
        N_points = [[[0,0],[0,0]]]
        
    return np.array(N_points, dtype=np.int64)
