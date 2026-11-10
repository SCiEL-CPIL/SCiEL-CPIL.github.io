# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = 49 (p=7).
    Uses a greedy local search to maximize the complement S (points NOT in N),
    starting from a unital-based seed. The goal is to make N as small as possible.
    """
    p = 7
    q = p * p
    total_points = q * q
    deadline = time.time() + 580.0  # Safety margin for 600s limit

    # Field arithmetic setup: F_49 = F_7[alpha], alpha^2 = 3
    k = 3

    # Helper functions for field arithmetic
    def mul(u, v):
        ua, ub = u
        va, vb = v
        return ((ua*va + ub*vb*k) % p, (ua*vb + ub*va) % p)
        
    def add(u, v):
        return ((u[0]+v[0])%p, (u[1]+v[1])%p)
        
    def sub(u, v):
        return ((u[0]-v[0])%p, (u[1]-v[1])%p)
        
    def get_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    # Precompute lines
    # A line is defined by direction v and a point x0.
    # We need to check for each point x if there is a line L through x such that L \ {x} is subset of N.
    # Equivalent: L \cap S = {x} (if x in S) or L \cap S = empty (if x in N).
    # Wait, the condition is:
    # For every point x in F_q^2, there exists a line L through x such that L \ {x} is subset of N.
    # This means:
    # If x in N: There is a line L through x such that (L \ {x}) subset N. Since x in N, this is equivalent to L subset N.
    #             i.e., L \cap S = empty.
    # If x in S: There is a line L through x such that (L \ {x}) subset N.
    #             i.e., (L \ {x}) \cap S = empty.
    #             i.e., L \cap S = {x}.
    
    # So the condition is:
    # For all x in N: exists line L through x with count_S(L) == 0.
    # For all x in S: exists line L through x with count_S(L) == 1.

    # We will maintain:
    # S: set of points in S (complement of N)
    # in_S: boolean array of size total_points
    # line_count: array of size num_lines, count_S(L)
    # point_isolated_S: boolean array, True if x in S and there is a line L through x with count_S(L)==1
    # point_ok_N: boolean array, True if x in N and there is a line L through x with count_S(L)==0
    
    # We need to generate all lines.
    # Directions: v = (1, m) for m in F_q, and v = (0, 1).
    # Total directions: q + 1 = 50.
    # For each direction, there are q^2 / q = q parallel lines? No.
    # In AG(2, q), number of lines in a parallel class is q.
    # Total lines = q * (q + 1) = 49 * 50 = 2450.
    
    lines = []
    line_membership = [[] for _ in range(total_points)]
    
    # Generate lines for v = (0, 1) (vertical lines)
    # x is constant.
    for xa in range(p):
        for xb in range(p):
            pts = []
            for ya in range(p):
                for yb in range(p):
                    idx = get_idx(xa, xb, ya, yb)
                    pts.append(idx)
            lines.append(pts)
            for pt in pts:
                line_membership[pt].append(len(lines) - 1)
                
    # Generate lines for v = (1, m)
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            # Line equation: y = m*x + b.
            # We can iterate over all possible intercepts b.
            # Or iterate over all points and find the unique line through it with slope m.
            # To avoid duplicates, we can just iterate over all points (xa, xb, ya, yb)
            # and compute the line. But we need to ensure we don't add the same line multiple times.
            # A line is uniquely determined by its set of points.
            # Since we are iterating over all points, we will generate the same line q times.
            # We can use a set of frozensets to track seen lines.
            
            seen_lines = set()
            for xa in range(p):
                for xb in range(p):
                    for ya in range(p):
                        for yb in range(p):
                            # Line through (xa, xb, ya, yb) with direction (1, m)
                            # Points: (x0 + t, y0 + m*t) for t in F_q
                            pts = []
                            for ta in range(p):
                                for tb in range(p):
                                    t = (ta, tb)
                                    x = add((xa, xb), t)
                                    y = add((ya, yb), mul(m, t))
                                    idx = get_idx(x[0], x[1], y[0], y[1])
                                    pts.append(idx)
                            line_set = frozenset(pts)
                            if line_set not in seen_lines:
                                seen_lines.add(line_set)
                                lines.append(list(line_set))
                                for pt in line_set:
                                    line_membership[pt].append(len(lines) - 1)

    num_lines = len(lines)
    
    # Initialize S based on the unital seed
    # U = {(x, y) : Trace(x) + Norm(y) = 0} => 2*xa + ya^2 - yb^2*k = 0
    # Y = {(u^2, v) : u, v in F_p} => ya is a square mod p.
    squares = {0, 1, 2, 4}
    
    S = set()
    in_S = np.zeros(total_points, dtype=bool)
    
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2*xa + ya*ya - yb*yb*k) % p == 0:
                        if ya not in squares:
                            idx = get_idx(xa, xb, ya, yb)
                            S.add(idx)
                            in_S[idx] = True
                            
    # Initialize line_count
    line_count = np.zeros(num_lines, dtype=np.int32)
    for x in S:
        for l_idx in line_membership[x]:
            line_count[l_idx] += 1
            
    # Initialize point_isolated_S and point_ok_N
    point_isolated_S = np.zeros(total_points, dtype=bool)
    point_ok_N = np.zeros(total_points, dtype=bool)
    
    def update_point_status(x):
        # Update status for point x
        if in_S[x]:
            # Check if exists line L through x with count_S(L) == 1
            isolated = False
            for l_idx in line_membership[x]:
                if line_count[l_idx] == 1:
                    isolated = True
                    break
            point_isolated_S[x] = isolated
        else:
            # Check if exists line L through x with count_S(L) == 0
            ok = False
            for l_idx in line_membership[x]:
                if line_count[l_idx] == 0:
                    ok = True
                    break
            point_ok_N[x] = ok

    # Initial status computation
    for x in range(total_points):
        update_point_status(x)
        
    # Check if current S is valid
    def is_valid():
        for x in range(total_points):
            if in_S[x]:
                if not point_isolated_S[x]:
                    return False
            else:
                if not point_ok_N[x]:
                    return False
        return True

    # Greedy Search: Try to add points to S (remove from N)
    # A point x can be added to S if:
    # 1. x is not already in S.
    # 2. After adding x, x is isolated (exists L through x with count_S(L)==1).
    #    Note: Before adding, count_S(L) for L through x was 0 (since x not in S, and if it was isolated in N, it had a line with 0).
    #    Wait, if x is in N, it has a line L0 with count 0.
    #    If we add x to S, count_S(L0) becomes 1. So x is isolated in S via L0.
    #    However, we must ensure that adding x does NOT break the validity of any other point.
    #    Adding x to S increases count_S(L) by 1 for all L through x.
    #    This might break:
    #    - Points y in S that were isolated via a line L through x (count was 1, now 2).
    #    - Points z in N that were ok via a line L through x (count was 0, now 1).
    
    # So we need to check the impact.
    # Let L_x be the set of lines through x.
    # For each L in L_x:
    #   If count_S(L) == 0:
    #       All points z in L \ {x} that were relying on L for their "ok" status (i.e., L was their only 0-count line) will break.
    #       Actually, if a point z in N had count_S(L)==0, it was ok. If we add x to S, count_S(L) becomes 1.
    #       z is still ok if it has ANOTHER line L' through z with count_S(L')==0.
    #   If count_S(L) == 1:
    #       The unique point y in S on L was relying on L for isolation.
    #       y is still isolated if it has ANOTHER line L' through y with count_S(L')==1.
    
    # This check is expensive. We can approximate or just try and roll back.
    # Given q=49, total points 2401.
    # Number of lines through a point is 50.
    # We can try adding x, update counts, check validity of affected points, and rollback if invalid.
    
    # Affected points are those on lines through x.
    # There are 50 lines, each with 49 points. Total affected points <= 50*49 = 2450 (almost all).
    # So checking all points is O(2401 * 50) = 120k operations.
    # We can do this for each candidate x.
    # How many candidates? We can try all points in N.
    
    # Optimization:
    # Precompute for each point z, the list of lines through z.
    # We maintain:
    # zero_count_lines[z] = number of lines through z with count_S == 0.
    # one_count_lines[z] = number of lines through z with count_S == 1.
    
    zero_count_lines = np.zeros(total_points, dtype=np.int32)
    one_count_lines = np.zeros(total_points, dtype=np.int32)
    
    for x in range(total_points):
        for l_idx in line_membership[x]:
            if line_count[l_idx] == 0:
                zero_count_lines[x] += 1
            elif line_count[l_idx] == 1:
                one_count_lines[x] += 1
                
    # Now, update_point_status can be O(1) using these counts.
    # x in S is valid if one_count_lines[x] >= 1.
    # x in N is valid if zero_count_lines[x] >= 1.
    
    def is_valid_fast():
        for x in range(total_points):
            if in_S[x]:
                if one_count_lines[x] < 1:
                    return False
            else:
                if zero_count_lines[x] < 1:
                    return False
        return True

    # Let's verify the seed is valid with this fast check.
    # If not, we need to fix it or start from empty S?
    # Empty S: N = F_q^2.
    # For any x in N, is there a line L with count 0? Yes, all lines have count 0.
    # So empty S is valid.
    # The seed S might not be valid. Let's check.
    
    if not is_valid_fast():
        # Seed is invalid. Start from empty S? Or try to repair?
        # Starting from empty S is safe.
        S.clear()
        in_S.fill(False)
        line_count.fill(0)
        zero_count_lines.fill(0)
        one_count_lines.fill(0)
        # Recalculate counts for empty S
        for x in range(total_points):
            for l_idx in line_membership[x]:
                # count is 0 for all
                zero_count_lines[x] = len(line_membership[x])
                one_count_lines[x] = 0
        
    # Greedy Loop
    # Try to add points to S one by one.
    # Order: Random or by some heuristic.
    # Let's try random restarts with greedy.
    
    best_S_size = len(S)
    best_S = set(S)
    
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Reset to seed or empty?
        # If seed is valid, use it. Else empty.
        if is_valid_fast():
            current_S = set(best_S) # Start from best found so far? Or seed?
            # Let's start from seed for each restart to diversify?
            # Or start from current best and try to improve.
            # Let's start from the seed for the first restart, then from best for others?
            # Actually, let's just try to improve the current state.
            
        # If we started from empty (because seed was invalid), we are at empty.
        # Let's just run greedy on the current state.
        
        # To make it faster, we can pre-filter candidates.
        # A candidate x in N is "good" if adding it doesn't break too many points.
        # But let's just iterate through all points in N and try to add them.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle order to avoid bias
            candidates = [x for x in range(total_points) if not in_S[x]]
            np.random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline:
                    break
                
                # Try adding x to S
                # Save state
                lines_through_x = line_membership[x]
                affected_points = set()
                for l_idx in lines_through_x:
                    for pt in lines[l_idx]:
                        affected_points.add(pt)
                
                # Save old counts for affected points
                old_zero = {pt: zero_count_lines[pt] for pt in affected_points}
                old_one = {pt: one_count_lines[pt] for pt in affected_points}
                old_line_count = {l_idx: line_count[l_idx] for l_idx in lines_through_x}
                
                # Apply change: add x to S
                in_S[x] = True
                S.add(x)
                
                # Update line counts
                for l_idx in lines_through_x:
                    line_count[l_idx] += 1
                    # Update zero/one counts for points on this line
                    for pt in lines[l_idx]:
                        old_c = old_line_count[l_idx]
                        new_c = old_c + 1
                        # This logic is tricky because we are updating multiple lines.
                        # We need to update zero/one counts based on the FINAL line counts.
                        # It's safer to recalculate zero/one counts for affected points from scratch.
                        pass
                
                # Recalculate zero/one counts for affected points
                # This is O(|affected| * 50)
                for pt in affected_points:
                    zc = 0
                    oc = 0
                    for l_idx in line_membership[pt]:
                        c = line_count[l_idx]
                        if c == 0:
                            zc += 1
                        elif c == 1:
                            oc += 1
                    zero_count_lines[pt] = zc
                    one_count_lines[pt] = oc
                    
                # Check validity of affected points
                valid = True
                for pt in affected_points:
                    if in_S[pt]:
                        if one_count_lines[pt] < 1:
                            valid = False
                            break
                    else:
                        if zero_count_lines[pt] < 1:
                            valid = False
                            break
                            
                if valid:
                    improved = True
                    if len(S) > best_S_size:
                        best_S_size = len(S)
                        best_S = set(S)
                else:
                    # Rollback
                    in_S[x] = False
                    S.remove(x)
                    for l_idx in lines_through_x:
                        line_count[l_idx] = old_line_count[l_idx]
                    for pt in affected_points:
                        zero_count_lines[pt] = old_zero[pt]
                        one_count_lines[pt] = old_one[pt]
                        
    # Return N = F_q^2 \ best_S
    points = []
    for x in range(total_points):
        if not in_S[x]:
            yb = x % p
            ya = (x // p) % p
            xb = (x // (p*p)) % p
            xa = (x // (p*p*p)) % p
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
