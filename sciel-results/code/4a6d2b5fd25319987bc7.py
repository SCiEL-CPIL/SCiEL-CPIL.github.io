import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by starting from an algebraic baseline
    and expanding the excluded set S using a greedy local search with
    efficient state tracking.
    """
    p = 7
    q = p * p
    total_points = q * q
    
    # Field arithmetic setup for F_49 = F_7[alpha], alpha^2 = 3
    k = 3
    
    # Precompute lines
    # A point is (x, y) where x, y in F_49.
    # Flat index: idx = ((xa * p + xb) * p + ya) * p + yb
    # But for line generation, it's easier to work with (xa, xb, ya, yb) tuples.
    
    def get_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    def get_coords(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p*p)) % p
        xa = (idx // (p*p*p)) % p
        return xa, xb, ya, yb

    # Field ops
    def mul(u, v):
        ua, ub = u
        va, vb = v
        return ((ua*va + ub*vb*k) % p, (ua*vb + ub*va) % p)
        
    def add(u, v):
        return ((u[0]+v[0])%p, (u[1]+v[1])%p)

    # Generate all lines
    # Directions: v = (1, m) for m in F_49, and v = (0, 1)
    lines = []
    line_membership = [[] for _ in range(total_points)]
    
    # Helper to add a line
    def add_line(pts):
        line_idx = len(lines)
        lines.append(pts)
        for pt in pts:
            line_membership[pt].append(line_idx)
            
    # 1. Vertical lines (v = (0, 1)): x is constant
    for xa in range(p):
        for xb in range(p):
            pts = []
            for ya in range(p):
                for yb in range(p):
                    pts.append(get_idx(xa, xb, ya, yb))
            add_line(pts)
            
    # 2. Lines with slope m = (ma, mb)
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            seen_lines = set()
            # Iterate over all points as potential anchors
            # To avoid duplicates, we can just generate all and use a set of frozensets
            # Or better, iterate over intercepts?
            # Line: y = m*x + c.
            # (ya, yb) = m * (xa, xb) + (ca, cb)
            # For each (xa, xb), (ya, yb) is determined by c.
            # There are q choices for c.
            # So for each m, there are q lines.
            
            # Let's generate lines by choosing a point (xa, xb) and calculating the intercept?
            # Actually, simpler: just generate the set of points for a line through (0,0) with slope m,
            # then translate.
            # Line L_0: { t*m : t in F_q }? No, x=t, y=m*t.
            # Points: (t, m*t).
            
            base_pts = []
            for ta in range(p):
                for tb in range(p):
                    t = (ta, tb)
                    x = t
                    y = mul(m, t)
                    base_pts.append(get_idx(x[0], x[1], y[0], y[1]))
            
            # Translate by (xa, xb, ya, yb)
            # We need q unique lines.
            # We can pick q points that are not on the same line.
            # E.g., (0,0), (1,0), ..., (0,1) etc?
            # Simpler: iterate over all points, generate the line, check if seen.
            # Since q=49, q^2=2401. 49 slopes * 2401 points = 117k checks.
            # Each check generates 49 points. 5.7M point generations.
            # This might be slow in Python.
            
            # Optimization:
            # For a fixed slope m, the lines are parallel.
            # We can generate one line, then add shifts.
            # Shifts by (dx, dy) such that (dx, dy) are representatives of F_q^2 / span(m).
            # Actually, just iterating all points and using a set of frozensets is robust.
            # Let's try to optimize the inner loop.
            
            for xa in range(p):
                for xb in range(p):
                    for ya in range(p):
                        for yb in range(p):
                            # Line through (xa, xb, ya, yb) with slope m
                            # Points: (x0 + t, y0 + m*t)
                            pts = []
                            for ta in range(p):
                                for tb in range(p):
                                    t = (ta, tb)
                                    x = add((xa, xb), t)
                                    y = add((ya, yb), mul(m, t))
                                    pts.append(get_idx(x[0], x[1], y[0], y[1]))
                            line_set = frozenset(pts)
                            if line_set not in seen_lines:
                                seen_lines.add(line_set)
                                add_line(list(line_set))
                                
    # Baseline S from Inspiration
    # U: Trace(x) + Norm(y) = 0 => 2*xa + ya^2 - yb^2*k = 0
    # Y: ya is a square mod p. Squares in F_7: {0, 1, 2, 4}
    squares = {0, 1, 2, 4}
    
    in_S = np.zeros(total_points, dtype=bool)
    S = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = get_idx(xa, xb, ya, yb)
                    if (2*xa + ya*ya - yb*yb*k) % p == 0:
                        if ya not in squares:
                            in_S[idx] = True
                            S.add(idx)
                            
    # Initialize state arrays
    line_count = np.zeros(len(lines), dtype=np.int32)
    witness_count = np.zeros(total_points, dtype=np.int32) # For points in S
    free_count = np.zeros(total_points, dtype=np.int32)    # For points not in S
    
    # Count points in S per line
    for x in S:
        for l_idx in line_membership[x]:
            line_count[l_idx] += 1
            
    # Calculate witness_count for x in S
    # x is isolated if exists l through x with line_count[l] == 1
    for x in S:
        cnt = 0
        for l_idx in line_membership[x]:
            if line_count[l_idx] == 1:
                cnt += 1
        witness_count[x] = cnt
        
    # Calculate free_count for y not in S
    # y is free if exists l through y with line_count[l] == 0
    for y in range(total_points):
        if not in_S[y]:
            cnt = 0
            for l_idx in line_membership[y]:
                if line_count[l_idx] == 0:
                    cnt += 1
            free_count[y] = cnt
            
    # Check initial validity (should be valid by construction of baseline)
    def is_valid_state():
        # All x in S must have witness_count[x] > 0
        for x in S:
            if witness_count[x] == 0:
                return False
        # All y not in S must have free_count[y] > 0
        for y in range(total_points):
            if not in_S[y] and free_count[y] == 0:
                return False
        return True
        
    # Local Search
    deadline = time.time() + 580
    best_S_size = len(S)
    best_S_copy = set(S)
    
    # Precompute list of lines for each point for faster access?
    # line_membership is already list of lists.
    
    # To speed up "find the unique x in S on line l", we can maintain a list of S points per line?
    # Or just iterate. Since line size is 49, it's fast.
    
    # Greedy expansion
    # We try to add points to S.
    # If we add y to S:
    # 1. in_S[y] = True
    # 2. For each l in line_membership[y]:
    #    old_c = line_count[l]
    #    new_c = old_c + 1
    #    line_count[l] = new_c
    #    
    #    If old_c == 0:
    #       # l was a free line for all z in l \ {y} not in S
    #       # Now it is not free.
    #       for z in lines[l]:
    #           if z != y and not in_S[z]:
    #               free_count[z] -= 1
    #               # If free_count[z] becomes 0, z is no longer valid (unless we fix it, but we are adding to S, so z stays in N)
    #               # So if free_count[z] == 0, the move is invalid.
    #               
    #    If old_c == 1:
    #       # l was a witness line for the unique x in S on l
    #       # Now it is not a witness.
    #       # Find x in S on l.
    #       x_found = None
    #       for x in lines[l]:
    #           if in_S[x]:
    #               x_found = x
    #               break
    #       witness_count[x_found] -= 1
    #       # If witness_count[x_found] == 0, x is no longer valid. Move invalid.
        
    # We implement a simple greedy: iterate candidates, if valid add.
    # To escape local optima, we can randomize the order and perhaps allow some "bad" moves (Simulated Annealing)
    # but given the strict validity constraint, "bad" moves are just invalid.
    # So we just do random restarts or random ordering.
    
    # Let's do random ordering of candidates in each pass.
    candidates = list(range(total_points))
    
    while time.time() < deadline:
        progress = False
        random.shuffle(candidates)
        
        for y in candidates:
            if time.time() > deadline:
                break
            if in_S[y]:
                continue
                
            # Check if y can be added
            # 1. y must be free (free_count[y] > 0)
            if free_count[y] == 0:
                continue
                
            # Simulate addition
            valid = True
            changes = [] # (type, obj, old_val, new_val) for rollback if needed? 
            # Actually, we can just check validity before committing.
            # If invalid, we don't change anything.
            
            # We need to check if any z in N loses its last free line
            # or any x in S loses its last witness line.
            
            # Affected lines: L_y = line_membership[y]
            
            # 1. Check for x in S losing witness
            for l_idx in L_y:
                old_c = line_count[l_idx]
                if old_c == 1:
                    # Find x in S on l
                    for x in lines[l_idx]:
                        if in_S[x]:
                            # This x loses one witness
                            if witness_count[x] == 1:
                                valid = False
                                break
                            break # Found the x, no need to check others on this line
                if not valid:
                    break
                    
            if not valid:
                continue
                
            # 2. Check for z in N losing free line
            # z is affected if z is in a line l through y where old_c == 0
            # Because then free_count[z] decreases.
            for l_idx in L_y:
                old_c = line_count[l_idx]
                if old_c == 0:
                    # All z in l \ {y} not in S lose a free line
                    for z in lines[l_idx]:
                        if z != y and not in_S[z]:
                            if free_count[z] == 1:
                                valid = False
                                break
                    if not valid:
                        break
                        
            if not valid:
                continue
                
            # Move is valid! Commit it.
            in_S[y] = True
            S.add(y)
            progress = True
            
            # Update state
            for l_idx in L_y:
                old_c = line_count[l_idx]
                line_count[l_idx] = old_c + 1
                
                if old_c == 0:
                    # Update free_count for z in l \ {y}, not in S
                    for z in lines[l_idx]:
                        if z != y and not in_S[z]:
                            free_count[z] -= 1
                            
                elif old_c == 1:
                    # Update witness_count for x in S on l
                    for x in lines[l_idx]:
                        if in_S[x]:
                            witness_count[x] -= 1
                            break
                            
            # Update y's own status
            # y is now in S.
            # Its witness_count is the number of lines through y with new line_count == 1.
            # We don't strictly need to update witness_count[y] for validity of OTHER points,
            # but if we ever remove y (we don't), we would need it.
            # However, for future additions, y is in S, so its witness_count matters if it's on a line with count 1.
            # Let's calculate witness_count[y] correctly.
            wc_y = 0
            for l_idx in L_y:
                if line_count[l_idx] == 1:
                    wc_y += 1
            witness_count[y] = wc_y
            
            # If we improved, update best
            if len(S) > best_S_size:
                best_S_size = len(S)
                best_S_copy = set(S)
                
        # If no progress in a full pass, we might be stuck.
        # But since we shuffle, we might find different orders.
        # If stuck, we could try a different strategy, e.g., removing a point?
        # Removing a point from S is harder to validate (need to ensure S points stay isolated).
        # Given time, greedy addition is the primary driver.
        
        # If no progress in the entire pass, break? 
        # Or continue shuffling? 
        # Let's allow a few passes with no progress before giving up on this specific state.
        # But since we are maximizing, and we started from a valid state, 
        # if we can't add ANY point, we are at a local maximum.
        if not progress:
             # Try a random restart? No, we lost the state.
             # We can try to "shake" by removing a random point from S and re-adding?
             # Too complex. 
             # Let's just break. The baseline is likely strong.
             # But wait, the baseline is fixed. If we can't expand it, we return it.
             # Is it possible to be stuck? Yes.
             # Let's try one more thing: if stuck, try removing a point that has witness_count > 1?
             # No, let's keep it simple.
             break
             
    # Return the best S found
    # N = all points not in best_S_copy
    N_points = []
    for i in range(total_points):
        if i not in best_S_copy:
            xa, xb, ya, yb = get_coords(i)
            N_points.append([[xa, xb], [ya, yb]])
            
    return np.array(N_points, dtype=np.int64)

# EVOLVE-BLOCK-END
