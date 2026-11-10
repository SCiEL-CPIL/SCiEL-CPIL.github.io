# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 with q = 49.
    Uses incremental local search with efficient validity tracking to maximize the complement S.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill

    p = 7
    q = p * p
    k = 3  # alpha^2 = 3

    # Field arithmetic setup
    # F_49 = F_7[alpha], alpha^2 = 3
    # Elements are represented as (a, b) meaning a + b*alpha
    
    def f_add(u, v):
        return ((u[0] + v[0]) % p, (u[1] + v[1]) % p)

    def f_mul(u, v):
        ua, ub = u
        va, vb = v
        return ((ua*va + ub*vb*k) % p, (ua*vb + ub*va) % p)

    def f_sub(u, v):
        return ((u[0] - v[0]) % p, (u[1] - v[1]) % p)

    # Point indexing
    # idx = ((xa * p + xb) * p + ya) * p + yb
    def get_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    def get_coords(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p*p)) % p
        xa = (idx // (p*p*p)) % p
        return xa, xb, ya, yb

    # Precompute lines
    # Directions v = (1, m) for m in F_49, and v = (0, 1)
    # Total lines: q (vertical) + q * q (slanted) = 49 + 2401 = 2450
    
    lines = [] # list of sets of point indices
    line_membership = [[] for _ in range(q*q)] # for each point, list of line indices

    # 1. Vertical lines (v = (0, 1))
    # x is constant, y varies
    for xa in range(p):
        for xb in range(p):
            line_idx = len(lines)
            pts = []
            for ya in range(p):
                for yb in range(p):
                    pts.append(get_idx(xa, xb, ya, yb))
            lines.append(set(pts))
            for pt in pts:
                line_membership[pt].append(line_idx)

    # 2. Slanted lines (v = (1, m))
    # For each slope m, there are q distinct parallel lines.
    # A line is determined by m and an intercept.
    # Equation: y = m*x + b. 
    # To avoid duplicates, we can iterate over all points and generate the line,
    # using a set of frozensets to track unique lines, or more efficiently:
    # For a fixed m, the lines partition the space. 
    # We can generate the q lines for a slope m by picking q "base" points
    # that are not on the same line.
    # A simple way: for each m, iterate over all points, generate line, if not seen, add.
    # Since q=49, q^2=2401 points, and 49 slopes, this is 49 * 2401 = ~117k line generations.
    # Each generation involves 49 points. Total ~5.7M point operations. Fast enough.
    
    seen_lines = set()
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            # Iterate over all points as potential "anchors"
            # To be efficient, we only need to process one point per line.
            # We can just iterate all points and check if the line is new.
            for xa in range(p):
                for xb in range(p):
                    for ya in range(p):
                        for yb in range(p):
                            # Generate line through (xa, xb, ya, yb) with slope m
                            # Points: (x, y) = (x0 + t, y0 + m*t) for t in F_49
                            pts = []
                            for ta in range(p):
                                for tb in range(p):
                                    t = (ta, tb)
                                    x = f_add((xa, xb), t)
                                    y = f_add((ya, yb), f_mul(m, t))
                                    pts.append(get_idx(x[0], x[1], y[0], y[1]))
                            line_set = frozenset(pts)
                            if line_set not in seen_lines:
                                seen_lines.add(line_set)
                                line_idx = len(lines)
                                lines.append(set(line_set))
                                for pt in line_set:
                                    line_membership[pt].append(line_idx)

    num_lines = len(lines)
    
    # Baseline S
    # S = { (x,y) in U : y not in Y }
    # U: Trace(x) + Norm(y) = 0 => 2*xa + ya^2 - yb^2*k = 0
    # Y: ya is a square mod p. Squares in F_7: 0, 1, 4, 2
    squares = {0, 1, 2, 4}
    
    S = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2*xa + ya*ya - yb*yb*k) % p == 0:
                        if ya not in squares:
                            S.add(get_idx(xa, xb, ya, yb))
                            
    # State variables for incremental search
    in_S = np.zeros(q*q, dtype=bool)
    in_S[list(S)] = True
    
    # line_count[l] = number of points of S in line l
    line_count = np.zeros(num_lines, dtype=np.int32)
    for x in S:
        for l_idx in line_membership[x]:
            line_count[l_idx] += 1
            
    # witness_count[x] for x in S: number of lines l through x with line_count[l] == 1
    witness_count = np.zeros(q*q, dtype=np.int32)
    # zero_count[z] for z not in S: number of lines l through z with line_count[l] == 0
    zero_count = np.zeros(q*q, dtype=np.int32)
    
    # Initialize witness_count and zero_count
    for l_idx in range(num_lines):
        if line_count[l_idx] == 1:
            for x in lines[l_idx]:
                if in_S[x]:
                    witness_count[x] += 1
        elif line_count[l_idx] == 0:
            for z in lines[l_idx]:
                if not in_S[z]:
                    zero_count[z] += 1
                    
    # Check initial validity (should be true for baseline)
    # All x in S must have witness_count[x] >= 1
    # All z not in S must have zero_count[z] >= 1
    # We assume baseline is valid.
    
    # Local Search
    # We want to add points to S to maximize |S|.
    # A point y can be added to S if:
    # 1. y is not in S.
    # 2. Adding y keeps S valid.
    #    Conditions for validity after adding y:
    #    a. y itself must be isolated: exists line l through y with line_count[l] == 0.
    #       (After adding y, count becomes 1, so y has a witness).
    #    b. No existing x in S loses its last witness.
    #       x loses a witness if there is a line l through x and y such that line_count[l] was 1.
    #       (After adding y, count becomes 2, so l is no longer a witness for x).
    #       So we need: for all x in S, witness_count[x] > 0 OR no line l through x,y has count 1.
    #       More precisely: if we add y, for any line l through y:
    #          - if old_count[l] == 1, let x be the point in S on l. witness_count[x] decreases by 1.
    #            If it becomes 0, validity is broken.
    #    c. No existing z in N (z != y) loses its last zero-line.
    #       z loses a zero-line if there is a line l through z and y such that line_count[l] was 0.
    #       (After adding y, count becomes 1, so l is no longer a zero-line for z).
    #       So we need: for all z in N \ {y}, zero_count[z] > 0 OR no line l through z,y has count 0.
    #       More precisely: if we add y, for any line l through y:
    #          - if old_count[l] == 0, for all z in l \ {y} with z not in S, zero_count[z] decreases by 1.
    #            If any becomes 0, validity is broken.
            
    # Optimization:
    # Precompute for each point y, the list of lines through y.
    # We can just use line_membership[y].
    
    # To check if adding y is valid efficiently:
    # 1. Check y_isolated: exists l in line_membership[y] with line_count[l] == 0.
    # 2. Check witnesses:
    #    For each l in line_membership[y] with line_count[l] == 1:
    #       Find x in S on l.
    #       If witness_count[x] == 1, then adding y breaks x's isolation. -> Invalid.
    # 3. Check zero-lines:
    #    For each l in line_membership[y] with line_count[l] == 0:
    #       For each z in lines[l] \ {y} with not in_S[z]:
    #          If zero_count[z] == 1, then adding y breaks z's n_ok status. -> Invalid.
    
    # Note: step 3 is O(q) per such line. There are up to q lines through y. So O(q^2) per candidate.
    # q=49, so ~2400 ops per candidate.
    # Number of candidates ~2400. Total ~5.7M ops per full pass.
    # This is very fast. We can do many passes.
    
    candidates = [y for y in range(q*q) if not in_S[y]]
    
    added_any = True
    while added_any and time.time() < deadline:
        added_any = False
        random.shuffle(candidates)
        
        for y in candidates:
            if time.time() > deadline:
                break
            if in_S[y]:
                continue
                
            # Check 1: y is isolated
            y_isolated = False
            for l_idx in line_membership[y]:
                if line_count[l_idx] == 0:
                    y_isolated = True
                    break
            if not y_isolated:
                continue
                
            # Check 2: No witness loss
            valid = True
            # We need to check if any x in S on a line l (count 1) through y has witness_count 1.
            # And if any z in N on a line l (count 0) through y has zero_count 1.
            
            # Let's simulate the changes to see if they break anything.
            # We don't actually apply changes until we are sure.
            
            # Check witnesses
            for l_idx in line_membership[y]:
                if line_count[l_idx] == 1:
                    # Find x in S on l
                    for x in lines[l_idx]:
                        if in_S[x]:
                            if witness_count[x] == 1:
                                valid = False
                                break
                    if not valid:
                        break
                
            if not valid:
                continue
                
            # Check zero-lines
            for l_idx in line_membership[y]:
                if line_count[l_idx] == 0:
                    for z in lines[l_idx]:
                        if z != y and not in_S[z]:
                            if zero_count[z] == 1:
                                valid = False
                                break
                    if not valid:
                        break
                
            if not valid:
                continue
                
            # Valid to add y. Apply changes.
            in_S[y] = True
            S.add(y)
            
            for l_idx in line_membership[y]:
                old_c = line_count[l_idx]
                line_count[l_idx] += 1
                new_c = old_c + 1
                
                if old_c == 0:
                    # y gains a witness (we don't track witness_count for y explicitly in the same way, 
                    # but y is now in S, so it needs witness_count[y] >= 1.
                    # Since old_c was 0, new_c is 1, so l is a witness for y.
                    # We should increment witness_count[y] for each such line.
                    witness_count[y] += 1
                    
                    # z loses a zero-line
                    for z in lines[l_idx]:
                        if z != y and not in_S[z]:
                            zero_count[z] -= 1
                            
                elif old_c == 1:
                    # x loses a witness
                    for x in lines[l_idx]:
                        if in_S[x]:
                            witness_count[x] -= 1
                            break
                            
                # If old_c >= 2, no status change for counts 1 or 0.
                
            # Remove y from candidates list? 
            # It's easier to just leave it and check in_S[y] at the start of the loop.
            # But we can remove it to save time.
            # Since we are iterating over a static list `candidates`, we can't easily remove.
            # We just skip if in_S[y] is True.
            
            added_any = True

    # Return N = complement of S
    N_points = []
    for i in range(q*q):
        if not in_S[i]:
            xa, xb, ya, yb = get_coords(i)
            N_points.append([[xa, xb], [ya, yb]])
            
    if not N_points:
        # Fallback if something went wrong (shouldn't happen)
        return np.zeros((1, 2, 2), dtype=np.int64)
        
    return np.array(N_points, dtype=np.int64)

# EVOLVE-BLOCK-END
