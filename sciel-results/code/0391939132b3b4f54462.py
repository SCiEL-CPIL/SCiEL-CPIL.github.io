import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    DEADLINE = 600  # seconds
    p = 7
    q = 49
    k = 3  # alpha^2 = 3

    # Precompute field operations using integer indices 0..48
    # idx = a * p + b represents a + b*alpha
    add_tbl = np.zeros((q, q), dtype=np.int64)
    sub_tbl = np.zeros((q, q), dtype=np.int64)
    mul_tbl = np.zeros((q, q), dtype=np.int64)
    
    for i in range(q):
        a0, a1 = i // p, i % p
        for j in range(q):
            b0, b1 = j // p, j % p
            r0, r1 = (a0 + b0) % p, (a1 + b1) % p
            add_tbl[i, j] = r0 * p + r1
            r0, r1 = (a0 - b0) % p, (a1 - b1) % p
            sub_tbl[i, j] = r0 * p + r1
            r0 = (a0 * b0 + a1 * b1 * k) % p
            r1 = (a0 * b1 + a1 * b0) % p
            mul_tbl[i, j] = r0 * p + r1

    total_points = q * q  # 2401

    # Construct Unital U: 2*xa + ya^2 - k*yb^2 = 0
    U = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - k * yb * yb) % p == 0:
                        x_idx = xa * p + xb
                        y_idx = ya * p + yb
                        U.add(x_idx * q + y_idx)

    all_points_set = set(range(total_points))
    
    # Build lines: 49 vertical + 49*49 slanted = 2450 lines
    lines = []
    # Vertical lines: x = c
    for c in range(q):
        pts = []
        for y in range(q):
            pts.append(c * q + y)
        lines.append(pts)
    
    # Slanted lines: y = m*x + b
    for m in range(q):
        for b in range(q):
            pts = []
            for x in range(q):
                y = add_tbl[mul_tbl[m, x], b]
                pts.append(x * q + y)
            lines.append(pts)
    
    num_lines = len(lines)
    
    # Precompute: for each point, which lines contain it
    # point_lines[pt_idx] = list of line indices
    point_lines = [[] for _ in range(total_points)]
    for lid, L in enumerate(lines):
        for pt in L:
            point_lines[pt].append(lid)
    
    # For each line, precompute which points are in U
    line_U_pts = []
    line_nonU_pts = []
    for L in lines:
        u_pts = [pt for pt in L if pt in U]
        nonU_pts = [pt for pt in L if pt not in U]
        line_U_pts.append(u_pts)
        line_nonU_pts.append(nonU_pts)
    
    # Initial state: S = empty, N = all points
    # We want to maximize |S| while maintaining validity
    # Validity: for each x in S, exists line through x with S-count = 1 (only x)
    #            for each x in N, exists line through x with S-count = 0
    
    # Start with S = empty (trivially valid)
    S_mask = np.zeros(total_points, dtype=bool)
    line_S_count = np.zeros(num_lines, dtype=np.int32)
    
    # Points in U are candidates for S (since non-U points can't be in S initially? 
    # Actually any point can be in S, but let's focus on U for structure)
    
    # Better approach: start with a known valid S and grow it
    # Baseline: take some lines, put their U-intersection points in S
    # But we need to ensure validity
    
    # Let's use the greedy approach from current but with fast validity check
    # State: S_mask (bool array), line_S_count (int array)
    
    def check_validity(S_mask, line_S_count):
        """Check if current S is a valid Nikodym complement."""
        # For each point x:
        #   if x in S: need some line through x with S-count == 1
        #   if x in N: need some line through x with S-count == 0
        for x in range(total_points):
            if S_mask[x]:
                found = False
                for lid in point_lines[x]:
                    if line_S_count[lid] == 1:
                        found = True
                        break
                if not found:
                    return False
            else:
                found = False
                for lid in point_lines[x]:
                    if line_S_count[lid] == 0:
                        found = True
                        break
                if not found:
                    return False
        return True
    
    # Greedy: try to add points to S one by one
    # Priority: points in U first (they have structure), then others
    # For each candidate point, check if adding it maintains validity
    
    # Candidates: all points, but prioritize U points
    candidates = list(U) + [pt for pt in range(total_points) if pt not in U]
    
    # Sort candidates by some heuristic: points with fewer "conflicting" lines first
    # A point p conflicts if adding it to S would break validity for other points
    # Specifically, adding p to S increases S-count by 1 for all lines through p
    # This could break:
    #   - Points in N that relied on those lines having count 0
    #   - Points in S that relied on those lines having count 1 (now 2)
    
    # For efficiency, track for each line how many N-points rely on it having count 0
    # and how many S-points rely on it having count 1
    
    # n_relying_on_zero[lid] = number of N-points x where lid is the ONLY line with count 0
    # s_relying_on_one[lid] = number of S-points x where lid is the ONLY line with count 1
    
    # But this is complex. Let's just do the full check for now with time limit.
    
    added_count = 0
    max_added = 2000  # safety limit
    
    for pt in candidates:
        if time.time() - start_time > DEADLINE - 60:
            break
        if added_count >= max_added:
            break
        
        if S_mask[pt]:
            continue
        
        # Try adding pt to S
        S_mask[pt] = True
        for lid in point_lines[pt]:
            line_S_count[lid] += 1
        
        # Check validity (expensive but we have time budget)
        if check_validity(S_mask, line_S_count):
            added_count += 1
        else:
            # Revert
            S_mask[pt] = False
            for lid in point_lines[pt]:
                line_S_count[lid] -= 1
    
    # Construct N = all_points - S
    N_points = []
    for pt in range(total_points):
        if not S_mask[pt]:
            x = pt // q
            y = pt % q
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            N_points.append([[xa, xb], [ya, yb]])
    
    return np.array(N_points, dtype=np.int64)
