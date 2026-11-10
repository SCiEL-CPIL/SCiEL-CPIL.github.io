# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    Q2 = q * q  # 2401
    k = 3  # alpha^2 = 3 in F_7

    # Flat index: idx = xa*49 + xb*7 + ya*7 + yb  (i.e., xa*49 + xb*7 + ya*7 + yb)
    # Actually: idx = xa*49 + xb*7 + ya*7 + yb doesn't work cleanly.
    # Let's use: idx = xa*49 + xb*7 + ya*7 + yb? No.
    # idx = ((xa * p + xb) * p + ya) * p + yb = xa*49 + xb*7 + ya*7 + yb? 
    # (xa*7+xb)*49 + (ya*7+yb) = xa*343 + xb*49 + ya*7 + yb. That's 0..2400.
    # Wait, 7^4 = 2401. So idx = xa*343 + xb*49 + ya*7 + yb.
    
    def get_idx(xa, xb, ya, yb):
        return xa * 343 + xb * 49 + ya * 7 + yb

    def get_coords(idx):
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        return xa, xb, ya, yb

    # Field ops
    def mul(u, v):
        ua, ub = u
        va, vb = v
        return ((ua * va + ub * vb * k) % p, (ua * vb + ub * va) % p)

    def add(u, v):
        return ((u[0] + v[0]) % p, (u[1] + v[1]) % p)

    # Precompute all lines
    # Directions: v=(1,m) for m in F_49, and v=(0,1)
    lines = []  # list of numpy arrays of point indices
    line_membership = [[] for _ in range(Q2)]

    # Vertical lines: x constant
    for xa in range(p):
        for xb in range(p):
            pts = np.array([get_idx(xa, xb, ya, yb) for ya in range(p) for yb in range(p)], dtype=np.int32)
            li = len(lines)
            lines.append(pts)
            for pt in pts:
                line_membership[pt].append(li)

    # Sloped lines: y = y0 + m*(x - x0)
    # For slope m, line is determined by intercept b = y0 - m*x0
    # Two points with same m and same b are on same line.
    # To generate unique lines: for each m, iterate over all possible b values.
    # b = (ya - m*xa, yb - m*xb) for some reference point.
    # Actually, for a given m, the lines are parameterized by b in F_49.
    # Line: (x, y) such that y - m*x = b.
    # For each m, there are q lines.
    
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            for ba in range(p):
                for bb in range(p):
                    b = (ba, bb)
                    pts = []
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            mx = mul(m, x)
                            y = add(mx, b)
                            pts.append(get_idx(xa, xb, y[0], y[1]))
                    pts = np.array(pts, dtype=np.int32)
                    # Check if this line already exists (it shouldn't for unique b)
                    # But let's just add it. Each (m, b) gives a unique line.
                    li = len(lines)
                    lines.append(pts)
                    for pt in pts:
                        line_membership[pt].append(li)

    num_lines = len(lines)
    # Each point is on: 1 vertical + 49 sloped = 50 lines
    # Total lines: 49 + 49*49 = 49 + 2401 = 2450
    
    # Convert line_membership to numpy arrays for speed
    line_membership_arr = [np.array(lm, dtype=np.int32) for lm in line_membership]
    
    # Convert lines to lists for faster iteration in some cases
    # Actually keep as numpy arrays

    # Baseline S
    squares = {0, 1, 2, 4}
    S = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - yb * yb * k) % p == 0:
                        if ya not in squares:
                            S.add(get_idx(xa, xb, ya, yb))

    # State arrays
    in_S = np.zeros(Q2, dtype=bool)
    for x in S:
        in_S[x] = True

    line_count = np.zeros(num_lines, dtype=np.int32)
    for x in S:
        for li in line_membership_arr[x]:
            line_count[li] += 1

    # witness_count[x]: number of lines through x with line_count == 1 (for x in S)
    # zero_count[z]: number of lines through z with line_count == 0 (for z not in S)
    witness_count = np.zeros(Q2, dtype=np.int32)
    zero_count = np.zeros(Q2, dtype=np.int32)

    for li in range(num_lines):
        lc = line_count[li]
        if lc == 1:
            for x in lines[li]:
                if in_S[x]:
                    witness_count[x] += 1
        elif lc == 0:
            for z in lines[li]:
                if not in_S[z]:
                    zero_count[z] += 1

    # Verify baseline validity
    # All x in S: witness_count[x] >= 1
    # All z not in S: zero_count[z] >= 1
    assert np.all(witness_count[in_S] >= 1), "Baseline S not isolated"
    assert np.all(zero_count[~in_S] >= 1), "Baseline N not covered"

    deadline = time.time() + 580

    # Local search: try to add points to S (removing from N)
    # We want to maximize |N| = Q2 - |S|, so we want to MINIMIZE |S|.
    # Wait, the metric is complement size = 2401 - |N|. 
    # N is the Nikodym set. The complement is S.
    # "Metric (higher is better): the complement size 2401 - |N|"
    # So we want to MAXIMIZE 2401 - |N| = |S|.
    # So we want to ADD points to S (remove from N) while maintaining validity.
    
    # But wait: the validity condition says:
    # "every point x of F_49^2 (in N or not) must lie on a line ... whose points other than x all belong to N"
    # For x in N: line L through x, L \ {x} subset N. i.e., L intersect S = empty (since x in N means x not in S).
    #   Wait: L \ {x} all in N means no other point of L is in S. So |L intersect S| = 0.
    # For x not in N (x in S): line L through x, L \ {x} subset N. 
    #   L \ {x} all in N means no other point of L is in S. So |L intersect S| = 1 (just x).
    
    # So:
    # x in S: exists line L through x with |L ∩ S| = 1. (witness_count[x] >= 1)
    # x in N: exists line L through x with |L ∩ S| = 0. (zero_count[x] >= 1)
    
    # This matches my setup. Good.
    # We want to maximize |S|. So we add points to S.
    
    # When we add y to S:
    # - y was in N, now in S.
    # - For each line l through y: line_count[l] increases by 1.
    # - If old_count[l] == 0: new_count[l] = 1.
    #   - l is no longer a zero line for any z in N on l (z != y).
    #   - l becomes a witness line for y (if y is the only S point on l).
    #   - For z in N on l (z != y): zero_count[z] -= 1. If zero_count[z] becomes 0, z is invalid.
    # - If old_count[l] == 1: new_count[l] = 2.
    #   - Let x be the S point on l. l was a witness for x. Now it's not.
    #   - witness_count[x] -= 1. If witness_count[x] becomes 0, x is invalid.
    # - If old_count[l] >= 2: new_count[l] >= 3. No change to witness/zero status.
    
    # Also: y itself. y is new to S. y needs witness_count[y] >= 1.
    #   y gets a witness from each l with old_count[l] == 0 (new_count = 1).
    #   So if there's at least one l through y with old_count[l] == 0, y is fine.
    
    # Check: can we add y?
    # 1. Exists l through y with line_count[l] == 0. (Otherwise y won't be isolated.)
    # 2. For all l through y with line_count[l] == 1: the S point x on l still has witness_count[x] >= 2.
    # 3. For all l through y with line_count[l] == 0: for all z in N on l (z != y), zero_count[z] >= 2.
    
    # Let's implement this check efficiently.
    
    # Precompute for each line l: the S point on l (if line_count[l] == 1)
    # And for each line l: list of N points on l (if line_count[l] == 0)
    # These change as we modify S, so we need to update them.
    
    # Alternative: just check on the fly.
    # For condition 2: for each l through y with count 1, find the S point. 
    #   We can precompute `s_point_on_line[l]` when count is 1.
    # For condition 3: for each l through y with count 0, check all N points.
    #   That's q-1 = 48 points per line. With up to 50 lines, that's 2400 checks per candidate.
    #   With 2400 candidates, that's ~5.7M checks per iteration. Doable.
    
    # Let's maintain s_point_on_line for lines with count 1.
    s_point_on_line = np.full(num_lines, -1, dtype=np.int32)
    for li in range(num_lines):
        if line_count[li] == 1:
            for x in lines[li]:
                if in_S[x]:
                    s_point_on_line[li] = x
                    break

    # Try to add points
    candidates = list(range(Q2))
    random.shuffle(candidates)
    
    additions = 0
    while time.time() < deadline:
        progress = False
        random.shuffle(candidates)
        for y in candidates:
            if time.time() > deadline:
                break
            if in_S[y]:
                continue
                
            # Check 1: y can be isolated
            has_zero_line = False
            for li in line_membership_arr[y]:
                if line_count[li] == 0:
                    has_zero_line = True
                    break
            if not has_zero_line:
                continue
                
            # Check 2: S points on count-1 lines remain isolated
            ok = True
            for li in line_membership_arr[y]:
                if line_count[li] == 1:
                    x = s_point_on_line[li]
                    if witness_count[x] <= 1:
                        ok = False
                        break
            if not ok:
                continue
                
            # Check 3: N points on count-0 lines remain covered
            for li in line_membership_arr[y]:
                if line_count[li] == 0:
                    for z in lines[li]:
                        if z != y and not in_S[z]:
                            if zero_count[z] <= 1:
                                ok = False
                                break
                    if not ok:
                        break
            if not ok:
                continue
                
            # Add y to S
            in_S[y] = True
            S.add(y)
            additions += 1
            progress = True
            
            # Update line_count and witness/zero counts
            for li in line_membership_arr[y]:
                old_c = line_count[li]
                line_count[li] = old_c + 1
                
                if old_c == 0:
                    # l becomes count 1. y is the S point.
                    s_point_on_line[li] = y
                    # y gets a witness
                    witness_count[y] += 1
                    # N points on l lose a zero line
                    for z in lines[li]:
                        if z != y and not in_S[z]:
                            zero_count[z] -= 1
                elif old_c == 1:
                    # l becomes count 2. s_point_on_line no longer valid.
                    x = s_point_on_line[li]
                    witness_count[x] -= 1
                    s_point_on_line[li] = -1
                # else: no change to witness/zero structures

            # Remove y from candidates
            # (we'll just skip it next time via in_S check)
            
        if not progress:
            break

    # Build N = complement of S
    N_points = []
    for i in range(Q2):
        if not in_S[i]:
            xa, xb, ya, yb = get_coords(i)
            N_points.append([[xa, xb], [ya, yb]])

    return np.array(N_points, dtype=np.int64)

# EVOLVE-BLOCK-END
