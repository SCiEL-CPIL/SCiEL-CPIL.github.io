import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by greedily maximizing the complement set S.
    """
    p = 7
    q = p * p
    k_const = 3 # alpha^2 = 3, smallest QR for 7 is 2? 
    # Check QRs for 7: 1^2=1, 2^2=4, 3^2=2. So 2 is QR. 3 is QNR.
    # The prompt says alpha^2 = 3. So k_const = 3.
    
    # Field operations for F_49 = F_7[alpha], alpha^2 = 3
    # Element is (a, b) representing a + b*alpha
    # We map (a, b) to index a * 7 + b
    
    def add_f(a, b, c, d):
        return ((a + c) % p, (b + d) % p)
    
    def mul_f(a, b, c, d):
        # (a + b*alpha) * (c + d*alpha) = ac + ad*alpha + bc*alpha + bd*alpha^2
        # = (ac + 3bd) + (ad + bc)*alpha
        return ((a * c + 3 * b * d) % p, (a * d + b * c) % p)
    
    # Precompute lines
    # A line is defined by a point (x0, y0) and a direction v = (1, m) or (0, 1)
    # Directions:
    # 1. Vertical: v = (0, 1). Equation x = x0.
    # 2. Slope m: v = (1, m). Equation y - y0 = m * (x - x0).
    
    # Total points: 2401
    # We represent points as flat indices 0..2400
    # point index idx -> xa = idx // 49, xb = (idx // 7) % 7? 
    # Let's use a consistent mapping.
    # Let point P be (x, y) where x=(xa, xb), y=(ya, yb).
    # Flat index: ((xa * p + xb) * p + ya) * p + yb ? No, 4^2 = 16? No, 7^4 = 2401.
    # idx = xa * 343 + xb * 49 + ya * 7 + yb
    
    def to_idx(xa, xb, ya, yb):
        return xa * 343 + xb * 49 + ya * 7 + yb
    
    def from_idx(idx):
        yb = idx % 7
        idx //= 7
        ya = idx % 7
        idx //= 7
        xb = idx % 7
        idx //= 7
        xa = idx
        return xa, xb, ya, yb

    # Generate all lines
    # lines is a list of lists of point indices
    lines = []
    
    # Vertical lines: x = (xa, xb) fixed, y varies
    for xa in range(p):
        for xb in range(p):
            line_pts = []
            for ya in range(p):
                for yb in range(p):
                    line_pts.append(to_idx(xa, xb, ya, yb))
            lines.append(line_pts)
            
    # Sloped lines: y - y0 = m * (x - x0)
    # m is in F_49
    # We iterate over all m and all base points (x0, y0), but avoid duplicates.
    # A line is uniquely defined by its set of points.
    # To avoid duplicates, we can generate lines by slope and intercept?
    # Equation: y = m*x + c.
    # For each m, and each c, we have a line.
    # There are q slopes and q intercepts -> q^2 lines.
    # Plus q vertical lines. Total q^2 + q = 49^2 + 49 = 2401 + 49 = 2450 lines.
    
    # We need to map m and c to the set of points.
    # For a fixed m, the line y = m*x + c contains points (x, m*x+c).
    
    # We need to iterate m in F_49.
    # m = (ma, mb).
    # c = (ca, cb).
    
    for ma in range(p):
        for mb in range(p):
            for ca in range(p):
                for cb in range(p):
                    line_pts = []
                    for xa in range(p):
                        for xb in range(p):
                            # y = m * x + c
                            # (ya, yb) = m*(xa, xb) + (ca, cb)
                            ya, yb = mul_f(ma, mb, xa, xb)
                            ya, yb = add_f(ya, yb, ca, cb)
                            line_pts.append(to_idx(xa, xb, ya, yb))
                    lines.append(line_pts)
                    
    # Total lines should be 2450.
    # Check: 49 vertical + 49*49 sloped = 49 + 2401 = 2450. Correct.
    
    # Precompute for each point, which lines pass through it.
    point_lines = [[] for _ in range(q*q)]
    for i, line_pts in enumerate(lines):
        for pt in line_pts:
            point_lines[pt].append(i)
            
    # Initialize S (complement of N) as empty set
    S = np.zeros(q*q, dtype=bool)
    
    # W[y] = number of witness lines for y
    # A line L is a witness for y if y in L and (L \ {y}) subset of N
    # (L \ {y}) subset of N  <=>  (L \ {y}) is disjoint from S
    # <=>  L \cap S is subset of {y}
    
    # Initially S is empty, so every line is a witness for every point on it.
    # Each point is on q+1 = 50 lines.
    W = np.full(q*q, q+1, dtype=np.int32)
    
    start_time = time.time()
    time_limit = 580.0 # Safety margin
    
    # Greedy loop
    # We want to add points to S.
    # Candidate x is valid if:
    # 1. x not in S
    # 2. There exists a line L through x such that L \cap S = {x} (i.e., L contains no other S points)
    #    This ensures x has at least 1 witness line after addition.
    # 3. For all y != x, if W[y] would become 0, we cannot add x.
    #    W[y] decreases by 1 for each line L through x and y such that L was a witness for y.
    #    L is a witness for y iff L \cap S \subseteq {y}.
    #    Since x \in L and x \notin S (currently), L \cap S \subseteq {y} implies y \in S or L \cap S is empty.
    #    Wait, if y \notin S, L \cap S is empty (if no other S in L). Then L is a witness.
    #    If y \in S, L \cap S contains y. If it contains only y, L is a witness.
    #    If L contains x and y, and x \notin S, y \in S, and no other S in L, then L \cap S = {y}.
    #    So, for a fixed x, the lines L through x that cause W[y] to drop are those where:
    #    - y is on L
    #    - L \cap S \subseteq {y}
    
    # To speed up, we can just iterate through all points x not in S, and check validity.
    # If valid, add to S.
    # To avoid O(N^2) or worse, we can be a bit smart, but N=2401 is small.
    # Number of iterations: |S| will grow to maybe 200-500.
    # Each iteration checks 2401 candidates.
    # Check cost: For a candidate x, we look at 50 lines. For each line, we look at 49 points.
    # Total check cost per candidate: 50 * 49 = 2450 ops.
    # Total cost per iteration: 2401 * 2450 ~ 6M ops.
    # 500 iterations * 6M = 3B ops. This might be too slow in Python (600s).
    
    # Optimization:
    # We can maintain a list of "valid" candidates.
    # Or, we can just try a few random candidates per step.
    # Or, we can precompute which lines are "clean" (contain no S).
    # A line L is "clean" for x if L \cap S \subseteq {x}.
    
    # Let's try a simpler heuristic:
    # Randomly pick a point x not in S.
    # Check if it can be added.
    # If yes, add it.
    # If no, try next.
    # To make it faster, we can limit the number of checks per step.
    
    # Better approach:
    # Iterate through points. If a point is "safe" to add, add it.
    # "Safe" means:
    # 1. It has a clean line (no other S).
    # 2. It doesn't kill any witness for any y.
    
    # Let's implement the check efficiently.
    # For a given x:
    # 1. Check if x has a clean line.
    #    For each line L through x:
    #       count_S_in_L = 0
    #       for y in L:
    #          if S[y]: count_S_in_L += 1
    #    If any L has count_S_in_L == 0 (since x is not in S yet, it's 0 other points), then x has a witness.
    #    Actually, if x is not in S, "L \cap S" is just the set of S points in L.
    #    We need L \cap S to be empty for L to be a witness for x (since x \notin S, witness means L \ {x} \subseteq N, which means L \cap S is empty).
    #    So condition 1: There is a line L through x with no points in S.
    
    # 2. Check impact on others.
    #    For each line L through x:
    #       For each y in L (y != x):
    #          If L was a witness for y (i.e., L \cap S \subseteq {y}), then W[y] decreases.
    #          Note: Since x \notin S, L \cap S \subseteq {y} is equivalent to (L \cap S) is empty OR (L \cap S) == {y}.
    #          If L \cap S is empty, then L is a witness for ALL points on L.
    #          If L \cap S == {y}, then L is a witness for y only (and not for other points on L).
    
    #    So, if we add x to S:
    #    - Any line L through x that was "empty" (no S) becomes "containing x".
    #    - For such a line L, it was a witness for all y in L.
    #    - After adding x, L is no longer a witness for any y in L (because x \in L \cap S and x \neq y).
    #    - So W[y] decreases by 1 for ALL y in L (y != x).
    #    - What if L already had some S points?
    #      If L had S points, it was NOT a witness for any y (unless the only S point was y itself).
    #      If L \cap S = {y}, then L was a witness for y.
    #      After adding x, L \cap S = {y, x}. Not a witness for y.
    #      So W[y] decreases by 1.
    #      For other z in L, L was not a witness for z (since y \in S, z \neq y). So W[z] unchanged.
    
    #    So, the condition is:
    #    For every line L through x:
    #       Let S_in_L = L \cap S.
    #       If S_in_L is empty:
    #          All y in L (y != x) lose 1 witness.
    #       If S_in_L == {y} for some y in L:
    #          y loses 1 witness.
    #       Else:
    #          No one loses a witness from this line.
    
    #    We need to ensure W[y] >= 1 for all y after these decrements.
    
    # Implementation:
    # We'll maintain W.
    # To check candidate x:
    #   1. Find if there is a line L through x with S_in_L empty. If not, skip x.
    #   2. Calculate the decrements for all y.
    #   3. If any W[y] - decrement[y] < 1, skip x.
    #   4. Else, apply decrements, add x to S, W[x] = 0 (or doesn't matter), continue.
    
    # To speed up step 2, we can precompute the list of lines through x.
    
    # We will do a single pass of greedy addition.
    # To ensure we finish in time, we will limit the number of candidates checked per "round" or just break if time is up.
    
    # Let's shuffle the points to avoid bias.
    order = np.arange(q*q)
    np.random.shuffle(order)
    
    added = 0
    for x in order:
        if time.time() - start_time > time_limit:
            break
        if S[x]:
            continue
            
        # Check condition 1: Has a clean line?
        has_clean = False
        for line_idx in point_lines[x]:
            line_pts = lines[line_idx]
            # Check if any point in line is in S
            # Since x is not in S, we just check others
            clean = True
            for y in line_pts:
                if y != x and S[y]:
                    clean = False
                    break
            if clean:
                has_clean = True
                break
                
        if not has_clean:
            continue
            
        # Check condition 2: Does it kill any witness?
        # We need to know for each y, how many witnesses it loses.
        # loss = np.zeros(q*q, dtype=np.int32)
        # This is too expensive to allocate.
        # Instead, we can just check if W[y] > 1 for all y that would lose a witness.
        # Or, we can simulate the loss.
        
        # Let's collect the set of (y, count) pairs.
        # Since max loss per y is small (at most 1 per line, 50 lines), we can use a dict or just check on the fly.
        # Actually, a point y can be on multiple lines through x? No, only one line through x and y.
        # So each y loses at most 1 witness from the lines through x.
        # So we just need to check if W[y] >= 2 for all y that lose a witness.
        
        # Identify which y's lose a witness.
        # For each line L through x:
        #   S_in_L = [y for y in L if S[y]]
        #   if len(S_in_L) == 0:
        #       All y in L (y!=x) lose 1.
        #   elif len(S_in_L) == 1:
        #       y0 = S_in_L[0]
        #       y0 loses 1.
        
        # We can just iterate and check W[y] >= 2.
        # If we find any y with W[y] == 1 that loses a witness, we reject x.
        
        reject = False
        # We need to be careful not to double count if a y is affected by multiple lines?
        # No, y is on exactly one line with x.
        
        for line_idx in point_lines[x]:
            line_pts = lines[line_idx]
            s_count = 0
            s_point = -1
            for y in line_pts:
                if y != x and S[y]:
                    s_count += 1
                    s_point = y
            if s_count == 0:
                # All y in L (y!=x) lose 1
                for y in line_pts:
                    if y != x:
                        if W[y] < 2:
                            reject = True
                            break
            elif s_count == 1:
                # s_point loses 1
                if W[s_point] < 2:
                    reject = True
                    break
            else:
                pass # No one loses a witness from this line
                
            if reject:
                break
                
        if reject:
            continue
            
        # It is safe to add x
        S[x] = True
        added += 1
        
        # Update W
        for line_idx in point_lines[x]:
            line_pts = lines[line_idx]
            s_count = 0
            s_point = -1
            for y in line_pts:
                if y != x and S[y]:
                    s_count += 1
                    s_point = y
            if s_count == 0:
                for y in line_pts:
                    if y != x:
                        W[y] -= 1
            elif s_count == 1:
                W[s_point] -= 1
                
    # Construct N
    N_indices = []
    for i in range(q*q):
        if not S[i]:
            N_indices.append(i)
            
    # Convert to array of shape (k, 2, 2)
    result = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for i, idx in enumerate(N_indices):
        xa, xb, ya, yb = from_idx(idx)
        result[i, 0, 0] = xa
        result[i, 0, 1] = xb
        result[i, 1, 0] = ya
        result[i, 1, 1] = yb
        
    return result
