# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    Strategy:
    1. Precompute all lines in F_49^2.
    2. Start with S = empty (complement of N is empty, so N is the whole plane).
       Wait, the metric is 2401 - |N|, so we want to maximize |S| where S = F^2 \ N.
       This means we want to find the largest set S such that N = F^2 \ S is a Nikodym set.
       Condition: For every x in F^2, there is a line L through x such that L \ {x} subset N.
       Equivalently: For every x in F^2, there is a line L through x such that L \ {x} is disjoint from S.
       Or: For every x in F^2, there exists a line L through x such that S \cap L subset {x}.
       
       Let's rephrase:
       A set S is "valid" (meaning its complement is a Nikodym set) if:
       For every point x in F^2, there exists a line L passing through x such that
       (L \ {x}) ∩ S = ∅.
       
       This implies:
       - If x is in S: We need a line L through x such that no other point of S is on L.
         i.e., x is an isolated point of S on L.
       - If x is not in S: We need a line L through x such that no point of S is on L \ {x}.
         Since x not in S, this means L ∩ S = ∅.
         i.e., there is a line through x disjoint from S.
       
       So we want to maximize |S| subject to:
       1. For every x ∉ S, ∃ line L ∋ x such that L ∩ S = ∅.
       2. For every x ∈ S, ∃ line L ∋ x such that L ∩ S = {x}.
       
       This looks like we can greedily add points to S.
       Start with S = ∅.
       At each step, check if we can add a point x to S.
       If we add x:
       - Condition 2 for x: We need a line L through x such that L ∩ (S ∪ {x}) = {x}.
         i.e., L ∩ S = ∅.
       - Condition 2 for existing points y ∈ S:
         For each y, we previously had a line L_y such that L_y ∩ S_old = {y}.
         Now S_new = S_old ∪ {x}.
         We need L_y ∩ S_new ⊆ {y}.
         This holds if x ∉ L_y.
         If x ∈ L_y, then L_y ∩ S_new = {y, x} ≠ {y}, so the witness for y is broken.
         However, y might have another witness line L'_y such that L'_y ∩ S_new = {y}.
       - Condition 1 for points z ∉ S_new:
         If z was previously covered by a line L_z disjoint from S_old.
         If x ∉ L_z, L_z is still disjoint from S_new.
         If x ∈ L_z, L_z is no longer disjoint. We need a new line for z.
       
       This is complex to maintain incrementally.
       
       Alternative approach:
       The condition "For every x ∉ S, ∃ line L ∋ x with L ∩ S = ∅" is strong.
       It means the set of lines disjoint from S must cover F^2 \ S.
       Let D(S) be the set of lines disjoint from S.
       The union of lines in D(S) must contain F^2 \ S.
       Actually, for each x ∉ S, there must be *at least one* line through x in D(S).
       
       Let's try a different greedy strategy.
       Start with S = F^2.
       Try to remove points from S one by one.
       If we remove x from S:
       - x is now in N.
       - Condition 2 for x (now in N): Does not apply to S membership, but x needs a witness line L such that L \ {x} ⊆ N.
         This means L ∩ S ⊆ {x}. But x ∉ S, so L ∩ S = ∅.
         So x must lie on a line disjoint from S.
       - Condition 1 for x (now in N): Same as above.
       - Condition for other points:
         If y ∈ S: y needs a line L_y with L_y ∩ S = {y}. Removing x from S only helps this (L_y ∩ S_new ⊆ L_y ∩ S_old). So existing S points are fine.
         If z ∈ N (z ≠ x): z needs a line L_z with L_z ∩ S = ∅.
         If x was on L_z, removing x from S might break the disjointness? No, L_z ∩ S_old = ∅ implies L_z ∩ S_new = ∅ since S_new ⊆ S_old.
         Wait, S_new = S_old \ {x}. So S_new is smaller.
         If L_z ∩ S_old = ∅, then L_z ∩ S_new = ∅.
         So removing points from S never breaks the condition for other points.
         It only adds the requirement that the removed point x must lie on a line disjoint from the *new* S.
       
       So the algorithm is:
       S = F^2
       For each point x (in some order):
           If x lies on a line L such that L ∩ S = ∅:
               Remove x from S.
       
       Wait, if S = F^2, no line is disjoint from S (since every line has points in S).
       So we can't remove anything initially?
       
       Let's re-read:
       N = F^2 \ S.
       Condition: For every x in F^2, ∃ line L ∋ x such that L \ {x} ⊆ N.
       
       If x ∈ N: L \ {x} ⊆ N. Since x ∈ N, this is equivalent to L ⊆ N ∪ {x}.
       Since x ∈ N, L ⊆ N.
       So if x ∈ N, x must lie on a line L fully contained in N.
       
       If x ∈ S: L \ {x} ⊆ N. Since x ∈ S, L ∩ S ⊆ {x}.
       So if x ∈ S, x must lie on a line L that intersects S only at x.
       
       So:
       1. Every point in N lies on a line fully contained in N.
       2. Every point in S lies on a line that intersects S only at that point.
       
       This implies:
       - If S is non-empty, any line containing a point of S can contain at most one point of S.
       - If N is non-empty, any point in N is on a line fully in N.
       
       Let's try to construct S greedily by *adding* points to S?
       Start S = ∅.
       N = F^2.
       Condition 1: Every x in N lies on a line fully in N.
       Since N = F^2, every line is fully in N. So Condition 1 holds for all x.
       Condition 2: Vacuously true for S = ∅.
       
       Now try to add x to S.
       New S' = S ∪ {x}. New N' = N \ {x}.
       Check Condition 1 for N':
       For every y in N', y must lie on a line L fully in N'.
       Previously y was on some line L_y fully in N.
       If x ∉ L_y, then L_y is fully in N' (since N' = N \ {x} and x not in L_y).
       If x ∈ L_y, then L_y is not fully in N' (it contains x which is in S').
       So y loses its witness if x is on all witness lines of y.
       
       Check Condition 2 for S':
       For x: x must lie on a line L_x such that L_x ∩ S' = {x}.
       i.e., L_x ∩ S = ∅.
       For y in S: y must lie on a line L_y such that L_y ∩ S' = {y}.
       i.e., L_y ∩ S ⊆ {y} and x ∉ L_y (or x=y which is impossible).
       So if x is on the witness line of y, y loses its witness.
       
       So adding x to S is valid if:
       1. There exists a line L_x through x such that L_x ∩ S = ∅. (x has a witness in new S)
       2. For all y in S, there exists a witness line L_y for y in S' (i.e., L_y ∩ S' = {y}).
          This means if x is on the *only* witness line of y, y loses validity.
          But y might have multiple witness lines.
       3. For all z in N', there exists a witness line L_z for z in N' (i.e., L_z ⊆ N').
          This means if x is on the *only* witness line of z, z loses validity.
       
       This is getting complicated to track "only witness".
       
       Let's simplify.
       Notice that if S is a set of points no two of which are on the same line, then Condition 2 is satisfied for all y in S by any line through y that doesn't contain another point of S.
       Actually, if no two points of S are collinear, then for any y in S, ANY line through y contains no other point of S. So Condition 2 is automatically satisfied for all y in S.
       
       What about Condition 1?
       If S is a set of points no two of which are collinear, does it help N?
       N = F^2 \ S.
       We need every z in N to be on a line fully in N.
       A line is fully in N if it contains no points of S.
       So we need: For every z in N, there is a line through z disjoint from S.
       
       So the problem reduces to:
       Find the largest set S such that:
       1. No two points in S are collinear. (S is a cap set / arc)
       2. Every point z not in S lies on a line disjoint from S.
       
       Condition 1 implies S is an arc. In F_q^2, the maximum size of an arc is q+1 (if q is even) or q (if q is odd)?
       For q=49 (odd), max arc size is q=49? Or q+1?
       Actually, a conic in PG(2, q) has q+1 points, no three collinear.
       But here we are in affine plane A(2, q).
       The maximum size of a cap (no 3 collinear) in A(2, q) is...
       Wait, "no two points collinear" is impossible for more than 2 points?
       No, "no two points on the same line" means every line contains at most 1 point of S.
       This is a set of points such that every line contains at most one point of S.
       This is a "1-arc" or just a set with no two collinear?
       If every line contains at most one point of S, then S is a set of points such that no line contains two points of S.
       This means S is a set of points with pairwise distinct slopes from any origin?
       Actually, if x, y in S, the line xy contains x and y. So if |S| > 1, the line through any two points of S contains two points of S.
       So Condition 1 "No two points in S are collinear" means |S| <= 1?
       No, "collinear" usually means on the same line.
       If x, y in S, they are on the line xy. So the line xy contains x and y.
       So if |S| >= 2, there is a line containing two points of S.
       So Condition 2 for y in S: "L_y ∩ S = {y}".
       This means there must be a line through y that does NOT contain any other point of S.
       This is possible even if other lines through y contain other points of S.
       So S does NOT need to be an arc (no 3 collinear) or a cap.
       It just needs that for each y in S, there is at least one line through y with no other points of S.
       
       Let's restart the greedy addition with this understanding.
       
       State: S (set of points), N = F^2 \ S.
       For each x in F^2, maintain:
       - witness_lines_S[x]: set of lines through x such that L ∩ S = {x} (if x in S) or L ∩ S = ∅ (if x not in S? No, for x in S, we need L ∩ S = {x}).
         Actually, let's define:
         valid_lines(x, S):
           if x in S: { L : x in L and L ∩ S = {x} }
           if x not in S: { L : x in L and L ∩ S = ∅ }
         
         x is "covered" if valid_lines(x, S) is not empty.
         
       We want to add x to S if:
       1. x is covered in the new state S' = S ∪ {x}.
          For x (now in S'): valid_lines(x, S') = { L : x in L and L ∩ S' = {x} } = { L : x in L and L ∩ S = ∅ }.
          So we need at least one line through x disjoint from S.
       2. All other points y are still covered in S'.
          For y in S: valid_lines(y, S') = { L : y in L and L ∩ S' = {y} } = { L : y in L and L ∩ S = {y} and x notin L }.
          So y loses coverage if all its valid lines in S contain x.
          For z in N' (z notin S'): valid_lines(z, S') = { L : z in L and L ∩ S' = ∅ } = { L : z in L and L ∩ S = ∅ and x notin L }.
          So z loses coverage if all its valid lines in S contain x.
       
       So we can greedily add x if:
       - There is a line through x disjoint from S.
       - For all y in S, there is a line through y disjoint from S ∪ {x} (i.e., disjoint from S and not containing x).
       - For all z in N, there is a line through z disjoint from S ∪ {x} (i.e., disjoint from S and not containing x).
       
       This is still heavy to check for all y, z.
       
       However, note that if we pick x such that x is on a line L disjoint from S,
       then for any y in S, if y is not on L, then L is not a witness for y.
       But y needs *some* witness.
       
       Let's try a simpler heuristic:
       Start with S = ∅.
       While True:
          Find a point x such that adding x to S is valid.
          If found, add it.
          Else, stop.
       
       To speed up, we can pick x randomly or in a fixed order.
       Checking validity of adding x:
       - Check if ∃ line L ∋ x with L ∩ S = ∅.
         If not, x cannot be added.
       - If yes, check if any other point loses coverage.
         This is the expensive part.
         
       Maybe we can just try to add points and if it fails, skip?
       Or use a local search / hill climbing.
       
       Given the time limit (600s), and q=49, total points 2401.
       Lines: 49 * 50 = 2450.
       
       Let's implement the greedy addition with full validity check.
       If it's too slow, we'll optimize.
       
       Precomputation:
       - List of all lines (as sets of point indices).
       - For each point, list of lines passing through it.
       
       State:
       - S: set of point indices.
       - For each point x, count of valid lines.
         valid_count[x] = number of lines L through x such that:
           if x in S: L ∩ S = {x}
           if x notin S: L ∩ S = ∅
         
       When we add x to S:
       - Update S.
       - For each line L through x:
         - Before: L ∩ S = ∅ (since we checked this for x to be addable? No, we check if *at least one* is disjoint).
           Actually, if x is added, for lines through x:
           - If L ∩ S_old = ∅, then L ∩ S_new = {x}.
             So for x, these lines become valid witnesses (for x in S).
             For other points y in L (y notin S_old), L was a valid witness (disjoint). Now L contains x, so L ∩ S_new = {x} ≠ ∅. So L is no longer a valid witness for y (since y notin S, needs disjoint).
             So for each y in L \ {x}, valid_count[y] decreases by 1.
           - If L ∩ S_old != ∅, then L ∩ S_new = L ∩ S_old ∪ {x}.
             If L ∩ S_old had exactly one point y (y != x), then L ∩ S_new has 2 points.
             For y (in S_old): L was a valid witness (L ∩ S_old = {y}). Now L ∩ S_new = {y, x} != {y}. So L is no longer a valid witness for y. valid_count[y] decreases by 1.
             For x: L is not a valid witness (L ∩ S_new != {x}).
             For other z in L: L was not a valid witness (L ∩ S_old != ∅). Still not.
             
             If L ∩ S_old had >= 2 points, nothing changes for validity counts?
             For y in S_old on L: L was not a valid witness (L ∩ S_old != {y}). Still not.
             For z notin S_old on L: L was not a valid witness. Still not.
             
       So the update is local to the lines through x.
       
       Algorithm:
       1. Initialize S = ∅.
       2. Initialize valid_count[x] for all x.
          For x notin S: valid_count[x] = number of lines through x (all disjoint from S=∅).
          So valid_count[x] = 50 for all x.
       3. Create a list of candidate points x with valid_count[x] > 0.
          Actually, for x to be added, we need at least one line through x disjoint from S.
          This is equivalent to: x has a line L with L ∩ S = ∅.
          Note: valid_count[x] (for x notin S) is exactly the number of such lines.
          So candidates are x notin S with valid_count[x] > 0.
       4. Pick a candidate x (e.g., one with smallest valid_count? or random?).
          Let's pick one with smallest valid_count to "commit" faster? Or largest to be safe?
          Let's try smallest valid_count, as it's most constrained.
       5. Check if adding x keeps all valid_count > 0.
          Simulate the update:
          - For each line L through x:
            - If L ∩ S = ∅:
              - For each y in L \ {x}:
                - valid_count[y] -= 1
                - If valid_count[y] == 0, adding x is invalid.
            - If L ∩ S != ∅:
              - Let Y = L ∩ S.
              - If |Y| == 1:
                - Let y = the single point in Y.
                - valid_count[y] -= 1
                - If valid_count[y] == 0, adding x is invalid.
              - If |Y| >= 2:
                - No change to valid_count.
          - Also, x itself:
            - valid_count[x] becomes the number of lines through x with L ∩ S = ∅.
            - This is the old valid_count[x].
            - We need valid_count[x] > 0 for x to be a candidate, which we already checked.
            - After adding x, x is in S. Its valid witnesses are lines with L ∩ S_new = {x}, which are exactly the lines with L ∩ S_old = ∅.
            - So new valid_count[x] = old valid_count[x].
            - We need new valid_count[x] > 0, which is true.
            
       6. If valid, commit the change:
          - Add x to S.
          - Apply the valid_count updates.
          - Remove x from candidates.
          - Update candidates for affected points y.
       7. Repeat until no candidates or time limit.
       
       This should be fast.
       Number of points 2401.
       Each addition affects at most 50 lines * 49 points = 2450 points.
       Total operations ~ 2401 * 2450 ~ 6 million. Very fast.
       
       Let's implement this.

    """
    start_time = time.time()
    deadline = start_time + 550.0  # Safety margin

    p = 7
    q = p * p
    k = 3  # alpha^2 = 3

    # Field operations
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def mul(a, b):
        return ((a[0] * b[0] + a[1] * b[1] * k) % p, (a[0] * b[1] + a[1] * b[0]) % p)

    # Generate all points in F_49^2
    # Index: x = (xa, xb), y = (ya, yb)
    # idx = ((xa * p + xb) * p + ya) * p + yb
    
    num_points = q * q
    
    # Precompute lines
    # Directions: (1, m) for m in F_49, and (0, 1).
    # Total q+1 = 50 directions.
    
    # lines: list of sets of point indices
    # point_to_lines: dict mapping point index to list of line indices
    lines = []
    point_to_lines = {i: [] for i in range(num_points)}
    
    seen_lines = set()
    
    # Directions
    # v = (vx, vy)
    # (1, m) for m in F_49
    # (0, 1)
    
    # For (1, m): y = m*x + c
    # For (0, 1): x = c
    
    # Iterate over all lines
    # For (1, m):
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    line_points = []
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            mx = mul(m, x)
                            y = add(mx, c)
                            idx = ((xa * p + xb) * p + y[0]) * p + y[1]
                            line_points.append(idx)
                    
                    line_tuple = tuple(sorted(line_points))
                    if line_tuple not in seen_lines:
                        seen_lines.add(line_tuple)
                        line_set = set(line_points)
                        lines.append(line_set)
                        line_id = len(lines) - 1
                        for idx in line_points:
                            point_to_lines[idx].append(line_id)
    
    # For (0, 1): x = c
    for ca in range(p):
        for cb in range(p):
            c = (ca, cb)
            line_points = []
            for ya in range(p):
                for yb in range(p):
                    idx = ((ca * p + cb) * p + ya) * p + yb
                    line_points.append(idx)
                    
            line_tuple = tuple(sorted(line_points))
            if line_tuple not in seen_lines:
                seen_lines.add(line_tuple)
                line_set = set(line_points)
                lines.append(line_set)
                line_id = len(lines) - 1
                for idx in line_points:
                    point_to_lines[idx].append(line_id)

    # Initialize S and valid_count
    S = set()
    valid_count = [0] * num_points
    
    # For S = empty, valid_count[x] = number of lines through x = 50
    for x in range(num_points):
        valid_count[x] = len(point_to_lines[x])
        
    # Candidates: points not in S with valid_count > 0
    # We will use a list and update it
    # To pick "smallest valid_count", we can just scan or use a heap.
    # Given the small size, scanning is fine.
    
    # To speed up, we can maintain a list of active candidates.
    # But checking if a point is still a candidate (not in S, valid_count > 0) is O(1).
    
    # We'll just iterate through all points and pick the best candidate each time.
    # Or better, maintain a set of candidates and pick one.
    
    # Let's use a simple loop:
    # While True:
    #   Find x notin S with valid_count[x] > 0 that minimizes valid_count[x] (or just pick the first one).
    #   Check if adding x is valid.
    #   If valid, add x.
    #   Else, mark x as "bad" (cannot be added now, and won't be added later? No, S grows, so valid_count decreases. If it's not addable now, it won't be addable later. So we can skip it forever.)
    
    # Actually, if x is not addable now because some y would lose coverage,
    # as S grows, y's valid_count only decreases. So x will never be addable.
    # So we can permanently remove x from consideration.
    
    candidates = set(range(num_points))
    
    while candidates:
        if time.time() > deadline:
            break
            
        # Pick a candidate
        # Heuristic: pick one with smallest valid_count
        best_x = None
        best_count = float('inf')
        for x in candidates:
            if valid_count[x] > 0 and valid_count[x] < best_count:
                best_count = valid_count[x]
                best_x = x
                
        if best_x is None:
            break
            
        x = best_x
        
        # Check if adding x is valid
        # Simulate updates
        # We need to check if any y would have valid_count[y] become 0.
        
        # Collect points that would be affected
        # affected = {} # point -> delta
        
        valid = True
        # We need to know for each line L through x, what is L ∩ S?
        # We can compute this on the fly.
        
        # To avoid double counting, we can just check the conditions directly.
        
        # 1. For each line L through x:
        for line_id in point_to_lines[x]:
            L = lines[line_id]
            # Count points in L that are in S
            # Since we only need to know if |L ∩ S| is 0, 1, or >=2,
            # and we need to identify the single point if |L ∩ S| == 1.
            
            s_points = []
            for y in L:
                if y in S:
                    s_points.append(y)
                    if len(s_points) > 1:
                        break
            
            if len(s_points) == 0:
                # L ∩ S = ∅
                # For each y in L \ {x}, valid_count[y] decreases by 1.
                for y in L:
                    if y == x:
                        continue
                    # Check if y is still a candidate or in S?
                    # We only care about points that are currently "covered" (valid_count > 0).
                    # If valid_count[y] == 1, it becomes 0 -> invalid.
                    if valid_count[y] == 1:
                        valid = False
                        break
                if not valid:
                    break
                    
            elif len(s_points) == 1:
                # L ∩ S = {y0}
                # valid_count[y0] decreases by 1.
                y0 = s_points[0]
                if valid_count[y0] == 1:
                    valid = False
                    break
            else:
                # L ∩ S >= 2, no change to valid_count
                pass
        
        if not valid:
            # Remove x from candidates
            candidates.discard(x)
            continue
            
        # Commit: add x to S
        S.add(x)
        candidates.discard(x)
        
        # Update valid_count
        for line_id in point_to_lines[x]:
            L = lines[line_id]
            s_points = []
            for y in L:
                if y in S and y != x: # x is just added, so exclude it from "old S" logic?
                    # Wait, S now contains x.
                    # We need to update based on the transition S_old -> S_new.
                    # It's easier to just recompute the change.
                    pass
            
            # Let's re-derive the update for S_new = S_old ∪ {x}
            # We are in the commit phase, so S already contains x.
            # We need to apply the delta calculated in the check phase.
            
            # Let's just recompute the intersection with S_old?
            # We don't have S_old.
            # But we know x was not in S_old.
            # So L ∩ S_old = (L ∩ S_new) \ {x}.
            
            s_new_points = [y for y in L if y in S]
            # Remove x from consideration for "old S" logic
            s_old_points = [y for y in s_new_points if y != x]
            
            if len(s_old_points) == 0:
                # L ∩ S_old = ∅
                # For each y in L \ {x}, valid_count[y] -= 1
                for y in L:
                    if y != x:
                        valid_count[y] -= 1
            elif len(s_old_points) == 1:
                # L ∩ S_old = {y0}
                y0 = s_old_points[0]
                valid_count[y0] -= 1
            else:
                # No change
                pass
            
            # Update x's valid_count
            # x is in S. valid_count[x] = number of lines through x with L ∩ S_new = {x}
            # This is the number of lines through x with L ∩ S_old = ∅.
            # We can compute this once outside the loop.
            
        # Recompute valid_count[x]
        # x is in S.
        # valid_count[x] = number of lines L through x such that L ∩ S = {x}
        # i.e., L ∩ (S \ {x}) = ∅
        count_x = 0
        for line_id in point_to_lines[x]:
            L = lines[line_id]
            # Check if L contains any point of S other than x
            has_other = False
            for y in L:
                if y != x and y in S:
                    has_other = True
                    break
            if not has_other:
                count_x += 1
        valid_count[x] = count_x

    # Construct N = F^2 \ S
    N = [i for i in range(num_points) if i not in S]
    
    # Convert indices to points
    points = []
    for idx in N:
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        points.append([[xa, xb], [ya, yb]])
        
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
