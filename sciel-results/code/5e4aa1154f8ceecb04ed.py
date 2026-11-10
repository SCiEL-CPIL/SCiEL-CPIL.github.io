# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 610.0  # safety margin before 630s kill

p = 7
K = 3  # alpha^2 = 3
N_TOTAL = p**4  # 2401
N_DIRS = p * p + 1  # 50 directions: (0,1) and (1, m)

def _point_to_idx(xa, xb, ya, yb):
    return ((xa * p + xb) * p * p + (ya * p + yb))

def _idx_to_point(idx):
    yb = idx % p
    ya = (idx // p) % p
    xb = (idx // (p * p)) % p
    xa = (idx // (p * p * p)) % p
    return xa, xb, ya, yb

def _precompute_lines():
    """
    Precompute all lines in F_49^2.
    Returns:
      lines: list of arrays, each of shape (49,), containing point indices on that line.
      line_dirs: list of direction indices for each line.
      point_to_lines: for each point, list of line indices it belongs to.
    """
    lines = []
    line_dirs = []
    point_to_lines = [[] for _ in range(N_TOTAL)]
    
    # Direction 0: v = (0, 1)
    # Lines parameterized by (xa, xb)
    # c_a, c_b are effectively ya, yb for the "base" but since x is fixed, 
    # the line is just all y for a fixed x.
    # Actually, for v=(0,1), x is constant. So line is determined by (xa, xb).
    for xa in range(p):
        for xb in range(p):
            line_idx = len(lines)
            line_points = np.empty(49, dtype=np.int32)
            for i, ya in enumerate(range(p)):
                for j, yb in enumerate(range(p)):
                    idx = _point_to_idx(xa, xb, ya, yb)
                    line_points[i * p + j] = idx
                    point_to_lines[idx].append(line_idx)
            lines.append(line_points)
            line_dirs.append(0)
            
    # Directions 1..49: v = (1, m) for m = ma + mb*alpha
    # Line determined by (c_a, c_b) where:
    # c_a = ya - ma*xa - 3*mb*xb
    # c_b = yb - ma*xb - mb*xa
    for ma in range(p):
        for mb in range(p):
            dir_idx = 1 + ma * p + mb
            for ca in range(p):
                for cb in range(p):
                    line_idx = len(lines)
                    line_points = np.empty(49, dtype=np.int32)
                    for i, xa in enumerate(range(p)):
                        for j, xb in enumerate(range(p)):
                            ya = (ca + ma * xa + 3 * mb * xb) % p
                            yb = (cb + ma * xb + mb * xa) % p
                            idx = _point_to_idx(xa, xb, ya, yb)
                            line_points[i * p + j] = idx
                            point_to_lines[idx].append(line_idx)
                    lines.append(line_points)
                    line_dirs.append(dir_idx)
    
    return lines, line_dirs, point_to_lines

def nikodym_set() -> np.ndarray:
    start = time.time()
    
    # Precomputation
    lines, line_dirs, point_to_lines = _precompute_lines()
    n_lines = len(lines) # 2450
    
    # Convert lines to a 2D array for vectorization
    # lines_arr: shape (n_lines, 49)
    lines_arr = np.array(lines, dtype=np.int32)
    
    # Initial S: Empty set? Or small seed?
    # Let's start with empty S and try to grow it.
    # Or start with a random sparse set.
    # Actually, starting with empty S is valid (N = all). 
    # We want to maximize |S|.
    
    # Strategy: 
    # 1. Initialize S as empty.
    # 2. Compute bad counts. Initially all 0.
    # 3. Greedily add points that are "safe".
    # 4. To escape local minima, use a "shake" step: randomly remove some points from S, then re-add.
    
    # However, the greedy addition in the previous program was slow because it checked 2400 points per addition.
    # With vectorization, we can check all points at once.
    
    # Let's define the state:
    # in_S: boolean array of shape (2401,)
    # bad: int array of shape (2401,), bad[x] = number of directions where punctured line through x hits S.
    
    # Updating bad counts:
    # If we add point c to S:
    # For each line L containing c:
    #   Let k = count of S points on L before adding c.
    #   For each point x on L, x != c:
    #     The punctured line through x in direction dir(L) now contains c (which is in S).
    #     So bad[x] increases by 1 IF this was the first S point on L (k==0) OR if we are tracking "exists S".
    #     Wait, bad[x] is the count of DIRECTIONS.
    #     A direction v is "bad" for x if there exists s in S on the line through x in dir v, s != x.
    #     So bad[x] is the number of lines L through x such that (count_S(L) - (1 if x in S else 0)) >= 1.
    
    # This is tricky to update incrementally with simple counts because of the "exists" logic per direction.
    # Let's stick to the definition: bad[x] = number of lines L through x such that count_S(L) > (1 if x in S else 0).
    
    # Let's try a different vectorized approach:
    # Maintain count_S_on_line: shape (n_lines,)
    # Maintain in_S: shape (N_TOTAL,)
    
    # To check if a point x is "valid" (bad[x] < 50):
    # We need to know, for each of the 50 lines through x, if count_S(line) > (1 if x in S else 0).
    
    # This is still a bit heavy to compute from scratch for all x.
    
    # Let's go back to the greedy addition but vectorized.
    # Candidate set C = points not in S.
    # For each c in C, we want to know if adding c makes any x invalid.
    # x becomes invalid if bad[x] was 49 and adding c increments bad[x].
    # bad[x] increments if c is on a line L through x, and L had 0 S points (if x not in S) or 1 S point (if x in S, and x != c).
    # Actually, if x in S and x != c, and L had 1 S point (which was x), then adding c makes count_S(L)=2.
    # Then for x, the condition is count_S(L) > 1. 2 > 1 is True. Previously 1 > 1 was False.
    # So bad[x] increases.
    
    # So, adding c increases bad[x] for all x such that:
    # 1. x is on a line L with c.
    # 2. x != c.
    # 3. count_S(L) == 0 (if x not in S) OR count_S(L) == 1 (if x in S).
    #    Note: if x in S and count_S(L)==1, the only S point was x. Adding c makes it 2.
    #    if x not in S and count_S(L)==0. Adding c makes it 1. Then for x, condition is 1 > 0. True.
    
    # So the "dangerous" lines for adding c are those lines L containing c where count_S(L) is small (0 or 1).
    
    # Let's implement this.
    
    in_S = np.zeros(N_TOTAL, dtype=np.bool_)
    count_S_on_line = np.zeros(n_lines, dtype=np.int32)
    
    # We need to track bad[x] to know if it hits 50.
    # bad[x] = number of lines L through x where count_S(L) > (1 if x in S else 0).
    
    # Let's initialize bad[x] = 0 for all x.
    bad = np.zeros(N_TOTAL, dtype=np.int32)
    
    # Helper to update bad counts when adding/removing a point.
    # This is complex. Let's try a simpler heuristic first:
    # Randomized Greedy with Restarts.
    
    # Actually, let's just try to add ALL points that are currently "safe" in a single vectorized pass.
    
    # A point c is safe to add if for all lines L through c, adding c doesn't make any x on L invalid.
    # This is hard to vectorize directly for "all x on L".
    
    # Let's try a different tactic: 
    # Start with a large random S, then remove "conflicting" points.
    # A point x is "conflicting" if bad[x] == 50.
    # If we remove a point from S, bad counts can only decrease.
    # So if we start with S = All Points, bad[x] = 50 for all x (since every line has 49 other S points).
    # We need to remove points until no x has bad[x] == 50.
    # This is equivalent to finding a large Independent Set in a hypergraph?
    
    # Let's try:
    # 1. S = All Points.
    # 2. Compute bad[x] for all x.
    # 3. While exists x with bad[x] == 50:
    #    Pick one such x.
    #    Remove a random point y from S that is on a line with x and is "responsible" for the conflict.
    #    Actually, if bad[x] == 50, it means for EVERY line L through x, count_S(L) > (1 if x in S else 0).
    #    To fix x, we need to find at least one line L through x such that we can reduce count_S(L) to <= (1 if x in S else 0).
    #    If x in S, we need count_S(L) <= 1. Since x is in S, we need to remove all other S points on L.
    #    If x not in S, we need count_S(L) <= 0. We need to remove all S points on L.
    
    # This suggests a strategy:
    # 1. Start with S = All Points.
    # 2. Identify all "bad" points (bad[x] == 50).
    # 3. For each bad point x, pick a line L through x.
    # 4. Remove all points from S on L, except possibly x if x in S.
    # 5. Repeat.
    
    # This might remove too many points.
    
    # Let's try a "Maximal S" approach via iterative removal.
    # To maximize |S|, we want to remove as FEW points as possible.
    
    # Let's try a simple loop:
    # 1. S = All Points.
    # 2. Compute bad.
    # 3. If any bad[x] == 50, remove a point from S that is on a line with x.
    #    Which point to remove? The one that appears in the most "conflicting" lines?
    
    # Let's implement this.
    
    in_S = np.ones(N_TOTAL, dtype=np.bool_)
    count_S_on_line = np.full(n_lines, 49, dtype=np.int32)
    
    # Compute bad
    # bad[x] = sum over lines L through x of (count_S(L) > (1 if x in S else 0))
    
    # We can compute this vectorized.
    # For each line L, let c_L = count_S(L).
    # For each point x on L, if c_L > (1 if x in S else 0), then L contributes 1 to bad[x].
    
    # Let's build a matrix M of shape (N_TOTAL, n_lines)? No, too big (2401 * 2450 ~ 6M).
    # But we have lines_arr (n_lines, 49) and point_to_lines.
    
    # Let's just compute bad directly.
    # bad = np.zeros(N_TOTAL, dtype=np.int32)
    # for L in range(n_lines):
    #   c = count_S_on_line[L]
    #   pts = lines_arr[L]
    #   for x in pts:
    #     threshold = 1 if in_S[x] else 0
    #     if c > threshold:
    #       bad[x] += 1
    
    # This is O(n_lines * 49) = 2450 * 49 ~ 120k ops. Very fast.
    
    def compute_bad():
        bad = np.zeros(N_TOTAL, dtype=np.int32)
        for L in range(n_lines):
            c = count_S_on_line[L]
            if c == 0:
                continue
            pts = lines_arr[L]
            # For each x in pts, check if c > (1 if in_S[x] else 0)
            # in_S[pts] is a boolean array of length 49.
            # threshold = in_S[pts].astype(np.int32)
            # condition = c > threshold
            # bad[pts[condition]] += 1
            thresholds = in_S[pts].astype(np.int32)
            conditions = c > thresholds
            if np.any(conditions):
                affected_pts = pts[conditions]
                bad[affected_pts] += 1
        return bad
    
    bad = compute_bad()
    
    # Now, iteratively remove points to resolve conflicts.
    # We want to remove points from S to make bad[x] < 50 for all x.
    
    # While there are bad points:
    #   Pick a bad point x.
    #   Find a line L through x such that removing a point from L helps.
    #   Actually, if bad[x] == 50, ALL lines through x are "bad" for x.
    #   We need to fix at least one line L through x.
    #   To fix L for x, we need count_S(L) <= (1 if x in S else 0).
    #   If x in S, we need count_S(L) <= 1. We must remove all S points on L except x.
    #   If x not in S, we need count_S(L) <= 0. We must remove all S points on L.
    
    # Strategy:
    # 1. Find all x with bad[x] == 50.
    # 2. If none, done.
    # 3. Pick one such x.
    # 4. Pick a line L through x.
    # 5. Remove points from S on L to satisfy the condition for x.
    #    To minimize removals, remove the "least useful" points?
    #    Or just remove all required points.
    
    # Let's try:
    # For a bad point x, pick a random line L through x.
    # If x in S: remove all points y in S on L, y != x.
    # If x not in S: remove all points y in S on L.
    
    # Update in_S and count_S_on_line.
    # Recompute bad.
    
    # This is O(1) per removal? No, recomputing bad is O(120k).
    # If we do this many times, it might be slow.
    # But we only do it for bad points.
    
    # Let's try.
    
    max_iters = 10000
    for _ in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        
        bad = compute_bad()
        bad_pts = np.where(bad == 50)[0]
        if len(bad_pts) == 0:
            break
        
        # Pick a random bad point
        x = bad_pts[np.random.randint(len(bad_pts))]
        
        # Pick a random line through x
        lines_through_x = point_to_lines[x]
        L = lines_through_x[np.random.randint(len(lines_through_x))]
        
        # Determine which points to remove from L
        pts_on_L = lines_arr[L]
        if in_S[x]:
            # Remove all y in S on L, y != x
            to_remove = pts_on_L[in_S[pts_on_L]]
            to_remove = to_remove[to_remove != x]
        else:
            # Remove all y in S on L
            to_remove = pts_on_L[in_S[pts_on_L]]
        
        for y in to_remove:
            if in_S[y]:
                in_S[y] = False
                # Update count_S_on_line for all lines through y
                for Ly in point_to_lines[y]:
                    count_S_on_line[Ly] -= 1
    
    # Final check
    bad = compute_bad()
    if np.any(bad == 50):
        # Still invalid. Remove more?
        # Just return what we have, but it's invalid.
        # To ensure validity, we can remove ALL points from S?
        # No, let's try to fix it properly.
        # If we still have bad points, we can remove points more aggressively.
        # For each bad point x, remove all points from S on one of its lines.
        bad_pts = np.where(bad == 50)[0]
        for x in bad_pts:
            if time.time() > _DEADLINE - 5:
                break
            lines_through_x = point_to_lines[x]
            L = lines_through_x[0]
            pts_on_L = lines_arr[L]
            if in_S[x]:
                to_remove = pts_on_L[in_S[pts_on_L]]
                to_remove = to_remove[to_remove != x]
            else:
                to_remove = pts_on_L[in_S[pts_on_L]]
            for y in to_remove:
                if in_S[y]:
                    in_S[y] = False
                    for Ly in point_to_lines[y]:
                        count_S_on_line[Ly] -= 1
    
    # Build result: N = complement of S
    points = []
    for idx in range(N_TOTAL):
        if not in_S[idx]:
            xa, xb, ya, yb = _idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
