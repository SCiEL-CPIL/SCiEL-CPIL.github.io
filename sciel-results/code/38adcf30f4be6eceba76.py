# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 610.0  # safety margin before 630s kill

p = 7
K = 3  # alpha^2 = 3
N_TOTAL = p**4  # 2401
N_LINES_TOTAL = 2450  # 49 vertical + 2401 slanted
N_PER_LINE = p*p  # 49

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
    Precompute all 2450 lines.
    Returns a numpy array of shape (N_LINES_TOTAL, N_PER_LINE) with point indices.
    Also returns point_lines: a list of lists, where point_lines[idx] contains indices of lines containing idx.
    """
    lines = np.zeros((N_LINES_TOTAL, N_PER_LINE), dtype=np.int32)
    point_lines = [[] for _ in range(N_TOTAL)]
    
    line_idx = 0
    # Direction 0: v = (0, 1), vertical
    # Lines parameterized by (xa, xb)
    for xa in range(p):
        for xb in range(p):
            for k in range(N_PER_LINE):
                ya = k // p
                yb = k % p
                pt_idx = _point_to_idx(xa, xb, ya, yb)
                lines[line_idx, k] = pt_idx
                point_lines[pt_idx].append(line_idx)
            line_idx += 1
    
    # Directions 1..49: v = (1, m) for m = ma + mb*alpha
    for ma in range(p):
        for mb in range(p):
            # Lines parameterized by (ca, cb) - the y-intercept essentially
            for ca in range(p):
                for cb in range(p):
                    for k in range(N_PER_LINE):
                        xa = k // p
                        xb = k % p
                        # y = c + x*m
                        # ya = ca + ma*xa + mb*xb*alpha^2 = ca + ma*xa + 3*mb*xb
                        # yb = cb + ma*xb + mb*xa
                        ya = (ca + ma * xa + 3 * mb * xb) % p
                        yb = (cb + ma * xb + mb * xa) % p
                        pt_idx = _point_to_idx(xa, xb, ya, yb)
                        lines[line_idx, k] = pt_idx
                        point_lines[pt_idx].append(line_idx)
                    line_idx += 1
    
    return lines, point_lines

def _generate_seed_excluded():
    """Return boolean array where True = point is in S (excluded from N)."""
    # Squares mod 7: 0^2=0, 1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1
    squares = set((u * u) % p for u in range(p))
    excluded = np.zeros(N_TOTAL, dtype=np.int8)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Unital condition: 2*xa + ya^2 - yb^2*K = 0 mod 7
                    if (2 * xa + ya * ya - yb * yb * K) % p == 0:
                        # y is special if ya is NOT a square mod 7
                        if ya not in squares:
                            excluded[_point_to_idx(xa, xb, ya, yb)] = 1
    return excluded

def _compute_initial_bad(lines, point_lines, in_S):
    """Compute bad counts for all points given initial S."""
    # line_s_counts[i] = number of points in S on line i
    line_s_counts = np.sum(in_S[lines], axis=1)
    
    bad = np.zeros(N_TOTAL, dtype=np.int32)
    
    # For each line, if count > threshold for a point, increment bad[point]
    # Threshold for point x on line L is 1 if x in S, 0 otherwise.
    # Condition: line_s_counts[L] > (1 if x in S else 0)
    
    # Vectorized approach per line is faster than per point
    for i in range(N_LINES_TOTAL):
        cnt = line_s_counts[i]
        if cnt > 0:
            line_pts = lines[i, :]
            # We need to add 1 to bad[x] for x in line_pts where cnt > in_S[x]
            mask = cnt > in_S[line_pts]
            bad[line_pts[mask]] += 1
            
    return bad, line_s_counts

def nikodym_set() -> np.ndarray:
    start = time.time()
    
    # Precomputation
    lines, point_lines = _precompute_lines()
    
    # Initialize S
    in_S = _generate_seed_excluded().astype(np.int8)
    
    # Initial bad counts
    bad, line_s_counts = _compute_initial_bad(lines, point_lines, in_S)
    
    # Check if initial set is valid
    # Valid if max(bad) <= 49 (since there are 50 directions total)
    # Actually, the condition is: for every x, there exists a line L such that L\{x} is in N.
    # This means for every x, there is at least one direction where S does not hit L\{x}.
    # bad[x] counts the number of directions where S DOES hit L\{x}.
    # So we need bad[x] < 50 for all x.
    
    current_badness = np.max(bad)
    if current_badness > 49:
        # Seed is invalid, need to remove points or start empty?
        # The seed is usually valid. If not, we'll handle it in SA.
        pass
        
    best_S = in_S.copy()
    best_size = np.sum(best_S)
    
    # Simulated Annealing Parameters
    # We want to maximize |S|.
    # Move: Add a point to S or Remove a point from S.
    # Acceptance:
    #   If new set is valid and |S| increases: Always accept.
    #   If new set is valid and |S| same: Accept with prob 0.5 (explore).
    #   If new set is invalid: Accept with prob exp(-delta_badness / T).
    #   If new set is valid but |S| decreases: Accept with prob exp(-1 / T) (to escape local min).
    
    # Actually, a simpler objective is to minimize "Invalidity" and maximize |S|.
    # Let's define a score: Score = |S| - M * (max(bad) - 49) if max(bad) > 49 else |S|.
    # We want to maximize Score.
    
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.1
    
    np.random.seed(42)
    
    # Precompute list of points for fast shuffling
    all_points = np.arange(N_TOTAL)
    
    while time.time() < _DEADLINE - 10:
        if T < min_T:
            break
            
        # Pick a random point to toggle
        c = np.random.choice(N_TOTAL)
        
        # Determine if adding or removing
        is_in_S = bool(in_S[c])
        
        # Calculate change in bad counts
        # We need to update bad and line_s_counts for this move.
        # To avoid modifying the global state if we reject, we can compute the delta.
        
        # However, computing the delta for bad[] is complex because it depends on line counts.
        # Let's try a "copy and check" approach if it's fast enough?
        # N_TOTAL=2401, lines=2450.
        # Copying arrays of size 2401/2450 is very fast (~microseconds).
        # Checking validity (max bad) is O(1) if we maintain max? No, we have to recompute max or update it.
        # Recomputing max(bad) is O(2401).
        # Total cost per step: ~ O(2401 + 50*49) ~ 5000 ops.
        # 600s / 5000 ops = 720,000 steps. This is feasible.
        
        new_in_S = in_S.copy()
        new_line_s_counts = line_s_counts.copy()
        
        if is_in_S:
            # Remove c
            new_in_S[c] = 0
            for l_idx in point_lines[c]:
                new_line_s_counts[l_idx] -= 1
        else:
            # Add c
            new_in_S[c] = 1
            for l_idx in point_lines[c]:
                new_line_s_counts[l_idx] += 1
        
        # Recompute bad counts efficiently?
        # The bad count for a point x depends on the counts of all 50 lines through x.
        # If we change line counts for 50 lines, we only need to update bad[x] for x on those 50 lines.
        # But bad[x] is a sum over 50 lines.
        # We can recompute bad[x] for all x on the affected lines.
        
        affected_points = set()
        for l_idx in point_lines[c]:
            for x in lines[l_idx]:
                affected_points.add(x)
        
        # Convert to list for vectorized update?
        # Or just loop over affected points.
        # Max affected points: 50 lines * 49 pts/line = 2450 (almost all points).
        # So we might as well recompute bad for ALL points?
        # Recomputing bad for all points:
        # for x in range(N_TOTAL):
        #    bad[x] = sum(1 for l in point_lines[x] if new_line_s_counts[l] > new_in_S[x])
        # This is 2401 * 50 = 120,000 ops.
        # 600s / 120k = 3000 steps. Might be too slow in Python.
        
        # Optimization: Update bad incrementally.
        # bad[x] changes only if the condition `count > in_S[x]` changes for any line through x.
        # For a line L, if count changes from C to C+1 (add) or C-1 (remove):
        # If adding (C -> C+1):
        #   For x in L:
        #     If in_S[x] == 0: condition was C > 0. Now C+1 > 0.
        #       If C == 0: False -> True. bad[x] increases by 1.
        #       If C > 0: True -> True. No change.
        #     If in_S[x] == 1: condition was C > 1. Now C+1 > 1.
        #       If C == 1: False -> True. bad[x] increases by 1.
        #       If C > 1: True -> True. No change.
        #       If C == 0: Impossible if x in S and we are adding c (unless c=x, but c is new).
        #       Wait, if x is in S, C >= 1.
        #       So for x in S, bad[x] increases by 1 if C == 1.
        
        # If removing (C -> C-1):
        #   For x in L:
        #     If in_S[x] == 0: condition was C > 0. Now C-1 > 0.
        #       If C == 1: True -> False. bad[x] decreases by 1.
        #       If C > 1: True -> True. No change.
        #     If in_S[x] == 1: condition was C > 1. Now C-1 > 1.
        #       If C == 2: True -> False. bad[x] decreases by 1.
        #       If C > 2: True -> True. No change.
        
        # So, we can update bad[x] for x in affected lines based on the OLD count C.
        
        new_bad = bad.copy()
        
        if is_in_S:
            # Remove c
            for l_idx in point_lines[c]:
                old_cnt = line_s_counts[l_idx]
                # new_cnt = old_cnt - 1
                line_pts = lines[l_idx, :]
                # For x in line_pts:
                # If in_S[x] == 0 and old_cnt == 1: bad[x] -= 1
                # If in_S[x] == 1 and old_cnt == 2: bad[x] -= 1
                
                if old_cnt == 1:
                    # Affected: x not in S
                    mask = (line_pts != c) & (in_S[line_pts] == 0)
                    new_bad[line_pts[mask]] -= 1
                elif old_cnt == 2:
                    # Affected: x in S
                    mask = (line_pts != c) & (in_S[line_pts] == 1)
                    new_bad[line_pts[mask]] -= 1
                # Note: c itself?
                # For x=c:
                # Old: in_S[c]=1. Condition: old_cnt > 1.
                # New: in_S[c]=0. Condition: old_cnt-1 > 0.
                # If old_cnt=1: Old: 1>1 (F). New: 0>0 (F). No change.
                # If old_cnt=2: Old: 2>1 (T). New: 1>0 (T). No change.
                # So c is never affected in terms of bad count change?
                # Let's check.
                # bad[c] counts lines where S hits L\{c}.
                # If we remove c, S loses c.
                # For a line L through c:
                # Old hit: S \ {c} intersect L != empty? No, S intersect L \ {c}.
                # If c was the ONLY point in S on L (old_cnt=1), then S \ {c} intersect L is empty.
                # So old bad contrib was 0 (since 1 > 1 is false).
                # New S' doesn't have c. S' intersect L is empty.
                # New bad contrib is 0 (since 0 > 0 is false).
                # So bad[c] doesn't change for lines where c was the only point.
                # If old_cnt > 1, there is another point in S on L.
                # Old: S intersect L \ {c} is non-empty. Contrib 1.
                # New: S' intersect L \ {c} is same as S intersect L \ {c} (since c removed, but other points remain).
                # So contrib remains 1.
                # So bad[c] is indeed invariant under removal of c?
                # Wait. The definition is "S intersect (L \ {x})".
                # If x=c, and we remove c from S:
                # Old set S has c. L\{c} contains other points.
                # If S had other points on L, then S intersect (L\{c}) is non-empty.
                # New set S' = S \ {c}. S' intersect (L\{c}) is same as S intersect (L\{c}) \ {c}.
                # Since c is not in L\{c}, this is identical.
                # So yes, bad[c] does not change when c is removed from S.
                
        else:
            # Add c
            for l_idx in point_lines[c]:
                old_cnt = line_s_counts[l_idx]
                # new_cnt = old_cnt + 1
                line_pts = lines[l_idx, :]
                
                if old_cnt == 0:
                    # Affected: x not in S
                    mask = (line_pts != c) & (in_S[line_pts] == 0)
                    new_bad[line_pts[mask]] += 1
                elif old_cnt == 1:
                    # Affected: x in S
                    mask = (line_pts != c) & (in_S[line_pts] == 1)
                    new_bad[line_pts[mask]] += 1
                # Again, bad[c] invariant.
        
        new_max_bad = np.max(new_bad)
        new_size = np.sum(new_in_S)
        
        # Acceptance Criteria
        # We want to maximize new_size, subject to new_max_bad <= 49.
        # If new_max_bad > 49, the set is invalid.
        
        accept = False
        
        if new_max_bad <= 49:
            # Valid move
            if new_size > np.sum(in_S):
                accept = True
            elif new_size == np.sum(in_S):
                accept = np.random.rand() < 0.5
            else:
                # Size decreased
                accept = np.random.rand() < np.exp(-1.0 / T)
        else:
            # Invalid move
            # Penalty based on how invalid it is
            # delta = new_max_bad - 49
            # accept = np.random.rand() < np.exp(-delta / T)
            delta = new_max_bad - 49
            if delta < 100: # Avoid overflow in exp
                accept = np.random.rand() < np.exp(-delta / T)
        
        if accept:
            in_S = new_in_S
            bad = new_bad
            line_s_counts = new_line_s_counts
            
            if new_size > best_size:
                best_size = new_size
                best_S = in_S.copy()
        
        T *= cooling_rate
        
    # Build result from best_S
    points = []
    for idx in range(N_TOTAL):
        if not best_S[idx]:
            xa, xb, ya, yb = _idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
