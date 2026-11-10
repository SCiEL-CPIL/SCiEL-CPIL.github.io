# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

_START_TIME = time.time()
_DEADLINE = _START_TIME + 615.0  # safety margin before 630s kill

p = 7
K = 3  # alpha^2 = 3
N_TOTAL = p**4  # 2401
N_LINES_TOTAL = 2450
N_PER_LINE = p*p  # 49

def _point_to_idx(xa, xb, ya, yb):
    return ((xa * p + xb) * p * p + (ya * p + yb))

def _idx_to_point(idx):
    yb = idx % p
    ya = (idx // p) % p
    xb = (idx // (p * p)) % p
    xa = (idx // (p * p * p)) % p
    return xa, xb, ya, yb

def _generate_seed_excluded():
    """Return boolean array where True = point is in S (excluded from N)."""
    squares = set((u * u) % p for u in range(p))
    excluded = np.zeros(N_TOTAL, dtype=np.int8)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - yb * yb * K) % p == 0:
                        if ya not in squares:
                            excluded[_point_to_idx(xa, xb, ya, yb)] = 1
    return excluded

def _precompute_lines():
    """
    Precompute all 2450 lines.
    Returns a numpy array of shape (N_LINES_TOTAL, N_PER_LINE) with point indices.
    """
    lines = np.zeros((N_LINES_TOTAL, N_PER_LINE), dtype=np.int32)
    
    line_idx = 0
    # Direction 0: v = (0, 1), vertical
    for xa in range(p):
        for xb in range(p):
            for k in range(N_PER_LINE):
                ya = k // p
                yb = k % p
                lines[line_idx, k] = _point_to_idx(xa, xb, ya, yb)
            line_idx += 1
    
    # Directions 1..49: v = (1, m) for m = ma + mb*alpha
    for ma in range(p):
        for mb in range(p):
            for ca in range(p):
                for cb in range(p):
                    for k in range(N_PER_LINE):
                        xa = k // p
                        xb = k % p
                        ya = (ca + ma * xa + 3 * mb * xb) % p
                        yb = (cb + ma * xb + mb * xa) % p
                        lines[line_idx, k] = _point_to_idx(xa, xb, ya, yb)
                    line_idx += 1
    
    return lines

def nikodym_set() -> np.ndarray:
    start = time.time()
    
    # Precompute lines
    lines = _precompute_lines()
    
    # Build point_lines: for each point, list of lines containing it
    point_lines = [[] for _ in range(N_TOTAL)]
    for i in range(N_LINES_TOTAL):
        for idx in lines[i]:
            point_lines[idx].append(i)
    
    # Build incidence matrix: A[line, point] = 1 if point is on line
    # This is 2450 x 2401, which is manageable
    A = np.zeros((N_LINES_TOTAL, N_TOTAL), dtype=np.float64)
    for i in range(N_LINES_TOTAL):
        for pt in lines[i]:
            A[i, pt] = 1.0
    
    # LP formulation:
    # Maximize sum(x_p) for p in 0..2400
    # Subject to:
    #   For each line L: sum_{x in L} x_x <= 1
    #   0 <= x_p <= 1 for all p
    
    # c = -1 for all variables (we minimize -sum)
    c = -np.ones(N_TOTAL, dtype=np.float64)
    
    # Inequality constraints: A @ x <= 1 for all lines
    # A is already (N_LINES_TOTAL, N_TOTAL)
    b_ub = np.ones(N_LINES_TOTAL, dtype=np.float64)
    
    # Bounds: 0 <= x_p <= 1
    bounds = [(0, 1)] * N_TOTAL
    
    # Solve LP
    try:
        res = linprog(c, A_ub=A, b_ub=b_ub, bounds=bounds, method='highs', 
                      options={'time_limit': 300})
    except Exception as e:
        # Fallback to greedy if LP fails
        return _greedy_fallback(lines, point_lines)
    
    if res.status != 0:
        # LP didn't converge, fallback
        return _greedy_fallback(lines, point_lines)
    
    x_lp = res.x  # fractional solution
    
    # Randomized rounding: for each point, include with probability x_lp[p]
    # Use multiple random seeds to get different rounded solutions
    best_S = None
    best_size = -1
    
    rng = np.random.RandomState(42)
    
    for trial in range(20):
        if time.time() > _DEADLINE - 60:
            break
        
        # Round the LP solution
        S = (rng.rand(N_TOTAL) < x_lp).astype(np.int8)
        
        # Now do local search: try to add points that don't violate constraints
        # Constraint: for each line L, sum_{x in L} S[x] <= 1
        # This means at most one point per line can be in S.
        
        # Wait, the constraint is sum <= 1, so at most one point per line.
        # But the Nikodym condition is different!
        
        # Let me re-read the task.
        # "every point x of F_49^2 (in N or not) must lie on a line {x + t*v : t in F_49}, 
        #  with v = (1, m) for some m in F_49 or v = (0, 1), whose points other than x all belong to N."
        
        # So for every point x, there exists a direction v such that all other points on the line 
        # through x in direction v are in N (i.e., not in S).
        
        # This is NOT the same as "at most one point per line in S".
        # It's a much weaker condition.
        
        # The LP I formulated is too restrictive. Let me reconsider.
        
        # The actual constraint is:
        # For every point x, there exists at least one line L through x such that S ∩ (L \ {x}) = empty.
        # Equivalently: for every point x, bad[x] < 50, where bad[x] = number of lines L through x 
        # such that S ∩ (L \ {x}) is non-empty.
        
        # This is a global constraint, not a per-line constraint.
        # The LP approach with per-line constraints is wrong.
        
        # Let me use the greedy approach instead, but with a better heuristic.
        
        # Actually, let me just use the greedy approach with multiple passes and randomization.
        pass
    
    # Since the LP formulation was incorrect, let's use a better greedy approach.
    # The key insight: we want to maximize |S| such that for every point x, 
    # there exists a line L through x with S ∩ (L \ {x}) = empty.
    
    # This is equivalent to: the set S is a "Nikodym complement".
    # A sufficient condition: S is an independent set in the "line graph" where 
    # two points are adjacent if they share a line. But that's too restrictive.
    
    # Actually, the condition is: for every x, there exists a line L_x through x 
    # such that L_x \ {x} is disjoint from S.
    
    # This means: for every x, x is NOT "covered" by S in all directions.
    # x is covered in direction v if there exists y ≠ x on the line through x in direction v 
    # such that y ∈ S.
    
    # So we need: for every x, there exists at least one direction v such that 
    # no other point on the line through x in direction v is in S.
    
    # Let's use a different approach: start with S = empty, and greedily add points.
    # When we add a point c to S, we need to check that for every point x, 
    # if x was previously covered only by the line through x in direction v_c (the line 
    # through c and x), then x must still be covered by some other direction.
    
    # This is complex. Let's use the original greedy approach but with better heuristics.
    
    return _greedy_fallback(lines, point_lines)

def _greedy_fallback(lines, point_lines):
    """Greedy approach with multiple passes and randomization."""
    N_TOTAL = 2401
    N_LINES_TOTAL = 2450
    
    # Initialize S as empty
    in_S = np.zeros(N_TOTAL, dtype=np.int8)
    
    # Compute line_s_counts and bad
    line_s_counts = np.zeros(N_LINES_TOTAL, dtype=np.int32)
    bad = np.zeros(N_TOTAL, dtype=np.int32)
    
    np.random.seed(42)
    
    # Multiple passes
    for pass_num in range(10):
        if time.time() > _DEADLINE - 60:
            break
        
        progress = False
        candidates = list(range(N_TOTAL))
        np.random.shuffle(candidates)
        
        for c_idx in candidates:
            if time.time() > _DEADLINE - 30:
                break
            
            if in_S[c_idx]:
                continue
            
            can_add = True
            
            for l_idx in point_lines[c_idx]:
                cnt = line_s_counts[l_idx]
                
                if cnt == 0:
                    # Adding c makes cnt=1. 
                    # For all x in line, x != c, x not in S: bad[x] increases by 1.
                    # Check if any such x has bad[x] >= 49.
                    line_pts = lines[l_idx, :]
                    for x in line_pts:
                        if x != c_idx and not in_S[x]:
                            if bad[x] >= 49:
                                can_add = False
                                break
                    if not can_add:
                        break
                elif cnt == 1:
                    # Adding c makes cnt=2.
                    # For all x in line, x != c, x in S: bad[x] increases by 1.
                    # There is exactly one such x (the one that made cnt=1).
                    line_pts = lines[l_idx, :]
                    for x in line_pts:
                        if x != c_idx and in_S[x]:
                            if bad[x] >= 49:
                                can_add = False
                                break
                            break  # Only one such point
                    if not can_add:
                        break
                # else cnt > 1: no effect on bad counts
            
            if can_add:
                in_S[c_idx] = 1
                progress = True
                
                for l_idx in point_lines[c_idx]:
                    cnt = line_s_counts[l_idx]
                    line_s_counts[l_idx] += 1
                    line_pts = lines[l_idx, :]
                    
                    if cnt == 0:
                        for x in line_pts:
                            if x != c_idx and not in_S[x]:
                                bad[x] += 1
                    elif cnt == 1:
                        for x in line_pts:
                            if x != c_idx and in_S[x]:
                                bad[x] += 1
                                break
        
        if not progress:
            break
    
    # Build result
    points = []
    for idx in range(N_TOTAL):
        if not in_S[idx]:
            xa, xb, ya, yb = _idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
