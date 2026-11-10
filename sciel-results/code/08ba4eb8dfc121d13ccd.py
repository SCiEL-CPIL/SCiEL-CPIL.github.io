# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 with q = 49 (p=7).
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
        
    def get_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    # Precompute lines
    # A line is defined by direction v and a point x0.
    # We need to check for each point x if there is a line L through x such that L \ {x} is subset of N.
    # Equivalent: L \cap S = {x} (if x in S) or L \cap S = empty (if x in N).
    
    # So the condition is:
    # For all x in N: exists line L through x with count_S(L) == 0.
    # For all x in S: exists line L through x with count_S(L) == 1.
    
    # We will maintain:
    # S: set of points in S (complement of N)
    # in_S: boolean array of size total_points
    # line_count: array of size num_lines, count_S(L)
    # zero_count_lines: array of size total_points, number of lines through x with count 0
    # one_count_lines: array of size total_points, number of lines through x with count 1
    
    # We need to generate all lines.
    # Directions: v = (1, m) for m in F_q, and v = (0, 1).
    # Total directions: q + 1 = 50.
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
            seen_lines = set()
            # Optimization: We only need to generate unique lines.
            # A line with slope m is determined by its y-intercept at x=0, i.e., the point (0, b).
            # There are q possible intercepts.
            # So we iterate over all possible intercepts b = (ba, bb).
            for ba in range(p):
                for bb in range(p):
                    # Line: y = m*x + (ba, bb)
                    pts = []
                    for ta in range(p):
                        for tb in range(p):
                            t = (ta, tb)
                            x = t  # since x starts at 0
                            y = add((ba, bb), mul(m, t))
                            idx = get_idx(x[0], x[1], y[0], y[1])
                            pts.append(idx)
                    line_set = frozenset(pts)
                    # This generation method (intercepts) guarantees uniqueness for a fixed slope m.
                    # No need for seen_lines set if we trust the math, but let's keep it safe or just append.
                    # Actually, iterating over intercepts generates each line exactly once.
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
            
    # Initialize zero_count_lines and one_count_lines
    zero_count_lines = np.zeros(total_points, dtype=np.int32)
    one_count_lines = np.zeros(total_points, dtype=np.int32)
    
    for x in range(total_points):
        for l_idx in line_membership[x]:
            if line_count[l_idx] == 0:
                zero_count_lines[x] += 1
            elif line_count[l_idx] == 1:
                one_count_lines[x] += 1
                
    # Check if current S is valid
    def is_valid_fast():
        for x in range(total_points):
            if in_S[x]:
                if one_count_lines[x] < 1:
                    return False
            else:
                if zero_count_lines[x] < 1:
                    return False
        return True

    # If seed is invalid, reset to empty S (which is always valid)
    if not is_valid_fast():
        S.clear()
        in_S.fill(False)
        line_count.fill(0)
        zero_count_lines.fill(0)
        one_count_lines.fill(0)
        for x in range(total_points):
            zero_count_lines[x] = len(line_membership[x])
            one_count_lines[x] = 0
        
    # Greedy Loop
    # Try to add points to S one by one.
    # A point x can be added to S if adding it keeps the configuration valid.
    # Adding x to S:
    # 1. in_S[x] = True
    # 2. For all lines L through x, line_count[L] += 1
    # 3. Update zero/one counts for all points on those lines.
    # 4. Check if all affected points are still valid.
    
    # To speed up, we precompute the list of lines through each point.
    # We also precompute the list of points on each line.
    # line_points = [set of points on line i for i in range(num_lines)]
    
    best_S_size = len(S)
    best_S_state = (in_S.copy(), line_count.copy(), zero_count_lines.copy(), one_count_lines.copy())
    
    # We will perform multiple passes or random restarts.
    # Since the space is large, simple greedy might get stuck.
    # Let's try a few random restarts from the seed or empty.
    
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Reset state for this restart
        # For restart 0, use the initial seed state (or empty if seed was invalid)
        # For subsequent restarts, we can try to start from the best found so far, 
        # or from the seed again with some randomization.
        # Given the complexity, let's just run greedy on the current state.
        
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
                lines_through_x = line_membership[x]
                
                # Identify affected points: union of points on all lines through x
                # This is almost the whole plane, but we can optimize by just checking validity.
                # However, updating counts requires iterating over lines through x.
                
                # Save state for rollback
                old_line_count = {l_idx: line_count[l_idx] for l_idx in lines_through_x}
                # We don't need to save zero/one counts for ALL points, only those affected.
                # Affected points are those on lines through x.
                affected_points = set()
                for l_idx in lines_through_x:
                    affected_points.update(lines[l_idx])
                
                old_zero = {pt: zero_count_lines[pt] for pt in affected_points}
                old_one = {pt: one_count_lines[pt] for pt in affected_points}
                
                # Apply change: add x to S
                in_S[x] = True
                
                # Update line counts and zero/one counts
                # First update line counts
                for l_idx in lines_through_x:
                    line_count[l_idx] += 1
                    
                # Recalculate zero/one counts for affected points
                # This is O(|affected| * 50). |affected| is roughly 2401.
                # 2401 * 50 = 120k operations. This is acceptable for a few thousand iterations.
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
                        best_S_state = (in_S.copy(), line_count.copy(), zero_count_lines.copy(), one_count_lines.copy())
                else:
                    # Rollback
                    in_S[x] = False
                    for l_idx in lines_through_x:
                        line_count[l_idx] = old_line_count[l_idx]
                    for pt in affected_points:
                        zero_count_lines[pt] = old_zero[pt]
                        one_count_lines[pt] = old_one[pt]

    # Restore best state
    in_S, line_count, zero_count_lines, one_count_lines = best_S_state
    
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
