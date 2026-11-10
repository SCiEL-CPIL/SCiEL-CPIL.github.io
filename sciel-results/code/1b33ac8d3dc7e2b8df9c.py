import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    total_points = q * q
    deadline = time.time() + 580.0
    k = 3  # alpha^2 = 3

    # Field arithmetic
    def mul(u, v):
        ua, ub = u
        va, vb = v
        return ((ua*va + ub*vb*k) % p, (ua*vb + ub*va) % p)
    
    def add(u, v):
        return ((u[0]+v[0])%p, (u[1]+v[1])%p)
    
    def get_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    # Generate all lines
    lines = []
    line_membership = [[] for _ in range(total_points)]
    
    # Vertical lines (v = (0, 1)): x is constant
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
    
    # Lines with direction (1, m) for each m in F_q
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            seen_lines = set()
            for xa in range(p):
                for xb in range(p):
                    for ya in range(p):
                        for yb in range(p):
                            pts = []
                            for ta in range(p):
                                for tb in range(p):
                                    t = (ta, tb)
                                    x = add((xa, xb), t)
                                    y = add((ya, yb), mul(m, t))
                                    idx = get_idx(x[0], x[1], y[0], y[1])
                                    pts.append(idx)
                            line_set = frozenset(pts)
                            if line_set not in seen_lines:
                                seen_lines.add(line_set)
                                lines.append(list(line_set))
                                for pt in line_set:
                                    line_membership[pt].append(len(lines) - 1)

    num_lines = len(lines)
    
    # State: in_S[x] = True means x is in complement S (not in N)
    # Validity condition:
    # For x in S: exists line L through x with count_S(L) == 1 (only x from S on L)
    # For x in N: exists line L through x with count_S(L) == 0 (no points from S on L)
    
    in_S = np.zeros(total_points, dtype=bool)
    line_count = np.zeros(num_lines, dtype=np.int32)
    
    # For each point, track:
    # zero_lines[x] = number of lines through x with count_S == 0
    # one_lines[x] = number of lines through x with count_S == 1
    zero_lines = np.zeros(total_points, dtype=np.int32)
    one_lines = np.zeros(total_points, dtype=np.int32)
    
    def init_counts():
        """Initialize zero_lines and one_lines from current state"""
        zero_lines.fill(0)
        one_lines.fill(0)
        for x in range(total_points):
            for l_idx in line_membership[x]:
                c = line_count[l_idx]
                if c == 0:
                    zero_lines[x] += 1
                elif c == 1:
                    one_lines[x] += 1
    
    def is_valid():
        for x in range(total_points):
            if in_S[x]:
                if one_lines[x] < 1:
                    return False
            else:
                if zero_lines[x] < 1:
                    return False
        return True
    
    def update_after_toggle(x):
        """Update counts after toggling point x"""
        for l_idx in line_membership[x]:
            old_c = line_count[l_idx]
            new_c = old_c + (1 if in_S[x] else -1)
            line_count[l_idx] = new_c
            
            # Update zero_lines and one_lines for all points on this line
            for pt in lines[l_idx]:
                # For this line, the count changed from old_c to new_c
                if old_c == 0:
                    zero_lines[pt] -= 1
                elif old_c == 1:
                    one_lines[pt] -= 1
                if new_c == 0:
                    zero_lines[pt] += 1
                elif new_c == 1:
                    one_lines[pt] += 1
    
    # Start from empty S (all points in N) - always valid
    # Then use simulated annealing to add points to S
    
    best_S_size = 0
    best_state = in_S.copy()
    
    # Simulated annealing
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute line indices for fast access
    # For each point, we have line_membership[x] = list of line indices
    
    while T > min_T and time.time() < deadline:
        # Pick a random point to toggle
        x = random.randint(0, total_points - 1)
        
        old_in_S = in_S[x]
        
        # Compute delta in objective (|S|)
        if old_in_S:
            delta_obj = -1  # removing from S
        else:
            delta_obj = 1   # adding to S
        
        # Toggle
        in_S[x] = not in_S[x]
        update_after_toggle(x)
        
        # Check validity
        valid = True
        # Only need to check affected points (on lines through x)
        affected = set()
        for l_idx in line_membership[x]:
            for pt in lines[l_idx]:
                affected.add(pt)
        
        for pt in affected:
            if in_S[pt]:
                if one_lines[pt] < 1:
                    valid = False
                    break
            else:
                if zero_lines[pt] < 1:
                    valid = False
                    break
        
        if valid:
            # Accept if improves or with probability exp(delta_obj / T)
            if delta_obj > 0 or random.random() < np.exp(delta_obj / T):
                if sum(in_S) > best_S_size:
                    best_S_size = sum(in_S)
                    best_state = in_S.copy()
            else:
                # Reject: rollback
                in_S[x] = old_in_S
                update_after_toggle(x)
        else:
            # Invalid: rollback
            in_S[x] = old_in_S
            update_after_toggle(x)
        
        T *= cooling
    
    # Use best state found
    in_S = best_state
    
    # Return N = complement of S
    points = []
    for x in range(total_points):
        if not in_S[x]:
            yb = x % p
            ya = (x // p) % p
            xb = (x // (p*p)) % p
            xa = (x // (p*p*p)) % p
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
