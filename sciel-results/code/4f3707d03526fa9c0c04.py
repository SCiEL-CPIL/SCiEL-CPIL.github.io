# EVOLVE-BLOCK-START
import numpy as np
import time

_P = 7
_Q = 49
_Q2 = 2401
_T0 = time.time()
_DEADLINE = _T0 + 600.0
_MARGIN = 30.0


def _time_left():
    return _DEADLINE - time.time() - _MARGIN


def nikodym_set() -> np.ndarray:
    p = _P
    q = _Q
    q2 = _Q2
    k = 3  # alpha^2 = 3

    # Field operations using precomputed tables
    # Index: a * p + b for a + b*alpha
    add_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)
    for x in range(q):
        xa, xb = x // p, x % p
        for y in range(q):
            ya, yb = y // p, y % p
            add_table[x, y] = ((xa + ya) % p) * p + ((xb + yb) % p)
            mul_table[x, y] = ((xa * ya + xb * yb * k) % p) * p + ((xa * yb + xb * ya) % p)

    # Point index: x * q + y
    def pt_idx(x, y):
        return x * q + y

    def pt_xy(idx):
        return idx // q, idx % q

    # Build all lines
    line_points = []

    # Vertical lines: 49 lines
    for x in range(q):
        pts = np.array([pt_idx(x, y) for y in range(q)], dtype=np.int64)
        line_points.append(pts)

    # Slanted lines: 49*49 = 2401 lines
    for m in range(q):
        for c in range(q):
            pts = np.array([pt_idx(t, int(add_table[mul_table[m, t], c])) for t in range(q)], dtype=np.int64)
            line_points.append(pts)

    n_lines = len(line_points)  # 2450

    # For each point, list of lines containing it
    point_lines = [[] for _ in range(q2)]
    for li in range(n_lines):
        for pi in line_points[li]:
            point_lines[pi].append(li)

    # Precompute line membership as a boolean matrix (n_lines x q2)
    line_member = np.zeros((n_lines, q2), dtype=np.int8)
    for li in range(n_lines):
        for pi in line_points[li]:
            line_member[li, pi] = 1

    # S is the complement of N (the "bad" set)
    # N is valid iff: for every point z, there exists a line L through z such that
    #   - if z in S: L has exactly 1 point in S (which is z)
    #   - if z not in S: L has 0 points in S
    # We want to maximize |S| to minimize |N|

    # Start with S = all points (trivially valid: each line has 49 S-points, 
    # but wait - that's not valid. Let me reconsider.)
    # Actually, S = all points is NOT valid because for z in S, we need a line with exactly 1 S-point.
    # If S = all points, every line has 49 S-points, so no line has exactly 1.
    
    # Start with S = empty set. Then for z not in S, we need a line with 0 S-points.
    # With S empty, every line has 0 S-points, so all points are valid.
    # This gives |N| = 2401, |S| = 0. Not great.
    
    # Better: start with S = graph of a function (49 points)
    # For z in S (on the graph): the vertical line through z has 1 S-point (z itself) if no other graph point is on that vertical.
    # Actually, the vertical line through (x, y) contains (x, y') for all y'. If S is a function graph, 
    # then for each x, there's exactly one y with (x,y) in S. So the vertical line through (x,y) in S 
    # has exactly 1 S-point. Good.
    # For z not in S: we need a line with 0 S-points. This is harder to guarantee.
    
    # Let's use a different approach: start with S = empty, then greedily ADD points to S.
    # When S is empty, all points are valid (each has 49 lines with 0 S-points).
    # We can add a point z to S if after adding, all points remain valid.
    
    S = np.zeros(q2, dtype=bool)
    line_S_count = np.zeros(n_lines, dtype=np.int64)
    
    # good_line_count[z] = number of lines through z that are "good" for z
    # If z in S: good line has S_count == 1
    # If z not in S: good line has S_count == 0
    good_line_count = np.zeros(q2, dtype=np.int64)
    
    # Initialize: S is empty, so all points not in S need lines with S_count == 0
    # All lines have S_count == 0, so each point has 49 good lines
    for z in range(q2):
        good_line_count[z] = len(point_lines[z])
    
    def is_valid():
        return np.all(good_line_count >= 1)
    
    def add_point(z):
        """Try to add z to S. Returns True if valid after addition."""
        # Check if adding z keeps validity
        # After adding z: for each line L through z, S_count[L] increases by 1
        # This affects good_line_count for all points on those lines
        
        # Temporarily add z
        S[z] = True
        affected = set()
        for li in point_lines[z]:
            old_count = line_S_count[li]
            new_count = old_count + 1
            line_S_count[li] = new_count
            
            # Update good_line_count for all points on this line
            for w in line_points[li]:
                if w == z:
                    continue
                affected.add(w)
                if S[w]:
                    # w in S: good if S_count == 1
                    if old_count == 1:
                        good_line_count[w] -= 1
                    elif new_count == 1:
                        good_line_count[w] += 1
                else:
                    # w not in S: good if S_count == 0
                    if old_count == 0:
                        good_line_count[w] -= 1
                    elif new_count == 0:
                        good_line_count[w] += 1
        
        # Update z's good_line_count
        # z is now in S, so z needs lines with S_count == 1
        z_good = 0
        for li in point_lines[z]:
            if line_S_count[li] == 1:
                z_good += 1
        good_line_count[z] = z_good
        
        if is_valid():
            return True
        else:
            # Revert
            S[z] = False
            for li in point_lines[z]:
                old_count = line_S_count[li]
                new_count = old_count - 1
                line_S_count[li] = new_count
                
                for w in line_points[li]:
                    if w == z:
                        continue
                    if S[w]:
                        if new_count == 1:
                            good_line_count[w] += 1
                        elif old_count == 1:
                            good_line_count[w] -= 1
                    else:
                        if new_count == 0:
                            good_line_count[w] += 1
                        elif old_count == 0:
                            good_line_count[w] -= 1
            
            # Revert z's good_line_count
            z_good = 0
            for li in point_lines[z]:
                if line_S_count[li] == 0:
                    z_good += 1
            good_line_count[z] = z_good
            return False
    
    def remove_point(z):
        """Remove z from S. Returns True if valid after removal."""
        # Temporarily remove z
        S[z] = False
        for li in point_lines[z]:
            old_count = line_S_count[li]
            new_count = old_count - 1
            line_S_count[li] = new_count
            
            for w in line_points[li]:
                if w == z:
                    continue
                if S[w]:
                    if old_count == 1:
                        good_line_count[w] -= 1
                    elif new_count == 1:
                        good_line_count[w] += 1
                else:
                    if old_count == 0:
                        good_line_count[w] -= 1
                    elif new_count == 0:
                        good_line_count[w] += 1
        
        # Update z's good_line_count
        # z is now not in S, so z needs lines with S_count == 0
        z_good = 0
        for li in point_lines[z]:
            if line_S_count[li] == 0:
                z_good += 1
        good_line_count[z] = z_good
        
        if is_valid():
            return True
        else:
            # Revert
            S[z] = True
            for li in point_lines[z]:
                old_count = line_S_count[li]
                new_count = old_count + 1
                line_S_count[li] = new_count
                
                for w in line_points[li]:
                    if w == z:
                        continue
                    if S[w]:
                        if old_count == 1:
                            good_line_count[w] += 1
                        elif new_count == 1:
                            good_line_count[w] -= 1
                    else:
                        if old_count == 0:
                            good_line_count[w] += 1
                        elif new_count == 0:
                            good_line_count[w] -= 1
            
            # Revert z's good_line_count
            z_good = 0
            for li in point_lines[z]:
                if line_S_count[li] == 1:
                    z_good += 1
            good_line_count[z] = z_good
            return False
    
    # Greedy: repeatedly try to add points to S
    # Priority: points with low good_line_count (more constrained) first
    rng = np.random.default_rng(42)
    
    for iteration in range(10000):
        if _time_left() <= 0:
            break
        
        # Find candidate points not in S
        candidates = [z for z in range(q2) if not S[z]]
        if not candidates:
            break
        
        # Sort by good_line_count (ascending) - most constrained first
        # But this is expensive; use random sampling instead
        sample_size = min(len(candidates), 50)
        sampled = rng.choice(candidates, size=sample_size, replace=False)
        
        best_z = -1
        for z in sampled:
            if add_point(int(z)):
                best_z = int(z)
                break
        
        if best_z == -1:
            # Try all remaining candidates
            for z in candidates:
                if add_point(z):
                    best_z = z
                    break
            if best_z == -1:
                break
    
    # Build output: N = all points not in S
    pts_list = []
    for z in range(q2):
        if not S[z]:
            x, y = pt_xy(z)
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            pts_list.append([[xa, xb], [ya, yb]])

    result = np.array(pts_list, dtype=np.int64)
    return result


# EVOLVE-BLOCK-END
