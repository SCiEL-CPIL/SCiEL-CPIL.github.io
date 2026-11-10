import numpy as np
import time

_P = 7
_Q = 49
_Q2 = 2401
_DEADLINE = time.time() + 600.0
_MARGIN = 15.0

def _build_tables():
    p = _P
    k = 3
    
    def pt_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p * p + ya * p + yb)

    line_points = []
    # Vertical
    for xa in range(p):
        for xb in range(p):
            pts = []
            for t in range(p * p):
                ta, tb = t // p, t % p
                pts.append(pt_idx(xa, xb, ta, tb))
            line_points.append(np.array(pts, dtype=np.int64))
    # Slanted
    for m in range(p * p):
        ma, mb = m // p, m % p
        for oa in range(p):
            for ob in range(p):
                pts = []
                for t in range(p * p):
                    ta, tb = t // p, t % p
                    ya = (ma * ta + mb * tb * k + oa) % p
                    yb = (ma * tb + mb * ta + ob) % p
                    pts.append(pt_idx(ta, tb, ya, yb))
                line_points.append(np.array(pts, dtype=np.int64))
    
    n_lines = len(line_points)
    
    point_lines = [[] for _ in range(_Q2)]
    for li, pts in enumerate(line_points):
        for pi in pts:
            point_lines[pi].append(li)

    # Baseline S
    squares = set((u * u) % p for u in range(p))
    S = np.zeros(_Q2, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - yb * yb * k) % p == 0:
                        if ya not in squares:
                            S[pt_idx(xa, xb, ya, yb)] = True

    return p, k, line_points, point_lines, S

def nikodym_set() -> np.ndarray:
    p, k, line_points, point_lines, S = _build_tables()
    q = _Q
    q2 = _Q2
    n_lines = len(line_points)
    
    # empty_line_count[x] = number of lines through x with 0 points in S
    empty_line_count = np.zeros(q2, dtype=np.int32)
    
    # Initialize counts
    # A line is empty if no point in S is on it.
    # For each line, if it's empty, increment count for all points on it.
    line_has_S = np.zeros(n_lines, dtype=bool)
    for li in range(n_lines):
        for pi in line_points[li]:
            if S[pi]:
                line_has_S[li] = True
                break
    for li in range(n_lines):
        if not line_has_S[li]:
            for pi in line_points[li]:
                empty_line_count[pi] += 1

    # Verify baseline
    # For x in S: need line with exactly 1 S (x).
    # For x in N: need line with 0 S.
    # This is guaranteed by construction of baseline?
    # We assume baseline is valid.

    def try_add(z):
        if S[z]:
            return False
        if empty_line_count[z] == 0:
            return False
        
        # Check if adding z breaks any x in N
        # z affects lines through z.
        # For each line L through z:
        #   If L was empty (count 0 S), it becomes count 1 S.
        #   All points x on L (x != z) lose this empty line.
        #   So empty_line_count[x] decreases by 1.
        #   We need new_count[x] > 0 for all x in N.
        #   So old_count[x] >= 2 for all x on empty lines through z.
        
        affected_points = set()
        for li in point_lines[z]:
            if not line_has_S[li]:
                for pi in line_points[li]:
                    if pi != z:
                        affected_points.add(pi)
        
        for x in affected_points:
            if not S[x] and empty_line_count[x] < 2:
                return False
        
        # Commit
        S[z] = True
        for li in point_lines[z]:
            if not line_has_S[li]:
                line_has_S[li] = True
                for pi in line_points[li]:
                    empty_line_count[pi] -= 1
        return True

    # Greedy loop
    # Heuristic: pick z that minimizes the number of affected points
    # or maximizes "safety"
    
    for _ in range(1000):
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        best_z = -1
        best_score = -1
        
        candidates = [z for z in range(q2) if not S[z]]
        
        # Randomize to avoid local optima
        np.random.shuffle(candidates)
        
        for z in candidates:
            if empty_line_count[z] == 0:
                continue
            
            affected = 0
            valid = True
            for li in point_lines[z]:
                if not line_has_S[li]:
                    for pi in line_points[li]:
                        if pi != z:
                            if not S[pi] and empty_line_count[pi] < 2:
                                valid = False
                                break
                            affected += 1
                    if not valid:
                        break
            
            if valid:
                # Score: prefer z with fewer affected points
                # Or prefer z with higher empty_line_count[z]?
                score = -affected
                if score > best_score:
                    best_score = score
                    best_z = z
        
        if best_z == -1:
            break
        
        try_add(best_z)

    # Build output
    pts_list = []
    for z in range(q2):
        if not S[z]:
            x = z // (p * p)
            y = z % (p * p)
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            pts_list.append([[xa, xb], [ya, yb]])
    
    result = np.array(pts_list, dtype=np.int64)
    return result
