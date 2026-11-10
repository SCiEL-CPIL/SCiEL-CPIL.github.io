import numpy as np
import time
import random

_P = 7
_Q = 49
_Q2 = 2401
_DEADLINE = time.time() + 600.0
_MARGIN = 10.0

def nikodym_set() -> np.ndarray:
    p = _P
    q = _Q
    q2 = _Q2
    
    # Build field elements: index = a * 7 + b for a + b*alpha
    # Point index = x_idx * 49 + y_idx where x_idx = xa*7+xb, y_idx = ya*7+yb
    
    # Precompute all lines: for each slope m (0..48) and vertical
    # Line with direction (1, m): points {(x, y + t*m) : t in F_49} for fixed (x0, y0) on line
    # Actually: line through (x0,y0) with direction (1,m) is {(x0 + t, y0 + t*m) : t in F_49}
    # Vertical line through (x0, y0) is {(x0, y0 + t) : t in F_49}
    
    # For efficiency, represent each line as a set of point indices
    
    # Field multiplication: (a1+b1*alpha)*(a2+b2*alpha) = (a1*a2+3*b1*b2) + (a1*b2+a2*b1)*alpha
    def field_mul(i, j):
        a1, b1 = i // p, i % p
        a2, b2 = j // p, j % p
        ra = (a1 * a2 + 3 * b1 * b2) % p
        rb = (a1 * b2 + a2 * b1) % p
        return ra * p + rb
    
    def field_add(i, j):
        a1, b1 = i // p, i % p
        a2, b2 = j // p, j % p
        ra = (a1 + a2) % p
        rb = (b1 + b2) % p
        return ra * p + rb
    
    def field_sub(i, j):
        a1, b1 = i // p, i % p
        a2, b2 = j // p, j % p
        ra = (a1 - a2) % p
        rb = (b1 - b2) % p
        return ra * p + rb
    
    # Build all lines
    # For each slope m (0..48) and each intercept, plus vertical lines
    # Line with direction (1,m) through point (x0, y0): {(x0+t, y0 + t*m) : t in F_49}
    # Two points (x1,y1) and (x2,y2) are on the same non-vertical line iff (x1-x2) != 0 and (y1-y2)/(x1-x2) is consistent
    # For direction (1,m): (y2-y1) = m*(x2-x1)
    
    # Precompute: for each point, which lines pass through it
    # There are 49 non-vertical directions * 49 lines per direction + 49 vertical lines = 49*49+49 = 2450 lines
    # Each line has 49 points. Total point-line incidences: 2450*49 = 120050 = 2401*50. Makes sense.
    
    # Let's precompute for each point, the list of lines it belongs to
    # Line ID: for non-vertical, line_id = m*49 + c where the line is y = m*x + c
    # For vertical, line_id = 49*49 + x0 (x = x0)
    
    # Actually, let me use a different representation.
    # For non-vertical lines with direction (1,m): 
    #   A point (x,y) is on the line if y - m*x = c (constant)
    #   So for each m, group points by c = y - m*x
    # For vertical lines: group points by x coordinate
    
    # Precompute line membership
    # point_lines[point_idx] = list of line_ids containing this point
    
    # Line IDs:
    # Non-vertical: m in 0..48, c in 0..48 -> line_id = m*49 + c (total 2401)
    # Vertical: x0 in 0..48 -> line_id = 2401 + x0 (total 49)
    # Total: 2450 lines
    
    point_lines = [[] for _ in range(q2)]
    line_points = [set() for _ in range(2450)]
    
    for xi in range(q):
        for yi in range(q):
            pi = xi * q + yi
            # Vertical line: x = xi
            vl = 2401 + xi
            point_lines[pi].append(vl)
            line_points[vl].add(pi)
            
            # Non-vertical lines: for each m, c = yi - m*xi
            for m in range(q):
                mc = field_sub(yi, field_mul(m, xi))
                nl = m * q + mc
                point_lines[pi].append(nl)
                line_points[nl].add(pi)
    
    # Start with parabola removed
    in_N = np.ones(q2, dtype=bool)
    
    for xa in range(p):
        for xb in range(p):
            ya = (xa * xa + 3 * xb * xb) % p
            yb = (2 * xa * xb) % p
            x_idx = xa * p + xb
            y_idx = ya * p + yb
            pi = x_idx * q + y_idx
            in_N[pi] = False
    
    # For each point, count how many valid witness lines it has
    # A line L is a valid witness for point x if:
    #   - x is on L
    #   - All other points on L are in N
    #   - The direction of L is (1,m) or (0,1) (which is always true for our lines)
    
    def count_witnesses(point_idx, in_N_arr):
        """Count valid witness lines for point_idx given in_N_arr."""
        count = 0
        for lid in point_lines[point_idx]:
            lp = line_points[lid]
            # Check if all other points on this line are in N
            valid = True
            for q_idx in lp:
                if q_idx != point_idx and not in_N_arr[q_idx]:
                    valid = False
                    break
            if valid:
                count += 1
        return count
    
    # Initial witness counts
    witness_counts = np.zeros(q2, dtype=np.int32)
    for i in range(q2):
        witness_counts[i] = count_witnesses(i, in_N)
    
    # Now try to remove additional points
    # A point p can be removed if:
    #   - p is currently in N
    #   - For every point x that shares a line with p, if that line was the ONLY witness for x, 
    #     then removing p would break x's witness
    #   - More precisely: removing p from N means p is no longer in N. 
    #     For any line L containing p, L is no longer a valid witness for any point on L 
    #     (since p is not in N, and the witness requires all OTHER points to be in N, 
    #     but wait - the witness for x on line L requires all points on L EXCEPT x to be in N.
    #     If p is on L and p != x, then p must be in N for L to be a witness for x.
    #     So removing p invalidates L as a witness for all x on L where x != p.)
    
    # So: removing p is safe if for every line L containing p, and every point x on L (x != p),
    #     x has at least one OTHER valid witness line (not L, or L but p is still... no, p is removed).
    #     Actually: after removing p, line L is no longer a valid witness for x (x on L, x != p) 
    #     because p is not in N. So x loses L as a witness.
    #     x is safe if x has at least one other valid witness line that doesn't contain p, 
    #     OR another valid witness line that contains p but... no, if it contains p, it's also invalidated.
    #     
    #     So: x is safe if x has at least one valid witness line L' (L' != L) that does NOT contain p.
    #     Equivalently: the number of valid witness lines for x that do NOT contain p is >= 1.
    
    # This is expensive to check for all x. Let me use a heuristic: 
    # Only try to remove points that have many neighbors with multiple witnesses.
    
    # Alternative simpler check: after removing p, recompute witness counts for affected points.
    # Affected points: all points on lines through p (at most 50 lines * 49 points, but with overlaps).
    
    # Let me try a greedy approach: repeatedly find a point that can be safely removed.
    
    removed_count = 49
    max_removed = 2400  # Can't remove all points (N must be non-empty for the set to make sense, actually k>=1)
    
    rng = random.Random(42)
    
    # Create a list of points currently in N
    candidates = [i for i in range(q2) if in_N[i]]
    rng.shuffle(candidates)
    
    # For efficiency, precompute for each point p, the set of points affected if p is removed
    # affected[p] = set of points x such that x is on some line with p, x != p
    # This is all points except p, since any two points determine a line... 
    # Wait, in F_q^2, any two points determine exactly one line (if they have the same x, it's vertical; 
    # otherwise it's non-vertical). So every other point is on some line with p.
    # So removing p affects ALL other points. That's too expensive to check naively.
    
    # Better approach: for each point x, track which lines are valid witnesses.
    # When we remove p, for each line L containing p, L becomes invalid for all x on L (x != p).
    # So we need to check: for each x != p, does x still have at least one valid witness line 
    # that does NOT contain p?
    
    # Let me maintain witness_lines[x] = set of line IDs that are valid witnesses for x.
    # Initially compute this.
    
    witness_lines = [set() for _ in range(q2)]
    for i in range(q2):
        for lid in point_lines[i]:
            lp = line_points[lid]
            valid = True
            for q_idx in lp:
                if q_idx != i and not in_N[q_idx]:
                    valid = False
                    break
            if valid:
                witness_lines[i].add(lid)
    
    # Now try to remove points
    # For a candidate point p to remove:
    #   For each line L containing p:
    #     For each x on L (x != p):
    #       If L is in witness_lines[x], then x loses L as a witness.
    #       Check if x still has at least one witness in witness_lines[x] \ {L} that doesn't contain p.
    #       Actually, all witnesses of x that contain p will be invalidated.
    #       So x is safe if |witness_lines[x] \ {L : L contains p}| >= 1.
    
    # Precompute: for each point p, the set of lines containing p (already have point_lines[p])
    # For each point x, the set of lines containing both x and p: 
    #   If x and p have the same x-coordinate, the vertical line.
    #   Otherwise, the non-vertical line through x and p.
    #   So there's exactly ONE line containing both x and p (for x != p).
    
    # So: removing p invalidates exactly one witness line for each x != p (the line through x and p).
    # x is safe if x has at least 2 witness lines, OR x has exactly 1 witness line and that line is NOT the line through x and p.
    
    # This is much simpler! Let me implement this.
    
    def line_through(x_idx, p_idx):
        """Get the line ID through points x_idx and p_idx (x_idx != p_idx)."""
        x1 = x_idx // q
        y1 = x_idx % q
        x2 = p_idx // q
        y2 = p_idx % q
        
        if x1 == x2:
            return 2401 + x1
        
        # m = (y2 - y1) / (x2 - x1)
        dx = field_sub(x2, x1)
        dy = field_sub(y2, y1)
        # Need dx^{-1}
        # In F_49, every non-zero element has an inverse
        # Compute inverse of dx
        # dx = a + b*alpha, inverse = (a - b*alpha) / (a^2 - 3*b^2)
        a, b = dx // p, dx % p
        denom = (a * a - 3 * b * b) % p
        # denom should be non-zero since dx != 0
        # Find inverse of denom in F_7
        denom_inv = pow(denom, 5, 7)  # Fermat's little theorem: x^6 = 1, so x^5 = x^{-1}
        # inverse = (a - b*alpha) * denom_inv
        ia = (a * denom_inv) % p
        ib = (-b * denom_inv) % p
        dx_inv = ia * p + ib
        
        m = field_mul(dy, dx_inv)
        # c = y1 - m * x1
        c = field_sub(y1, field_mul(m, x1))
        return m * q + c
    
    # Try to remove additional points
    # For each candidate p (in shuffled order):
    #   Check if removing p is safe:
    #     For each x != p:
    #       L = line through x and p
    #       If L is in witness_lines[x] and len(witness_lines[x]) == 1:
    #         Then x would have 0 witnesses -> not safe
    #       If L is NOT in witness_lines[x] or len(witness_lines[x]) >= 2:
    #         Safe for x
    #   If safe for all x, remove p and update witness_lines
    
    # This is O(q2 * q2) per candidate, which is 2401*2401 ≈ 5.7M operations per candidate.
    # Too slow. Let me optimize.
    
    # Optimization: only check x where the line through x and p is actually a witness for x.
    # Precompute: for each point x, the set of points p such that line(x,p) is in witness_lines[x].
    # That's expensive to precompute for all pairs.
    
    # Alternative: for each candidate p, iterate over lines L containing p.
    # For each such line L, iterate over points x on L (x != p).
    # For each such x, check if L is in witness_lines[x] and len(witness_lines[x]) == 1.
    # If so, not safe.
    
    # Number of lines through p: 50. Each line has 48 other points. Total checks: 50*48 = 2400.
    # That's fast!
    
    def can_remove(p_idx, in_N_arr, witness_lines_list):
        """Check if removing p_idx is safe."""
        for lid in point_lines[p_idx]:
            for x_idx in line_points[lid]:
                if x_idx == p_idx:
                    continue
                if lid in witness_lines_list[x_idx] and len(witness_lines_list[x_idx]) == 1:
                    return False
        return True
    
    def remove_point(p_idx, in_N_arr, witness_lines_list):
        """Remove p_idx from N and update witness_lines."""
        in_N_arr[p_idx] = False
        # For each line L containing p_idx:
        #   L is no longer a valid witness for any x on L (x != p_idx)
        #   Remove L from witness_lines[x] for all x on L (x != p_idx)
        for lid in point_lines[p_idx]:
            for x_idx in line_points[lid]:
                if x_idx == p_idx:
                    continue
                if lid in witness_lines_list[x_idx]:
                    witness_lines_list[x_idx].discard(lid)
    
    # Try to remove points in shuffled order
    for p_idx in candidates:
        if time.time() > _DEADLINE - _MARGIN:
            break
        if not in_N[p_idx]:
            continue
        if can_remove(p_idx, in_N, witness_lines):
            remove_point(p_idx, in_N, witness_lines)
            removed_count += 1
    
    # Build result
    N = []
    for i in range(q2):
        if in_N[i]:
            x = i // q
            y = i % q
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            N.append([[xa, xb], [ya, yb]])
    
    return np.array(N, dtype=np.int64)
