"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import itertools
from collections import defaultdict
import math

# EVOLVE-BLOCK-START

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def count_four_lines(points):
    """Count lines with exactly 4 points, return 0 if any line has 5+."""
    n = len(points)
    if n < 4:
        return 0
    
    # Map each line (normalized) to the set of points on it
    line_map = defaultdict(set)
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
            
            g = _gcd(abs(dx), abs(dy))
            if g > 0:
                dx_n = dx // g
                dy_n = dy // g
            else:
                dx_n, dy_n = dx, dy
            
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            
            # Normal vector to direction (dx_n, dy_n) is (dy_n, -dx_n)
            nx = dy_n
            ny = -dx_n
            g2 = _gcd(abs(nx), abs(ny))
            if g2 > 0:
                nx = nx // g2
                ny = ny // g2
            if nx < 0 or (nx == 0 and ny < 0):
                nx = -nx
                ny = -ny
            c = nx * x1 + ny * y1
            
            key = (nx, ny, c)
            line_map[key].add(i)
            line_map[key].add(j)
    
    count = 0
    for key, pt_set in line_map.items():
        if len(pt_set) >= 5:
            return 0
        if len(pt_set) == 4:
            count += 1
    
    return count

def check_no_five_concurrent(lines):
    """Check that no point in dual plane is on 5+ lines (i.e., no 5 original points collinear).
    In dual: line (a,b,c) in primal becomes point (a,b) with intercept c.
    Two primal lines intersect at a point that lies on both.
    For dual: we check if any intersection point of dual lines has 5+ lines passing through it.
    
    Actually, let's think differently:
    - Primal point (x,y) -> dual line: y = m*x + b where... 
    - Standard dual: point (a,b) -> line y = ax - b
    - Line y = mx + c -> point (m, -c)
    
    Collinearity of primal points (x1,y1),...,(xk,yk) means their dual lines are concurrent.
    So "no 5 collinear" = "no 5 dual lines concurrent".
    
    To check concurrency: for each pair of dual lines, compute intersection, then check how many lines pass through that point.
    """
    n = len(lines)
    if n < 5:
        return True
    
    # For each pair of lines, find intersection and count how many lines pass through it
    # Lines are (a, b, c) representing ax + by = c
    # Intersection of a1*x + b1*y = c1 and a2*x + b2*y = c2:
    # det = a1*b2 - a2*b1
    # x = (c1*b2 - c2*b1) / det
    # y = (a1*c2 - a2*c1) / det
    
    # To avoid floating point, we can use rational arithmetic or check via cross products.
    # A point (x,y) is on line (a,b,c) iff a*x + b*y = c.
    # For intersection of lines i and j:
    # x = (ci*bj - cj*bi) / (ai*bj - aj*bi)
    # y = (ai*cj - aj*ci) / (ai*bj - aj*bi)
    
    # Check if line k passes through this point:
    # ak * x + bk * y = ck
    # ak * (ci*bj - cj*bi) + bk * (ai*cj - aj*ci) = ck * (ai*bj - aj*bi)
    
    concurrent_count = 0
    for i in range(n):
        for j in range(i+1, n):
            ai, bi, ci = lines[i]
            aj, bj, cj = lines[j]
            det = ai * bj - aj * bi
            if det == 0:
                continue  # parallel or same line
            
            # Check how many lines pass through intersection of i and j
            count = 0
            for k in range(n):
                ak, bk, ck = lines[k]
                # ak*(ci*bj - cj*bi) + bk*(ai*cj - aj*ci) == ck*(ai*bj - aj*bi)
                lhs = ak * (ci * bj - cj * bi) + bk * (ai * cj - aj * ci)
                rhs = ck * det
                if lhs == rhs:
                    count += 1
            
            if count >= 5:
                return False
    
    return True

def get_four_line_count_from_lines(lines):
    """Given dual lines (each representing a primal point), count how many intersection points
    have exactly 4 lines passing through them. Each such intersection corresponds to a line 
    in the primal plane containing exactly 4 points."""
    n = len(lines)
    if n < 4:
        return 0
    
    # For each pair of lines, compute the intersection point (rational)
    # and count how many lines pass through it
    intersection_counts = defaultdict(int)
    
    for i in range(n):
        for j in range(i+1, n):
            ai, bi, ci = lines[i]
            aj, bj, cj = lines[j]
            det = ai * bj - aj * bi
            if det == 0:
                continue
            
            # Intersection point represented as (x_num/det, y_num/det)
            # x_num = ci*bj - cj*bi
            # y_num = ai*cj - aj*ci
            x_num = ci * bj - cj * bi
            y_num = ai * cj - aj * ci
            
            # Normalize the rational point to create a canonical key
            # Reduce by gcd
            g = _gcd(abs(x_num), abs(y_num))
            if g == 0:
                g = 1
            x_num_r = x_num // g
            y_num_r = y_num // g
            
            # Also normalize the denominator
            det_r = det
            if det_r < 0:
                det_r = -det_r
                x_num_r = -x_num_r
                y_num_r = -y_num_r
            
            # Further reduce by gcd of all three
            g2 = _gcd(_gcd(abs(x_num_r), abs(y_num_r)), det_r)
            if g2 > 0:
                x_num_r //= g2
                y_num_r //= g2
                det_r //= g2
            
            key = (x_num_r, y_num_r, det_r)
            intersection_counts[key] += 1
    
    # But wait: each intersection point is counted once per pair of lines through it.
    # If k lines pass through a point, it's counted C(k,2) times.
    # We need to find points where exactly 4 lines pass through.
    
    # Better approach: for each intersection point, count how many lines actually pass through it.
    four_count = 0
    seen_points = set()
    
    for i in range(n):
        for j in range(i+1, n):
            ai, bi, ci = lines[i]
            aj, bj, cj = lines[j]
            det = ai * bj - aj * bi
            if det == 0:
                continue
            
            x_num = ci * bj - cj * bi
            y_num = ai * cj - aj * ci
            
            # Canonical key for the point
            g = _gcd(abs(x_num), abs(y_num))
            if g == 0:
                g = 1
            x_num_r = x_num // g
            y_num_r = y_num // g
            det_r = det
            if det_r < 0:
                det_r = -det_r
                x_num_r = -x_num_r
                y_num_r = -y_num_r
            g2 = _gcd(_gcd(abs(x_num_r), abs(y_num_r)), det_r)
            if g2 > 0:
                x_num_r //= g2
                y_num_r //= g2
                det_r //= g2
            
            key = (x_num_r, y_num_r, det_r)
            if key in seen_points:
                continue
            seen_points.add(key)
            
            # Count how many lines pass through this point
            count = 0
            for k in range(n):
                ak, bk, ck = lines[k]
                lhs = ak * x_num + bk * y_num
                rhs = ck * det
                if lhs == rhs:
                    count += 1
            
            if count == 4:
                four_count += 1
            elif count >= 5:
                return 0
    
    return four_count

def construct():
    random.seed(42)
    
    # We'll work in the dual plane.
    # A primal point (x,y) maps to a dual line.
    # Standard dual: point (a,b) -> line y = ax - b, or in form: ax - y - b = 0
    # So dual line coefficients: (a, -1, b) representing a*x - y = b
    
    # We need 16 dual lines with integer coefficients (a, -1, b) where a,b are integers.
    # Constraint: no 5 dual lines concurrent (no 5 primal points collinear).
    # Objective: maximize number of intersection points with exactly 4 lines.
    
    # Initialize with random lines
    best_lines = None
    best_score = 0
    
    # Try multiple random initializations
    for init_attempt in range(20):
        random.seed(42 + init_attempt)
        
        lines = []
        for _ in range(16):
            a = random.randint(-50, 50)
            b = random.randint(-50, 50)
            # Dual line: a*x - y = b, so coefficients (a, -1, b)
            lines.append((a, -1, b))
        
        # Check if valid (no 5 concurrent)
        if not check_no_five_concurrent(lines):
            continue
        
        score = get_four_line_count_from_lines(lines)
        if score > best_score:
            best_score = score
            best_lines = [l[:] for l in lines]
    
    if best_lines is None:
        # Fallback: use a known good configuration
        # 4x4 grid in primal maps to specific dual lines
        # Primal point (i,j) -> dual line: i*x - y = j
        best_lines = [(i, -1, j) for i in range(4) for j in range(4)]
        best_score = get_four_line_count_from_lines(best_lines)
    
    # Simulated annealing on dual lines
    T = 10.0
    T_min = 0.1
    cooling = 0.998
    max_iters = 30000
    
    current_lines = [l[:] for l in best_lines]
    current_score = best_score
    
    for it in range(max_iters):
        if T < T_min:
            break
        
        # Pick a random line to modify
        idx = random.randint(0, 15)
        old_line = current_lines[idx][:]
        
        # Modify the line: change a or b slightly
        move_range = random.choice([1, 2, 3, 5, 8])
        new_a = old_line[0] + random.randint(-move_range, move_range)
        new_b = old_line[2] + random.randint(-move_range, move_range)
        
        # Keep coefficients within reasonable bounds
        new_a = max(-200, min(200, new_a))
        new_b = max(-200, min(200, new_b))
        
        current_lines[idx] = (new_a, -1, new_b)
        
        # Check validity
        if not check_no_five_concurrent(current_lines):
            current_lines[idx] = old_line
            T *= cooling
            continue
        
        new_score = get_four_line_count_from_lines(current_lines)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_lines = [l[:] for l in current_lines]
        else:
            current_lines[idx] = old_line
        
        T *= cooling
    
    # Convert dual lines back to primal points
    # Dual line (a, -1, b) represents a*x - y = b
    # This corresponds to primal point... wait, I need to reconsider the duality.
    
    # Actually, the standard point-line duality:
    # Primal point (p, q) -> dual line y = px - q, i.e., px - y - q = 0
    # Primal line y = mx + c -> dual point (m, -c)
    
    # So if we have dual lines (a, -1, b) representing a*x - y = b,
    # these correspond to primal points (a, b) because:
    # Primal point (a, b) -> dual line a*x - y - b = 0, i.e., a*x - y = b.
    
    # So the primal points are simply (a, b) for each dual line (a, -1, b).
    
    points = [[l[0], l[2]] for l in best_lines]
    
    # Verify
    final_score = count_four_lines(points)
    if final_score < best_score:
        # Something went wrong, use fallback
        points = [[i, j] for i in range(4) for j in range(4)]
    
    return points

# EVOLVE-BLOCK-END
