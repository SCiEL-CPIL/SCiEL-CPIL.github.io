import time
import random
import math
from fractions import Fraction

def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def count_ordinary_lines(points):
    """
    Count the number of lines containing exactly two of the given points.
    points: list of [x, y]
    """
    n = len(points)
    if n < 2:
        return 0
    
    lines = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y2 - y1
            B = x1 - x2
            C = -(A * x1 + B * y1)
            
            g = gcd(gcd(A, B), C)
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            lines[key] = lines.get(key, 0) + 1
    
    ordinary_count = 0
    for key, count in lines.items():
        if count == 1:
            ordinary_count += 1
            
    return ordinary_count

def is_all_collinear(points):
    if len(points) < 3:
        return False
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    
    if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
        return False
    
    for i in range(3, len(points)):
        xi, yi = points[i]
        if (x2 - x1) * (yi - y1) != (y2 - y1) * (xi - x1):
            return False
            
    return True

def line_through_two_points(p1, p2):
    """Return (A, B, C) normalized such that A*x + B*y + C = 0, with gcd(A,B,C)=1 and canonical sign."""
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = -(A * x1 + B * y1)
    g = gcd(gcd(A, B), C)
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def find_integer_points_on_line(line_key, x_range=(-50, 50), y_range=(-50, 50)):
    """
    Find integer points (x, y) within range that lie on the line A*x + B*y + C = 0.
    Returns list of [x, y].
    """
    A, B, C = line_key
    points = []
    if A == 0 and B == 0:
        return points
    
    if A == 0:
        # B*y + C = 0 => y = -C/B
        if B != 0 and C % B == 0:
            y = -C // B
            if y_range[0] <= y <= y_range[1]:
                for x in range(x_range[0], x_range[1] + 1):
                    points.append([x, y])
        return points
    
    if B == 0:
        # A*x + C = 0 => x = -C/A
        if A != 0 and C % A == 0:
            x = -C // A
            if x_range[0] <= x <= x_range[1]:
                for y in range(y_range[0], y_range[1] + 1):
                    points.append([x, y])
        return points
    
    # General case: A*x + B*y + C = 0
    # Solve for y given x: y = -(A*x + C) / B
    # Need B | (A*x + C)
    # Let g = gcd(A, B). Then A*x + C ≡ 0 (mod B) has solutions iff g | C.
    g = gcd(A, B)
    if C % g != 0:
        return points
    
    A_reduced = A // g
    B_reduced = B // g
    C_reduced = C // g
    
    # A_reduced * x ≡ -C_reduced (mod B_reduced)
    # Since gcd(A_reduced, B_reduced) = 1, we can find inverse of A_reduced mod B_reduced
    if B_reduced == 0:
        return points
    
    # Find modular inverse of A_reduced modulo B_reduced
    def modinv(a, m):
        if m == 1:
            return 0
        g, x, _ = extended_gcd(a, m)
        if g != 1:
            return None
        return x % m
    
    def extended_gcd(a, b):
        if b == 0:
            return (a, 1, 0)
        g, x1, y1 = extended_gcd(b, a % b)
        return (g, y1, x1 - (a // b) * y1)
    
    inv = modinv(A_reduced % B_reduced, B_reduced)
    if inv is None:
        return points
    
    # x ≡ (-C_reduced * inv) (mod B_reduced)
    x0 = (-C_reduced * inv) % B_reduced
    
    # All solutions: x = x0 + k * B_reduced for integer k
    # For each such x in range, compute y = -(A*x + C) / B and check if integer and in range
    k_min = (x_range[0] - x0) // B_reduced
    if x0 + k_min * B_reduced < x_range[0]:
        k_min += 1
    k_max = (x_range[1] - x0) // B_reduced
    
    for k in range(k_min, k_max + 1):
        x = x0 + k * B_reduced
        num = -(A * x + C)
        if num % B == 0:
            y = num // B
            if y_range[0] <= y <= y_range[1]:
                points.append([x, y])
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 135s safety margin for 150s limit
    
    # Start with a 3x4 grid (12 points) - known to have moderate ordinary lines
    # 3x4 grid: points (0,0) to (2,3)
    initial_points = [[x, y] for x in range(3) for y in range(4)]
    
    best_points = [p[:] for p in initial_points]
    best_score = count_ordinary_lines(best_points)
    
    if is_all_collinear(best_points):
        # Fallback: use a different configuration
        best_points = [[0,0], [1,0], [2,0], [0,1], [1,1], [2,1], [0,2], [1,2], [2,2], [3,0], [3,1], [3,2]]
        best_score = count_ordinary_lines(best_points)
    
    # Line-aware local search
    current_points = [p[:] for p in initial_points]
    current_score = best_score
    
    # Precompute all lines through pairs of current points
    def get_all_lines(pts):
        lines_dict = {}
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                key = line_through_two_points(pts[i], pts[j])
                if key != (0, 0, 0):
                    lines_dict[key] = lines_dict.get(key, set())
                    lines_dict[key].add(i)
                    lines_dict[key].add(j)
        return lines_dict
    
    # Simulated annealing with line-aware moves
    T = 5.0
    cooling = 0.998
    min_T = 0.05
    
    while time.time() < deadline and T > min_T:
        if time.time() > deadline - 5:
            break
        
        i = random.randint(0, 11)
        orig_x, orig_y = current_points[i]
        
        # Strategy: try to move point i to lie on a line through two other points
        # This creates a new collinear triple, potentially reducing ordinary lines
        
        # Find all lines through pairs of other points
        other_indices = [j for j in range(12) if j != i]
        candidate_lines = set()
        for a in range(len(other_indices)):
            for b in range(a + 1, len(other_indices)):
                ia = other_indices[a]
                ib = other_indices[b]
                key = line_through_two_points(current_points[ia], current_points[ib])
                if key != (0, 0, 0):
                    candidate_lines.add(key)
        
        # For each candidate line, find integer points on it near the original position
        candidates = []
        for line_key in candidate_lines:
            # Search in a window around the original point
            x_lo = max(-10**9, orig_x - 100)
            x_hi = min(10**9, orig_x + 100)
            y_lo = max(-10**9, orig_y - 100)
            y_hi = min(10**9, orig_y + 100)
            
            pts_on_line = find_integer_points_on_line(line_key, (x_lo, x_hi), (y_lo, y_hi))
            for pt in pts_on_line:
                if pt != [orig_x, orig_y]:
                    # Check distinctness
                    distinct = True
                    for j in range(12):
                        if j == i:
                            continue
                        if current_points[j] == pt:
                            distinct = False
                            break
                    if distinct:
                        candidates.append(pt)
        
        # Also add some random perturbations for diversity
        for _ in range(3):
            dx = random.randint(-50, 50)
            dy = random.randint(-50, 50)
            new_x = orig_x + dx
            new_y = orig_y + dy
            if -10**9 <= new_x <= 10**9 and -10**9 <= new_y <= 10**9:
                pt = [new_x, new_y]
                distinct = True
                for j in range(12):
                    if j == i:
                        continue
                    if current_points[j] == pt:
                        distinct = False
                        break
                if distinct:
                    candidates.append(pt)
        
        if not candidates:
            T *= cooling
            continue
        
        new_point = random.choice(candidates)
        
        old_point = current_points[i]
        current_points[i] = new_point
        
        if is_all_collinear(current_points):
            current_points[i] = old_point
            T *= cooling
            continue
            
        new_score = count_ordinary_lines(current_points)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            delta = new_score - current_score
            if delta > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    current_points[i] = old_point
        
        T *= cooling
    
    # Final polishing: greedy local search
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(12):
            if time.time() > deadline - 5:
                break
            orig_x, orig_y = current_points[i]
            
            # Try small perturbations
            best_local = None
            best_local_score = current_score
            
            for dx in [-3, -2, -1, 1, 2, 3]:
                for dy in [-3, -2, -1, 1, 2, 3]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x = orig_x + dx
                    new_y = orig_y + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    new_point = [new_x, new_y]
                    is_distinct = True
                    for j in range(12):
                        if j == i:
                            continue
                        if current_points[j] == new_point:
                            is_distinct = False
                            break
                    if not is_distinct:
                        continue
                    old_point = current_points[i]
                    current_points[i] = new_point
                    if is_all_collinear(current_points):
                        current_points[i] = old_point
                        continue
                    new_score = count_ordinary_lines(current_points)
                    if new_score < best_local_score:
                        best_local_score = new_score
                        best_local = [new_x, new_y]
                    current_points[i] = old_point
            
            if best_local is not None:
                current_points[i] = best_local
                current_score = best_local_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
    
    # Validate final result
    if is_all_collinear(best_points):
        # Fix: perturb one point
        best_points[0][1] += 1
        while len(set(map(tuple, best_points))) < 12:
            best_points[0][0] += 1
    
    return best_points
