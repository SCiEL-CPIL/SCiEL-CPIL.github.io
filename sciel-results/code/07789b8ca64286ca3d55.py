import random
import time
import math
from itertools import combinations

def count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly two points.
    points: list of [x, y]
    """
    n = len(points)
    if n < 3:
        return 0
    
    # Use a set of line keys to identify lines
    # A line can be represented by (A, B, C) where Ax + By + C = 0
    # To avoid floating point issues, use integer coefficients.
    # For points (x1, y1) and (x2, y2):
    # A = y2 - y1
    # B = x1 - x2
    # C = x2*y1 - x1*y2
    # We must normalize (A, B, C) by dividing by their GCD and ensuring a canonical sign.
    
    lines = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            # Normalize the line equation
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            # Canonical sign: first non-zero of (A, B) should be positive
            if A < 0:
                A, B, C = -A, -B, -C
            elif A == 0 and B < 0:
                A, B, C = -A, -B, -C
            elif A == 0 and B == 0 and C < 0:
                A, B, C = -A, -B, -C
                
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary_count = 0
    for key, pts_in_line in lines.items():
        if len(pts_in_line) == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_score = float('inf')
    
    # Initial candidates
    candidates = []
    
    # 1. 3x4 Grid
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # 2. 4x3 Grid
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # 3. 2x6 Grid
    candidates.append([[x, y] for x in range(6) for y in range(2)])
    
    # 4. 6x2 Grid
    candidates.append([[x, y] for x in range(2) for y in range(6)])
    
    # 5. Random points in a small box
    for _ in range(5):
        pts = []
        while len(pts) < 12:
            p = [random.randint(-5, 5), random.randint(-5, 5)]
            if p not in pts:
                pts.append(p)
        candidates.append(pts)
        
    # 6. Hexagonal lattice approximation (integer coords)
    # Points in a hexagon-like shape
    hex_pts = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2],
        [0, 3], [1, 3], [2, 3]
    ]
    candidates.append(hex_pts)
    
    # Local Search Function
    def local_search(points):
        current_pts = [p[:] for p in points]
        current_score = count_ordinary_lines(current_pts)
        
        while time.time() < deadline:
            improved = False
            # Shuffle points to avoid bias
            random.shuffle(current_pts)
            
            for i in range(12):
                if time.time() > deadline:
                    break
                x, y = current_pts[i]
                # Try moving point i in a neighborhood
                # Radius 10 as per knowledge
                for dx in range(-10, 11):
                    for dy in range(-10, 11):
                        if dx == 0 and dy == 0:
                            continue
                        new_x, new_y = x + dx, y + dy
                        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                            continue
                        
                        # Check for duplicates
                        is_dup = False
                        for j in range(12):
                            if i != j and current_pts[j] == [new_x, new_y]:
                                is_dup = True
                                break
                        if is_dup:
                            continue
                        
                        # Try the move
                        old_pt = current_pts[i]
                        current_pts[i] = [new_x, new_y]
                        new_score = count_ordinary_lines(current_pts)
                        
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                        else:
                            current_pts[i] = old_pt
                        
                        if current_score == 6:
                            return current_pts, current_score
                
                if current_score == 6:
                    return current_pts, current_score
                    
            if not improved:
                break
                
        return current_pts, current_score

    # Run local search on each candidate
    for candidate in candidates:
        if time.time() > deadline:
            break
        pts, score = local_search(candidate)
        if score < best_score:
            best_score = score
            best_points = pts
        if best_score == 6:
            break
            
    # If we didn't find a good solution, try random restarts
    if best_score > 6:
        attempts = 0
        while time.time() < deadline and attempts < 50:
            attempts += 1
            # Random start
            pts = []
            while len(pts) < 12:
                p = [random.randint(-20, 20), random.randint(-20, 20)]
                if p not in pts:
                    pts.append(p)
            
            pts, score = local_search(pts)
            if score < best_score:
                best_score = score
                best_points = pts
            if best_score == 6:
                break

    # Fallback: Return the best found, or the seed if none
    if best_points is None:
        return [[x, y] for x in range(4) for y in range(3)]
        
    return best_points
