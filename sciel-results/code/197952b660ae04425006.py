"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def count_ordinary_lines(points):
    """Count lines containing exactly 2 points."""
    n = len(points)
    if n < 3:
        return 0
    
    # Use a set of normalized line representations
    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Normalize line representation
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            # Find gcd to normalize
            g = _gcd(abs(dx), abs(dy))
            dx_norm = dx // g
            dy_norm = dy // g
            
            # Ensure consistent sign
            if dx_norm < 0 or (dx_norm == 0 and dy_norm < 0):
                dx_norm = -dx_norm
                dy_norm = -dy_norm
            
            # Line equation: dy_norm*x - dx_norm*y = c
            c = dy_norm * x1 - dx_norm * y1
            
            key = (dx_norm, dy_norm, c)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    
    ordinary = 0
    for key, pts_in_line in lines.items():
        if len(pts_in_line) == 2:
            ordinary += 1
    
    return ordinary

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _is_valid(points):
    """Check if points are valid: distinct, integer, |x|,|y| <= 10^9, not all collinear."""
    if len(points) != 12:
        return False
    
    seen = set()
    for p in points:
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    
    # Check not all collinear
    if len(points) < 3:
        return True
    
    x1, y1 = points[0]
    x2, y2 = points[1]
    dx = x2 - x1
    dy = y2 - y1
    
    for i in range(2, len(points)):
        xi, yi = points[i]
        if dx * (yi - y1) != dy * (xi - x1):
            return True
    
    return False

def _hexagonal_lattice_points():
    """Generate 12 points on a hexagonal lattice pattern."""
    # Use a hexagonal grid: center + 6 neighbors + 6 at distance 2
    # Hexagonal lattice basis: (1,0), (0,1), (1,1) in transformed coords
    # Actually, use standard hexagonal lattice with 60-degree angles
    
    # Points at distance 0, 1, 2 from center in hexagonal lattice
    # Hex lattice: (a + b/2, b*sqrt(3)/2) but we need integers
    # Use axial coordinates: (q, r) -> (q + r/2, r*sqrt(3)/2)
    # For integer coords, use: x = 2*q + r, y = r (this gives a valid integer lattice)
    
    # Let's use a simpler approach: 3x4 grid with some diagonal points
    # Start with a 4x3 grid
    points = []
    for i in range(4):
        for j in range(3):
            points.append([i, j])
    return points

def _try_improve(points, max_time=100):
    """Local search to improve the configuration."""
    start_time = time.time()
    current_points = [list(p) for p in points]
    current_score = count_ordinary_lines(current_points)
    best_points = [list(p) for p in current_points]
    best_score = current_score
    
    # Try moving individual points
    while time.time() - start_time < max_time:
        improved = False
        random.shuffle(current_points)
        
        for i in range(len(current_points)):
            if time.time() - start_time > max_time - 10:
                break
            
            x, y = current_points[i]
            
            # Try nearby positions
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    
                    new_x = x + dx
                    new_y = y + dy
                    
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    
                    # Check if position is already taken
                    if any(current_points[j] == [new_x, new_y] for j in range(len(current_points)) if j != i):
                        continue
                    
                    # Try the move
                    old = current_points[i]
                    current_points[i] = [new_x, new_y]
                    
                    if not _is_valid(current_points):
                        current_points[i] = old
                        continue
                    
                    new_score = count_ordinary_lines(current_points)
                    
                    if new_score < current_score:
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_points = [list(p) for p in current_points]
                        improved = True
                    else:
                        current_points[i] = old
        
        if not improved:
            # Try swapping pairs
            improved = False
            for i in range(len(current_points)):
                if time.time() - start_time > max_time - 10:
                    break
                for j in range(i+1, len(current_points)):
                    if time.time() - start_time > max_time - 10:
                        break
                    
                    # Try swapping positions
                    current_points[i], current_points[j] = current_points[j], current_points[i]
                    
                    if _is_valid(current_points):
                        new_score = count_ordinary_lines(current_points)
                        if new_score < current_score:
                            current_score = new_score
                            if current_score < best_score:
                                best_score = current_score
                                best_points = [list(p) for p in current_points]
                            improved = True
                        else:
                            current_points[i], current_points[j] = current_points[j], current_points[i]
            
            if not improved:
                break
    
    return best_points, best_score

def construct():
    """
    Construct 12 integer points with a minimized number of ordinary lines.
    Strategy: Start with a hexagonal lattice pattern and use local search
    to maximize 3-point collinearities.
    """
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline with 10s safety margin
    
    # Initial configuration: 4x3 grid
    points = _hexagonal_lattice_points()
    
    # Run local search
    best_points, best_score = _try_improve(points, max_time=120)
    
    # Ensure we're within deadline
    if time.time() > deadline:
        # Return the best we have
        return best_points
    
    return best_points
# EVOLVE-BLOCK-END
