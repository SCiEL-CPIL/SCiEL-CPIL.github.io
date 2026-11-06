"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = _gcd(dx, dy)
    if g == 0:
        return None
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _count_4_lines(points):
    n = len(points)
    lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            dx, dy, c = key
            count = 0
            for k in range(n):
                x, y = points[k]
                if dy * x - dx * y == c:
                    count += 1
            if count == 4:
                lines.add(key)
    return len(lines)

def _is_valid(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            dx, dy, c = key
            count = 0
            for k in range(n):
                x, y = points[k]
                if dy * x - dx * y == c:
                    count += 1
            if count >= 5:
                return False
    return True

def _ag24_points():
    """
    Generate 16 points from AG(2,4) affine plane embedded in integer lattice.
    AG(2,4) has 16 points and 20 lines of size 4.
    We map (a,b) where a,b in GF(4) to integer coordinates.
    GF(4) = {0, 1, alpha, alpha+1} where alpha^2 = alpha + 1.
    We use a linear embedding: (a,b) -> (x, y) where we represent GF(4) elements as pairs.
    Actually, let's use a simpler approach: enumerate all 16 points of AG(2,4) 
    and find an integer embedding.
    
    AG(2,4) points can be represented as (i, j) where i, j in {0,1,2,3}.
    The lines are:
    - Horizontal: {(i, j) for fixed j}
    - Vertical: {(i, j) for fixed i}  
    - Sloped lines with various slopes in GF(4)
    
    Let's use a known embedding. The 16 points of AG(2,4) can be mapped to:
    (0,0), (1,0), (2,0), (3,0),
    (0,1), (1,1), (2,1), (3,1),
    (0,2), (1,2), (2,2), (3,2),
    (0,3), (1,3), (2,3), (3,3)
    but this is just a 4x4 grid which doesn't have the right line structure.
    
    Instead, let's use the fact that AG(2,4) is isomorphic to the vector space GF(4)^2.
    We can map each element of GF(4) to a pair of bits, so GF(4)^2 maps to GF(2)^4.
    Then we can use a linear transformation to get integer coordinates.
    
    Actually, the simplest approach: use the 4x4 grid but recognize that it has 
    exactly 20 lines of size 4 (4 horizontal + 4 vertical + 12 diagonal/sloped).
    Let's verify this.
    """
    # 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    return points

def _ag24_lines(points):
    """
    Find all lines of exactly 4 points in the given configuration.
    For AG(2,4) embedded as 4x4 grid, we expect 20 such lines.
    """
    n = len(points)
    lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            dx, dy, c = key
            count = 0
            for k in range(n):
                x, y = points[k]
                if dy * x - dx * y == c:
                    count += 1
            if count == 4:
                lines.add(key)
    return lines

def _local_search(points, max_time=100.0):
    """
    Local search to improve the configuration.
    """
    start_time = time.time()
    limit = max_time
    best_points = [p[:] for p in points]
    best_score = _count_4_lines(best_points)
    
    while time.time() - start_time < limit:
        improved = False
        for idx in range(16):
            if time.time() - start_time > limit - 5.0:
                break
            x, y = points[idx]
            # Try nearby moves
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    
                    # Check collision
                    collision = False
                    for i in range(16):
                        if i == idx:
                            continue
                        if points[i][0] == nx and points[i][1] == ny:
                            collision = True
                            break
                    if collision:
                        continue
                    
                    new_points = [p[:] for p in points]
                    new_points[idx] = [nx, ny]
                    
                    # Check validity (no 5 collinear)
                    valid = True
                    for i in range(16):
                        if i == idx:
                            continue
                        key = _line_key(new_points[idx], new_points[i])
                        if key is None:
                            continue
                        dx_l, dy_l, c = key
                        count = 0
                        for j in range(16):
                            x_j, y_j = new_points[j]
                            if dy_l * x_j - dx_l * y_j == c:
                                count += 1
                        if count >= 5:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    score = _count_4_lines(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points
                        points = new_points
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try a random jump
            if random.random() < 0.1:
                idx = random.randint(0, 15)
                nx = random.randint(-15, 15)
                ny = random.randint(-15, 15)
                if abs(nx) <= 1000 and abs(ny) <= 1000:
                    collision = False
                    for i in range(16):
                        if i == idx:
                            continue
                        if points[i][0] == nx and points[i][1] == ny:
                            collision = True
                            break
                    if not collision:
                        new_points = [p[:] for p in points]
                        new_points[idx] = [nx, ny]
                        if _is_valid(new_points):
                            score = _count_4_lines(new_points)
                            if score > best_score:
                                best_score = score
                                best_points = new_points
                                points = new_points
            else:
                break
                
    return best_points, best_score

def construct():
    start_time = time.time()
    limit = 110.0  # Safety margin
    
    # Start with AG(2,4) embedding (4x4 grid)
    points = _ag24_points()
    best_points = [p[:] for p in points]
    best_score = _count_4_lines(best_points)
    
    # Verify the initial score
    # The 4x4 grid should have 20 lines of size 4
    # 4 horizontal + 4 vertical + 12 diagonal (slopes 1, -1, 2, -2, 1/2, -1/2, etc.)
    
    # Run local search to try to improve
    if time.time() - start_time < limit - 10.0:
        best_points, best_score = _local_search(best_points, max_time=limit - (time.time() - start_time) - 5.0)
    
    return best_points
# EVOLVE-BLOCK-END
