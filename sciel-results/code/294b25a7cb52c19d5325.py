import time
import random
from math import gcd
from collections import Counter, defaultdict

def _line_key(p1, p2):
    """Compute canonical line key from two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    ndx = dx // g
    ndy = dy // g
    if ndx < 0 or (ndx == 0 and ndy < 0):
        ndx = -ndx
        ndy = -ndy
    c = ndy * x1 - ndx * y1
    return (ndx, ndy, c)

def _evaluate(points):
    """
    Evaluate a list of 16 [x,y] points.
    Returns (is_valid, score).
    is_valid: 16 distinct integer points, |x|,|y|<=1000, no 5 collinear.
    score: number of lines containing exactly 4 points.
    """
    n = len(points)
    if n != 16:
        return False, 0
    
    point_set = set()
    for p in points:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return False, 0
        pt = (x, y)
        if pt in point_set:
            return False, 0
        point_set.add(pt)
        
    lines = defaultdict(int)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            key = _line_key(points[i], points[j])
            if key is not None:
                lines[key] += 1
            
    score = 0
    for count in lines.values():
        if count >= 10:  # C(5,2)=10 means 5+ collinear
            return False, 0
        if count == 6:   # C(4,2)=6 means exactly 4 collinear
            score += 1
            
    return True, score

def _build_line_counts(points):
    """Build line count dictionary from points."""
    lines = defaultdict(int)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                lines[key] += 1
    return lines

def _score_from_line_counts(lines):
    """Compute score and validity from line counts."""
    score = 0
    for count in lines.values():
        if count >= 10:
            return False, 0
        if count == 6:
            score += 1
    return True, score

def _incremental_evaluate(points, idx_old, old_pt, new_pt):
    """
    Incrementally update line counts when point idx_old moves from old_pt to new_pt.
    Returns (is_valid, score, updated_line_counts).
    """
    n = len(points)
    
    # Check new point validity
    nx, ny = new_pt
    if abs(nx) > 1000 or abs(ny) > 1000:
        return False, 0, None
    
    # Check distinctness
    for i in range(n):
        if i != idx_old:
            if points[i][0] == nx and points[i][1] == ny:
                return False, 0, None
    
    # Build current line counts
    lines = _build_line_counts(points)
    
    # Remove contributions of old point
    for j in range(n):
        if j != idx_old:
            key = _line_key(old_pt, points[j])
            if key is not None:
                lines[key] -= 1
                if lines[key] == 0:
                    del lines[key]
    
    # Add contributions of new point
    for j in range(n):
        if j != idx_old:
            key = _line_key(new_pt, points[j])
            if key is not None:
                lines[key] += 1
    
    is_valid, score = _score_from_line_counts(lines)
    return is_valid, score, lines

def _generate_affine_plane_points():
    """
    Generate 16 points from the affine plane AG(2,4) mapped to integers.
    AG(2,4) has 16 points and 20 lines of 4 points each.
    
    We represent points as (a, b) where a, b are in GF(4).
    GF(4) = {0, 1, alpha, alpha+1} where alpha^2 = alpha + 1.
    
    Map GF(4) elements to integers: 0->0, 1->1, alpha->2, alpha+1->3
    This gives a 4x4 grid, but we need to verify the line structure.
    
    Actually, let's use a more careful construction.
    """
    # GF(4) arithmetic: elements are 0, 1, 2, 3 representing 0, 1, alpha, alpha+1
    # alpha = 2, alpha^2 = alpha + 1 = 3, alpha^3 = 1
    # Addition table (XOR-like but not exactly):
    # We'll define addition and multiplication for GF(4)
    
    # Element representation: 0, 1, 2, 3
    # alpha = 2, so:
    # 0+0=0, 0+1=1, 0+2=2, 0+3=3
    # 1+1=0, 1+2=3, 1+3=2
    # 2+2=0, 2+3=1
    # 3+3=0
    
    add_table = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0]
    ]
    
    mul_table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    
    # Points of AG(2,4): (x, y) where x, y in GF(4)
    # Map to integer coordinates
    # Use a nonlinear mapping to avoid trivial grid structure
    
    # Simple mapping: (x, y) -> (x * 10 + y, x * 10 + y * 2) or similar
    # Let's try: (x, y) -> (x, y) with x, y in {0,1,2,3} scaled
    
    # For now, use a 4x4 grid but with perturbation to break symmetries
    # The 4x4 grid already has 20 four-point lines (4 rows, 4 cols, 4+4 diagonals... actually let's count)
    
    # 4x4 grid: 4 horizontal lines, 4 vertical lines, 2 main diagonals = 10 lines of 4
    # But there are also other 4-point lines in a 4x4 grid?
    # Actually in a 4x4 grid, the lines of 4 points are:
    # - 4 rows
    # - 4 columns  
    # - 2 main diagonals
    # Total: 10 lines of 4 points
    
    # So the 4x4 grid gives score 10, not 20.
    # To get closer to the affine plane structure, we need a different mapping.
    
    # Let's use a different approach: generate points that form the affine plane structure
    # by using a projective transformation.
    
    # For now, start with the 4x4 grid and let the search improve it.
    return [[x, y] for x in range(4) for y in range(4)]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial configuration: 4x4 grid
    current_points = _generate_affine_plane_points()
    is_valid, current_score = _evaluate(current_points)
    if not is_valid:
        current_points = [[x, y] for x in range(4) for y in range(4)]
        is_valid, current_score = _evaluate(current_points)
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # Simulated annealing parameters
    temperature = 1.0
    cooling_rate = 0.9995
    min_temperature = 0.01
    
    no_improve_steps = 0
    max_no_improve = 300
    restart_count = 0
    
    while time.time() < deadline:
        prev_points = [p[:] for p in current_points]
        
        # Generate a move
        r = random.random()
        if r < 0.5:
            # Small local move
            idx = random.randint(0, 15)
            ox, oy = current_points[idx]
            nx = ox + random.randint(-3, 3)
            ny = oy + random.randint(-3, 3)
            if abs(nx) <= 1000 and abs(ny) <= 1000:
                current_points[idx] = [nx, ny]
            else:
                continue
        elif r < 0.7:
            # Medium move
            idx = random.randint(0, 15)
            nx = random.randint(-50, 50)
            ny = random.randint(-50, 50)
            current_points[idx] = [nx, ny]
        elif r < 0.85:
            # Large move
            idx = random.randint(0, 15)
            nx = random.randint(-200, 200)
            ny = random.randint(-200, 200)
            current_points[idx] = [nx, ny]
        else:
            # Swap two points
            i, j = random.sample(range(16), 2)
            current_points[i], current_points[j] = current_points[j], current_points[i]
        
        is_valid, new_score = _evaluate(current_points)
        
        if is_valid:
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
                no_improve_steps = 0
            elif new_score == current_score:
                no_improve_steps += 1
                # Accept equal score to allow random walk
            else:
                # Worse score: accept with simulated annealing probability
                delta = current_score - new_score
                if random.random() < min(1.0, temperature / max(1, delta)):
                    no_improve_steps += 1
                else:
                    # Revert
                    current_points = prev_points
                    no_improve_steps += 1
        else:
            # Invalid, revert
            current_points = prev_points
            no_improve_steps += 1
        
        # Cool down
        temperature = max(min_temperature, temperature * cooling_rate)
        
        # Check for restart condition
        if no_improve_steps > max_no_improve:
            # Restart from best
            current_points = [p[:] for p in best_points]
            current_score = best_score
            no_improve_steps = 0
            temperature = 0.5  # Reset temperature for restart
            restart_count += 1
            
            # If we've restarted many times, increase temperature for more exploration
            if restart_count > 5:
                temperature = 1.0

    return best_points
