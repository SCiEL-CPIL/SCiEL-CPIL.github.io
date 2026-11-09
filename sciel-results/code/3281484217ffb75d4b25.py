"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _count_ordinary_lines(points):
    """Count lines containing exactly two of the given points."""
    n = len(points)
    if n < 3:
        return 0
    # Use exact rational arithmetic to avoid floating point issues
    ordinary = 0
    # For each pair, check if any third point lies on the line
    # A line through points i and j is ordinary if no other point k is collinear
    # To count efficiently: for each pair (i,j), count collinear points
    # But that's O(n^3). For n=12, that's 12*11*10/2 * 10 = 6600 checks, very fast.
    
    # Alternative: for each point i, group other points by slope/intercept
    # A line through i and j is ordinary if the group size is exactly 2 (just i and j)
    
    for i in range(n):
        xi, yi = points[i]
        # Group all other points by the line through (xi, yi)
        line_map = {}
        for j in range(n):
            if j == i:
                continue
            xj, yj = points[j]
            # Direction vector, normalized to a canonical form
            dx = xj - xi
            dy = yj - yi
            # Normalize: divide by gcd, ensure consistent sign
            g = _gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            else:
                # dx=dy=0, shouldn't happen for distinct points
                continue
            # Ensure canonical sign: first non-zero component positive
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            key = (dx, dy)
            # We also need to distinguish parallel lines with same direction but different intercepts
            # Actually, for lines through (xi, yi), the direction uniquely determines the line
            # since they all pass through (xi, yi). So direction is sufficient.
            if key not in line_map:
                line_map[key] = []
            line_map[key].append(j)
        
        # For each direction from point i, if exactly one other point shares it,
        # then the line through i and that point has exactly 2 points total?
        # No! We need to check if any OTHER point (not i) is also on that line.
        # Since all lines here pass through i, if line_map[key] has size k, 
        # then the line contains i plus those k points = k+1 points.
        # It's ordinary if k+1 == 2, i.e., k == 1.
        
        for key, pts_on_line in line_map.items():
            if len(pts_on_line) == 1:
                ordinary += 1
    
    # Each ordinary line is counted twice (once from each endpoint)
    # Wait, no. We count for each point i, and for each line through i with exactly 1 other point.
    # An ordinary line has exactly 2 points, say p and q.
    # When i=p, the line through p and q has exactly 1 other point (q), so it's counted.
    # When i=q, the line through q and p has exactly 1 other point (p), so it's counted.
    # So each ordinary line is counted exactly twice.
    return ordinary // 2

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _evaluate(points):
    """Return the number of ordinary lines for the given set of points."""
    return _count_ordinary_lines(points)

def _not_all_collinear(points):
    """Check that not all points are collinear."""
    n = len(points)
    if n < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    # Find a third point not on the line through first two
    for i in range(2, n):
        xi, yi = points[i]
        # Cross product
        if (x1 - x0) * (yi - y0) - (y1 - y0) * (xi - x0) != 0:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Known good configuration: 3x4 grid has some number of ordinary lines
    # Let's start with a known configuration and try to improve
    
    # The 3x4 grid:
    grid = [[x, y] for x in range(4) for y in range(3)]
    best = grid[:]
    best_score = _evaluate(grid)
    
    # Try various structured configurations
    # 1. 4x3 grid
    grid2 = [[x, y] for x in range(3) for y in range(4)]
    score2 = _evaluate(grid2)
    if score2 < best_score:
        best = grid2[:]
        best_score = score2
    
    # 2. Points on a conic (parabola y = x^2) - but need 12 points
    # For points on a parabola, any line intersects at most 2 points,
    # so all lines are ordinary. That gives C(12,2) = 66 ordinary lines. Bad.
    
    # 3. Try configurations based on known optimal or near-optimal sets
    # The minimum for 12 points over reals is 6.
    # A configuration achieving 6 would have every line contain either 2 or >=4 points,
    # with exactly 6 lines containing exactly 2 points.
    
    # Let's try a 2x6 arrangement or other structured sets
    # Actually, let's try to search intelligently.
    
    # Try: points arranged in rows with some structure
    # Configuration: 4 points on y=0, 4 on y=1, 4 on y=2, with x-coordinates chosen
    # to create many 3-point lines.
    
    # Known: the Pappus configuration and other combinatorial designs
    # For 12 points, one candidate is based on the affine plane of order 3 (9 points)
    # plus 3 more points.
    
    # Let me try a specific known configuration:
    # The "dual" of a 12-point configuration with 6 ordinary lines
    
    # Try: 3 groups of 4 points, each group on a line, arranged so that
    # there are many transversals with 3+ points.
    
    # Configuration A: 4x3 grid already tried.
    
    # Configuration B: Points at (i, j) for i in 0..3, j in 0..2, but shifted
    # to create more 3-point lines.
    
    # Let's try a local search / hill climbing approach
    import random
    random.seed(42)
    
    # Start from the best known
    current = best[:]
    current_score = best_score
    
    # Local search: perturb one point at a time
    max_iter = 5000
    for it in range(max_iter):
        if time.time() > deadline - 10.0:
            break
        
        # Try moving a random point to a nearby integer location
        i = random.randint(0, 11)
        x, y = current[i]
        # Try small perturbations
        improved = False
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                # Check if new point is distinct from others
                new_points = current[:]
                new_points[i] = [nx, ny]
                # Check distinctness
                pts_set = set()
                valid = True
                for p in new_points:
                    if tuple(p) in pts_set:
                        valid = False
                        break
                    pts_set.add(tuple(p))
                if not valid:
                    continue
                if not _not_all_collinear(new_points):
                    continue
                s = _evaluate(new_points)
                if s < current_score:
                    current = new_points
                    current_score = s
                    if s < best_score:
                        best = current[:]
                        best_score = s
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try a larger jump occasionally
            if it % 100 == 0:
                i = random.randint(0, 11)
                x, y = current[i]
                for _ in range(10):
                    nx = x + random.randint(-10, 10)
                    ny = y + random.randint(-10, 10)
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    new_points = current[:]
                    new_points[i] = [nx, ny]
                    pts_set = set()
                    valid = True
                    for p in new_points:
                        if tuple(p) in pts_set:
                            valid = False
                            break
                        pts_set.add(tuple(p))
                    if not valid:
                        continue
                    if not _not_all_collinear(new_points):
                        continue
                    s = _evaluate(new_points)
                    if s < current_score:
                        current = new_points
                        current_score = s
                        if s < best_score:
                            best = current[:]
                            best_score = s
                        break
    
    # Try a few more structured configurations
    structured_configs = [
        # 2x6 grid
        [[x, y] for x in range(6) for y in range(2)],
        # 6x2 grid
        [[x, y] for x in range(2) for y in range(6)],
        # 3x4 with offset
        [[x, y] for x in [0, 1, 3, 4] for y in [0, 1, 2]],
        # 4x3 with offset
        [[x, y] for x in [0, 1, 2] for y in [0, 1, 3, 4]],
        # Symmetric cross
        [[0, i] for i in range(-3, 4)] + [[1, 0], [-1, 0]],
        # 12 points on two lines: 6 on each
        [[i, 0] for i in range(6)] + [[i, 1] for i in range(6)],
        # 12 points: 4 on y=0, 4 on y=1, 4 on y=2 with aligned x
        [[i, 0] for i in range(4)] + [[i, 1] for i in range(4)] + [[i, 2] for i in range(4)],
    ]
    
    for config in structured_configs:
        if len(config) != 12:
            continue
        # Check distinctness
        pts_set = set()
        valid = True
        for p in config:
            if tuple(p) in pts_set:
                valid = False
                break
            pts_set.add(tuple(p))
        if not valid:
            continue
        if not _not_all_collinear(config):
            continue
        s = _evaluate(config)
        if s < best_score:
            best = config[:]
            best_score = s
    
    return best
# EVOLVE-BLOCK-END
