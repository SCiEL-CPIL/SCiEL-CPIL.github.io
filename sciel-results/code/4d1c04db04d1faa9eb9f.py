"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
from itertools import combinations
import random
import math

def gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def normalize_line(A, B, C):
    """Normalize line Ax + By + C = 0 to canonical form."""
    g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A != 0 or B != 0 or C != 0) else 1
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def count_ordinary_lines(points):
    """Count the number of lines that contain exactly two of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    line_to_points = {}
    
    for i, j in combinations(range(n), 2):
        x1, y1 = points[i]
        x2, y2 = points[j]
        
        A = y2 - y1
        B = -(x2 - x1)
        C = x2 * y1 - y2 * x1
        
        key = normalize_line(A, B, C)
        if key not in line_to_points:
            line_to_points[key] = set()
        line_to_points[key].add(i)
        line_to_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_to_points.items():
        if len(pts) == 2:
            ordinary += 1
    
    return ordinary

def get_line_info(points):
    """Return dict mapping line key to set of point indices on that line."""
    n = len(points)
    line_to_points = {}
    
    for i, j in combinations(range(n), 2):
        x1, y1 = points[i]
        x2, y2 = points[j]
        
        A = y2 - y1
        B = -(x2 - x1)
        C = x2 * y1 - y2 * x1
        
        key = normalize_line(A, B, C)
        if key not in line_to_points:
            line_to_points[key] = set()
        line_to_points[key].add(i)
        line_to_points[key].add(j)
    
    return line_to_points

def evaluate_move(points, idx, new_pos):
    """
    Evaluate the change in ordinary lines if we move point idx to new_pos.
    Returns the delta (new_ordinary - old_ordinary).
    """
    x, y = new_pos
    
    # Check if new position is already taken
    for i, p in enumerate(points):
        if i != idx and p[0] == x and p[1] == y:
            return float('inf')
    
    old_x, old_y = points[idx]
    if x == old_x and y == old_y:
        return 0
    
    # Compute old line info
    old_lines = get_line_info(points)
    old_ordinary = sum(1 for pts in old_lines.values() if len(pts) == 2)
    
    # Apply the move
    points[idx] = [x, y]
    new_lines = get_line_info(points)
    new_ordinary = sum(1 for pts in new_lines.values() if len(pts) == 2)
    
    # Revert
    points[idx] = [old_x, old_y]
    
    return new_ordinary - old_ordinary

def construct():
    """
    Construct 12 integer points minimizing the number of ordinary lines.
    
    Strategy: Use hexagonal lattice symmetry to create initial configuration with
    many 3-point collinearities, then apply targeted local search to maximize
    3-point lines while minimizing 2-point (ordinary) lines.
    
    The theoretical minimum for 12 points over reals is 6 ordinary lines.
    """
    
    random.seed(123)
    
    # Start with a hexagonal lattice configuration.
    # Use a scaled hexagon: center + 6 vertices + 6 midpoints of edges
    # For a regular hexagon with "radius" r (distance from center to vertex),
    # vertices are at angles 0, 60, 120, 180, 240, 300 degrees.
    
    # To get integer coordinates, use the hexagonal lattice basis:
    # v1 = (2, 0), v2 = (1, sqrt(3)) - but sqrt(3) is irrational.
    
    # Instead, use a rectangular approximation or a known good configuration.
    
    # Let's try a 4x3 grid with some strategic modifications.
    # Or better: use points that form multiple 3-point lines.
    
    # Configuration 1: Based on affine plane of order 3 (9 points) + 3 more
    # Affine plane AG(2,3) has 9 points with 12 lines of 3 points each.
    # We can add 3 more points to create additional 3-point lines.
    
    # AG(2,3) points (mod 3): all pairs (i,j) for i,j in {0,1,2}
    # Scale by a large factor to separate them, then add 3 more points.
    
    # Let's use a different approach: start with a symmetric configuration.
    
    # Hexagon-based: center (0,0), and 11 other points arranged symmetrically.
    # Use a 3x3 grid centered at origin, plus 3 more points.
    
    # Try: 3x3 grid = 9 points, add 3 points to make 12.
    # The 3x3 grid has many 3-point lines (rows, columns, diagonals).
    
    # Base: 3x3 grid centered at (0,0)
    base_points = [[x, y] for x in [-1, 0, 1] for y in [-1, 0, 1]]
    
    # Add 3 more points to create additional 3-point lines.
    # Try adding points that align with existing points.
    
    # For example, add (2, 0), (0, 2), (2, 2) or similar.
    
    # Let's try several initial configurations and pick the best.
    
    candidate_configs = [
        # 3x3 grid + 3 points
        [[x, y] for x in [-1, 0, 1] for y in [-1, 0, 1]] + [[2, 0], [0, 2], [2, 2]],
        # 4x3 grid
        [[x, y] for x in range(4) for y in range(3)],
        # 3x4 grid
        [[x, y] for x in range(3) for y in range(4)],
        # Hexagonal pattern: center + 6 at distance 2 + 5 more
        [[0, 0], [2, 0], [1, 2], [-1, 2], [-2, 0], [-1, -2], [1, -2],
         [3, 0], [0, 3], [-3, 0], [0, -3], [3, 3]],
        # Another symmetric config
        [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1],
         [1, -1], [-1, -1], [2, 0], [0, 2], [2, 2]],
    ]
    
    best_points = None
    best_ordinary = float('inf')
    
    for config in candidate_configs:
        if len(config) != 12:
            continue
        # Check distinctness
        if len(set(tuple(p) for p in config)) != 12:
            continue
        ordinary = count_ordinary_lines(config)
        if ordinary < best_ordinary:
            best_ordinary = ordinary
            best_points = [p[:] for p in config]
    
    if best_points is None:
        # Fallback to 4x3 grid
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_ordinary = count_ordinary_lines(best_points)
    
    points = [p[:] for p in best_points]
    
    # Local search: try moving each point to improve the count.
    # Focus on moves that create 3-point lines.
    
    max_iter = 20000
    stall_count = 0
    
    for iteration in range(max_iter):
        improved = False
        
        # Try each point, and for each, try a few candidate new positions
        for idx in range(12):
            old_x, old_y = points[idx]
            
            # Generate candidate positions: nearby points and strategic positions
            candidates = []
            
            # Nearby positions
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    nx, ny = old_x + dx, old_y + dy
                    if abs(nx) <= 100 and abs(ny) <= 100:
                        candidates.append((nx, ny))
            
            # Also try some strategic positions that might create 3-point lines
            # For each pair of other points, find the third point that would
            # complete a 3-point line.
            for i, j in combinations(range(12), 2):
                if i == idx or j == idx:
                    continue
                x1, y1 = points[i]
                x2, y2 = points[j]
                
                # Check if there's a third point that would make a 3-point line
                # with i and j, and if moving idx to that position would help.
                # The line through (x1,y1) and (x2,y2) has equation:
                # (y2-y1)x - (x2-x1)y + (x2*y1 - y2*x1) = 0
                A = y2 - y1
                B = -(x2 - x1)
                C = x2 * y1 - y2 * x1
                
                # If A and B are both even, we can find integer solutions.
                # Actually, we want to find a point (x,y) on this line such that
                # moving idx there would create a 3-point line.
                # The point must be distinct from all other points.
                
                # For a 3-point line, we need idx to be on the same line as i and j.
                # So we need to find an integer point (x,y) on this line that is
                # not already occupied.
                
                # Parametric form: if (x2-x1, y2-y1) = (dx, dy), then
                # points on the line are (x1 + t*dx, y1 + t*dy) for integer t.
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
                if g > 0:
                    dx //= g
                    dy //= g
                
                # Try t = -2, -1, 0, 1, 2, 3 to find nearby integer points
                for t in range(-3, 4):
                    nx = x1 + t * dx
                    ny = y1 + t * dy
                    if abs(nx) <= 100 and abs(ny) <= 100:
                        candidates.append((nx, ny))
            
            # Try each candidate
            for nx, ny in candidates:
                if nx == old_x and ny == old_y:
                    continue
                
                # Check if position is taken
                if any(p[0] == nx and p[1] == ny for i, p in enumerate(points) if i != idx):
                    continue
                
                # Evaluate the move
                delta = evaluate_move(points, idx, (nx, ny))
                
                if delta < 0:
                    points[idx] = [nx, ny]
                    best_ordinary += delta
                    improved = True
                    break  # Move to next point
            
            if improved:
                break
        
        if not improved:
            stall_count += 1
            if stall_count > 50:
                # Try a random restart with a different initial config
                for config in candidate_configs:
                    if len(config) != 12:
                        continue
                    if len(set(tuple(p) for p in config)) != 12:
                        continue
                    ordinary = count_ordinary_lines(config)
                    if ordinary < best_ordinary:
                        best_ordinary = ordinary
                        points = [p[:] for p in config]
                        improved = True
                        break
                if not improved:
                    # Try random perturbation
                    idx = random.randint(0, 11)
                    old_x, old_y = points[idx]
                    for _ in range(100):
                        nx = random.randint(-100, 100)
                        ny = random.randint(-100, 100)
                        if any(p[0] == nx and p[1] == ny for i, p in enumerate(points) if i != idx):
                            continue
                        delta = evaluate_move(points, idx, (nx, ny))
                        if delta < 0:
                            points[idx] = [nx, ny]
                            best_ordinary += delta
                            improved = True
                            break
                    if not improved:
                        break
            stall_count = 0
        else:
            stall_count = 0
        
        if best_ordinary <= 6:
            break
    
    # Final verification
    result = [p[:] for p in points]
    
    # Ensure 12 distinct points
    assert len(result) == 12
    assert len(set(tuple(p) for p in result)) == 12
    
    # Ensure not all collinear
    if len(result) >= 3:
        (x1, y1), (x2, y2), (x3, y3) = result[0], result[1], result[2]
        area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        assert area != 0, "Points are all collinear"
    
    # Ensure coordinates are within bounds
    for x, y in result:
        assert abs(x) <= 10**9 and abs(y) <= 10**9
    
    return result
# EVOLVE-BLOCK-END
