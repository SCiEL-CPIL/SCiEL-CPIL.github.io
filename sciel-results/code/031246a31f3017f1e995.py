"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
from fractions import Fraction

def _count_ordinary(pts):
    """Count lines containing exactly 2 of the given points."""
    n = len(pts)
    if n < 2:
        return 0
    line_map = {}
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            # Line coefficients a*x + b*y + c = 0
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            # Normalize
            g = gcd(gcd(abs(a), abs(b)), abs(c))
            if g > 0:
                a //= g
                b //= g
                c //= g
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    return sum(1 for v in line_map.values() if len(v) == 2)

def _is_valid(pts):
    """Check 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(pts) != 12:
        return False
    seen = set()
    for p in pts:
        if not isinstance(p[0], int) or not isinstance(p[1], int):
            return False
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
        key = (p[0], p[1])
        if key in seen:
            return False
        seen.add(key)
    if len(pts) < 3:
        return True
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    x2, y2 = pts[2]
    # Check if first 3 are collinear. If they are, check all. If not, valid (not all collinear).
    # Actually, just check if all are collinear.
    # If (x1-x0)*(y2-y0) == (y1-y0)*(x2-x0), first 3 are collinear.
    # If they are, check all points against line 0-1.
    # If they are not, they are not all collinear.
    if (x1 - x0) * (y2 - y0) != (y1 - y0) * (x2 - x0):
        return True
    for i in range(3, len(pts)):
        xi, yi = pts[i]
        if (xi - x0) * (y1 - y0) != (yi - y0) * (x1 - x0):
            return True
    return False

def construct():
    deadline = time.time() + 145
    
    # Known good configuration: 3x4 grid
    # This has 10 ordinary lines.
    # We want to find one with 6.
    
    # Strategy:
    # 1. Try to construct a configuration with 6 ordinary lines using combinatorial design.
    # 2. If that fails, use a faster local search starting from the 3x4 grid.
    
    # --- Part 1: Combinatorial Search for 6 Ordinary Lines ---
    # A configuration with 6 ordinary lines for n=12.
    # Total pairs = 66.
    # Let k be the number of lines with >= 3 points.
    # Sum_{L} C(|L|, 2) = 66.
    # Number of ordinary lines = 6.
    # This implies a specific structure.
    # The Pappus configuration has 9 points.
    # A 12-point configuration with 6 ordinary lines is rare.
    # Actually, the minimum number of ordinary lines for n=12 is 6.
    # This is achieved by the "Pappus-like" or "Hesse-like" configurations?
    # The Hesse configuration has 9 points.
    # For n=12, one known configuration is the "12-point configuration" derived from the affine plane of order 4? No, that's 16 points.
    # Let's try to search for a specific structure:
    # 3 lines of 4 points?
    # If we have 3 lines of 4 points, they cover 12 points if they are disjoint? No, lines intersect.
    # If 3 lines of 4 points intersect at a common point?
    # Let's try a "star" structure or "triangle" structure.
    
    # Instead of guessing, let's use a randomized search with a very fast evaluation
    # and a smarter move set: move a point to a position that creates a new collinearity.
    
    # Initial best: 3x4 grid
    best_pts = [[x, y] for x in range(4) for y in range(3)]
    best_score = _count_ordinary(best_pts)
    
    # --- Part 2: Accelerated Local Search ---
    # Precompute line keys for the current configuration to speed up updates?
    # Or just use a faster move set.
    
    # Move set: 
    # 1. Move a point to align with two other points.
    # 2. Move a point to align with one other point on a line that already has 2 points (making it 3).
    
    import random
    random.seed(42)
    
    for restart in range(10):
        if time.time() > deadline - 10:
            break
            
        if restart == 0:
            current_pts = [list(p) for p in best_pts]
        else:
            # Random perturbation of best
            current_pts = [list(p) for p in best_pts]
            for _ in range(5):
                idx = random.randint(0, 11)
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                current_pts[idx][0] += dx
                current_pts[idx][1] += dy
            if not _is_valid(current_pts):
                continue
                
        current_score = _count_ordinary(current_pts)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for idx in range(12):
                if time.time() > deadline - 5:
                    break
                orig_x, orig_y = current_pts[idx]
                
                # Candidate moves:
                # 1. Align with two other points (i, j)
                for i, j in combinations(range(12), 2):
                    if i == idx or j == idx:
                        continue
                    x1, y1 = current_pts[i]
                    x2, y2 = current_pts[j]
                    # Check if i, j, idx are already collinear
                    if (x2 - x1) * (orig_y - y1) == (y2 - y1) * (orig_x - x1):
                        continue
                    # We want to move idx to a point on line ij.
                    # Pick a point on line ij that is close to orig.
                    # Parametric: P = P1 + t * (P2 - P1)
                    # We want integer coordinates.
                    # Try small integer steps along the line direction.
                    dx_line = x2 - x1
                    dy_line = y2 - y1
                    g = gcd(abs(dx_line), abs(dy_line))
                    if g == 0:
                        continue
                    dx_line //= g
                    dy_line //= g
                    
                    # Find t such that P is close to orig.
                    # Project orig onto line ij.
                    # Vector v = P2 - P1
                    # Vector w = orig - P1
                    # t = dot(w, v) / dot(v, v)
                    vx, vy = dx_line * g, dy_line * g # Use unnormalized for projection? No, use normalized.
                    # Actually, let's just try a few integer points on the line near orig.
                    # The line passes through (x1, y1) with direction (dx_line, dy_line).
                    # Points are (x1 + k*dx_line, y1 + k*dy_line).
                    # We want to find k such that (x1 + k*dx_line, y1 + k*dy_line) is close to (orig_x, orig_y).
                    
                    # Approximate k:
                    # k ~ (orig_x - x1) / dx_line
                    if dx_line != 0:
                        k_est = (orig_x - x1) // dx_line
                    elif dy_line != 0:
                        k_est = (orig_y - y1) // dy_line
                    else:
                        continue
                        
                    for k in range(k_est - 2, k_est + 3):
                        new_x = x1 + k * dx_line
                        new_y = y1 + k * dy_line
                        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                            continue
                        # Check distinctness
                        if any(new_x == p[0] and new_y == p[1] for p in current_pts if p[0] == new_x and p[1] == new_y and current_pts.index(p) != idx):
                             # Simpler distinctness check
                             pass
                        
                        new_pts = [list(p) for p in current_pts]
                        new_pts[idx] = [new_x, new_y]
                        
                        # Quick distinctness check
                        valid = True
                        seen = set()
                        for p in new_pts:
                            key = (p[0], p[1])
                            if key in seen:
                                valid = False
                                break
                            seen.add(key)
                        if not valid:
                            continue
                            
                        new_score = _count_ordinary(new_pts)
                        if new_score < current_score:
                            current_pts = new_pts
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
                    
        if current_score < best_score:
            best_score = current_score
            best_pts = [list(p) for p in current_pts]
            if best_score <= 6:
                return [list(p) for p in best_pts]
                
    return [list(p) for p in best_pts]
# EVOLVE-BLOCK-END
