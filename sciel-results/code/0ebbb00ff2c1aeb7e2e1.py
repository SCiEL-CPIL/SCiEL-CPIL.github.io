"""Combined program: hexagonal lattice init + directed 3-point collinearity search + simulated annealing."""
import time
import random
import math
from math import gcd

_DEADLINE = 140.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _count_ordinary_lines(points):
    """Count lines containing exactly two of the given points using pair counting."""
    n = len(points)
    if n < 3:
        return 0
    
    # Count pairs per line
    line_pair_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                g = 1
            A //= g
            B //= g
            C //= g
            
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            line_pair_counts[key] = line_pair_counts.get(key, 0) + 1
    
    # A line with k points has C(k,2) pairs. Ordinary lines have exactly 1 pair.
    ordinary_count = 0
    for count in line_pair_counts.values():
        if count == 1:
            ordinary_count += 1
    return ordinary_count

def _not_all_collinear(points):
    n = len(points)
    if n < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for i in range(2, n):
        xi, yi = points[i]
        if (x1 - x0) * (yi - y0) != (y1 - y0) * (xi - x0):
            return True
    return False

def _evaluate(points):
    """Return the number of ordinary lines, or None if invalid."""
    if len(points) != 12:
        return None
    seen = set()
    for p in points:
        t = (p[0], p[1])
        if t in seen:
            return None
        seen.add(t)
    if not _not_all_collinear(points):
        return None
    return _count_ordinary_lines(points)

def _make_hexagonal_lattice():
    """Generate 12 integer points on a hexagonal lattice with high symmetry.
    
    Use a hexagonal arrangement: center + 6 inner ring + 5 outer ring points.
    Scale to keep integer coordinates.
    """
    # Hexagonal lattice: use axial coordinates converted to Cartesian
    # For integer coords, use a scaled hex grid
    # Pattern: 3 rows of 4 points each, but with hex offset
    points = []
    # Row 0: 4 points
    for x in range(4):
        points.append([x * 2, 0])
    # Row 1: 4 points offset
    for x in range(4):
        points.append([x * 2 + 1, 3])
    # Row 2: 4 points
    for x in range(4):
        points.append([x * 2, 6])
    return points

def _make_alternative_hex():
    """Alternative hexagonal configuration with different symmetry."""
    points = []
    # Use a more compact hex-like structure
    # 4-4-4 grid but with slight perturbation to create diagonal collinearities
    base = [[x, y] for x in range(4) for y in range(3)]
    # Shift middle column to create diagonal lines
    for i, p in enumerate(base):
        if p[1] == 1:
            p[0] += 1
    return base

def _find_collinearity_moves(points, num_candidates=50):
    """Find candidate moves that create new 3-point collinearities.
    
    For each point i, and each pair (j, k) of other points, compute the line through j and k.
    Check if moving point i to a position on that line (at integer coordinates) would
    create a 3-point line.
    """
    candidates = []
    n = len(points)
    
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(j + 1, n):
                if k == i:
                    continue
                # Line through j and k
                xj, yj = points[j]
                xk, yk = points[k]
                dx = xk - xj
                dy = yk - yj
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx //= g
                dy //= g
                
                # Parametric: (xj + t*dx, yj + t*dy) for integer t
                # Try small t values to find integer positions
                for t in range(-3, 4):
                    nx = xj + t * dx
                    ny = yj + t * dy
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    # Check if this position is not already occupied
                    pos = (nx, ny)
                    if pos == (points[i][0], points[i][1]):
                        continue
                    occupied = False
                    for m in range(n):
                        if m != i and (points[m][0], points[m][1]) == pos:
                            occupied = True
                            break
                    if not occupied:
                        candidates.append((i, nx, ny))
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = (c[0], c[1], c[2])
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    # Return a random sample
    if len(unique_candidates) > num_candidates:
        return random.sample(unique_candidates, num_candidates)
    return unique_candidates

def _directed_local_search(points, max_iter=20000):
    """Directed local search: try moves that create 3-point collinearities."""
    current = [list(p) for p in points]
    current_score = _evaluate(current)
    if current_score is None:
        return current, current_score
    
    best = [list(p) for p in current]
    best_score = current_score
    
    for it in range(max_iter):
        if _elapsed() > _DEADLINE:
            break
        
        # Get candidate moves
        candidates = _find_collinearity_moves(current, num_candidates=20)
        
        if not candidates:
            # Fallback to random small move
            idx = random.randint(0, 11)
            x, y = current[idx]
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
            new_x = x + dx
            new_y = y + dy
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
            if (new_x, new_y) in [(p[0], p[1]) for p in current if p is not current[idx]]:
                continue
            old_point = current[idx][:]
            current[idx] = [new_x, new_y]
            new_score = _evaluate(current)
            if new_score is not None and new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best = [list(p) for p in current]
            else:
                current[idx] = old_point
            continue
        
        # Try a random candidate
        idx, new_x, new_y = random.choice(candidates)
        old_point = current[idx][:]
        current[idx] = [new_x, new_y]
        new_score = _evaluate(current)
        
        if new_score is not None and new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = [list(p) for p in current]
        else:
            current[idx] = old_point
    
    return best, best_score

def _simulated_annealing(points, max_iter=30000):
    """Simulated annealing with mixed move strategies."""
    current = [list(p) for p in points]
    current_score = _evaluate(current)
    if current_score is None:
        return current, current_score
    
    best = [list(p) for p in current]
    best_score = current_score
    
    T = 2.0
    cooling_rate = 0.9995
    
    for it in range(max_iter):
        if _elapsed() > _DEADLINE:
            break
        
        # Choose move strategy
        if random.random() < 0.5:
            # Directed move: try to create 3-point collinearity
            candidates = _find_collinearity_moves(current, num_candidates=10)
            if candidates:
                idx, new_x, new_y = random.choice(candidates)
            else:
                idx = random.randint(0, 11)
                x, y = current[idx]
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                new_x = x + dx
                new_y = y + dy
        else:
            # Random move
            idx = random.randint(0, 11)
            x, y = current[idx]
            if random.random() < 0.7:
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
            else:
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
            new_x = x + dx
            new_y = y + dy
        
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
        if (new_x, new_y) in [(p[0], p[1]) for p in current if p is not current[idx]]:
            continue
        
        old_point = current[idx][:]
        current[idx] = [new_x, new_y]
        new_score = _evaluate(current)
        
        if new_score is None:
            current[idx] = old_point
            continue
        
        delta = new_score - current_score
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < math.exp(-delta / T)):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best = [list(p) for p in current]
        else:
            current[idx] = old_point
        
        T *= cooling_rate
        if T < 0.05:
            T = 0.05
    
    return best, best_score

def _try_multiple_starts():
    """Try multiple starting configurations and search methods."""
    best_points = None
    best_score = float('inf')
    
    random.seed(42)
    
    # Initial configurations
    configs = []
    configs.append(_make_hexagonal_lattice())
    configs.append(_make_alternative_hex())
    configs.append([[x, y] for x in range(4) for y in range(3)])
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # Add random configs
    for _ in range(3):
        pts = set()
        while len(pts) < 12:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            pts.add((x, y))
        configs.append([list(p) for p in pts])
    
    for init_pts in configs:
        if _elapsed() > _DEADLINE:
            break
        
        # Validate
        if len(set(tuple(p) for p in init_pts)) != 12:
            continue
        if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in init_pts):
            continue
        
        # Try directed local search
        pts, score = _directed_local_search(init_pts, max_iter=10000)
        if score is not None and score < best_score:
            best_score = score
            best_points = pts
        
        if _elapsed() > _DEADLINE:
            break
        
        # Try simulated annealing from the result
        pts, score = _simulated_annealing(pts, max_iter=15000)
        if score is not None and score < best_score:
            best_score = score
            best_points = pts
    
    return best_points, best_score

def construct():
    global _start_time
    _start_time = time.time()
    
    best_points, best_score = _try_multiple_starts()
    
    if best_points is None:
        # Fallback: return a valid 3x4 grid
        return [[x, y] for x in range(4) for y in range(3)]
    
    return best_points
