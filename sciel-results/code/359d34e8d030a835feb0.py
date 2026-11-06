import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Return canonical line key for two points."""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    if a == 0 and b == 0:
        return None
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g == 0:
        g = 1
    a, b, c = a // g, b // g, c // g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _count_ordinary(points):
    """Count ordinary lines."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for v in line_counts.values() if v == 1)

def _is_valid(points):
    """Check: 12 distinct points, not all collinear."""
    if len(points) != 12:
        return False
    if len(set(tuple(p) for p in points)) != 12:
        return False
    # Check not all collinear
    (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
    if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
        return True
    for i in range(3, 12):
        (x4, y4) = points[i]
        if (x2 - x1) * (y4 - y1) != (y2 - y1) * (x4 - x1):
            return True
    return False

def _intersections_of_lines(lines):
    """
    Given a list of lines (a, b, c), return a list of unique integer intersection points.
    """
    pts = set()
    n = len(lines)
    for i in range(n):
        a1, b1, c1 = lines[i]
        for j in range(i + 1, n):
            a2, b2, c2 = lines[j]
            det = a1 * b2 - a2 * b1
            if det == 0:
                continue # Parallel
            x_num = b1 * c2 - b2 * c1
            y_num = a2 * c1 - a1 * c2
            if x_num % det != 0 or y_num % det != 0:
                continue
            x = x_num // det
            y = y_num // det
            if abs(x) <= 10**9 and abs(y) <= 10**9:
                pts.add((x, y))
    return list(pts)

def _random_line(max_coeff=5):
    """Generate a random line with small integer coefficients."""
    while True:
        a = random.randint(-max_coeff, max_coeff)
        b = random.randint(-max_coeff, max_coeff)
        c = random.randint(-max_coeff, max_coeff)
        if a == 0 and b == 0:
            continue
        g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
        if g == 0:
            g = 1
        a, b, c = a // g, b // g, c // g
        # Canonicalize
        if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
            a, b, c = -a, -b, -c
        return (a, b, c)

def _line_based_search(deadline):
    """
    Structural change: Search for a set of base lines such that their intersections
    yield 12 points with minimal ordinary lines.
    """
    best_points = None
    best_score = float('inf')
    
    # Start with a random set of 6-8 lines
    num_lines = 7
    current_lines = [_random_line() for _ in range(num_lines)]
    current_lines = list(set(current_lines)) # Remove duplicates
    
    # If too few unique lines, add more
    while len(current_lines) < 6:
        new_l = _random_line()
        if new_l not in current_lines:
            current_lines.append(new_l)
            
    # Calculate initial points
    pts = _intersections_of_lines(current_lines)
    if len(pts) < 12:
        return None, float('inf')
    
    # Select 12 points. We want to select points that maximize collinearity.
    # Greedy selection: pick points that lie on many lines?
    # Or just pick 12 random points from the intersections and optimize?
    # Let's pick the 12 points that are "central" or just random.
    # To be safe, we will evaluate the best subset of 12.
    # But 12 out of N is expensive if N is large.
    # Let's just take the first 12 or a random 12.
    
    # Better: Use the fact that we want to minimize ordinary lines.
    # This is equivalent to maximizing the number of "non-ordinary" lines (lines with >=3 points).
    
    # We will use Simulated Annealing on the set of lines.
    temperature = 10.0
    cooling = 0.995
    min_temp = 0.1
    
    # Initial candidate points
    if len(pts) > 12:
        # Randomly sample 12 points to start
        candidate_points = random.sample(pts, 12)
    else:
        candidate_points = pts
        
    if not _is_valid(candidate_points):
        return None, float('inf')
        
    current_score = _count_ordinary(candidate_points)
    best_score = current_score
    best_points = [list(p) for p in candidate_points]
    
    while time.time() < deadline and temperature > min_temp:
        # Perturb one line
        idx = random.randint(0, len(current_lines) - 1)
        old_line = current_lines[idx]
        
        # Generate a new line close to the old one?
        # Or just a random new line?
        # Random new line is simpler.
        new_line = _random_line(max_coeff=10)
        
        if new_line == old_line:
            continue
            
        current_lines[idx] = new_line
        
        # Recalculate intersections
        new_pts = _intersections_of_lines(current_lines)
        
        if len(new_pts) < 12:
            # Revert
            current_lines[idx] = old_line
            temperature *= cooling
            continue
            
        # We need to select 12 points from new_pts.
        # To maintain continuity, we try to keep as many old points as possible.
        # But if the structure changed significantly, we might need new points.
        # Simple strategy: If the set of points changed, pick 12 random ones.
        # This might be noisy.
        
        # Let's try to pick the 12 points from new_pts that are "closest" to the old candidate_points?
        # Or just random.
        
        # For robustness, let's just pick 12 random points from the new intersections.
        # This allows the "population" of points to evolve.
        new_candidate_points = random.sample(new_pts, 12)
        
        if not _is_valid(new_candidate_points):
            current_lines[idx] = old_line
            temperature *= cooling
            continue
            
        new_score = _count_ordinary(new_candidate_points)
        
        # Acceptance probability
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_score = new_score
            candidate_points = new_candidate_points
            if new_score < best_score:
                best_score = new_score
                best_points = [list(p) for p in candidate_points]
        else:
            # Revert
            current_lines[idx] = old_line
            
        temperature *= cooling
        
    return best_points, best_score

def _try_structured_configs():
    """Try several structured configurations and return the best."""
    candidates = []
    
    # 1. 3x4 grid
    candidates.append([[i, j] for i in range(4) for j in range(3)])
    
    # 2. 4x3 grid (transposed)
    candidates.append([[i, j] for i in range(3) for j in range(4)])
    
    # 3. Hesse-like: 3x3 grid plus 3 extra points on existing lines
    h9 = [[i, j] for i in range(3) for j in range(3)]
    candidates.append(h9 + [[3, 0], [3, 1], [3, 2]])
    candidates.append(h9 + [[0, 3], [1, 3], [2, 3]])
    candidates.append(h9 + [[3, 0], [2, 3], [0, 3]])
    
    # 4. Try a configuration with 3 lines of 4 points each (3x4 grid is this)
    
    # 5. Try: 3x3 grid plus (3,3), (4,0), (0,4)
    candidates.append(h9 + [[3, 3], [4, 0], [0, 4]])
    
    # 6. Try: 3x3 grid plus (3,0), (0,3), (3,3)
    candidates.append(h9 + [[3, 0], [0, 3], [3, 3]])
    
    best_pts = None
    best_score = float('inf')
    
    for pts in candidates:
        if not _is_valid(pts):
            continue
        score = _count_ordinary(pts)
        if score < best_score:
            best_score = score
            best_pts = [list(p) for p in pts]
    
    return best_pts, best_score

def _local_search_fast(points, deadline):
    """Fast local search with incremental line counting."""
    current = [tuple(p) for p in points]
    n = len(current)
    
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(current[i], current[j])
            if key is not None:
                line_counts[key] = line_counts.get(key, 0) + 1
    ordinary_count = sum(1 for v in line_counts.values() if v == 1)
    
    best_score = ordinary_count
    best_points = [list(p) for p in current]
    
    coords = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = current[i]
            x2, y2 = current[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            for t in range(-3, 4):
                coords.add((x1 + t * dx, y1 + t * dy))
    
    for p in current:
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                coords.add((p[0] + dx, p[1] + dy))
    
    candidate_list = list(coords)
    random.shuffle(candidate_list)
    
    no_improve = 0
    while time.time() < deadline and no_improve < 100:
        improved = False
        random.shuffle(candidate_list)
        
        for i in range(n):
            if time.time() >= deadline:
                break
            old_pt = current[i]
            for cand in candidate_list:
                if cand == old_pt:
                    continue
                if any(cand == current[j] for j in range(n) if j != i):
                    continue
                
                delta = 0
                
                for j in range(n):
                    if j == i:
                        continue
                    key = _line_key(old_pt, current[j])
                    if key is not None:
                        c = line_counts[key]
                        if c == 2:
                            delta -= 1
                        elif c == 3:
                            delta += 1
                        line_counts[key] = c - 1
                        if line_counts[key] == 0:
                            del line_counts[key]
                
                for j in range(n):
                    if j == i:
                        continue
                    key = _line_key(cand, current[j])
                    if key is not None:
                        c = line_counts.get(key, 0)
                        if c == 1:
                            delta += 1
                        elif c == 2:
                            delta -= 1
                        line_counts[key] = c + 1
                
                new_ordinary = ordinary_count + delta
                
                if new_ordinary < ordinary_count:
                    current[i] = cand
                    ordinary_count = new_ordinary
                    improved = True
                    no_improve = 0
                    if new_ordinary < best_score:
                        best_score = new_ordinary
                        best_points = [list(p) for p in current]
                    break
                else:
                    for j in range(n):
                        if j == i:
                            continue
                        key = _line_key(cand, current[j])
                        if key is not None:
                            c = line_counts[key]
                            line_counts[key] = c - 1
                            if line_counts[key] == 0:
                                del line_counts[key]
                    
                    for j in range(n):
                        if j == i:
                            continue
                        key = _line_key(old_pt, current[j])
                        if key is not None:
                            line_counts[key] = line_counts.get(key, 0) + 1
                    
            else:
                no_improve += 1
                continue
            break
        else:
            no_improve += 1
            continue
        
        if not improved:
            idx = random.randint(0, n - 1)
            cand = random.choice(candidate_list)
            if cand != current[idx] and not any(cand == current[j] for j in range(n) if j != idx):
                old_pt = current[idx]
                for j in range(n):
                    if j == idx:
                        continue
                    key = _line_key(old_pt, current[j])
                    if key is not None:
                        c = line_counts[key]
                        if c == 2:
                            ordinary_count -= 1
                        elif c == 3:
                            ordinary_count += 1
                        line_counts[key] = c - 1
                        if line_counts[key] == 0:
                            del line_counts[key]
                
                current[idx] = cand
                for j in range(n):
                    if j == idx:
                        continue
                    key = _line_key(cand, current[j])
                    if key is not None:
                        c = line_counts.get(key, 0)
                        if c == 1:
                            ordinary_count += 1
                        elif c == 2:
                            ordinary_count -= 1
                        line_counts[key] = c + 1
                
                no_improve = 0
    
    return best_points, best_score

def construct():
    deadline = time.time() + 135  # Safety margin
    random.seed(42)
    
    # Step 1: Try structured configurations
    best_points, best_score = _try_structured_configs()
    
    # Step 2: Local search from best structured config
    if time.time() < deadline - 30:
        ls_points, ls_score = _local_search_fast(best_points, deadline - 30)
        if ls_score < best_score:
            best_score = ls_score
            best_points = ls_points
            
    # Step 3: Line-based search (Structural Change)
    # This uses a different representation (lines defining points) and SA.
    if time.time() < deadline - 10:
        lb_points, lb_score = _line_based_search(deadline - 10)
        if lb_points and lb_score < best_score:
            best_score = lb_score
            best_points = lb_points

    # Final validation
    if not _is_valid(best_points):
        best_points = [[i, j] for i in range(4) for j in range(3)]
    
    return [[int(p[0]), int(p[1])] for p in best_points]

# EVOLVE-BLOCK-END
