# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd

def _normalize_line(A, B, C):
    """Normalize line (A,B,C) so first nonzero is positive, gcd=1."""
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_inc(points, lines):
    """Count incidences. points: (n,2) int array, lines: (m,3) int array."""
    # For each line, count points on it
    # Vectorized: for each line (A,B,C), compute A*x + B*y + C for all points
    # lines: (m,3), points: (n,2)
    # result[m, n] = A_i * x_j + B_i * y_j + C_i
    A = lines[:, 0:1]  # (m, 1)
    B = lines[:, 1:2]  # (m, 1)
    C = lines[:, 2:3]  # (m, 1)
    x = points[0:1, 0]  # (1, n)
    y = points[0:1, 1]  # (1, n)
    vals = A * x + B * y + C  # (m, n)
    return int(np.sum(vals == 0))

def _line_through(p1, p2):
    """Line through two points, normalized."""
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    return _normalize_line(A, B, C)

def _line_contains(line, pt):
    """Check if point is on line."""
    A, B, C = line
    x, y = pt
    return A * x + B * y + C == 0

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, use 600 for safety margin
    
    n_pts = 54
    n_lines = 27
    
    # Initial construction: rich lines + pair lines
    # Strategy: pick a few "rich" lines that pass through many points,
    # then fill remaining lines with lines through pairs of points.
    
    def _construct_and_score(rich_line_configs, max_time=550):
        """
        Given rich line configs, build points and lines, score.
        rich_line_configs: list of (slope_num, slope_den, intercept_num, intercept_den) or similar
        """
        pass
    
    # Better approach: direct construction
    # Place points on a few lines to create rich incidences.
    
    # Let's try: 3 rich lines, each with ~18 points, plus 24 pair lines
    # But points on multiple rich lines give bonus incidences.
    
    # Actually, let me think differently. 
    # If I put all 54 points on 2 lines: 2*54 = 108 from those 2 lines.
    # Then 25 more lines. Each additional line can pass through at most 2 points 
    # (if no 3 points are collinear outside the 2 main lines).
    # But if I arrange points so that many pairs define the same line, I get more.
    
    # Best strategy: use a grid-like structure but optimized.
    
    # Let me try: put points on 3 lines with intersections.
    # Line 1: y=0, points (1,0)...(18,0) -> 18 points
    # Line 2: y=1, points (1,1)...(18,1) -> 18 points  
    # Line 3: x=1, points (1,0),(1,1) already counted, add (1,2)...(1,17) -> 16 new points
    # Total: 18+18+16 = 52. Need 2 more.
    # Add (2,2) and (3,3) or something.
    
    # This is getting complex. Let me use a search approach.
    
    # Simpler: start with the grid construction and improve via local search.
    
    def _initial_grid():
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        lines = np.array(rows, dtype=np.int64)
        return points, lines
    
    def _try_rich_lines():
        """Try various rich-line constructions."""
        best_score = 0
        best_points = None
        best_lines = None
        
        # Try: put points on a few parallel lines
        # e.g., 3 lines y=0, y=1, y=2, with 18 points each
        # Then lines are: the 3 horizontal lines (54 incidences) 
        # plus 24 lines through pairs of points.
        
        for n_rich in [2, 3, 4, 5]:
            pts_per_rich = n_pts // n_rich
            remaining = n_pts - pts_per_rich * n_rich
            
            # Place points on n_rich horizontal lines
            points_list = []
            for i in range(n_rich):
                for j in range(pts_per_rich):
                    points_list.append((j + 1, i + 1))
            # Add remaining points
            for j in range(remaining):
                points_list.append((j + 1, n_rich + 1))
            
            points = np.array(points_list, dtype=np.int64)
            
            # Rich lines: the horizontal lines
            lines_list = []
            for i in range(n_rich):
                lines_list.append((0, 1, -(i + 1)))  # y = i+1
            if remaining > 0:
                lines_list.append((0, 1, -(n_rich + 1)))  # y = n_rich+1
            
            # Now add lines through pairs of points to reach 27 lines
            # Choose pairs that maximize additional incidences
            # A line through 2 points gives 2 incidences (if no other point on it)
            # A line through 3 points gives 3 incidences, etc.
            
            # For simplicity, add lines through pairs, avoiding duplicates
            existing_lines = set(lines_list)
            p = len(points_list)
            # Try to find lines through 3+ points first
            # Then fill with 2-point lines
            
            # This is complex; for now, just add random pairs
            import random
            random.seed(42)
            attempts = 0
            max_attempts = 10000
            while len(lines_list) < n_lines and attempts < max_attempts:
                i, j = random.sample(range(p), 2)
                line = _line_through(tuple(points_list[i]), tuple(points_list[j]))
                if line and line not in existing_lines:
                    lines_list.append(line)
                    existing_lines.add(line)
                attempts += 1
            
            if len(lines_list) < n_lines:
                continue
            
            lines = np.array(lines_list[:n_lines], dtype=np.int64)
            
            # Verify no duplicate lines
            norm_lines = []
            seen = set()
            valid = True
            for l in lines_list[:n_lines]:
                nl = _normalize_line(*l)
                if nl in seen:
                    valid = False
                    break
                seen.add(nl)
                norm_lines.append(nl)
            
            if not valid or len(norm_lines) != n_lines:
                continue
            
            lines = np.array(norm_lines, dtype=np.int64)
            
            score = _count_inc(points, lines)
            if score > best_score:
                best_score = score
                best_points = points
                best_lines = lines
        
        return best_score, best_points, best_lines
    
    # First, get the baseline
    base_points, base_lines = _initial_grid()
    base_score = _count_inc(base_points, base_lines)
    
    # Try rich line constructions
    rich_score, rich_points, rich_lines = _try_rich_lines()
    
    if rich_points is not None and rich_score > base_score:
        best_points = rich_points
        best_lines = rich_lines
        best_score = rich_score
    else:
        best_points = base_points
        best_lines = base_lines
        best_score = base_score
    
    # Now do local search / simulated annealing to improve
    # Move a point or change a line to improve incidence count
    
    if time.time() > deadline - 50:
        return best_points, best_lines
    
    # Simulated annealing on points
    import random
    random.seed(123)
    
    current_points = best_points.copy()
    current_lines = best_lines.copy()
    current_score = best_score
    
    temperature = 10.0
    cooling = 0.995
    min_temp = 0.1
    
    step = 0
    while temperature > min_temp and time.time() < deadline - 10:
        step += 1
        
        # Random move: move a random point to a nearby location
        idx = random.randint(0, n_pts - 1)
        new_x = current_points[idx, 0] + random.randint(-3, 3)
        new_y = current_points[idx, 1] + random.randint(-3, 3)
        
        old_pt = current_points[idx].copy()
        current_points[idx] = [new_x, new_y]
        
        # Check for duplicate points
        valid = True
        for i in range(n_pts):
            if i == idx:
                continue
            if current_points[i, 0] == new_x and current_points[i, 1] == new_y:
                valid = False
                break
        
        if not valid:
            current_points[idx] = old_pt
            temperature *= cooling
            continue
        
        new_score = _count_inc(current_points, current_lines)
        
        if new_score >= current_score or random.random() < np.exp((new_score - current_score) / temperature):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points.copy()
                best_lines = current_lines.copy()
        else:
            current_points[idx] = old_pt
        
        temperature *= cooling
    
    # Try moving lines too
    if time.time() > deadline - 20:
        return best_points, best_lines
    
    current_points = best_points.copy()
    current_lines = best_lines.copy()
    current_score = best_score
    
    temperature = 5.0
    cooling = 0.995
    min_temp = 0.1
    
    step = 0
    while temperature > min_temp and time.time() < deadline - 10:
        step += 1
        
        # Random move: change a line
        idx = random.randint(0, n_lines - 1)
        
        # Pick a random line through 2 random points
        i, j = random.sample(range(n_pts), 2)
        new_line = _line_through(tuple(current_points[i]), tuple(current_points[j]))
        
        if new_line is None:
            temperature *= cooling
            continue
        
        # Check if this line is already in the set (excluding the one we're replacing)
        valid = True
        for k in range(n_lines):
            if k == idx:
                continue
            if _normalize_line(*current_lines[k]) == new_line:
                valid = False
                break
        
        if not valid:
            temperature *= cooling
            continue
        
        old_line = current_lines[idx].copy()
        current_lines[idx] = new_line
        
        new_score = _count_inc(current_points, current_lines)
        
        if new_score >= current_score or random.random() < np.exp((new_score - current_score) / temperature):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points.copy()
                best_lines = current_lines.copy()
        else:
            current_lines[idx] = old_line
        
        temperature *= cooling
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
