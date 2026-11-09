# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
import random

def _count_incidentences(points, lines):
    """Count (point, line) pairs where A*x + B*y + C == 0 exactly."""
    n = len(points)
    m = len(lines)
    count = 0
    for i in range(n):
        x, y = points[i]
        for j in range(m):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                count += 1
    return count

def _normalize_line(A, B, C):
    """Normalize (A, B, C) so gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_key(A, B, C):
    return _normalize_line(A, B, C)

def _is_valid(points, lines):
    """Check all validity constraints."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    pt_set = set()
    for i in range(54):
        x, y = points[i]
        if not np.isfinite(x) or not np.isfinite(y):
            return False
        if abs(x - round(x)) > 1e-6 or abs(y - round(y)) > 1e-6:
            return False
        xi, yi = int(round(x)), int(round(y))
        if (xi, yi) in pt_set:
            return False
        pt_set.add((xi, yi))
    
    line_set = set()
    for i in range(27):
        A, B, C = lines[i]
        if not np.isfinite(A) or not np.isfinite(B) or not np.isfinite(C):
            return False
        if abs(A - round(A)) > 1e-6 or abs(B - round(B)) > 1e-6 or abs(C - round(C)) > 1e-6:
            return False
        Ai, Bi, Ci = int(round(A)), int(round(B)), int(round(C))
        if Ai == 0 and Bi == 0:
            return False
        key = _line_key(Ai, Bi, Ci)
        if key in line_set:
            return False
        line_set.add(key)
    return True

def _generate_initial_grid(cols, rows):
    """Generate a grid of points."""
    points = []
    for x in range(1, cols + 1):
        for y in range(1, rows + 1):
            points.append((x, y))
    return np.array(points[:54], dtype=np.int64)

def _best_lines_for_points(points, num_lines=27):
    """Find the best num_lines lines for a given set of points."""
    n = len(points)
    candidate_lines = set()
    
    # Add lines through pairs of points
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _line_key(A, B, C)
            if key[0] == 0 and key[1] == 0:
                continue
            candidate_lines.add(key)
            
    # Also add some obvious lines: vertical, horizontal, diagonals near the points
    xs = points[:, 0]
    ys = points[:, 1]
    for x in np.unique(xs):
        candidate_lines.add(_line_key(1, 0, -x))
    for y in np.unique(ys):
        candidate_lines.add(_line_key(0, 1, -y))
    for d in range(-10, 11):
        candidate_lines.add(_line_key(1, -1, d))
        candidate_lines.add(_line_key(1, 1, d))

    # Score each candidate line by number of incident points
    scored_lines = []
    for key in candidate_lines:
        A, B, C = key
        cnt = 0
        for x, y in points:
            if A * x + B * y + C == 0:
                cnt += 1
        scored_lines.append((cnt, key))
        
    scored_lines.sort(key=lambda x: x[0], reverse=True)
    
    # Select top num_lines
    selected = []
    seen = set()
    for cnt, key in scored_lines:
        if key not in seen:
            selected.append(key)
            seen.add(key)
            if len(selected) == num_lines:
                break
                
    # If we don't have enough, pad with arbitrary valid lines
    if len(selected) < num_lines:
        # Generate more lines
        for i in range(100):
            # Random line
            A = random.randint(-10, 10)
            B = random.randint(-10, 10)
            if A == 0 and B == 0:
                continue
            C = random.randint(-100, 100)
            key = _line_key(A, B, C)
            if key not in seen:
                selected.append(key)
                seen.add(key)
                if len(selected) == num_lines:
                    break

    return np.array(selected[:num_lines], dtype=np.int64)

def _local_search(points, lines, deadline, safety_margin=30):
    """Perform local search by swapping lines."""
    current_count = _count_incidentences(points, lines)
    best_count = current_count
    best_lines = lines.copy()
    
    while time.time() < deadline - safety_margin:
        improved = False
        # Try replacing each line with a better one
        for j in range(len(lines)):
            # Generate candidate lines through pairs of points
            n = len(points)
            candidates = set()
            # Sample some pairs to save time
            for _ in range(20):
                i1 = random.randint(0, n - 1)
                i2 = random.randint(0, n - 1)
                if i1 == i2:
                    continue
                x1, y1 = points[i1]
                x2, y2 = points[i2]
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                if A == 0 and B == 0:
                    continue
                key = _line_key(A, B, C)
                if key[0] == 0 and key[1] == 0:
                    continue
                candidates.add(key)
                
            # Add the current line's neighbors (perturbations)
            A, B, C = lines[j]
            for dA in [-1, 0, 1]:
                for dB in [-1, 0, 1]:
                    for dC in [-1, 0, 1]:
                        if dA == 0 and dB == 0 and dC == 0:
                            continue
                        key = _line_key(A + dA, B + dB, C + dC)
                        if key[0] == 0 and key[1] == 0:
                            continue
                        candidates.add(key)
                        
            # Try each candidate
            for key in candidates:
                if key == tuple(lines[j]):
                    continue
                
                new_lines = lines.copy()
                new_lines[j] = np.array(key, dtype=np.int64)
                
                if not _is_valid(points, new_lines):
                    continue
                
                new_count = _count_incidentences(points, new_lines)
                if new_count > current_count:
                    current_count = new_count
                    lines = new_lines
                    improved = True
                    if current_count > best_count:
                        best_count = current_count
                        best_lines = lines.copy()
                    break # Restart search from new state
                    
        if not improved:
            # No improvement found in this pass, break or do a random jump?
            # To avoid getting stuck, we could accept a worse move (simulated annealing)
            # But for simplicity and stability, let's just break if no improvement
            # Or try a random restart of the line set
            if random.random() < 0.1:
                # Random restart of lines
                lines = _best_lines_for_points(points)
                current_count = _count_incidentences(points, lines)
                if current_count > best_count:
                    best_count = current_count
                    best_lines = lines.copy()
            else:
                break
                
    return best_lines, best_count

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines on the integer lattice.
    Uses a Multi-Start strategy with different grid configurations and local search.
    """
    deadline = time.time() + 600
    safety_margin = 30
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Define several candidate point configurations
    configs = []
    
    # 1. 6x9 Grid
    pts = _generate_initial_grid(6, 9)
    if len(pts) == 54:
        configs.append(pts)
        
    # 2. 9x6 Grid
    pts = _generate_initial_grid(9, 6)
    if len(pts) == 54:
        configs.append(pts)
        
    # 3. 3x18 Grid
    pts = _generate_initial_grid(3, 18)
    if len(pts) == 54:
        configs.append(pts)
        
    # 4. 18x3 Grid
    pts = _generate_initial_grid(18, 3)
    if len(pts) == 54:
        configs.append(pts)
        
    # 5. 2x27 Grid
    pts = _generate_initial_grid(2, 27)
    if len(pts) == 54:
        configs.append(pts)
        
    # 6. 27x2 Grid
    pts = _generate_initial_grid(27, 2)
    if len(pts) == 54:
        configs.append(pts)
        
    # 7. 1x54 Grid (Line of points)
    pts = _generate_initial_grid(1, 54)
    if len(pts) == 54:
        configs.append(pts)
        
    # 8. 54x1 Grid
    pts = _generate_initial_grid(54, 1)
    if len(pts) == 54:
        configs.append(pts)
        
    # 9. Random Cluster
    # Generate a random set of 54 points in a small box
    random.seed(42)
    pts = []
    for _ in range(54):
        x = random.randint(1, 10)
        y = random.randint(1, 10)
        if (x, y) not in pts:
            pts.append((x, y))
    if len(pts) == 54:
        configs.append(np.array(pts, dtype=np.int64))
        
    # 10. Another Random Cluster
    random.seed(123)
    pts = []
    for _ in range(54):
        x = random.randint(1, 15)
        y = random.randint(1, 15)
        if (x, y) not in pts:
            pts.append((x, y))
    if len(pts) == 54:
        configs.append(np.array(pts, dtype=np.int64))

    # Evaluate each configuration
    for pts in configs:
        if time.time() > deadline - safety_margin:
            break
            
        # Ensure validity of points
        if not _is_valid(pts, np.zeros((27, 3), dtype=np.int64)): # Dummy lines for check
            # _is_valid checks lines too, so let's just check points manually
            pt_set = set()
            valid = True
            for i in range(54):
                x, y = pts[i]
                if not np.isfinite(x) or not np.isfinite(y):
                    valid = False
                    break
                if abs(x - round(x)) > 1e-6 or abs(y - round(y)) > 1e-6:
                    valid = False
                    break
                xi, yi = int(round(x)), int(round(y))
                if (xi, yi) in pt_set:
                    valid = False
                    break
                pt_set.add((xi, yi))
            if not valid:
                continue

        # Find best lines for these points
        lines = _best_lines_for_points(pts)
        
        if not _is_valid(pts, lines):
            continue
            
        score = _count_incidentences(pts, lines)
        
        # Local search to improve
        lines, score = _local_search(pts, lines, deadline, safety_margin)
        
        if score > best_score:
            best_score = score
            best_points = pts
            best_lines = lines
            
    # Fallback if no valid configuration found
    if best_points is None:
        pts = _generate_initial_grid(6, 9)
        lines = _best_lines_for_points(pts)
        best_points = pts
        best_lines = lines
        
    return best_points, best_lines

# EVOLVE-BLOCK-END
