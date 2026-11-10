# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
from itertools import combinations

def _normalize_line(A, B, C):
    A, B, C = int(A), int(B), int(C)
    g = math.gcd(abs(A), abs(B))
    g = math.gcd(g, abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidence(points, lines):
    """Count incidences between points (N,2) and lines (M,3)."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0]
    py = points[:, 1]
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    vals = A[:, None] * px[None, :] + B[:, None] * py[None, :] + C[:, None]
    return int(np.sum(vals == 0))

def _build_candidate_lines(points, extra_range=8):
    """Build candidate lines from point pairs plus small-coefficient lines."""
    n = len(points)
    points_int = points.astype(np.int64)
    
    seen_lines = set()
    candidate_lines = []
    
    for i in range(n):
        x1, y1 = int(points_int[i, 0]), int(points_int[i, 1])
        for j in range(i + 1, n):
            x2, y2 = int(points_int[j, 0]), int(points_int[j, 1])
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _normalize_line(A, B, C)
            if key not in seen_lines:
                seen_lines.add(key)
                candidate_lines.append(key)
    
    # Add small-coefficient lines with wider range
    for A in range(-extra_range, extra_range + 1):
        for B in range(-extra_range, extra_range + 1):
            if A == 0 and B == 0:
                continue
            for C in range(-80, 81):
                key = _normalize_line(A, B, C)
                if key not in seen_lines:
                    seen_lines.add(key)
                    candidate_lines.append(key)
    
    return np.array(candidate_lines, dtype=np.int64), seen_lines

def _find_best_lines_for_points(points, k=27, candidate_arr=None):
    """Find the k lines with maximum total incidence for the given points."""
    n = len(points)
    points_int = points.astype(np.int64)
    
    if candidate_arr is None:
        candidate_arr, _ = _build_candidate_lines(points)
    
    if len(candidate_arr) == 0:
        return np.array([[_normalize_line(0, 1, -c)] for c in range(1, 28)], dtype=np.int64)
    
    px = points_int[:, 0:1].T
    py = points_int[:, 1:2].T
    A = candidate_arr[:, 0:1]
    B = candidate_arr[:, 1:2]
    C = candidate_arr[:, 2:3]
    
    vals = A * px + B * py + C
    counts = np.sum(vals == 0, axis=1)
    
    if len(counts) > k:
        idx = np.argsort(-counts)
        top_idx = idx[:k]
    else:
        top_idx = np.arange(len(counts))
    
    return candidate_arr[top_idx].astype(np.int64)

def _joint_line_swap_refine(points, lines, deadline, max_steps=500):
    """Try swapping one line with a better candidate while keeping points fixed."""
    current_score = _count_incidence(points, lines)
    best_lines = lines.copy()
    best_score = current_score
    no_improve = 0
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Pick a random line to try replacing
        li = random.randint(0, 26)
        
        # Build candidates from current points
        candidate_arr, seen = _build_candidate_lines(points)
        
        # Exclude current lines from candidates
        current_keys = set()
        for l in best_lines:
            current_keys.add(_normalize_line(int(l[0]), int(l[1]), int(l[2])))
        
        # Find best replacement
        px = points.astype(np.int64)[:, 0:1].T
        py = points.astype(np.int64)[:, 1:2].T
        
        best_replace = None
        best_replace_gain = 0
        
        for ci in range(len(candidate_arr)):
            key = _normalize_line(int(candidate_arr[ci, 0]), int(candidate_arr[ci, 1]), int(candidate_arr[ci, 2]))
            if key in current_keys:
                continue
            
            A = candidate_arr[ci, 0:1]
            B = candidate_arr[ci, 1:2]
            C = candidate_arr[ci, 2:3]
            vals = A * px + B * py + C
            cnt = int(np.sum(vals == 0))
            
            # Current line's count
            A_cur = best_lines[li, 0:1]
            B_cur = best_lines[li, 1:2]
            C_cur = best_lines[li, 2:3]
            vals_cur = A_cur * px + B_cur * py + C_cur
            cnt_cur = int(np.sum(vals_cur == 0))
            
            gain = cnt - cnt_cur
            if gain > best_replace_gain:
                best_replace_gain = gain
                best_replace = candidate_arr[ci].copy()
        
        if best_replace is not None and best_replace_gain > 0:
            best_lines[li] = best_replace
            best_score = current_score + best_replace_gain
            current_score = best_score
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 50:
                break
    
    return best_lines, best_score

def _generate_point_sets(deadline):
    """Generate a variety of point configurations."""
    sets = []
    
    # 1. Rectangular Grid 6x9 with offsets
    for ox in range(0, 4):
        for oy in range(0, 4):
            pts = np.array([[ox + i, oy + j] for i in range(6) for j in range(9)], dtype=np.int64)
            sets.append(pts)
    
    # 2. Triangular Lattice
    for stagger_type in ['even', 'odd']:
        pts = []
        for j in range(9):
            for i in range(6):
                if stagger_type == 'even' and j % 2 == 1:
                    x = i + 1
                elif stagger_type == 'odd' and j % 2 == 0:
                    x = i + 1
                else:
                    x = i
                pts.append([x, j])
        pts = np.array(pts, dtype=np.int64)
        if len(set(map(tuple, pts))) == 54:
            sets.append(pts)
    
    # 3. Clique-star: vertical line + grid
    vpts = [[0, y] for y in range(27)]
    gpts = [[x, y] for x in range(2, 5) for y in range(9)]
    pts = np.array(vpts + gpts, dtype=np.int64)
    if len(set(map(tuple, pts))) == 54:
        sets.append(pts)
    
    # 4. Sheared grid
    for shear in [1, 2, 3]:
        pts = []
        for j in range(9):
            for i in range(6):
                x = i + shear * (j // 3)
                pts.append([x, j])
        pts = np.array(pts, dtype=np.int64)
        if len(set(map(tuple, pts))) == 54:
            sets.append(pts)
    
    # 5. Scaled grids
    for sx, sy in [(2, 1), (1, 2), (2, 2), (3, 1), (1, 3)]:
        pts = np.array([[i * sx, j * sy] for i in range(6) for j in range(9)], dtype=np.int64)
        if len(set(map(tuple, pts))) == 54:
            sets.append(pts)
    
    # 6. Random perturbations of 6x9 grid
    for _ in range(12):
        base = np.array([[i, j] for i in range(6) for j in range(9)], dtype=np.int64)
        pts = base.copy()
        for _ in range(5):
            idx = random.randint(0, 53)
            pts[idx, 0] += random.randint(-3, 3)
            pts[idx, 1] += random.randint(-3, 3)
        if len(set(map(tuple, pts))) == 54:
            sets.append(pts)
    
    # 7. Diagonal stripe configurations
    for diag_shift in [0, 1, 2]:
        pts = []
        for j in range(9):
            for i in range(6):
                x = i + diag_shift
                y = j
                # Create diagonal bands
                if (i + j) % 3 == 0:
                    x += 1
                pts.append([x, y])
        pts = np.array(pts, dtype=np.int64)
        if len(set(map(tuple, pts))) == 54:
            sets.append(pts)
    
    # 8. Compact cluster + sparse points
    cluster = [[i, j] for i in range(4) for j in range(4)]  # 16 points
    sparse = [[10 + i, 0] for i in range(18)]  # 18 points
    sparse2 = [[0, 10 + i] for i in range(20)]  # 20 points
    pts = np.array(cluster + sparse + sparse2, dtype=np.int64)
    if len(set(map(tuple, pts))) == 54:
        sets.append(pts)
    
    # 9. Two-row dense configuration
    row1 = [[i, 0] for i in range(27)]
    row2 = [[i, 1] for i in range(27)]
    pts = np.array(row1 + row2, dtype=np.int64)
    if len(set(map(tuple, pts))) == 54:
        sets.append(pts)
    
    # 10. L-shaped configuration
    horiz = [[i, 0] for i in range(27)]
    vert = [[0, i] for i in range(1, 28)]
    pts = np.array(horiz + vert, dtype=np.int64)
    if len(set(map(tuple, pts))) == 54:
        sets.append(pts)
    
    return sets

def _evaluate_set(points, deadline):
    """Evaluate a specific point set by finding best lines."""
    candidate_arr, _ = _build_candidate_lines(points)
    lines = _find_best_lines_for_points(points, 27, candidate_arr)
    score = _count_incidence(points, lines)
    return points, lines, score

def _refine_set(points, lines, deadline, max_steps=3000):
    """
    Refine the configuration by moving points and swapping lines.
    Uses incremental candidate line rebuilding for speed.
    """
    current_score = _count_incidence(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = current_score
    no_improve_count = 0
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Alternate between point moves and line swaps
        if step % 5 == 4:
            # Try line swap
            new_lines, new_score = _joint_line_swap_refine(best_points, best_lines, deadline, max_steps=50)
            if new_score > current_score:
                best_lines = new_lines
                current_score = new_score
                best_score = current_score
                no_improve_count = 0
            else:
                no_improve_count += 1
            continue
        
        # Pick a random point to move
        i = random.randint(0, 53)
        ox, oy = int(best_points[i, 0]), int(best_points[i, 1])
        
        moved = False
        # Try nearby locations in a spiral pattern
        for radius in [1, 2, 3]:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    
                    # Check conflict
                    conflict = False
                    for j in range(54):
                        if j != i and int(best_points[j, 0]) == nx and int(best_points[j, 1]) == ny:
                            conflict = True
                            break
                    if conflict:
                        continue
                    
                    new_points = best_points.copy()
                    new_points[i, 0] = nx
                    new_points[i, 1] = ny
                    
                    # Rebuild candidates and find best lines
                    candidate_arr, _ = _build_candidate_lines(new_points)
                    new_lines = _find_best_lines_for_points(new_points, 27, candidate_arr)
                    new_score = _count_incidence(new_points, new_lines)
                    
                    if new_score > current_score:
                        best_points = new_points
                        best_lines = new_lines
                        current_score = new_score
                        best_score = current_score
                        no_improve_count = 0
                        moved = True
                        break
                    elif new_score == current_score and random.random() < 0.05:
                        # Occasional equal-score move to escape local optima
                        best_points = new_points
                        best_lines = new_lines
                        no_improve_count = 0
                        moved = True
                        break
                
                if moved:
                    break
            if moved:
                break
        
        if not moved:
            no_improve_count += 1
            if no_improve_count > 80:
                # Big jump
                i = random.randint(0, 53)
                ox, oy = int(best_points[i, 0]), int(best_points[i, 1])
                nx = ox + random.randint(-6, 6)
                ny = oy + random.randint(-6, 6)
                
                conflict = False
                for j in range(54):
                    if j != i and int(best_points[j, 0]) == nx and int(best_points[j, 1]) == ny:
                        conflict = True
                        break
                if not conflict:
                    new_points = best_points.copy()
                    new_points[i, 0] = nx
                    new_points[i, 1] = ny
                    candidate_arr, _ = _build_candidate_lines(new_points)
                    new_lines = _find_best_lines_for_points(new_points, 27, candidate_arr)
                    new_score = _count_incidence(new_points, new_lines)
                    
                    if new_score > current_score:
                        best_points = new_points
                        best_lines = new_lines
                        current_score = new_score
                        best_score = current_score
                        no_improve_count = 0
                    elif new_score == current_score and random.random() < 0.1:
                        best_points = new_points
                        best_lines = new_lines
                        no_improve_count = 0
    
    return best_points, best_lines, best_score

def _polish_set(points, lines, deadline, max_steps=400):
    """
    Final high-precision polish with strict improvements only and unit steps.
    """
    current_score = _count_incidence(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = current_score
    
    for _ in range(max_steps):
        if time.time() > deadline:
            break
        
        i = random.randint(0, 53)
        ox, oy = int(best_points[i, 0]), int(best_points[i, 1])
        
        moved = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ox + dx, oy + dy
                
                conflict = False
                for j in range(54):
                    if j != i and int(best_points[j, 0]) == nx and int(best_points[j, 1]) == ny:
                        conflict = True
                        break
                if conflict:
                    continue
                
                new_points = best_points.copy()
                new_points[i, 0] = nx
                new_points[i, 1] = ny
                
                candidate_arr, _ = _build_candidate_lines(new_points)
                new_lines = _find_best_lines_for_points(new_points, 27, candidate_arr)
                new_score = _count_incidence(new_points, new_lines)
                
                if new_score > current_score:
                    best_points = new_points
                    best_lines = new_lines
                    current_score = new_score
                    best_score = current_score
                    moved = True
                    break
            if moved:
                break
    
    return best_points, best_lines, best_score

def _multi_start_refine(points, lines, deadline, num_starts=5):
    """
    Run multiple refinement passes from slightly perturbed starting points.
    """
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = _count_incidence(points, lines)
    
    for s in range(num_starts):
        if time.time() > deadline:
            break
        
        # Create a perturbed start
        perturbed = best_points.copy()
        for _ in range(random.randint(1, 4)):
            idx = random.randint(0, 53)
            perturbed[idx, 0] += random.randint(-3, 3)
            perturbed[idx, 1] += random.randint(-3, 3)
        
        # Ensure uniqueness
        if len(set(map(tuple, perturbed))) != 54:
            continue
        
        candidate_arr, _ = _build_candidate_lines(perturbed)
        lines_s = _find_best_lines_for_points(perturbed, 27, candidate_arr)
        score_s = _count_incidence(perturbed, lines_s)
        
        if score_s >= best_score:
            # Refine from this start
            p, l, sc = _refine_set(perturbed, lines_s, deadline, max_steps=1000)
            if sc > best_score:
                best_points = p
                best_lines = l
                best_score = sc
    
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Phase 1: Evaluate initial candidate sets
    candidate_sets = _generate_point_sets(deadline)
    
    for pts in candidate_sets:
        if time.time() > deadline:
            break
        pts, lines, score = _evaluate_set(pts, deadline)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lines.copy()
    
    # Phase 2: Multi-start refinement
    if best_points is not None:
        best_points, best_lines, best_score = _multi_start_refine(
            best_points, best_lines, deadline, num_starts=5
        )
        
        # Phase 3: Polish
        if time.time() < deadline - 30:
            best_points, best_lines, best_score = _polish_set(
                best_points, best_lines, deadline, max_steps=400
            )
    
    # Final Validation
    if best_points is None or len(best_points) != 54:
        best_points = np.array([[i, j] for i in range(6) for j in range(9)], dtype=np.int64)
        best_lines = _find_best_lines_for_points(best_points, 27)
    
    points = np.array(best_points, dtype=np.int64)
    lines = np.array(best_lines, dtype=np.int64)
    
    # Deduplicate points
    if len(points) > 54:
        points = points[:54]
    elif len(points) < 54:
        while len(points) < 54:
            points = np.vstack([points, [len(points) * 100, len(points) * 200]])
    
    # Deduplicate lines
    valid_lines = []
    seen = set()
    for l in lines:
        key = _normalize_line(int(l[0]), int(l[1]), int(l[2]))
        if key not in seen and (key[0] != 0 or key[1] != 0):
            seen.add(key)
            valid_lines.append(key)
    
    lines = np.array(valid_lines, dtype=np.int64)
    
    if len(lines) > 27:
        lines = lines[:27]
    elif len(lines) < 27:
        c = 1000
        while len(lines) < 27:
            key = _normalize_line(0, 1, -c)
            if key not in seen:
                seen.add(key)
                lines = np.vstack([lines, np.array(key, dtype=np.int64)])
            c += 1
    
    return points, lines

# EVOLVE-BLOCK-END
