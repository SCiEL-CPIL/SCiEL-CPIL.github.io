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

def _find_best_lines_for_points(points, k=27, deadline=None):
    """Find the k lines with maximum total incidence for the given points."""
    n = len(points)
    points_int = points.astype(np.int64)
    
    seen_lines = set()
    candidate_lines = []
    
    # All distinct lines from pairs of points
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
    
    # Add lines with small integer coefficients
    for A in range(-6, 7):
        for B in range(-6, 7):
            if A == 0 and B == 0:
                continue
            for C in range(-20, 21):
                key = _normalize_line(A, B, C)
                if key not in seen_lines:
                    seen_lines.add(key)
                    candidate_lines.append(key)
    
    if len(candidate_lines) == 0:
        return np.array([[_normalize_line(0, 1, -c)] for c in range(-27, 0)], dtype=np.int64)
    
    arr = np.array(candidate_lines, dtype=np.int64)
    
    px = points_int[:, 0:1].T
    py = points_int[:, 1:2].T
    A = arr[:, 0:1]
    B = arr[:, 1:2]
    C = arr[:, 2:3]
    
    vals = A * px + B * py + C
    counts = np.sum(vals == 0, axis=1)
    
    if len(counts) > k:
        idx = np.argsort(-counts)
        top_idx = idx[:k]
    else:
        top_idx = np.arange(len(counts))
    
    return arr[top_idx].astype(np.int64)

def _generate_point_sets(deadline):
    """Generate a variety of point configurations."""
    sets = []
    
    # 1. Rectangular Grids
    for w, h in [(6, 9), (9, 6), (7, 8), (8, 7)]:
        if w * h == 54:
            pts = np.array([[i, j] for i in range(w) for j in range(h)], dtype=np.int64)
            sets.append(pts)
    
    # 1b. Scaled grids
    for scale in [2, 3]:
        for w, h in [(6, 9), (9, 6)]:
            if w * h == 54:
                pts = np.array([[i * scale, j * scale] for i in range(w) for j in range(h)], dtype=np.int64)
                sets.append(pts)
    
    # 1c. Skewed grids
    for shear in [1, 2, -1]:
        for w, h in [(6, 9)]:
            pts = np.array([[i + shear * j, j] for i in range(w) for j in range(h)], dtype=np.int64)
            if len(set(map(tuple, pts.tolist()))) == 54:
                sets.append(pts)
    
    # 2. Clique-star hybrid: 27 on vertical line + 27 on 3x9 grid
    pts = []
    for j in range(27):
        pts.append([0, j])
    for i in range(3):
        for j in range(9):
            pts.append([2 + i, j])
    pts = np.array(pts, dtype=np.int64)
    if len(pts) == 54 and len(set(map(tuple, pts))) == 54:
        sets.append(pts)
    
    # 2b. Clique-star: 27 on horizontal line + 27 on 9x3 grid
    pts = []
    for i in range(27):
        pts.append([i, 0])
    for i in range(9):
        for j in range(3):
            pts.append([i, 2 + j])
    pts = np.array(pts, dtype=np.int64)
    if len(pts) == 54 and len(set(map(tuple, pts))) == 54:
        sets.append(pts)
    
    # 2c. Clique-star: 18 on vertical + 18 on horizontal + 18 on grid
    pts = []
    for j in range(18):
        pts.append([0, j])
    for i in range(1, 19):
        pts.append([i, 0])
    for i in range(3):
        for j in range(6):
            pts.append([2 + i, 2 + j])
    pts = np.array(pts, dtype=np.int64)
    if len(pts) == 54 and len(set(map(tuple, pts))) == 54:
        sets.append(pts)
    
    # 2d. Clique-star: 27 on diagonal + 27 on grid
    pts = []
    for j in range(27):
        pts.append([j, j])
    for i in range(3):
        for j in range(9):
            pts.append([30 + i, j])
    pts = np.array(pts, dtype=np.int64)
    if len(pts) == 54 and len(set(map(tuple, pts))) == 54:
        sets.append(pts)
    
    # 3. Two parallel lines of 27
    pts = []
    for j in range(27):
        pts.append([j, 0])
    for j in range(27):
        pts.append([j, 3])
    pts = np.array(pts, dtype=np.int64)
    if len(pts) == 54 and len(set(map(tuple, pts))) == 54:
        sets.append(pts)
    
    # 4. Three lines of 18
    pts = []
    for j in range(18):
        pts.append([j, 0])
    for j in range(18):
        pts.append([j, 3])
    for j in range(18):
        pts.append([j, 6])
    pts = np.array(pts, dtype=np.int64)
    if len(pts) == 54 and len(set(map(tuple, pts))) == 54:
        sets.append(pts)
    
    # 5. 6x9 grid with diagonal perturbation
    base = np.array([[i, j] for i in range(6) for j in range(9)], dtype=np.int64)
    perturbed = base.copy()
    perturbed[0, 0] = 6
    perturbed[1, 0] = 7
    perturbed[2, 0] = 8
    perturbed[3, 0] = 9
    perturbed[4, 0] = 10
    perturbed[5, 0] = 11
    if len(set(map(tuple, perturbed))) == 54:
        sets.append(perturbed)
    
    # 6. L-shape
    pts = []
    for i in range(18):
        pts.append([i, 0])
    for j in range(1, 37):
        pts.append([0, j])
    pts = np.array(pts, dtype=np.int64)
    if len(pts) == 54 and len(set(map(tuple, pts))) == 54:
        sets.append(pts)
    
    # 7. 3x18 grid
    pts = np.array([[i, j] for i in range(3) for j in range(18)], dtype=np.int64)
    sets.append(pts)
    
    # 8. 18x3 grid
    pts = np.array([[i, j] for i in range(18) for j in range(3)], dtype=np.int64)
    sets.append(pts)
    
    return sets

def _evaluate_set(points, deadline):
    """Evaluate a specific point set by finding best lines."""
    lines = _find_best_lines_for_points(points, 27, deadline)
    score = _count_incidence(points, lines)
    return points, lines, score

def _refine_set(points, lines, deadline, max_steps=5000):
    """
    Refine the configuration by moving points.
    Uses incremental incidence tracking for speed.
    """
    current_score = _count_incidence(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = current_score
    
    no_improve_count = 0
    point_set = set(map(tuple, best_points.tolist()))
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        i = random.randint(0, 53)
        ox, oy = int(best_points[i, 0]), int(best_points[i, 1])
        
        moved = False
        offsets = [(dx, dy) for dx in range(-4, 5) for dy in range(-4, 5) if (dx, dy) != (0, 0)]
        random.shuffle(offsets)
        
        for dx, dy in offsets:
            nx, ny = ox + dx, oy + dy
            
            if (nx, ny) in point_set:
                continue
            
            new_points = best_points.copy()
            new_points[i, 0] = nx
            new_points[i, 1] = ny
            
            new_lines = _find_best_lines_for_points(new_points, 27)
            new_score = _count_incidence(new_points, new_lines)
            
            if new_score > current_score:
                best_points = new_points
                best_lines = new_lines
                current_score = new_score
                best_score = current_score
                no_improve_count = 0
                point_set = set(map(tuple, best_points.tolist()))
                moved = True
                break
            elif new_score == current_score and random.random() < 0.05:
                best_points = new_points
                best_lines = new_lines
                point_set = set(map(tuple, best_points.tolist()))
                no_improve_count = 0
                moved = True
                break
        
        if not moved:
            no_improve_count += 1
            if no_improve_count > 80:
                i = random.randint(0, 53)
                ox, oy = int(best_points[i, 0]), int(best_points[i, 1])
                nx = ox + random.randint(-10, 10)
                ny = oy + random.randint(-10, 10)
                
                if (nx, ny) not in point_set:
                    new_points = best_points.copy()
                    new_points[i, 0] = nx
                    new_points[i, 1] = ny
                    new_lines = _find_best_lines_for_points(new_points, 27)
                    new_score = _count_incidence(new_points, new_lines)
                    
                    if new_score >= current_score:
                        best_points = new_points
                        best_lines = new_lines
                        current_score = new_score
                        best_score = max(best_score, current_score)
                        point_set = set(map(tuple, best_points.tolist()))
                        no_improve_count = 0
    
    return best_points, best_lines, best_score

def _polish_set(points, lines, deadline):
    """
    Final polish: try all single-point moves with larger radius, accept only improvements.
    """
    current_score = _count_incidence(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = current_score
    
    point_set = set(map(tuple, best_points.tolist()))
    
    for pass_num in range(3):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(54):
            if time.time() > deadline:
                break
            
            ox, oy = int(best_points[i, 0]), int(best_points[i, 1])
            
            best_move = None
            best_move_score = current_score
            
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    
                    if (nx, ny) in point_set:
                        continue
                    
                    new_points = best_points.copy()
                    new_points[i, 0] = nx
                    new_points[i, 1] = ny
                    
                    new_lines = _find_best_lines_for_points(new_points, 27)
                    new_score = _count_incidence(new_points, new_lines)
                    
                    if new_score > best_move_score:
                        best_move_score = new_score
                        best_move = (nx, ny)
            
            if best_move is not None:
                nx, ny = best_move
                best_points[i, 0] = nx
                best_points[i, 1] = ny
                point_set.remove((ox, oy))
                point_set.add((nx, ny))
                best_lines = _find_best_lines_for_points(best_points, 27)
                current_score = best_move_score
                best_score = max(best_score, current_score)
                improved = True
        
        if not improved:
            break
    
    return best_points, best_lines, best_score

def _refine_lines(points, lines, deadline, max_iters=200):
    """
    Joint refinement: swap lines to improve incidence while keeping points fixed.
    """
    current_score = _count_incidence(points, lines)
    best_lines = lines.copy()
    best_score = current_score
    n = len(points)
    
    # Precompute point coordinates
    px = points[:, 0].astype(np.int64)
    py = points[:, 1].astype(np.int64)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        # Try replacing each line with a better one
        for li in range(27):
            if time.time() > deadline:
                break
            
            # Generate candidate replacement lines through pairs of points
            candidates = set()
            for i in range(n):
                for j in range(i + 1, n):
                    A = int(py[j]) - int(py[i])
                    B = int(px[i]) - int(px[j])
                    C = int(px[j]) * int(py[i]) - int(px[i]) * int(py[j])
                    if A == 0 and B == 0:
                        continue
                    key = _normalize_line(A, B, C)
                    candidates.add(key)
            
            # Also try small-coefficient lines
            for A in range(-4, 5):
                for B in range(-4, 5):
                    if A == 0 and B == 0:
                        continue
                    for C in range(-15, 16):
                        candidates.add(_normalize_line(A, B, C))
            
            # Compute incidence for each candidate
            best_cand = None
            best_cand_score = current_score - int(np.sum(A * px + B * py + C == 0))  # remove current line's contribution
            A_cur, B_cur, C_cur = int(lines[li, 0]), int(lines[li, 1]), int(lines[li, 2])
            
            for cand in candidates:
                if cand == (A_cur, B_cur, C_cur):
                    continue
                ca, cb, cc = cand
                inc = int(np.sum(ca * px + cb * py + cc == 0))
                # Total score if we replace line li with cand
                new_score = current_score - int(np.sum(A_cur * px + B_cur * py + C_cur == 0)) + inc
                if new_score > best_cand_score:
                    best_cand_score = new_score
                    best_cand = cand
            
            if best_cand is not None:
                lines[li, 0] = best_cand[0]
                lines[li, 1] = best_cand[1]
                lines[li, 2] = best_cand[2]
                current_score = best_cand_score
                best_score = max(best_score, current_score)
                improved = True
        
        if not improved:
            break
    
    return lines, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 590
    
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
    
    # Phase 1b: Try random perturbations of the best set
    if best_points is not None:
        for trial in range(10):
            if time.time() > deadline:
                break
            perturbed = best_points.copy()
            for _ in range(10):
                idx = random.randint(0, 53)
                perturbed[idx, 0] += random.randint(-3, 3)
                perturbed[idx, 1] += random.randint(-3, 3)
            seen_pts = set()
            valid = []
            for p in perturbed:
                key = (int(p[0]), int(p[1]))
                if key not in seen_pts:
                    seen_pts.add(key)
                    valid.append(key)
            if len(valid) == 54:
                perturbed = np.array(valid, dtype=np.int64)
                pts, lines, score = _evaluate_set(perturbed, deadline)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
                    best_lines = lines.copy()
    
    # Phase 2: Refinement
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        
        refine_budget = int(remaining * 0.45)
        line_refine_budget = int(remaining * 0.10)
        polish_budget = int(remaining * 0.15)
        refine2_budget = int(remaining * 0.30)
        
        refine_deadline = start_time + elapsed + refine_budget
        max_steps = min(15000, int(refine_budget * 80))
        if max_steps < 1000:
            max_steps = 1000
        
        best_points, best_lines, best_score = _refine_set(
            best_points, best_lines, refine_deadline, max_steps
        )
        
        # Phase 2b: Line refinement (swap lines to improve)
        line_deadline = start_time + elapsed + refine_budget + line_refine_budget
        best_lines, best_score = _refine_lines(best_points, best_lines, line_deadline)
        
        # Phase 3: Polish
        polish_deadline = start_time + elapsed + refine_budget + line_refine_budget + polish_budget
        best_points, best_lines, best_score = _polish_set(
            best_points, best_lines, polish_deadline
        )
        
        # Phase 3b: Another line refinement pass
        if time.time() < deadline - 10:
            line_deadline2 = deadline - 5
            best_lines, best_score = _refine_lines(best_points, best_lines, line_deadline2)
        
        # Phase 4: Second refinement pass
        if time.time() < deadline - 5:
            random.seed(int(time.time() * 1000) % 100000)
            refine2_deadline = deadline - 5
            max_steps2 = min(10000, int(refine2_budget * 80))
            if max_steps2 < 500:
                max_steps2 = 500
            best_points, best_lines, best_score = _refine_set(
                best_points, best_lines, refine2_deadline, max_steps2
            )
    
    # Final Validation and Cleanup
    if best_points is None or len(best_points) != 54:
        best_points = np.array([[i, j] for i in range(6) for j in range(9)], dtype=np.int64)
        best_lines = _find_best_lines_for_points(best_points, 27)
    
    points = np.array(best_points, dtype=np.int64)
    lines = np.array(best_lines, dtype=np.int64)
    
    if len(points) > 54:
        points = points[:54]
    elif len(points) < 54:
        while len(points) < 54:
            points = np.vstack([points, [len(points)*100, len(points)*200]])
    
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
