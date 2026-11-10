# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize line (A,B,C) to canonical form."""
    if A == 0 and B == 0:
        return (0, 0, 0)
    g = gcd(abs(A), abs(B))
    g = gcd(g, abs(C))
    if g > 1:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count exact incidences."""
    count = 0
    for p in points:
        x, y = p
        for l in lines:
            A, B, C = l
            if A * x + B * y + C == 0:
                count += 1
    return count

def _baseline():
    """Szemeredi-Trotter grid construction."""
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

def _is_valid(points, lines):
    """Check validity constraints."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(lines)):
        return False
    if not np.allclose(points, np.round(points), atol=1e-6) or not np.allclose(lines, np.round(lines), atol=1e-6):
        return False
    # Check for repeated points
    pts = set(map(tuple, points))
    if len(pts) != 54:
        return False
    # Check lines
    seen_lines = set()
    for l in lines:
        A, B, C = int(l[0]), int(l[1]), int(l[2])
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        if norm in seen_lines:
            return False
        seen_lines.add(norm)
    return True

def _line_from_two_points(p1, p2):
    """Compute normalized line through two points."""
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return _normalize_line(A, B, C)

def _local_search(points, lines, max_iters=50000, time_limit=550):
    """Local search to improve incidences."""
    start_time = time.time()
    points = points.copy()
    lines = lines.copy()
    best_score = _count_incidences(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    
    n_pts = 54
    n_lines = 27
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        if random.random() < 0.5:
            idx = random.randint(0, n_pts - 1)
            old_x, old_y = int(points[idx, 0]), int(points[idx, 1])
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = old_x + dx, old_y + dy
                    if any(int(points[i,0]) == new_x and int(points[i,1]) == new_y for i in range(n_pts) if i != idx):
                        continue
                    points[idx, 0] = new_x
                    points[idx, 1] = new_y
                    score = _count_incidences(points, lines)
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
                        best_lines = lines.copy()
                    else:
                        points[idx, 0] = old_x
                        points[idx, 1] = old_y
        else:
            idx = random.randint(0, n_lines - 1)
            old_A, old_B, old_C = int(lines[idx, 0]), int(lines[idx, 1]), int(lines[idx, 2])
            for dA in range(-2, 3):
                for dB in range(-2, 3):
                    for dC in range(-5, 6):
                        if dA == 0 and dB == 0 and dC == 0:
                            continue
                        new_A, new_B, new_C = old_A + dA, old_B + dB, old_C + dC
                        if new_A == 0 and new_B == 0:
                            continue
                        norm_new = _normalize_line(new_A, new_B, new_C)
                        is_dup = False
                        for j in range(n_lines):
                            if j != idx:
                                norm_j = _normalize_line(int(lines[j,0]), int(lines[j,1]), int(lines[j,2]))
                                if norm_new == norm_j:
                                    is_dup = True
                                    break
                        if is_dup:
                            continue
                        lines[idx, 0] = new_A
                        lines[idx, 1] = new_B
                        lines[idx, 2] = new_C
                        score = _count_incidences(points, lines)
                        if score > best_score:
                            best_score = score
                            best_points = points.copy()
                            best_lines = lines.copy()
                        else:
                            lines[idx, 0] = old_A
                            lines[idx, 1] = old_B
                            lines[idx, 2] = old_C
    
    return best_points, best_lines, best_score

def _line_replacement_search(points, lines, time_limit=120):
    """Replace low-value lines with new lines through pairs of poorly-covered points."""
    start_time = time.time()
    points = points.copy()
    lines = lines.copy()
    n_pts = 54
    n_lines = 27
    
    def _line_incidence_count(line, pts):
        A, B, C = line
        cnt = 0
        for p in pts:
            if A * int(p[0]) + B * int(p[1]) + C == 0:
                cnt += 1
        return cnt
    
    def _point_coverage(pt, lns):
        x, y = int(pt[0]), int(pt[1])
        cnt = 0
        for l in lns:
            if int(l[0]) * x + int(l[1]) * y + int(l[2]) == 0:
                cnt += 1
        return cnt
    
    best_score = _count_incidences(points, lines)
    best_lines = lines.copy()
    
    for iteration in range(300):
        if time.time() - start_time > time_limit:
            break
        
        line_counts = []
        for i in range(n_lines):
            line_counts.append(_line_incidence_count(tuple(int(lines[i,j]) for j in range(3)), points))
        
        pt_cov = []
        for i in range(n_pts):
            pt_cov.append(_point_coverage(points[i], lines))
        
        min_line_idx = min(range(n_lines), key=lambda i: line_counts[i])
        min_line_count = line_counts[min_line_idx]
        
        low_cov_points = [i for i in range(n_pts) if pt_cov[i] <= 2]
        if not low_cov_points:
            low_cov_points = list(range(n_pts))
        
        best_new_score = best_score
        best_new_line = None
        
        candidates = []
        if len(low_cov_points) <= 20:
            for i in range(len(low_cov_points)):
                for j in range(i+1, len(low_cov_points)):
                    candidates.append((low_cov_points[i], low_cov_points[j]))
        else:
            for _ in range(80):
                i = random.choice(low_cov_points)
                j = random.choice(low_cov_points)
                if i != j:
                    candidates.append((min(i,j), max(i,j)))
        
        for pi, pj in candidates:
            if time.time() - start_time > time_limit:
                break
            new_line = _line_from_two_points(points[pi], points[pj])
            dup = False
            for k in range(n_lines):
                if k == min_line_idx:
                    continue
                if new_line == tuple(int(lines[k,j]) for j in range(3)):
                    dup = True
                    break
            if dup:
                continue
            
            new_count = _line_incidence_count(new_line, points)
            delta = new_count - min_line_count
            new_score = best_score + delta
            
            if new_score > best_new_score:
                best_new_score = new_score
                best_new_line = new_line
        
        if best_new_line is not None and best_new_score > best_score:
            lines[min_line_idx, 0] = best_new_line[0]
            lines[min_line_idx, 1] = best_new_line[1]
            lines[min_line_idx, 2] = best_new_line[2]
            best_score = best_new_score
            best_lines = lines.copy()
        else:
            # Try random replacement
            if random.random() < 0.4:
                idx = random.randint(0, n_lines - 1)
                pi = random.randint(0, n_pts - 1)
                pj = random.randint(0, n_pts - 1)
                if pi == pj:
                    continue
                new_line = _line_from_two_points(points[pi], points[pj])
                dup = False
                for k in range(n_lines):
                    if k == idx:
                        continue
                    if new_line == tuple(int(lines[k,j]) for j in range(3)):
                        dup = True
                        break
                if dup:
                    continue
                old_count = _line_incidence_count(tuple(int(lines[idx,j]) for j in range(3)), points)
                new_count = _line_incidence_count(new_line, points)
                if new_count > old_count:
                    lines[idx, 0] = new_line[0]
                    lines[idx, 1] = new_line[1]
                    lines[idx, 2] = new_line[2]
                    best_score += (new_count - old_count)
                    best_lines = lines.copy()
    
    return points, best_lines, best_score

def _try_grid_variants(deadline):
    """Try different grid-like constructions."""
    best_points = None
    best_lines = None
    best_score = 0
    
    # Try various rectangular grids
    for w in range(2, 8):
        for h in range(2, 12):
            if w * h != 54:
                continue
            if time.time() > deadline:
                break
            xs = np.arange(1, w + 1)
            ys = np.arange(1, h + 1)
            px, py = np.meshgrid(xs, ys, indexing="ij")
            points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
            
            # Find best 27 lines through these points
            # For each pair of points, compute the line and count incidences
            line_candidates = {}
            for i in range(54):
                for j in range(i+1, 54):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    A = y2 - y1
                    B = x1 - x2
                    C = x2 * y1 - x1 * y2
                    norm = _normalize_line(A, B, C)
                    if norm not in line_candidates:
                        line_candidates[norm] = 0
                    line_candidates[norm] += 1
            
            # Sort by count and take top 27
            sorted_lines = sorted(line_candidates.items(), key=lambda x: -x[1])[:27]
            lines = np.array([l for l, c in sorted_lines], dtype=np.int64)
            
            score = _count_incidences(points, lines)
            if score > best_score:
                best_score = score
                best_points = points.copy()
                best_lines = lines.copy()
    
    return best_points, best_lines, best_score

def _random_restart_search(deadline, num_restarts=20):
    """Try random configurations and local search from each."""
    best_points = None
    best_lines = None
    best_score = 0
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random points in a bounded region
        points = np.zeros((54, 2), dtype=np.int64)
        for i in range(54):
            points[i, 0] = random.randint(-10, 10)
            points[i, 1] = random.randint(-10, 10)
        
        # Remove duplicates
        seen = set()
        unique_points = []
        for p in points:
            t = (int(p[0]), int(p[1]))
            if t not in seen:
                seen.add(t)
                unique_points.append(t)
        while len(unique_points) < 54:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            t = (x, y)
            if t not in seen:
                seen.add(t)
                unique_points.append(t)
        points = np.array(unique_points[:54], dtype=np.int64)
        
        # Find best 27 lines
        line_candidates = {}
        for i in range(54):
            for j in range(i+1, 54):
                x1, y1 = points[i]
                x2, y2 = points[j]
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                norm = _normalize_line(A, B, C)
                if norm not in line_candidates:
                    line_candidates[norm] = 0
                line_candidates[norm] += 1
        
        sorted_lines = sorted(line_candidates.items(), key=lambda x: -x[1])[:27]
        lines = np.array([l for l, c in sorted_lines], dtype=np.int64)
        
        score = _count_incidences(points, lines)
        
        # Local search
        if time.time() < deadline - 60:
            p2, l2, s2 = _local_search(points, lines, max_iters=5000, time_limit=50)
            if s2 > score:
                score = s2
                points = p2
                lines = l2
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
            best_lines = lines.copy()
    
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    Returns (points, lines) where points is (54,2) int array and lines is (27,3) int array.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Start with baseline
    points, lines = _baseline()
    best_score = _count_incidences(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    
    # Try grid variants
    if time.time() < deadline - 200:
        gp, gl, gs = _try_grid_variants(deadline - 200)
        if gp is not None and gs > best_score:
            best_score = gs
            best_points = gp
            best_lines = gl
    
    # Line replacement search on best grid
    if time.time() < deadline - 130:
        lr_time = min(60, int(deadline - 130 - time.time()))
        if lr_time > 10:
            p2, l2, s2 = _line_replacement_search(best_points, best_lines, time_limit=lr_time)
            if s2 > best_score:
                best_score = s2
                best_points = p2
                best_lines = l2
    
    # Random restart search
    if time.time() < deadline - 80:
        rp, rl, rs = _random_restart_search(deadline - 80, num_restarts=12)
        if rp is not None and rs > best_score:
            best_score = rs
            best_points = rp
            best_lines = rl
    
    # Line replacement on random restart best
    if time.time() < deadline - 50:
        lr_time = min(30, int(deadline - 50 - time.time()))
        if lr_time > 5:
            p2, l2, s2 = _line_replacement_search(best_points, best_lines, time_limit=lr_time)
            if s2 > best_score:
                best_score = s2
                best_points = p2
                best_lines = l2
    
    # Final local search on best found
    if time.time() < deadline - 30:
        remaining = min(180, int(deadline - 30 - time.time()))
        if remaining > 10:
            fp, fl, fs = _local_search(best_points, best_lines, max_iters=100000, time_limit=remaining)
            if fs > best_score:
                best_score = fs
                best_points = fp
                best_lines = fl
    
    # Final line replacement polish
    if time.time() < deadline - 10:
        lr_time = min(20, int(deadline - 10 - time.time()))
        if lr_time > 5:
            p2, l2, s2 = _line_replacement_search(best_points, best_lines, time_limit=lr_time)
            if s2 > best_score:
                best_score = s2
                best_points = p2
                best_lines = l2
    
    # Ensure validity
    if not _is_valid(best_points, best_lines):
        # Fallback to baseline
        best_points, best_lines = _baseline()
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
