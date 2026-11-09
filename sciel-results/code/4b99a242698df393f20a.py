import numpy as np
import time
import math
import random
from math import gcd

def _normalize_line(A, B, C):
    """Normalize (A,B,C) so that gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    # Make first nonzero positive
    for v in (A, B, C):
        if v != 0:
            if v < 0:
                A, B, C = -A, -B, -C
            break
    return (A, B, C)

def _count_incidents(points, lines):
    """Count point-line incidences exactly."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    total = 0
    for i in range(len(lns)):
        A, B, C = lns[i]
        if A == 0 and B == 0:
            continue
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        total += int(np.sum(vals == 0))
    return total

def _validate(points, lines):
    if not (isinstance(points, (list, tuple, np.ndarray))):
        return False
    pts = np.array(points, dtype=np.int64)
    lns = np.array(lines, dtype=np.int64)
    if pts.shape != (54, 2) or lns.shape != (27, 3):
        return False
    if not np.all(np.isfinite(pts.astype(np.float64))):
        return False
    if not np.all(np.isfinite(lns.astype(np.float64))):
        return False
    # No repeated points
    if len(set(map(tuple, pts.tolist()))) != 54:
        return False
    # No line with A=B=0, and no duplicate lines (normalized)
    seen = set()
    for A, B, C in lns:
        if A == 0 and B == 0:
            return False
        key = _normalize_line(int(A), int(B), int(C))
        if key in seen:
            return False
        seen.add(key)
    return True

def _build_grid6x9():
    """6x9 grid with best 27 lines."""
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    lines = []
    for c in range(6):
        lines.append((1, 0, -c))
    for r in range(9):
        lines.append((0, 1, -r))
    for d in range(-5, 6):
        lines.append((-1, 1, d))
    for s in range(0, 15):
        lines.append((1, 1, -s))

    scores = []
    for idx, (A, B, C) in enumerate(lines):
        val = A * points[:, 0] + B * points[:, 1] + C
        cnt = int(np.sum(val == 0))
        scores.append((cnt, idx))
    scores.sort(reverse=True)
    top = [lines[idx] for _, idx in scores[:27]]
    lines_arr = np.array(top, dtype=np.int64)
    return points, lines_arr

def _build_sz6x18():
    """Szemeredi-Trotter style: 3 columns x 18 rows."""
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

def _generate_candidate_lines(points, max_coord=12, max_lines=200):
    """Generate a pool of candidate lines from point pairs and simple families."""
    pts = np.asarray(points, dtype=np.int64)
    candidates = set()
    n = len(pts)
    
    # Lines through pairs of points
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _normalize_line(A, B, C)
            if key[0] != 0 or key[1] != 0:
                # Check coefficients are within reasonable bounds
                if abs(key[0]) <= max_coord and abs(key[1]) <= max_coord and abs(key[2]) <= max_coord * max_coord:
                    candidates.add(key)
    
    # Add simple families: vertical, horizontal, diagonals
    for c in range(-max_coord, max_coord + 1):
        candidates.add(_normalize_line(1, 0, -c))
        candidates.add(_normalize_line(0, 1, -c))
    for d in range(-2 * max_coord, 2 * max_coord + 1):
        candidates.add(_normalize_line(1, -1, d))
        candidates.add(_normalize_line(1, 1, d))
    for a in range(-3, 4):
        for b in range(-max_coord, max_coord + 1):
            candidates.add(_normalize_line(a, -1, b))
    
    return list(candidates)

def _simulated_annealing(points, lines, deadline, seed=42):
    """Run simulated annealing to improve the point-line configuration."""
    rng = random.Random(seed)
    pts = np.array(points, dtype=np.int64).copy()
    lns = np.array(lines, dtype=np.int64).copy()
    
    best_pts = pts.copy()
    best_lns = lns.copy()
    best_inc = _count_incidents(pts, lns)
    
    current_inc = best_inc
    T = 50.0
    T_min = 0.1
    cooling = 0.995
    
    # Generate candidate line pool
    line_pool = _generate_candidate_lines(pts, max_coord=12, max_lines=300)
    line_pool_arr = np.array(line_pool, dtype=np.int64)
    
    step = 0
    while time.time() < deadline and T > T_min:
        step += 1
        move_type = rng.random()
        
        if move_type < 0.3:
            # Move a point
            idx = rng.randint(0, 53)
            new_x = int(pts[idx, 0]) + rng.randint(-3, 3)
            new_y = int(pts[idx, 1]) + rng.randint(-3, 3)
            
            # Check for duplicate
            dup = False
            for i in range(54):
                if i != idx and pts[i, 0] == new_x and pts[i, 1] == new_y:
                    dup = True
                    break
            if dup:
                continue
            
            old_x, old_y = int(pts[idx, 0]), int(pts[idx, 1])
            pts[idx, 0] = new_x
            pts[idx, 1] = new_y
            
            new_inc = _count_incidents(pts, lns)
            delta = new_inc - current_inc
            
            if delta > 0 or rng.random() < math.exp(delta / T):
                current_inc = new_inc
                if current_inc > best_inc:
                    best_inc = current_inc
                    best_pts = pts.copy()
                    best_lns = lns.copy()
            else:
                pts[idx, 0] = old_x
                pts[idx, 1] = old_y
                
        elif move_type < 0.6:
            # Replace a line with another from the pool
            idx = rng.randint(0, 26)
            new_line_idx = rng.randint(0, len(line_pool_arr) - 1)
            new_line = line_pool_arr[new_line_idx]
            
            # Check for duplicate
            dup = False
            old_key = _normalize_line(int(lns[idx, 0]), int(lns[idx, 1]), int(lns[idx, 2]))
            new_key = _normalize_line(int(new_line[0]), int(new_line[1]), int(new_line[2]))
            if old_key == new_key:
                continue
            for i in range(27):
                if i != idx:
                    k = _normalize_line(int(lns[i, 0]), int(lns[i, 1]), int(lns[i, 2]))
                    if k == new_key:
                        dup = True
                        break
            if dup:
                continue
            
            old_line = lns[idx].copy()
            lns[idx] = new_line
            
            new_inc = _count_incidents(pts, lns)
            delta = new_inc - current_inc
            
            if delta > 0 or rng.random() < math.exp(delta / T):
                current_inc = new_inc
                if current_inc > best_inc:
                    best_inc = current_inc
                    best_pts = pts.copy()
                    best_lns = lns.copy()
            else:
                lns[idx] = old_line
                
        else:
            # Swap two points
            i = rng.randint(0, 53)
            j = rng.randint(0, 53)
            if i == j:
                continue
            pts[[i, j], :] = pts[[j, i], :]
            new_inc = _count_incidents(pts, lns)
            delta = new_inc - current_inc
            if delta > 0 or rng.random() < math.exp(delta / T):
                current_inc = new_inc
                if current_inc > best_inc:
                    best_inc = current_inc
                    best_pts = pts.copy()
                    best_lns = lns.copy()
            else:
                pts[[i, j], :] = pts[[j, i], :]
        
        T *= cooling
    
    return best_pts, best_lns, best_inc

def szemeredi_trotter_incidences():
    deadline = time.time() + 570.0
    random.seed(12345)
    np.random.seed(12345)
    
    # Start from known good configurations
    candidates = [
        ("grid6x9", _build_grid6x9),
        ("sz6x18", _build_sz6x18),
    ]
    
    best_pts = None
    best_lns = None
    best_inc = -1
    
    # Evaluate initial candidates
    for name, fn in candidates:
        if time.time() > deadline:
            break
        try:
            pts, lns = fn()
            if _validate(pts, lns):
                inc = _count_incidents(pts, lns)
                if inc > best_inc:
                    best_inc = inc
                    best_pts = pts.copy()
                    best_lns = lns.copy()
        except Exception:
            pass
    
    if best_pts is None:
        pts, lns = _build_grid6x9()
        return pts, lns
    
    # Run simulated annealing from the best starting point
    if time.time() < deadline - 30:
        sa_pts, sa_lns, sa_inc = _simulated_annealing(
            best_pts, best_lns, deadline, seed=42
        )
        if sa_inc > best_inc:
            best_inc = sa_inc
            best_pts = sa_pts
            best_lns = sa_lns
    
    # Try a second SA run with different seed if time permits
    if time.time() < deadline - 30:
        sa_pts2, sa_lns2, sa_inc2 = _simulated_annealing(
            best_pts, best_lns, deadline, seed=999
        )
        if sa_inc2 > best_inc:
            best_inc = sa_inc2
            best_pts = sa_pts2
            best_lns = sa_lns2
    
    return best_pts, best_lns
