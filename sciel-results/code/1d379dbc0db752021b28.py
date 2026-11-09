import numpy as np
import time
import random
import math
from math import gcd
from itertools import product

def _normalize_line(A, B, C):
    """Normalize a line (A, B, C) by dividing by gcd and making first nonzero positive."""
    A, B, C = int(A), int(B), int(C)
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 1:
        A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count exact integer incidences between points and lines."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    total = 0
    for i in range(len(lns)):
        A, B, C = lns[i]
        val = A * pts[:, 0] + B * pts[:, 1] + C
        total += int(np.sum(val == 0))
    return total

def _is_valid(points, lines):
    """Validate points and lines per the task contract."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    if pts.shape != (54, 2) or lns.shape != (27, 3):
        return False
    # Check no NaN/inf
    if not np.all(np.isfinite(pts.astype(np.float64))):
        return False
    if not np.all(np.isfinite(lns.astype(np.float64))):
        return False
    # No repeated points
    seen_pts = set()
    for p in pts:
        t = (int(p[0]), int(p[1]))
        if t in seen_pts:
            return False
        seen_pts.add(t)
    # No invalid or duplicate lines
    seen_lines = set()
    for l in lns:
        A, B, C = int(l[0]), int(l[1]), int(l[2])
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        if norm is None:
            return False
        if norm in seen_lines:
            return False
        seen_lines.add(norm)
    return True

def _build_6x9():
    """Classical Szemeredi-Trotter k=3 construction."""
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    lines = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            lines.append((a, -1, b))
    lines_arr = np.array(lines, dtype=np.int64)
    return points, lines_arr

def _build_grid6x9():
    """6x9 grid with top-27 lines by incidence count."""
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    candidates = []
    # Vertical
    for c in range(6):
        candidates.append((1, 0, -c))
    # Horizontal
    for r in range(9):
        candidates.append((0, 1, -r))
    # Diagonals y - x = d
    for d in range(-5, 6):
        candidates.append((-1, 1, d))
    # Anti-diagonals x + y = s
    for s in range(0, 15):
        candidates.append((1, 1, -s))
    # Steeper slopes
    for slope in [2, -2]:
        for c in range(-9, 9):
            candidates.append((slope, -1, c))

    # Score each candidate
    scores = []
    for idx, (A, B, C) in enumerate(candidates):
        val = A * points[:, 0] + B * points[:, 1] + C
        cnt = int(np.sum(val == 0))
        scores.append((cnt, idx))
    scores.sort(reverse=True)
    top = [candidates[idx] for _, idx in scores[:27]]
    lines_arr = np.array(top, dtype=np.int64)
    return points, lines_arr

def _generate_candidate_pool(points):
    """Generate a pool of candidate lines based on the current points."""
    pts = points.astype(np.int64)
    candidates = set()
    
    # Add lines through pairs of points
    n = len(pts)
    # Limit pairs to keep pool size manageable, or use all if n is small
    # 54 points -> 1431 pairs. This is fine.
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            norm = _normalize_line(A, B, C)
            if norm:
                candidates.add(norm)
                
    # Add some standard grid lines in the vicinity
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)
    
    # Vertical lines
    for x in range(min_x - 2, max_x + 3):
        candidates.add((1, 0, -x))
    # Horizontal lines
    for y in range(min_y - 2, max_y + 3):
        candidates.add((0, 1, -y))
    # Diagonals
    for d in range(min_y - max_x - 2, max_y - min_x + 3):
        candidates.add((-1, 1, d))
    for s in range(min_x + min_y - 2, max_x + max_y + 3):
        candidates.add((1, 1, -s))
        
    return list(candidates)

def _simulated_annealing(points, lines, candidate_pool, deadline):
    """Run Simulated Annealing to improve incidences."""
    best_pts = points.copy()
    best_lns = lines.copy()
    current_pts = points.copy()
    current_lns = lines.copy()
    
    current_inc = _count_incidences(current_pts, current_lns)
    best_inc = current_inc
    
    # Pre-compute incidence matrix: inc[i, j] is 1 if point i is on line j
    # This helps in fast updates? Actually, recalculating for a single line/point is fast enough for 54x27.
    
    T = 1.0
    T_min = 1e-4
    cooling = 0.999
    
    start_time = time.time()
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.5:
            # Move a point
            idx = random.randint(0, 53)
            old_pt = current_pts[idx].copy()
            dx = random.choice([-2, -1, 1, 2])
            dy = random.choice([-2, -1, 1, 2])
            new_pt = old_pt.copy()
            new_pt[0] += dx
            new_pt[1] += dy
            
            # Check for duplicate points
            # Simple check: is new_pt in current_pts (excluding idx)?
            # For efficiency, we can just try it and validate at the end, or do a quick check.
            # Given 54 points, a set check is fast.
            pts_set = set(map(tuple, current_pts.tolist()))
            if tuple(new_pt) in pts_set and not np.array_equal(new_pt, old_pt):
                continue
            if tuple(new_pt) in pts_set:
                # If it's the same point, no change
                continue
            
            current_pts[idx] = new_pt
            
            # Calculate new incidence count
            # We need to recalculate incidences for all lines for this point
            new_inc = 0
            for j in range(27):
                A, B, C = current_lns[j]
                val = A * new_pt[0] + B * new_pt[1] + C
                if val == 0:
                    new_inc += 1
            
            # We need to subtract the old incidences for this point
            # It's easier to just recalculate total incidence if we don't track per-point-line state
            # But 27 lines * 54 points is small. Let's just recalculate total.
            # Actually, we can optimize:
            # old_point_contrib = sum(1 for line in lines if point on line)
            # new_point_contrib = sum(1 for line in lines if new_point on line)
            # delta = new_point_contrib - old_point_contrib
            
            old_contrib = 0
            for j in range(27):
                A, B, C = current_lns[j]
                val = A * old_pt[0] + B * old_pt[1] + C
                if val == 0:
                    old_contrib += 1
            
            delta = new_inc - old_contrib
            new_total = current_inc + delta
            
            if new_total > current_inc or random.random() < math.exp((new_total - current_inc) / T):
                current_inc = new_total
                if new_total > best_inc:
                    best_inc = new_total
                    best_pts = current_pts.copy()
                    best_lns = current_lns.copy()
            else:
                current_pts[idx] = old_pt
                
        else:
            # Move a line
            idx = random.randint(0, 26)
            old_line = current_lns[idx].copy()
            
            # Pick a new line from candidate pool
            # Ensure it's not already in current_lns
            current_lines_set = set(map(tuple, current_lns.tolist()))
            new_line = None
            attempts = 0
            while attempts < 10:
                cand = random.choice(candidate_pool)
                if cand not in current_lines_set:
                    new_line = np.array(cand, dtype=np.int64)
                    break
                attempts += 1
            
            if new_line is None:
                continue
                
            # Calculate incidence for new line
            A, B, C = new_line
            new_contrib = int(np.sum(A * current_pts[:, 0] + B * current_pts[:, 1] + C == 0))
            
            # Calculate incidence for old line
            A, B, C = old_line
            old_contrib = int(np.sum(A * current_pts[:, 0] + B * current_pts[:, 1] + C == 0))
            
            delta = new_contrib - old_contrib
            new_total = current_inc + delta
            
            if new_total > current_inc or random.random() < math.exp((new_total - current_inc) / T):
                current_lns[idx] = new_line
                current_inc = new_total
                if new_total > best_inc:
                    best_inc = new_total
                    best_pts = current_pts.copy()
                    best_lns = current_lns.copy()
                    # Update candidate pool occasionally?
                    # If we have time, we might regenerate pool, but it's expensive.
                    # Let's stick to the initial pool + maybe some local additions.
            else:
                pass # Revert (implicitly, as we didn't change current_lns)
            
        T *= cooling
        
        # Occasionally regenerate candidate pool to explore new areas
        if random.random() < 0.01:
             # Only do this if we have time
             if time.time() < deadline - 10:
                 candidate_pool = _generate_candidate_pool(best_pts)
                 
    return best_pts, best_lns, best_inc

def szemeredi_trotter_incidences():
    deadline = time.time() + 580.0
    
    # Start with the best static construction
    pts0, lns0 = _build_grid6x9()
    if not _is_valid(pts0, lns0):
        pts0, lns0 = _build_6x9()
        
    best_pts = pts0
    best_lns = lns0
    best_inc = _count_incidences(best_pts, best_lns)
    
    # Generate initial candidate pool
    pool = _generate_candidate_pool(best_pts)
    
    # Run SA
    if time.time() < deadline - 10:
        sa_pts, sa_lns, sa_inc = _simulated_annealing(best_pts, best_lns, pool, deadline)
        if sa_inc > best_inc:
            best_pts = sa_pts
            best_lns = sa_lns
            best_inc = sa_inc
            
    # Final validation and return
    if _is_valid(best_pts, best_lns):
        return best_pts, best_lns
    else:
        # Fallback
        return _build_6x9()
