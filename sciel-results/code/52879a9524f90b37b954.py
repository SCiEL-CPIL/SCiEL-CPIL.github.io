import numpy as np
import time
import random
import math
from itertools import combinations
from collections import defaultdict

def _canon_line(A, B, C):
    """Canonicalize a line (A, B, C) -> (A, B, C) such that gcd(|A|,|B|,|C|)=1 and first non-zero is positive."""
    if A == 0 and B == 0:
        return None
    g = math.gcd(abs(A), math.gcd(abs(B), abs(C)))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count incidences exactly."""
    I = 0
    for x, y in points:
        for A, B, C in lines:
            if A * x + B * y + C == 0:
                I += 1
    return I

def _build_candidate_lines(points):
    """Build all candidate lines determined by pairs of points."""
    candidates = {}
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            A = dy
            B = -dx
            C = dx*yi - dy*xi
            cl = _canon_line(A, B, C)
            if cl is not None:
                if cl not in candidates:
                    candidates[cl] = set()
    # Also add horizontal and vertical lines through each point
    for i in range(n):
        xi, yi = points[i]
        cl = _canon_line(0, 1, -yi)
        if cl is not None and cl not in candidates:
            candidates[cl] = set()
        cl = _canon_line(1, 0, -xi)
        if cl is not None and cl not in candidates:
            candidates[cl] = set()
    # Compute incidence sets
    for cl in list(candidates.keys()):
        A, B, C = cl
        for i in range(n):
            xi, yi = points[i]
            if A*xi + B*yi + C == 0:
                candidates[cl].add(i)
    return candidates

def _greedy_select_lines(points, candidates, num_lines=27):
    """Greedy select top lines by incidence count."""
    sorted_lines = sorted(candidates.items(), key=lambda x: len(x[1]), reverse=True)
    selected = []
    for cl, pts_on_line in sorted_lines[:num_lines]:
        selected.append(cl)
    return selected

def _evaluate_config(points, lines):
    """Evaluate total incidences."""
    I = 0
    for x, y in points:
        for A, B, C in lines:
            if A*x + B*y + C == 0:
                I += 1
    return I

def _local_search(points, lines, candidates, max_iters=5000, deadline=None):
    """Local search that can swap points or lines."""
    points = list(points)
    lines = list(lines)
    n_pts = len(points)
    n_lines = len(lines)
    
    # Build incidence matrix
    M = np.zeros((n_pts, n_lines), dtype=np.int8)
    for i in range(n_pts):
        xi, yi = points[i]
        for j in range(n_lines):
            A, B, C = lines[j]
            if A*xi + B*yi + C == 0:
                M[i, j] = 1
    current_I = int(np.sum(M))
    best_I = current_I
    best_points = [tuple(p) for p in points]
    best_lines = [tuple(l) for l in lines]
    
    point_set = set(points)
    line_set = set(lines)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        if random.random() < 0.5:
            i = random.randint(0, n_pts-1)
            old_x, old_y = points[i]
            
            found = False
            for _ in range(15):
                dx = random.randint(-4, 4)
                dy = random.randint(-4, 4)
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = old_x + dx, old_y + dy
                new_pt = (new_x, new_y)
                if new_pt in point_set:
                    continue
                
                delta = 0
                for j in range(n_lines):
                    A, B, C = lines[j]
                    old_inc = M[i, j]
                    new_inc = 1 if (A*new_x + B*new_y + C == 0) else 0
                    delta += new_inc - old_inc
                
                if delta > 0:
                    point_set.remove((old_x, old_y))
                    points[i] = new_pt
                    point_set.add(new_pt)
                    for j in range(n_lines):
                        A, B, C = lines[j]
                        M[i, j] = 1 if (A*new_x + B*new_y + C == 0) else 0
                    current_I += delta
                    if current_I > best_I:
                        best_I = current_I
                        best_points = [tuple(p) for p in points]
                        best_lines = [tuple(l) for l in lines]
                    found = True
                    break
            if not found:
                continue
        else:
            j = random.randint(0, n_lines-1)
            old_line = lines[j]
            
            if not candidates:
                continue
            
            cand_keys = list(candidates.keys())
            found = False
            for _ in range(8):
                if not cand_keys:
                    break
                new_line = random.choice(cand_keys)
                if new_line == old_line or new_line in line_set:
                    continue
                
                delta = 0
                A, B, C = new_line
                for i in range(n_pts):
                    xi, yi = points[i]
                    old_inc = M[i, j]
                    new_inc = 1 if (A*xi + B*yi + C == 0) else 0
                    delta += new_inc - old_inc
                
                if delta > 0:
                    line_set.remove(old_line)
                    lines[j] = new_line
                    line_set.add(new_line)
                    for i in range(n_pts):
                        xi, yi = points[i]
                        M[i, j] = 1 if (A*xi + B*yi + C == 0) else 0
                    current_I += delta
                    if current_I > best_I:
                        best_I = current_I
                        best_points = [tuple(p) for p in points]
                        best_lines = [tuple(l) for l in lines]
                    found = True
                    break
            if not found:
                continue
    
    return best_points, best_lines, best_I

def _seed_grid():
    """The baseline Szemeredi-Trotter grid."""
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    points = [tuple(p) for p in points]
    
    lines = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            lines.append(_canon_line(a, -1, b))
    return points, lines

def _seed_dense_cluster():
    """Create a dense cluster with high line multiplicity."""
    points = []
    for x in range(-3, 3):
        for y in range(-4, 5):
            points.append((x, y))
    
    lines = set()
    for y in range(-4, 5):
        lines.add(_canon_line(0, 1, -y))
    for x in range(-3, 3):
        lines.add(_canon_line(1, 0, -x))
    for d in range(-7, 8):
        lines.add(_canon_line(1, -1, -d))
    
    candidates = {}
    for cl in lines:
        A, B, C = cl
        count = sum(1 for (x, y) in points if A*x + B*y + C == 0)
        candidates[cl] = count
    
    sorted_lines = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    selected_lines = [cl for cl, _ in sorted_lines[:27]]
    
    return points, selected_lines

def _seed_lattice():
    """Create a lattice structure."""
    points = []
    for x in range(9):
        for y in range(6):
            points.append((x, y))
    
    candidates = {}
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            A = dy
            B = -dx
            C = dx*yi - dy*xi
            cl = _canon_line(A, B, C)
            if cl is not None:
                if cl not in candidates:
                    candidates[cl] = 0
                candidates[cl] += 1
    
    for x in range(9):
        cl = _canon_line(1, 0, -x)
        if cl:
            if cl not in candidates:
                candidates[cl] = 0
            candidates[cl] = max(candidates.get(cl, 0), 6)
    for y in range(6):
        cl = _canon_line(0, 1, -y)
        if cl:
            if cl not in candidates:
                candidates[cl] = 0
            candidates[cl] = max(candidates.get(cl, 0), 9)
    
    sorted_lines = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    selected_lines = [cl for cl, _ in sorted_lines[:27]]
    
    return points, selected_lines

def _seed_hexagonal():
    """Create a hexagonal-like lattice."""
    points = []
    # 9x6 with offset rows
    for y in range(6):
        offset = y % 2
        for x in range(9):
            points.append((x + offset, y))
    
    candidates = _build_candidate_lines(points)
    lines = _greedy_select_lines(points, candidates, 27)
    return points, lines

def _seed_spiral():
    """Create a spiral arrangement."""
    points = []
    # Dense center with expanding rings
    for r in range(0, 8):
        if r == 0:
            points.append((0, 0))
        else:
            # Place points on a circle of radius r
            n_pts_ring = min(2 * r + 1, 54 - len(points))
            for k in range(n_pts_ring):
                angle = 2 * math.pi * k / n_pts_ring
                x = round(r * math.cos(angle))
                y = round(r * math.sin(angle))
                if (x, y) not in points:
                    points.append((x, y))
        if len(points) >= 54:
            break
    
    # Pad if needed
    while len(points) < 54:
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        if (x, y) not in points:
            points.append((x, y))
    
    candidates = _build_candidate_lines(points)
    lines = _greedy_select_lines(points, candidates, 27)
    return points, lines

def _dual_space_optimize(points, deadline):
    """Use dual-space hypergraph approach to optimize."""
    candidates = _build_candidate_lines(points)
    lines = _greedy_select_lines(points, candidates, 27)
    I = _evaluate_config(points, lines)
    points, lines, I = _local_search(points, lines, candidates, max_iters=3000, deadline=deadline)
    return points, lines, I

def _polish_pass(points, lines, deadline):
    """High-precision polish: try all single-point moves and line swaps."""
    points = list(points)
    lines = list(lines)
    n_pts = len(points)
    n_lines = len(lines)
    
    M = np.zeros((n_pts, n_lines), dtype=np.int8)
    for i in range(n_pts):
        xi, yi = points[i]
        for j in range(n_lines):
            A, B, C = lines[j]
            if A*xi + B*yi + C == 0:
                M[i, j] = 1
    current_I = int(np.sum(M))
    best_I = current_I
    best_points = [tuple(p) for p in points]
    best_lines = [tuple(l) for l in lines]
    
    point_set = set(points)
    line_set = set(lines)
    
    # Build candidates
    candidates = _build_candidate_lines(points)
    cand_keys = list(candidates.keys())
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try all single-point moves
        for i in range(n_pts):
            if time.time() > deadline:
                break
            old_x, old_y = points[i]
            best_delta = 0
            best_move = None
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = old_x + dx, old_y + dy
                    new_pt = (new_x, new_y)
                    if new_pt in point_set:
                        continue
                    delta = 0
                    for j in range(n_lines):
                        A, B, C = lines[j]
                        old_inc = M[i, j]
                        new_inc = 1 if (A*new_x + B*new_y + C == 0) else 0
                        delta += new_inc - old_inc
                    if delta > best_delta:
                        best_delta = delta
                        best_move = (new_x, new_y)
            
            if best_move and best_delta > 0:
                point_set.remove((old_x, old_y))
                points[i] = best_move
                point_set.add(best_move)
                for j in range(n_lines):
                    A, B, C = lines[j]
                    M[i, j] = 1 if (A*best_move[0] + B*best_move[1] + C == 0) else 0
                current_I += best_delta
                if current_I > best_I:
                    best_I = current_I
                    best_points = [tuple(p) for p in points]
                    best_lines = [tuple(l) for l in lines]
                improved = True
        
        # Try all single-line swaps
        for j in range(n_lines):
            if time.time() > deadline:
                break
            old_line = lines[j]
            best_delta = 0
            best_line = None
            for new_line in cand_keys:
                if new_line == old_line or new_line in line_set:
                    continue
                A, B, C = new_line
                delta = 0
                for i in range(n_pts):
                    xi, yi = points[i]
                    old_inc = M[i, j]
                    new_inc = 1 if (A*xi + B*yi + C == 0) else 0
                    delta += new_inc - old_inc
                if delta > best_delta:
                    best_delta = delta
                    best_line = new_line
            
            if best_line and best_delta > 0:
                line_set.remove(old_line)
                lines[j] = best_line
                line_set.add(best_line)
                A, B, C = best_line
                for i in range(n_pts):
                    xi, yi = points[i]
                    M[i, j] = 1 if (A*xi + B*yi + C == 0) else 0
                current_I += best_delta
                if current_I > best_I:
                    best_I = current_I
                    best_points = [tuple(p) for p in points]
                    best_lines = [tuple(l) for l in lines]
                improved = True
    
    return best_points, best_lines, best_I

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines.
    """
    start_time = time.time()
    deadline = start_time + 590
    
    best_I = 0
    best_points = None
    best_lines = None
    
    seeds = [
        ("grid", _seed_grid()),
        ("cluster", _seed_dense_cluster()),
        ("lattice", _seed_lattice()),
        ("hex", _seed_hexagonal()),
    ]
    
    for name, (pts, lns) in seeds:
        if time.time() > deadline - 80:
            break
        
        res_pts, res_lns, res_I = _dual_space_optimize(pts, deadline - 60)
        
        if res_I > best_I:
            best_I = res_I
            best_points = res_pts
            best_lines = res_lns
    
    # Polish pass
    if best_points is not None and time.time() < deadline - 20:
        res_pts, res_lns, res_I = _polish_pass(best_points, best_lines, deadline - 5)
        if res_I > best_I:
            best_I = res_I
            best_points = res_pts
            best_lines = res_lns
    
    points_arr = np.array(best_points, dtype=np.int64)
    lines_arr = np.array(best_lines, dtype=np.int64)
    
    return points_arr, lines_arr

# EVOLVE-BLOCK-START
# The code above is the complete program.
# EVOLVE-BLOCK-END
