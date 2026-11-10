# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd
from itertools import combinations
from collections import defaultdict

def _normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) to a canonical form."""
    if A == 0 and B == 0:
        return (0, 0, 0)
    
    g = gcd(abs(A), abs(B))
    g = gcd(g, abs(C))
    
    if g > 1:
        A //= g
        B //= g
        C //= g
        
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0 and B < 0:
        B = -B
        C = -C
        
    return (A, B, C)

def _build_rich_pool():
    """Build a large pool of candidate lines with diverse slopes."""
    lines_pool = set()
    
    # Axis aligned
    for c in range(-8, 9):
        lines_pool.add(_normalize_line(1, 0, -c))
        lines_pool.add(_normalize_line(0, 1, -c))
    
    # Diagonals slope 1
    for c in range(-15, 16):
        lines_pool.add(_normalize_line(1, 1, -c))
    
    # Diagonals slope -1
    for c in range(-15, 16):
        lines_pool.add(_normalize_line(1, -1, -c))
    
    # Slope 2, 1/2, -2, -1/2
    for c in range(-20, 21):
        lines_pool.add(_normalize_line(2, 1, -c))
        lines_pool.add(_normalize_line(1, 2, -c))
        lines_pool.add(_normalize_line(2, -1, -c))
        lines_pool.add(_normalize_line(1, -2, -c))
    
    # Slope 3, 1/3, -3, -1/3
    for c in range(-20, 21):
        lines_pool.add(_normalize_line(3, 1, -c))
        lines_pool.add(_normalize_line(1, 3, -c))
        lines_pool.add(_normalize_line(3, -1, -c))
        lines_pool.add(_normalize_line(1, -3, -c))
    
    # Slope 4, 1/4
    for c in range(-20, 21):
        lines_pool.add(_normalize_line(4, 1, -c))
        lines_pool.add(_normalize_line(1, 4, -c))
    
    # Other slopes
    for c in range(-20, 21):
        lines_pool.add(_normalize_line(2, 3, -c))
        lines_pool.add(_normalize_line(3, 2, -c))
        lines_pool.add(_normalize_line(2, -3, -c))
        lines_pool.add(_normalize_line(3, -2, -c))
    
    return list(lines_pool)

def _fast_evaluate(lines_set, num_points=54):
    """
    Fast evaluation: find all integer intersections, compute degrees,
    select top num_points, return score.
    Uses vectorized operations where possible.
    """
    if not lines_set:
        return 0, []
    
    ls = list(lines_set)
    n_l = len(ls)
    
    # Find all integer intersection points
    candidates = set()
    for i in range(n_l):
        A1, B1, C1 = ls[i]
        for j in range(i + 1, n_l):
            A2, B2, C2 = ls[j]
            det = A1 * B2 - A2 * B1
            if det == 0:
                continue
            num_x = B1 * C2 - B2 * C1
            num_y = A2 * C1 - A1 * C2
            
            if num_x % det == 0 and num_y % det == 0:
                x = num_x // det
                y = num_y // det
                if -60 <= x <= 60 and -60 <= y <= 60:
                    candidates.add((x, y))
                    
    if not candidates:
        return 0, []
    
    # Vectorize: compute incidence matrix
    pts = list(candidates)
    n_pts = len(pts)
    
    if n_pts == 0:
        return 0, []
    
    # Build point array
    px = np.array([p[0] for p in pts], dtype=np.int64)
    py = np.array([p[1] for p in pts], dtype=np.int64)
    
    # Build line array
    lA = np.array([l[0] for l in ls], dtype=np.int64)
    lB = np.array([l[1] for l in ls], dtype=np.int64)
    lC = np.array([l[2] for l in ls], dtype=np.int64)
    
    # Compute all incidences: (n_pts, n_l)
    vals = px[:, None] * lA[None, :] + py[:, None] * lB[None, :] + lC[None, :]
    inc_matrix = (vals == 0)
    
    # Degree of each point
    degrees = inc_matrix.sum(axis=1)
    
    # Select top num_points by degree
    if n_pts <= num_points:
        selected_indices = list(range(n_pts))
        score = int(degrees.sum())
    else:
        # Get top num_points
        top_indices = np.argsort(degrees)[::-1][:num_points]
        selected_indices = list(top_indices)
        score = int(degrees[top_indices].sum())
    
    selected_points = [pts[i] for i in selected_indices]
    
    return score, selected_points

def _evaluate_with_padding(lines_set, num_points=54):
    """Evaluate and ensure we can form exactly num_points."""
    score, points = _fast_evaluate(lines_set, num_points)
    return score, points

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    """
    start_time = time.time()
    deadline = start_time + 590  # Leave 40s margin before 630s kill
    
    pool = _build_rich_pool()
    pool_set = set(pool)
    
    # Initial selection: strong grid core + diverse additions
    initial_lines = set()
    for c in range(7):
        initial_lines.add(_normalize_line(1, 0, -c))  # x=0..6
        initial_lines.add(_normalize_line(0, 1, -c))  # y=0..6
    
    # Add 13 more lines, biased towards diagonals and other useful slopes
    remaining_pool = [l for l in pool if l not in initial_lines]
    random.shuffle(remaining_pool)
    initial_lines.update(remaining_pool[:13])
    
    current_lines = set(initial_lines)
    current_score, current_points = _fast_evaluate(current_lines)
    best_score = current_score
    best_lines = set(current_lines)
    best_points = list(current_points)
    
    # Local Search with smart swap strategy
    steps = 0
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Reset: try a different random restart
            no_improve_count = 0
            current_lines = set(best_lines)
            current_score = best_score
        
        # Strategy: try multiple swaps per iteration
        n_swaps = 3
        improved = False
        
        for _ in range(n_swaps):
            if time.time() > deadline:
                break
            
            # Pick line to remove: prefer lines with lower marginal contribution
            # Approximate: remove random line from current
            l_out = random.choice(list(current_lines))
            
            # Pick line to add: prefer lines not in current, from pool
            candidates_in = [l for l in pool if l not in current_lines]
            if not candidates_in:
                break
            l_in = random.choice(candidates_in)
            
            new_set = (current_lines - {l_out}) | {l_in}
            new_score, new_points = _fast_evaluate(new_set)
            
            if new_score > current_score:
                current_lines = new_set
                current_score = new_score
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        if current_score > best_score:
            best_score = current_score
            best_lines = set(current_lines)
            best_points = list(current_points)
        
        steps += 1
        
        # Every 200 steps, try a larger perturbation
        if steps % 200 == 0:
            # Remove 3 random lines, add 3 new ones
            to_remove = random.sample(list(current_lines), 3)
            candidates_in = [l for l in pool if l not in current_lines]
            if len(candidates_in) >= 3:
                to_add = random.sample(candidates_in, 3)
                perturbed = (current_lines - set(to_remove)) | set(to_add)
                pert_score, pert_points = _fast_evaluate(perturbed)
                if pert_score > current_score:
                    current_lines = perturbed
                    current_score = pert_score
                    if current_score > best_score:
                        best_score = current_score
                        best_lines = set(current_lines)
                        best_points = list(pert_points)
    
    # Finalize: ensure exactly 54 points
    final_points = list(best_points)
    if len(final_points) < 54:
        used = set(final_points)
        # Add points from intersections of best_lines that weren't selected
        ls = list(best_lines)
        all_intersections = set()
        for i in range(len(ls)):
            A1, B1, C1 = ls[i]
            for j in range(i + 1, len(ls)):
                A2, B2, C2 = ls[j]
                det = A1 * B2 - A2 * B1
                if det == 0:
                    continue
                num_x = B1 * C2 - B2 * C1
                num_y = A2 * C1 - A1 * C2
                if num_x % det == 0 and num_y % det == 0:
                    x = num_x // det
                    y = num_y // det
                    if -60 <= x <= 60 and -60 <= y <= 60:
                        all_intersections.add((x, y))
        
        for p in all_intersections:
            if len(final_points) >= 54:
                break
            if p not in used:
                final_points.append(p)
                used.add(p)
        
        # Still need more? Pad with random
        while len(final_points) < 54:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in used:
                final_points.append(p)
                used.add(p)
    else:
        final_points = final_points[:54]
    
    points_arr = np.array(final_points, dtype=np.int64)
    lines_arr = np.array(list(best_lines), dtype=np.int64)
    
    # Validation
    if points_arr.shape != (54, 2) or lines_arr.shape != (27, 3):
        # Fallback: Simple grid
        pts = []
        for i in range(6):
            for j in range(9):
                pts.append((i, j))
        points_arr = np.array(pts, dtype=np.int64)
        
        ls = []
        for i in range(6):
            ls.append((1, 0, -i))
        for j in range(9):
            ls.append((0, 1, -j))
        # Need 27 lines, have 15. Add 12 more.
        for c in range(12):
            ls.append(_normalize_line(1, 1, -(c)))
        lines_arr = np.array(ls, dtype=np.int64)
        
        # Re-validate
        if points_arr.shape != (54, 2) or lines_arr.shape != (27, 3):
            # Last resort
            pts = [(i, j) for i in range(6) for j in range(9)]
            points_arr = np.array(pts, dtype=np.int64)
            ls = [(1, 0, -i) for i in range(6)] + [(0, 1, -j) for j in range(9)]
            for c in range(12):
                ls.append((1, 1, -c))
            lines_arr = np.array(ls[:27], dtype=np.int64)
    
    return points_arr, lines_arr

# EVOLVE-BLOCK-END
