# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from math import gcd

def _normalize_line(A, B, C):
    """Normalize a line so it's unique: divide by gcd, make first nonzero positive."""
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count exact integer incidences using vectorized computation."""
    # points: (N, 2) int64 array
    # lines: (M, 3) int64 array
    # For each (point, line) pair, check if A*x + B*y + C == 0
    # Use matrix multiplication: [x, y, 1] @ [A; B; C] == 0
    ones = np.ones((len(points), 1), dtype=np.int64)
    P = np.hstack([points, ones])  # (N, 3)
    L = lines  # (M, 3)
    # Compute all dot products: P @ L.T gives (N, M) matrix
    dots = P.astype(np.int64) @ L.T.astype(np.int64)
    return int(np.sum(dots == 0))

def _line_incidence_counts(points, lines_list):
    """For a list of lines, return an array of incidence counts (number of points on each line)."""
    if len(lines_list) == 0:
        return np.array([], dtype=np.int64)
    lines_arr = np.array(lines_list, dtype=np.int64)
    ones = np.ones((len(points), 1), dtype=np.int64)
    P = np.hstack([points, ones])  # (N, 3)
    dots = P.astype(np.int64) @ lines_arr.T.astype(np.int64)  # (N, M)
    return np.sum(dots == 0, axis=0).astype(np.int64)

def _generate_candidate_lines(points):
    """Generate all lines through pairs of points."""
    lines = set()
    n = len(points)
    for i in range(n):
        x1, y1 = int(points[i][0]), int(points[i][1])
        for j in range(i + 1, n):
            x2, y2 = int(points[j][0]), int(points[j][1])
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            norm = _normalize_line(A, B, C)
            if norm != (0, 0, 0):
                lines.add(norm)
    return list(lines)

def _select_best_lines(points, num_lines, all_candidate_lines, time_budget=300.0):
    """Select best lines using vectorized greedy + local search with 1-swap and 2-swap moves."""
    start_time = time.time()
    
    # Precompute incidence counts for all candidate lines
    line_counts = _line_incidence_counts(points, all_candidate_lines)
    
    # Sort by count descending
    sorted_indices = np.argsort(line_counts, kind='stable')[::-1]
    sorted_lines = [all_candidate_lines[i] for i in sorted_indices]
    
    # Initial selection: top num_lines by individual incidence count
    selected = sorted_lines[:num_lines]
    selected_set = set(selected)
    
    # Compute current total
    current_score = int(np.sum(line_counts[:num_lines]))
    
    # Local search: 1-swap moves
    improved = True
    while improved:
        if time.time() - start_time > time_budget:
            break
        improved = False
        
        # For each position in selected, try replacing with a better line
        for i in range(len(selected)):
            if time.time() - start_time > time_budget:
                break
            
            old_line = selected[i]
            old_A, old_B, old_C = old_line
            old_count = int(line_counts[all_candidate_lines.index(old_line)]) if old_line in all_candidate_lines else 0
            
            # Find best replacement line not already selected
            best_new_idx = -1
            best_new_count = -1
            
            for j, line in enumerate(all_candidate_lines):
                if line in selected_set:
                    continue
                new_count = int(line_counts[j])
                if new_count > best_new_count:
                    best_new_count = new_count
                    best_new_idx = j
            
            if best_new_idx >= 0 and best_new_count > old_count:
                old_line = selected[i]
                selected[i] = all_candidate_lines[best_new_idx]
                selected_set.remove(old_line)
                selected_set.add(all_candidate_lines[best_new_idx])
                current_score += best_new_count - old_count
                improved = True
                break
    
    # 2-swap polish pass: try swapping two lines at once
    if time.time() - start_time < time_budget:
        improved2 = True
        while improved2 and time.time() - start_time < time_budget:
            improved2 = False
            
            for i in range(len(selected)):
                if time.time() - start_time > time_budget:
                    break
                for k in range(i + 1, len(selected)):
                    if time.time() - start_time > time_budget:
                        break
                    
                    old_line1 = selected[i]
                    old_line2 = selected[k]
                    old_count1 = int(line_counts[all_candidate_lines.index(old_line1)]) if old_line1 in all_candidate_lines else 0
                    old_count2 = int(line_counts[all_candidate_lines.index(old_line2)]) if old_line2 in all_candidate_lines else 0
                    
                    # Find two best replacements
                    candidates = [(j, line_counts[j]) for j, line in enumerate(all_candidate_lines) 
                                  if line not in selected_set]
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    
                    best_pair = None
                    best_delta = 0
                    for j1, c1 in candidates[:10]:  # Only check top 10 candidates for speed
                        for j2, c2 in candidates[:10]:
                            if j1 == j2:
                                continue
                            delta = (c1 + c2) - (old_count1 + old_count2)
                            if delta > best_delta:
                                best_delta = delta
                                best_pair = (j1, j2)
                    
                    if best_pair is not None and best_delta > 0:
                        j1, j2 = best_pair
                        selected[i] = all_candidate_lines[j1]
                        selected[k] = all_candidate_lines[j2]
                        selected_set.remove(old_line1)
                        selected_set.remove(old_line2)
                        selected_set.add(all_candidate_lines[j1])
                        selected_set.add(all_candidate_lines[j2])
                        current_score += best_delta
                        improved2 = True
                        break
    
    return selected

def _generate_grid_points(nx, ny):
    """Generate grid points for given dimensions."""
    xs = np.arange(1, nx + 1)
    ys = np.arange(1, ny + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points

def _validate_result(points, lines):
    """Check if the result is valid according to the task contract."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    
    # Check for NaN/inf
    if not np.all(np.isfinite(points.astype(float))):
        return False
    if not np.all(np.isfinite(lines.astype(float))):
        return False
    
    # Check entries are close to integers
    if not np.allclose(points, np.round(points), atol=1e-6):
        return False
    if not np.allclose(lines, np.round(lines), atol=1e-6):
        return False
    
    # Check no repeated points
    point_set = set(map(tuple, points.tolist()))
    if len(point_set) != 54:
        return False
    
    # Check no A=B=0 and no duplicate lines
    norm_lines = []
    for line in lines:
        A, B, C = int(line[0]), int(line[1]), int(line[2])
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        norm_lines.append(norm)
    
    if len(set(norm_lines)) != 27:
        return False
    
    return True

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    Searches over grid dimensions and selects optimal lines using vectorized local search.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit
    
    best_points = None
    best_lines = None
    best_score = 0
    
    # Try different grid dimensions that multiply to 54
    grid_dims = [(6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2)]
    
    for (nx, ny) in grid_dims:
        if time.time() > deadline:
            break
        
        # Generate grid points
        points = _generate_grid_points(nx, ny)
        
        if points.shape != (54, 2):
            continue
        
        # Generate all candidate lines
        all_lines = _generate_candidate_lines(points)
        
        if len(all_lines) == 0:
            continue
        
        # Select best 27 lines with vectorized local search
        time_budget = min(90, deadline - time.time())
        if time_budget < 10:
            break
            
        selected = _select_best_lines(points, 27, all_lines, time_budget=time_budget)
        
        if len(selected) < 27:
            continue
        
        # Check for duplicates and pad if needed
        seen = set()
        unique_selected = []
        for line in selected:
            if line not in seen:
                seen.add(line)
                unique_selected.append(line)
        
        if len(unique_selected) < 27:
            for line in all_lines:
                if len(unique_selected) >= 27:
                    break
                if line not in seen:
                    unique_selected.append(line)
                    seen.add(line)
        
        if len(unique_selected) < 27:
            continue
        
        lines = np.array(unique_selected[:27], dtype=np.int64)
        
        # Verify validity
        if not _validate_result(points, lines):
            continue
        
        # Count incidences
        score = _count_incidences(points, lines)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
            best_lines = lines.copy()
    
    # Fallback to original if search didn't find anything
    if best_points is None:
        points = _generate_grid_points(6, 9)
        
        lines_list = []
        for i in range(1, 7):
            lines_list.append((1, 0, -i))
        for j in range(1, 10):
            lines_list.append((0, 1, -j))
        for k in [0, 1, 2, 3]:
            lines_list.append((-1, 1, -k))
        for k in [-1, 4]:
            lines_list.append((-1, 1, -k))
        for k in [7, 8, 9, 10]:
            lines_list.append((1, 1, -k))
        for k in [6, 11]:
            lines_list.append((1, 1, -k))
        best_lines = np.array(lines_list, dtype=np.int64)
        best_points = points
    
    assert best_points.shape == (54, 2)
    assert best_lines.shape == (27, 3)
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
