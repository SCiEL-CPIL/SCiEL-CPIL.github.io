import numpy as np
import time
from math import gcd
from itertools import combinations
from collections import Counter

def _normalize_line(A, B, C):
    A, B, C = int(A), int(B), int(C)
    if A == 0 and B == 0:
        return None
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g > 0:
        A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _generate_candidates(points, nx, ny):
    """Generate candidate lines for a grid configuration."""
    candidates = set()
    
    # Vertical lines
    for c in range(nx):
        candidates.add((1, 0, -c))
    
    # Horizontal lines
    for c in range(ny):
        candidates.add((0, 1, -c))
    
    # Diagonal lines with various slopes
    for dx in range(1, nx + 1):
        for dy in range(-(ny - 1), ny):
            if dy == 0:
                continue
            if gcd(dx, abs(dy)) != 1:
                continue
            
            A, B = dy, -dx
            C_vals = dx * points[:, 1] - dy * points[:, 0]
            for C in np.unique(C_vals):
                candidates.add((A, B, int(C)))
    
    # Also try some additional slopes that might capture more structure
    for dx in range(1, min(nx, 5) + 1):
        for dy in range(1, min(ny, 5) + 1):
            if gcd(dx, dy) != 1:
                continue
            # Both positive and negative slopes
            for sign in [1, -1]:
                A, B = sign * dy, -dx
                C_vals = dx * points[:, 1] - sign * dy * points[:, 0]
                for C in np.unique(C_vals):
                    candidates.add((A, B, int(C)))
    
    # Normalize and deduplicate
    unique_lines = []
    seen = set()
    for line in candidates:
        norm = _normalize_line(*line)
        if norm and norm not in seen:
            seen.add(norm)
            unique_lines.append(norm)
    
    return np.array(unique_lines, dtype=np.int64)

def _compute_incidences(lines, points):
    """Compute incidence counts for each line against points."""
    if len(lines) == 0 or len(points) == 0:
        return np.array([])
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    x = points[:, 0]
    y = points[:, 1]
    vals = A[:, None] * x[None, :] + B[:, None] * y[None, :] + C[:, None]
    return np.sum(vals == 0, axis=1)

def _select_top_lines(lines, points, k=27):
    """Select top k lines by incidence count."""
    incidences = _compute_incidences(lines, points)
    if len(incidences) >= k:
        top_indices = np.argsort(-incidences)[:k]
    else:
        top_indices = np.arange(len(incidences))
    return lines[top_indices], incidences[top_indices]

def _local_search(points, lines, deadline):
    """Try swapping lines to improve total incidence."""
    best_score = int(np.sum(_compute_incidences(lines, points)))
    best_lines = lines.copy()
    
    # Compute which lines are selected
    selected_set = set(map(tuple, best_lines.tolist()))
    
    # Get all candidate lines that weren't selected
    # We'll try swapping one selected line with one unselected line
    # But we need the full candidate pool, so this is limited
    
    # Instead, try small perturbations: check if removing a low-incidence line
    # and adding a nearby one helps
    return best_lines, best_score

def _try_grid_config(nx, ny, deadline):
    """Try a specific grid configuration and return best (points, lines, score)."""
    xs = np.arange(nx, dtype=np.int64)
    ys = np.arange(ny, dtype=np.int64)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    base_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    n_pts = len(base_points)
    
    # Generate candidates using full grid
    all_lines = _generate_candidates(base_points, nx, ny)
    
    if n_pts == 54:
        # No removal needed
        best_points = base_points
        selected, _ = _select_top_lines(all_lines, base_points, 27)
        score = int(np.sum(_compute_incidences(selected, base_points)))
        return base_points, selected, score
    
    elif n_pts == 56:
        # Remove 2 points - try all pairs
        best_score = -1
        best_points = None
        best_lines = None
        
        for i, j in combinations(range(n_pts), 2):
            if time.time() > deadline:
                break
            
            mask = np.ones(n_pts, dtype=bool)
            mask[i] = False
            mask[j] = False
            active_points = base_points[mask]
            
            selected, _ = _select_top_lines(all_lines, active_points, 27)
            score = int(np.sum(_compute_incidences(selected, active_points)))
            
            if score > best_score:
                best_score = score
                best_points = active_points.copy()
                best_lines = selected.copy()
        
        if best_points is None:
            best_points = base_points[:54]
            best_lines, _ = _select_top_lines(all_lines, best_points, 27)
            best_score = int(np.sum(_compute_incidences(best_lines, best_points)))
        
        return best_points, best_lines, best_score
    
    elif n_pts == 57:
        # Remove 1 point - try all single removals
        best_score = -1
        best_points = None
        best_lines = None
        
        for i in range(n_pts):
            if time.time() > deadline:
                break
            
            mask = np.ones(n_pts, dtype=bool)
            mask[i] = False
            active_points = base_points[mask]
            
            selected, _ = _select_top_lines(all_lines, active_points, 27)
            score = int(np.sum(_compute_incidences(selected, active_points)))
            
            if score > best_score:
                best_score = score
                best_points = active_points.copy()
                best_lines = selected.copy()
        
        if best_points is None:
            best_points = base_points[:54]
            best_lines, _ = _select_top_lines(all_lines, best_points, 27)
            best_score = int(np.sum(_compute_incidences(best_lines, best_points)))
        
        return best_points, best_lines, best_score
    
    else:
        return None, None, -1

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = 600.0  # Safety margin before 630s kill
    
    # Try multiple grid configurations
    configs = [
        (7, 8),   # 56 points, remove 2
        (6, 9),   # 54 points, no removal
        (8, 7),   # 56 points, remove 2
        (9, 6),   # 54 points, no removal
        (5, 11),  # 55 points, remove 1
        (11, 5),  # 55 points, remove 1
        (5, 10),  # 50 points - too few, skip
        (6, 10),  # 60 points, remove 6 - expensive
        (7, 9),   # 63 points, remove 9 - expensive
    ]
    
    best_points = None
    best_lines = None
    best_score = -1
    
    for nx, ny in configs:
        if time.time() > deadline:
            break
        
        pts, lns, score = _try_grid_config(nx, ny, deadline)
        
        if pts is not None and score > best_score:
            best_score = score
            best_points = pts
            best_lines = lns
    
    # Fallback
    if best_points is None:
        nx, ny = 6, 9
        xs = np.arange(nx, dtype=np.int64)
        ys = np.arange(ny, dtype=np.int64)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        all_lines = _generate_candidates(best_points, nx, ny)
        best_lines, _ = _select_top_lines(all_lines, best_points, 27)
    
    # Final validation and cleanup
    best_points = np.round(best_points).astype(np.int64)
    best_lines = np.round(best_lines).astype(np.int64)
    
    # Ensure exactly 27 unique lines
    seen = set()
    final_lines = []
    for line in best_lines:
        norm = _normalize_line(int(line[0]), int(line[1]), int(line[2]))
        if norm and norm not in seen:
            seen.add(norm)
            final_lines.append(norm)
    
    # If we have fewer than 27, add from all candidates
    if len(final_lines) < 27:
        nx, ny = 6, 9
        xs = np.arange(nx, dtype=np.int64)
        ys = np.arange(ny, dtype=np.int64)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        fallback_pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        all_lines = _generate_candidates(fallback_pts, nx, ny)
        for line in all_lines:
            if len(final_lines) >= 27:
                break
            norm = _normalize_line(int(line[0]), int(line[1]), int(line[2]))
            if norm and norm not in seen:
                seen.add(norm)
                final_lines.append(norm)
    
    best_lines = np.array(final_lines[:27], dtype=np.int64)
    
    return best_points, best_lines
