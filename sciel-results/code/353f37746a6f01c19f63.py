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
    """Generate candidate lines for a grid configuration using hypergraph approach."""
    candidates = set()
    n_pts = len(points)
    
    # Vertical lines
    for c in range(nx):
        candidates.add((1, 0, -c))
    
    # Horizontal lines
    for c in range(ny):
        candidates.add((0, 1, -c))
    
    # Diagonal lines with various slopes - expanded range
    max_dx = min(nx + 2, 12)
    max_dy = min(ny + 2, 12)
    
    for dx in range(1, max_dx + 1):
        for dy in range(-max_dy, max_dy + 1):
            if dy == 0:
                continue
            if gcd(dx, abs(dy)) != 1:
                continue
            
            A, B = dy, -dx
            C_vals = dx * points[:, 1] - dy * points[:, 0]
            for C in np.unique(C_vals):
                candidates.add((A, B, int(C)))
    
    # Additional slopes for better coverage
    for dx in range(1, min(nx, 6) + 1):
        for dy in range(1, min(ny, 6) + 1):
            if gcd(dx, dy) != 1:
                continue
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

def _compute_line_membership(lines, points):
    """Return a boolean matrix where [i,j] is True if point j lies on line i."""
    if len(lines) == 0 or len(points) == 0:
        return np.zeros((len(lines), len(points)), dtype=bool)
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    x = points[:, 0]
    y = points[:, 1]
    vals = A[:, None] * x[None, :] + B[:, None] * y[None, :] + C[:, None]
    return vals == 0

def _select_top_lines(lines, points, k=27):
    """Select top k lines by incidence count."""
    incidences = _compute_incidences(lines, points)
    if len(incidences) >= k:
        top_indices = np.argsort(-incidences)[:k]
    else:
        top_indices = np.arange(len(incidences))
    return lines[top_indices], incidences[top_indices]

def _evaluate_configuration(points, all_lines, k=27):
    """
    Evaluate a point configuration by finding the best k lines.
    Uses hypergraph perspective: maximize sum of top-k edge sizes.
    """
    n_pts = len(points)
    membership = _compute_line_membership(all_lines, points)
    incidences = np.sum(membership, axis=1)
    
    if len(incidences) >= k:
        top_indices = np.argsort(-incidences)[:k]
    else:
        top_indices = np.arange(len(incidences))
    
    total_incidence = int(np.sum(incidences[top_indices]))
    selected_lines = all_lines[top_indices]
    
    return selected_lines, total_incidence

def _local_search(points, lines, deadline):
    """Try swapping lines to improve total incidence."""
    best_score = int(np.sum(_compute_incidences(lines, points)))
    best_lines = lines.copy()
    return best_lines, best_score

def _swap_point_improve(points, all_lines, k=27, deadline=None):
    """
    Hypergraph-based local search: try replacing individual points
    to maximize the sum of top-k line incidences.
    """
    n_pts = len(points)
    nx = int(np.max(points[:, 0])) + 1
    ny = int(np.max(points[:, 1])) + 1
    
    # Generate candidate replacement points (nearby grid positions)
    candidate_positions = set()
    for x in range(max(0, nx - 2), min(nx + 2, nx + 3)):
        for y in range(max(0, ny - 2), min(ny + 2, ny + 3)):
            candidate_positions.add((x, y))
    
    # Remove current points from candidates
    current_set = set(map(tuple, points.tolist()))
    for pt in current_set:
        candidate_positions.discard(pt)
    
    candidate_points = np.array(list(candidate_positions), dtype=np.int64)
    if len(candidate_points) == 0:
        return points, int(np.sum(_compute_incidences(_select_top_lines(all_lines, points, k)[0], points)))
    
    best_score, best_points = -1, points.copy()
    
    # Compute membership for all lines
    membership_all = _compute_line_membership(all_lines, candidate_points)
    line_sizes = np.sum(membership_all, axis=1)
    
    # Try replacing each point with each candidate
    for i in range(n_pts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Remove point i
        mask = np.ones(n_pts, dtype=bool)
        mask[i] = False
        active_points = points[mask]
        
        # Find best replacement
        best_replace_score = -1
        best_replace_pt = None
        
        for j in range(len(candidate_points)):
            if candidate_points[j, 0] == points[i, 0] and candidate_points[j, 1] == points[i, 1]:
                continue
            
            # Compute score if we add candidate_points[j]
            new_points_list = list(active_points) + [candidate_points[j]]
            new_points = np.array(new_points_list, dtype=np.int64)
            
            # Quick estimate: how many top-k lines would this point be on?
            # This is an approximation for speed
            pt_idx_in_candidates = j
            lines_through_pt = np.where(membership_all[:, pt_idx_in_candidates])[0]
            
            # Estimate gain
            if len(lines_through_pt) > 0:
                # Count how many of these are in current top-k
                current_top_k = set(np.argsort(-line_sizes)[:k].tolist())
                overlap = len(set(lines_through_pt.tolist()) & current_top_k)
                est_gain = overlap * 2  # Rough heuristic
                
                if est_gain > 0:
                    # Full evaluation
                    selected, score = _evaluate_configuration(new_points, all_lines, k)
                    if score > best_replace_score:
                        best_replace_score = score
                        best_replace_pt = candidate_points[j].copy()
        
        # Apply best replacement
        if best_replace_pt is not None and best_replace_score > 0:
            new_points_list = list(active_points) + [best_replace_pt]
            new_points = np.array(new_points_list, dtype=np.int64)
            selected, score = _evaluate_configuration(new_points, all_lines, k)
            
            if score > best_score:
                best_score = score
                best_points = new_points
    
    return best_points, best_score

def _try_grid_config(nx, ny, deadline):
    """Try a specific grid configuration using hypergraph optimization."""
    xs = np.arange(nx, dtype=np.int64)
    ys = np.arange(ny, dtype=np.int64)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    base_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    n_pts = len(base_points)
    
    # Generate candidates using full grid
    all_lines = _generate_candidates(base_points, nx, ny)
    
    if n_pts == 54:
        # No removal needed - evaluate directly
        selected, score = _evaluate_configuration(base_points, all_lines, 27)
        return base_points, selected, score
    
    elif n_pts >= 55:
        # Need to remove (n_pts - 54) points
        n_remove = n_pts - 54
        
        # Use greedy removal: remove points that contribute least to top-k lines
        best_score = -1
        best_points = None
        best_lines = None
        
        # Compute initial membership
        membership = _compute_line_membership(all_lines, base_points)
        line_sizes = np.sum(membership, axis=1)
        
        # Greedy removal strategy
        current_points = base_points.copy()
        current_membership = membership.copy()
        current_line_sizes = line_sizes.copy()
        
        for removal_step in range(n_remove):
            if time.time() > deadline:
                break
            
            # For each point, estimate the loss if removed
            best_remove_idx = None
            best_remove_loss = float('inf')
            
            for i in range(len(current_points)):
                # Count how many top-k lines this point is on
                lines_through_pt = np.where(current_membership[:, i])[0]
                if len(lines_through_pt) == 0:
                    loss = 0
                else:
                    # Weight by line size (larger lines are more valuable)
                    top_k_set = set(np.argsort(-current_line_sizes)[:27].tolist())
                    weighted_loss = sum(current_line_sizes[l] for l in lines_through_pt if l in top_k_set)
                    loss = weighted_loss
            
            if loss < best_remove_loss:
                best_remove_loss = loss
                best_remove_idx = i
            
            # Remove the point
            if best_remove_idx is not None:
                mask = np.ones(len(current_points), dtype=bool)
                mask[best_remove_idx] = False
                current_points = current_points[mask]
                
                # Update membership
                current_membership = current_membership[:, mask]
                current_line_sizes = np.sum(current_membership, axis=1)
        
        # Evaluate final configuration
        if len(current_points) == 54:
            selected, score = _evaluate_configuration(current_points, all_lines, 27)
            best_points = current_points
            best_lines = selected
            best_score = score
        
        # Also try some random removals for diversity
        n_random_trials = 50
        for trial in range(n_random_trials):
            if time.time() > deadline:
                break
            
            # Random removal
            remove_indices = np.random.choice(n_pts, size=n_remove, replace=False)
            mask = np.ones(n_pts, dtype=bool)
            mask[remove_indices] = False
            trial_points = base_points[mask]
            
            selected, score = _evaluate_configuration(trial_points, all_lines, 27)
            
            if score > best_score:
                best_score = score
                best_points = trial_points.copy()
                best_lines = selected.copy()
        
        if best_points is None:
            best_points = base_points[:54]
            best_lines, _ = _select_top_lines(all_lines, best_points, 27)
            best_score = int(np.sum(_compute_incidences(best_lines, best_points)))
        
        # Local improvement via point swaps
        if time.time() < deadline - 30:
            best_points, best_score = _swap_point_improve(
                best_points, all_lines, 27, 
                deadline=min(deadline - 10, time.time() + 60)
            )
            if best_score > 0:
                best_lines, _ = _select_top_lines(all_lines, best_points, 27)
        
        return best_points, best_lines, best_score
    
    else:
        return None, None, -1

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = 580.0  # Safety margin before 630s kill
    
    # Try multiple grid configurations
    configs = [
        (6, 9),   # 54 points, no removal - start with exact fit
        (9, 6),   # 54 points, no removal
        (7, 8),   # 56 points, remove 2
        (8, 7),   # 56 points, remove 2
        (5, 11),  # 55 points, remove 1
        (11, 5),  # 55 points, remove 1
        (6, 10),  # 60 points, remove 6
        (10, 6),  # 60 points, remove 6
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
