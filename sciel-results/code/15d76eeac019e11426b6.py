import numpy as np
from math import gcd
import time
import random

def normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) to a canonical form."""
    if A == 0 and B == 0:
        return (0, 0, 0)
    
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
        
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def count_incidences(points, lines):
    """Count total incidences between points and lines."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    x = points[:, 0]
    y = points[:, 1]
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    
    vals = A[:, np.newaxis] * x[np.newaxis, :] + B[:, np.newaxis] * y[np.newaxis, :] + C[:, np.newaxis]
    inc = np.sum(vals == 0)
    return inc

def select_top_lines(points, k=27):
    """Select the k lines with the highest incidence counts."""
    n_pts = len(points)
    lines_set = set()
    
    # Generate all lines from pairs of points
    for i in range(n_pts):
        x1, y1 = points[i]
        for j in range(i + 1, n_pts):
            x2, y2 = points[j]
            A = int(y1 - y2)
            B = int(x2 - x1)
            C = int(x1 * y2 - x2 * y1)
            norm = normalize_line(A, B, C)
            if norm != (0, 0, 0):
                lines_set.add(norm)
                
    if not lines_set:
        return np.empty((0, 3), dtype=np.int64)
        
    lines_list = list(lines_set)
    counts = []
    x = points[:, 0]
    y = points[:, 1]
    
    # Count incidences for each line
    for line in lines_list:
        A, B, C = line
        vals = A * x + B * y + C
        counts.append(np.sum(vals == 0))
        
    # Sort indices by counts descending
    indices = np.argsort(counts)[::-1]
    top_indices = indices[:k]
    selected_lines = np.array([lines_list[i] for i in top_indices], dtype=np.int64)
    
    return selected_lines

def local_search(points, lines, time_limit=100):
    """Perform local search to improve incidences."""
    start_time = time.time()
    current_points = points.copy()
    current_lines = lines.copy()
    current_score = count_incidences(current_points, current_lines)
    
    occupied = set(map(tuple, current_points))
    
    # Precompute neighbors? No, just generate on the fly.
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try moving each point
        for i in range(len(current_points)):
            if time.time() - start_time > time_limit:
                break
                
            x, y = current_points[i]
            best_neighbor = None
            best_neighbor_score = current_score
            
            # Try a small grid around the point
            # Radius 2 or 3
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in occupied:
                        continue
                    
                    # Try the move
                    old_point = current_points[i].copy()
                    current_points[i] = [nx, ny]
                    occupied.remove((x, y))
                    occupied.add((nx, ny))
                    
                    # Recompute top lines and score
                    new_lines = select_top_lines(current_points, 27)
                    new_score = count_incidences(current_points, new_lines)
                    
                    if new_score > best_neighbor_score:
                        best_neighbor_score = new_score
                        best_neighbor = (nx, ny)
                        best_lines = new_lines.copy()
                    
                    # Revert
                    current_points[i] = old_point
                    occupied.remove((nx, ny))
                    occupied.add((x, y))
            
            if best_neighbor is not None:
                # Apply the best move
                x, y = current_points[i]
                occupied.remove((x, y))
                current_points[i] = best_neighbor
                occupied.add(best_neighbor)
                current_lines = best_lines
                current_score = best_neighbor_score
                improved = True
        
        if not improved:
            break
            
    return current_points, current_lines, current_score

def szemeredi_trotter_incidences():
    best_points = None
    best_lines = None
    best_score = -1
    
    start_time = time.time()
    deadline = 600
    
    # 1. Generate initial candidates
    candidates = []
    
    # 6x9 grid
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    p1 = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    candidates.append(p1)
    
    # 7x8 grid minus 2 points
    xs = np.arange(7)
    ys = np.arange(8)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    full_grid = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Remove corners
    # (0,0) is index 0
    # (6,7) is index 55
    p2 = np.delete(full_grid, [0, 55], axis=0)
    candidates.append(p2)
    
    # Remove (0,0) and (6,0)
    p3 = np.delete(full_grid, [0, 6], axis=0) # (0,0) and (1,0)? No, index 6 is (1,0).
    # Index 0: (0,0). Index 6: (1,0).
    # (6,0) is index 6*8+0 = 48.
    p3 = np.delete(full_grid, [0, 48], axis=0)
    candidates.append(p3)
    
    # 3x18 grid
    xs = np.arange(3)
    ys = np.arange(18)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    p4 = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    candidates.append(p4)
    
    # Evaluate candidates
    for cand in candidates:
        if time.time() - start_time > deadline:
            break
        lines = select_top_lines(cand, 27)
        score = count_incidences(cand, lines)
        if score > best_score:
            best_score = score
            best_points = cand.copy()
            best_lines = lines.copy()
            
    # Local search on the best candidate
    if best_points is not None:
        # Use remaining time for local search
        time_spent = time.time() - start_time
        remaining_time = deadline - time_spent - 10 # 10s margin
        if remaining_time > 5:
            best_points, best_lines, best_score = local_search(best_points, best_lines, time_limit=remaining_time)
            
    return best_points, best_lines
