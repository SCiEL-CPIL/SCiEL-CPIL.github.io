# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from itertools import combinations

def normalize_line(A, B, C):
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def get_best_lines(points):
    n = len(points)
    lines = {}
    # Generate all lines through pairs
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            key = normalize_line(A, B, C)
            if key not in lines:
                lines[key] = 0
            
    # Count incidences
    for key in lines:
        A, B, C = key
        count = 0
        for x, y in points:
            if A * x + B * y + C == 0:
                count += 1
        lines[key] = count
        
    # Sort and get top 27
    sorted_lines = sorted(lines.items(), key=lambda x: -x[1])
    best_lines = [l for l, c in sorted_lines[:27]]
    total_score = sum(c for l, c in sorted_lines[:27])
    return np.array(best_lines, dtype=np.int64), total_score

def evaluate_configuration(points):
    # Check validity: unique points
    if len(set(map(tuple, points))) != len(points):
        return -1, None
    lines, score = get_best_lines(points)
    return score, lines

def local_search(points, time_limit):
    start_time = time.time()
    current_points = points.copy()
    current_score, current_lines = evaluate_configuration(current_points)
    
    if current_score < 0:
        return current_points, current_lines, current_score

    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(len(current_points)):
            if (time.time() - start_time) > time_limit:
                break
            x, y = current_points[i]
            # Try moving to neighbors
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = x + dx, y + dy
                    # Create new points array
                    new_points = current_points.copy()
                    new_points[i] = [new_x, new_y]
                    
                    # Check uniqueness
                    if len(set(map(tuple, new_points))) != len(new_points):
                        continue
                    
                    new_score, new_lines = evaluate_configuration(new_points)
                    if new_score > current_score:
                        current_points = new_points
                        current_score = new_score
                        current_lines = new_lines
                        improved = True
                        # Restart scan from 0 or continue? Continuing is fine, but restart might find more.
                        # Let's just break out of neighbor loops for this point to re-evaluate neighbors of new pos?
                        # Actually, just accepting and continuing the loop for this point's new neighbors is tricky.
                        # Simpler: accept and break out to restart the outer loop? 
                        # Or just mark improved and continue checking other points.
                        # Let's break out of the dx/dy loops for this point to avoid complex state management.
                        break
                if improved:
                    break
            # If we made a move, we should probably restart the scan to find more improvements
            # because the landscape changed.
            # But restarting is expensive. Let's just continue and see if we hit a local optimum.
            # Actually, if we move point i, it might affect the best lines for other points.
            # A standard hill climb: find one improvement, accept, and restart the search for next improvement.
            if improved:
                break
                
    return current_points, current_lines, current_score

def generate_grid(a, b):
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points

def generate_grid_minus(a, b, remove_indices):
    points = generate_grid(a, b)
    # Remove points
    mask = np.ones(len(points), dtype=bool)
    for idx in remove_indices:
        mask[idx] = False
    return points[mask]

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600
    
    # Candidate initial configurations
    candidates = []
    
    # Standard grids
    for a, b in [(6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2), (1, 54), (54, 1)]:
        candidates.append(generate_grid(a, b))
        
    # 7x8 grid (56 points) minus 2
    # Try removing corners (0,0), (55,55) equivalent in flat index
    # Indices: 0, 55
    candidates.append(generate_grid_minus(7, 8, [0, 55]))
    # Try removing (0,1), (54,55)?
    candidates.append(generate_grid_minus(7, 8, [1, 54]))
    
    # 5x11 grid (55 points) minus 1
    candidates.append(generate_grid_minus(5, 11, [0]))
    candidates.append(generate_grid_minus(5, 11, [27])) # Middle
    
    best_points = None
    best_lines = None
    best_score = -1
    
    for points in candidates:
        if len(points) != 54:
            continue
            
        # Evaluate initial
        score, lines = evaluate_configuration(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            best_lines = lines
            
        # Time check
        if time.time() > deadline - 100:
            break
            
    # Local search on the best candidate found so far
    # We can also run local search on top few candidates if time permits
    # Let's run on the best one
    if best_points is not None:
        # Reserve 10s for final checks
        remaining_time = deadline - time.time() - 10
        if remaining_time > 10:
            best_points, best_lines, best_score = local_search(best_points, remaining_time)
            
    # Fallback if something went wrong
    if best_points is None:
        best_points = generate_grid(6, 9)
        best_lines, _ = get_best_lines(best_points)
        
    return best_points, best_lines

# EVOLVE-BLOCK-END
