"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 21

# EVOLVE-BLOCK-START

def _valid(arr):
    """Check if the array represents valid non-overlapping circles in a perimeter-4 rect."""
    if arr is None or len(arr) != N:
        return False
    try:
        xs = arr[:, 0]
        ys = arr[:, 1]
        rs = arr[:, 2]
    except Exception:
        return False
    if np.any(rs <= 0):
        return False
    # Find bounding box
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    if w < 0 or h < 0:
        return False
    perim = 2 * (w + h)
    if perim > 4.0 + 1e-9:
        return False
    # Check non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-7:
                return False
    return True

def _objective(arr, aspect):
    """
    Objective: maximize sum of radii subject to perimeter constraint.
    We parameterize the rectangle by aspect ratio r = w/h.
    Perimeter = 2(w+h) = 4 => w+h = 2.
    w = 2*r/(1+r), h = 2/(1+r).
    
    Actually, let's use a different approach.
    We place circles in a coordinate system where the rectangle is [0, W] x [0, H].
    Constraint: 2*(W+H) <= 4 => W+H <= 2.
    We want to maximize sum(r_i).
    
    Let's fix the rectangle dimensions W, H with W+H=2.
    Circles must be inside: x_i - r_i >= 0, x_i + r_i <= W, y_i - r_i >= 0, y_i + r_i <= H.
    Non-overlap: (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2.
    
    To make it a smooth optimization, we can use a penalty method or SLSQP with constraints.
    Given the complexity, let's use a grid-based initial guess and then local search.
    """
    pass

def _score(arr):
    """Calculate the sum of radii for a valid configuration."""
    if not _valid(arr):
        return -1e9
    return np.sum(arr[:, 2])

def _initial_grid(n, W, H):
    """Create an initial grid of n circles in a W x H rectangle."""
    # Approximate grid
    side = int(np.ceil(np.sqrt(n)))
    rows = side
    cols = side
    # We might have more cells than n, so we take the first n
    cells = []
    for i in range(rows):
        for j in range(cols):
            if len(cells) >= n:
                break
            cx = (j + 0.5) * W / cols
            cy = (i + 0.5) * H / rows
            # Max radius is half the min cell dimension, but we start small
            r = 0.4 * min(W / cols, H / rows)
            cells.append([cx, cy, r])
    return np.array(cells[:n])

def _local_search(arr, W, H, max_iter=1000, step_size=0.01, lr=0.1):
    """
    Simple local search to maximize sum of radii.
    We treat positions and radii as variables.
    Constraints:
    1. Inside rectangle: x_i >= r_i, x_i <= W - r_i, y_i >= r_i, y_i <= H - r_i.
    2. Non-overlap: dist(i,j) >= r_i + r_j.
    
    We use a penalty function for constraints and gradient ascent for sum(r_i).
    """
    arr = arr.copy()
    n = len(arr)
    
    for it in range(max_iter):
        # Calculate forces
        forces = np.zeros_like(arr)
        # Gradient of objective: d(sum r_i)/d r_i = 1
        # So we want to increase r_i.
        # But we must satisfy constraints.
        
        # Penalty for constraints
        penalty = 0.0
        for i in range(n):
            x, y, r = arr[i]
            # Inside rectangle
            if x < r:
                forces[i, 0] += 100 * (r - x)
                penalty += 100 * (r - x) ** 2
            if x > W - r:
                forces[i, 0] -= 100 * (x - (W - r))
                penalty += 100 * (x - (W - r)) ** 2
            if y < r:
                forces[i, 1] += 100 * (r - y)
                penalty += 100 * (r - y) ** 2
            if y > H - r:
                forces[i, 1] -= 100 * (y - (H - r))
                penalty += 100 * (y - (H - r)) ** 2
            
        for i in range(n):
            for j in range(i + 1, n):
                dx = arr[i, 0] - arr[j, 0]
                dy = arr[i, 1] - arr[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = arr[i, 2] + arr[j, 2]
                if dist < min_dist:
                    # Repulsive force
                    if dist > 1e-6:
                        f = 100 * (min_dist - dist)
                        fx = f * dx / dist
                        fy = f * dy / dist
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
                    else:
                        # Random nudge
                        forces[i, 0] += np.random.randn()
                        forces[i, 1] += np.random.randn()
                        forces[j, 0] -= np.random.randn()
                        forces[j, 1] -= np.random.randn()
                    penalty += 100 * (min_dist - dist) ** 2
        
        # Update positions
        arr[:, 0] += lr * forces[:, 0]
        arr[:, 1] += lr * forces[:, 1]
        
        # Increase radii slightly if valid
        # Check validity
        valid = True
        for i in range(n):
            x, y, r = arr[i]
            if x < r or x > W - r or y < r or y > H - r:
                valid = False
                break
        if valid:
            for i in range(n):
                for j in range(i + 1, n):
                    dx = arr[i, 0] - arr[j, 0]
                    dy = arr[i, 1] - arr[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < arr[i, 2] + arr[j, 2]:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            # Increase radii
            for i in range(n):
                # Max possible radius increase is limited by distance to walls and neighbors
                max_inc = min(W - 2 * arr[i, 0], 2 * arr[i, 0], H - 2 * arr[i, 1], 2 * arr[i, 1])
                if max_inc > 0:
                    arr[i, 2] += 0.001
        else:
            # Rollback if invalid
            # This is a simplification; a proper implementation would use line search
            pass
            
    return arr

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to leave 10s margin
    
    best_arr = None
    best_score = -1e9
    
    # Try different aspect ratios
    aspects = [1.0, 1.5, 0.5, 2.0, 0.25, 4.0, 1.2, 0.8]
    
    for aspect in aspects:
        if time.time() > deadline:
            break
            
        W = 2 * aspect / (1 + aspect)
        H = 2 / (1 + aspect)
        
        # Initial grid
        arr = _initial_grid(N, W, H)
        
        # Local search
        arr = _local_search(arr, W, H, max_iter=500)
        
        score = _score(arr)
        if score > best_score:
            best_score = score
            best_arr = arr.copy()
            
    if best_arr is None:
        # Fallback to simple grid in square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_arr

# EVOLVE-BLOCK-END
