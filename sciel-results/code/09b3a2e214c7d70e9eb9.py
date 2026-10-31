"""Rewritten program for circle packing in perimeter-4 rectangle using force-directed relaxation."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random


def _initial_layout(n):
    """Generate an initial layout of n circles in a rectangle with perimeter 4."""
    # Rectangle with perimeter 4: 2*(w+h) = 4 => w+h = 2
    # Start with a roughly square-ish rectangle
    w, h = 1.0, 1.0
    
    # Place circles in a grid pattern initially
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    centers = []
    for i in range(rows):
        for j in range(cols):
            if len(centers) >= n:
                break
            x = (j + 0.5) * w / cols
            y = (i + 0.5) * h / rows
            centers.append([x, y])
    
    centers = np.array(centers[:n])
    return centers, w, h


def _evaluate(centers, w, h, radii):
    """Evaluate validity and objective. Returns (is_valid, objective)."""
    n = len(centers)
    
    # Check bounding box perimeter
    if w + h > 2.0 + 1e-6:
        return False, 0.0
    
    # Check all circles are within the rectangle
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > w + 1e-6 or y - r < -1e-6 or y + r > h + 1e-6:
            return False, 0.0
    
    # Check no overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - 1e-6:
                return False, 0.0
    
    # Objective: sum of radii
    return True, np.sum(radii)


def _penalized_objective(params, n, w, h):
    """
    Penalized objective for optimization.
    params layout: [x0, y0, x1, y1, ..., x_{n-1}, y_{n-1}, r0, r1, ..., r_{n-1}]
    We optimize positions and radii, with w and h fixed for now.
    """
    positions = params[:2 * n].reshape(n, 2)
    radii = params[2 * n:]
    
    # Penalty for going outside the rectangle
    penalty = 0.0
    eps = 1e-4
    
    for i in range(n):
        x, y = positions[i]
        r = radii[i]
        # Distance to walls
        if x - r < 0:
            penalty += (x - r) ** 2
        if x + r > w:
            penalty += (x + r - w) ** 2
        if y - r < 0:
            penalty += (y - r) ** 2
        if y + r > h:
            penalty += (y + r - h) ** 2
    
    # Penalty for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    
    # Objective: maximize sum of radii, so minimize negative sum + penalty
    objective = -np.sum(radii) + 100.0 * penalty
    return objective


def _force_directed_relaxation(centers, w, h, radii, iterations=5000, lr=0.01):
    """
    Force-directed relaxation to satisfy distance constraints.
    """
    n = len(centers)
    centers = centers.copy()
    radii = radii.copy()
    
    for it in range(iterations):
        forces = np.zeros_like(centers)
        
        # Repulsive forces between overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-10:
                    # Overlap: push apart
                    overlap = min_dist - dist
                    fx = (dx / dist) * overlap
                    fy = (dy / dist) * overlap
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                elif dist < 1e-10:
                    # Coincident: random push
                    angle = random.uniform(0, 2 * np.pi)
                    fx = np.cos(angle) * 0.01
                    fy = np.sin(angle) * 0.01
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Attractive forces toward wall contacts (to maximize radii)
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Push against walls to maximize radius
            # Left wall
            if x - r < 0.1:
                forces[i, 0] += 0.01
            # Right wall
            if x + r > w - 0.1:
                forces[i, 0] -= 0.01
            # Bottom wall
            if y - r < 0.1:
                forces[i, 1] += 0.01
            # Top wall
            if y + r > h - 0.1:
                forces[i, 1] -= 0.01
        
        # Apply forces with learning rate decay
        current_lr = lr * (1.0 - it / iterations)
        centers += current_lr * forces
        
        # Clamp to rectangle
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], w - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], h - radii[i])
        
        # Gradually increase radii
        radii *= (1.0 + 1e-5)
    
    return centers, radii


def construct():
    """
    Pack 21 circles with disjoint interiors in a variable rectangle of perimeter 4
    and maximize the sum of their radii.
    
    Returns: array of shape (n, 3) with [x, y, r] for each circle.
    """
    n = N
    
    best_solution = None
    best_score = -np.inf
    
    # Try multiple initial configurations
    for trial in range(20):
        random.seed(trial * 12345 + 678)
        np.random.seed(trial * 12345 + 678)
        
        # Different aspect ratios
        if trial < 5:
            w = 1.0
            h = 1.0
        elif trial < 10:
            w = 0.8
            h = 1.2
        elif trial < 15:
            w = 1.2
            h = 0.8
        else:
            w = 0.9
            h = 1.1
        
        # Initial layout
        centers, w, h = _initial_layout(n)
        
        # Initial radii: estimate from grid spacing
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        base_r = min(w / cols, h / rows) * 0.45
        radii = np.full(n, base_r)
        
        # Force-directed relaxation
        centers, radii = _force_directed_relaxation(centers, w, h, radii, iterations=3000, lr=0.05)
        
        # Check validity and get score
        is_valid, score = _evaluate(centers, w, h, radii)
        
        if is_valid and score > best_score:
            best_score = score
            best_solution = np.column_stack([centers, radii])
        
        # Also try scipy optimization from this starting point
        params = np.concatenate([centers.flatten(), radii])
        
        def obj(p):
            pos = p[:2 * n].reshape(n, 2)
            r = p[2 * n:]
            return _penalized_objective(p, n, w, h)
        
        try:
            result = minimize(obj, params, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-8})
            
            if result.success or result.fun < 0:  # Check if improvement
                opt_params = result.x
                opt_centers = opt_params[:2 * n].reshape(n, 2)
                opt_radii = opt_params[2 * n:]
                
                is_valid, score = _evaluate(opt_centers, w, h, opt_radii)
                if is_valid and score > best_score:
                    best_score = score
                    best_solution = np.column_stack([opt_centers, opt_radii])
        except Exception:
            pass
    
    if best_solution is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_solution
# EVOLVE-BLOCK-END
