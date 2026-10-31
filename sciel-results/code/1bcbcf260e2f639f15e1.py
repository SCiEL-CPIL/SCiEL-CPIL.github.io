"""Rewritten program for circle packing in perimeter-4 rectangle using force-directed relaxation."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random


def _initial_layout(n, w, h):
    """Generate an initial layout of n circles in a rectangle with perimeter 4."""
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
    return centers


def _build_contact_graph(centers, radii, w, h, tolerance=1e-4):
    """
    Build contact graph: edges where circles touch or are very close.
    Returns list of (i, j) pairs.
    """
    n = len(centers)
    contacts = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if abs(dist - min_dist) < tolerance:
                contacts.append((i, j))
    
    # Also add wall contacts
    wall_contacts = []
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if abs(x - r) < tolerance:
            wall_contacts.append((i, 'L'))
        if abs(x + r - w) < tolerance:
            wall_contacts.append((i, 'R'))
        if abs(y - r) < tolerance:
            wall_contacts.append((i, 'B'))
        if abs(y + r - h) < tolerance:
            wall_contacts.append((i, 'T'))
    
    return contacts, wall_contacts


def _force_directed_relaxation(centers, w, h, radii, iterations=10000, lr=0.01, 
                                growth_rate=1e-5, max_radius_increase=0.05):
    """
    Force-directed relaxation with contact graph awareness.
    Uses structural rigidity to guide the search.
    """
    n = len(centers)
    centers = centers.copy()
    radii = radii.copy()
    
    # Build initial contact graph
    contacts, wall_contacts = _build_contact_graph(centers, radii, w, h, tolerance=1e-3)
    
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
        
        # Wall constraint forces
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x - r < 0:
                forces[i, 0] += (r - x) * 2.0
            # Right wall
            if x + r > w:
                forces[i, 0] -= (x + r - w) * 2.0
            # Bottom wall
            if y - r < 0:
                forces[i, 1] += (r - y) * 2.0
            # Top wall
            if y + r > h:
                forces[i, 1] -= (y + r - h) * 2.0
        
        # Apply forces with learning rate decay
        current_lr = lr * (1.0 - it / iterations) ** 0.5
        centers += current_lr * forces
        
        # Clamp to rectangle
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], w - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], h - radii[i])
        
        # Gradually increase radii
        radii *= (1.0 + growth_rate)
        
        # Rebuild contact graph every 100 iterations
        if it % 100 == 0:
            contacts, wall_contacts = _build_contact_graph(centers, radii, w, h, tolerance=1e-3)
    
    return centers, radii


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
    """
    positions = params[:2 * n].reshape(n, 2)
    radii = params[2 * n:]
    
    # Penalty for going outside the rectangle
    penalty = 0.0
    
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


def construct():
    """
    Pack 21 circles with disjoint interiors in a variable rectangle of perimeter 4
    and maximize the sum of their radii.
    
    Returns: array of shape (n, 3) with [x, y, r] for each circle.
    """
    n = N
    
    best_solution = None
    best_score = -np.inf
    
    # Try multiple initial configurations with different aspect ratios
    aspect_ratios = [
        (1.0, 1.0),
        (0.9, 1.1),
        (1.1, 0.9),
        (0.8, 1.2),
        (1.2, 0.8),
        (0.7, 1.3),
        (1.3, 0.7),
        (0.85, 1.15),
        (1.15, 0.85),
        (0.75, 1.25),
        (1.25, 0.75),
        (0.95, 1.05),
        (1.05, 0.95),
    ]
    
    for trial, (w, h) in enumerate(aspect_ratios):
        random.seed(trial * 7919 + 13)
        np.random.seed(trial * 7919 + 13)
        
        # Initial layout
        centers = _initial_layout(n, w, h)
        
        # Initial radii: estimate from grid spacing
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        base_r = min(w / cols, h / rows) * 0.42
        radii = np.full(n, base_r)
        
        # Force-directed relaxation with multiple restarts
        for restart in range(3):
            random.seed(trial * 7919 + 13 + restart)
            np.random.seed(trial * 7919 + 13 + restart)
            
            # Perturb initial layout
            if restart > 0:
                perturbation = 0.05 * (1.0 + restart * 0.5)
                centers = centers + np.random.normal(0, perturbation, centers.shape)
                # Clamp to valid range
                for i in range(n):
                    centers[i, 0] = np.clip(centers[i, 0], radii[i], w - radii[i])
                    centers[i, 1] = np.clip(centers[i, 1], radii[i], h - radii[i])
            
            # Force-directed relaxation
            centers_relaxed, radii_relaxed = _force_directed_relaxation(
                centers, w, h, radii, 
                iterations=5000, lr=0.03,
                growth_rate=1e-5
            )
            
            # Check validity and get score
            is_valid, score = _evaluate(centers_relaxed, w, h, radii_relaxed)
            
            if is_valid and score > best_score:
                best_score = score
                best_solution = np.column_stack([centers_relaxed, radii_relaxed])
        
        # Try scipy optimization from the best relaxed solution
        if best_solution is not None:
            params = np.concatenate([centers_relaxed.flatten(), radii_relaxed])
            
            def obj(p):
                return _penalized_objective(p, n, w, h)
            
            try:
                result = minimize(obj, params, method='L-BFGS-B', 
                                 options={'maxiter': 300, 'ftol': 1e-9})
                
                if result.fun < 0:  # Check if improvement
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
