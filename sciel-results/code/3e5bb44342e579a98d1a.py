"""Rewritten program for circle packing in perimeter-4 rectangle using topological graph model."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import random
import time
from scipy.optimize import minimize


def _initial_hex_layout(n, w, h):
    """Generate an initial hexagonal layout of n circles in a rectangle."""
    cols = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
    rows = int(np.ceil(n / cols))
    
    dx = w / cols
    dy = h / rows
    spacing = min(dx, dy * 2 / np.sqrt(3))
    
    centers = []
    for i in range(rows):
        for j in range(cols):
            if len(centers) >= n:
                break
            x = (j + 0.5) * dx + (0.5 * dx if i % 2 == 1 else 0)
            y = (i + 0.5) * dy
            x = min(max(x, 0), w)
            y = min(max(y, 0), h)
            centers.append([x, y])
    
    centers = np.array(centers[:n])
    return centers


def _build_contact_graph(centers, radii, w, h, threshold=1.05):
    """
    Build a contact graph where edges connect circles that are nearly touching
    or touching walls. Returns adjacency list and wall contacts.
    """
    n = len(centers)
    adj = {i: [] for i in range(n)}
    wall_contacts = {i: [] for i in range(n)}  # 'L', 'R', 'B', 'T'
    
    # Circle-circle contacts
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            
            if dist < min_dist * threshold:
                adj[i].append(j)
                adj[j].append(i)
    
    # Wall contacts
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < 0.01 * r:
            wall_contacts[i].append('L')
        if x + r > w - 0.01 * r:
            wall_contacts[i].append('R')
        if y - r < 0.01 * r:
            wall_contacts[i].append('B')
        if y + r > h - 0.01 * r:
            wall_contacts[i].append('T')
    
    return adj, wall_contacts


def _topological_relaxation(centers, radii, w, h, iterations=2000, lr=0.02, 
                            stiffness=10.0, wall_stiffness=5.0):
    """
    Topological force-directed relaxation using contact graph.
    Circles that are in contact act as rigid bars.
    """
    n = len(centers)
    centers = centers.copy()
    radii = radii.copy()
    
    for it in range(iterations):
        forces = np.zeros_like(centers)
        
        # Rebuild contact graph periodically
        if it % 50 == 0 or it == 0:
            adj, wall_contacts = _build_contact_graph(centers, radii, w, h, threshold=1.05)
        
        # Rigid bar constraints between contacting circles
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = radii[i] + radii[j]
                    
                    if dist < 1e-10:
                        angle = random.uniform(0, 2 * np.pi)
                        fx = np.cos(angle) * 0.01
                        fy = np.sin(angle) * 0.01
                    else:
                        # Force to maintain distance = min_dist
                        error = min_dist - dist
                        fx = (dx / dist) * error
                        fy = (dy / dist) * error
                    
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Wall contact forces
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            if 'L' in wall_contacts[i]:
                forces[i, 0] += wall_stiffness * (r - x)
            if 'R' in wall_contacts[i]:
                forces[i, 0] -= wall_stiffness * (x + r - w)
            if 'B' in wall_contacts[i]:
                forces[i, 1] += wall_stiffness * (r - y)
            if 'T' in wall_contacts[i]:
                forces[i, 1] -= wall_stiffness * (y + r - h)
        
        # Apply forces with decay
        current_lr = lr * (1.0 - 0.5 * it / iterations)
        centers += current_lr * forces
        
        # Clamp to valid region
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], w - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], h - radii[i])
        
        # Adaptive radius growth: only grow if constraints are satisfied
        if it % 100 == 0:
            # Check if all constraints are satisfied
            constraints_ok = True
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                if x - r < -1e-6 or x + r > w + 1e-6 or y - r < -1e-6 or y + r > h + 1e-6:
                    constraints_ok = False
                    break
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < radii[i] + radii[j] - 1e-6:
                        constraints_ok = False
                        break
                if not constraints_ok:
                    break
            
            if constraints_ok:
                # Grow radii slightly
                growth_factor = 1.0 + 1e-5 * (1.0 - it / iterations)
                radii *= growth_factor
    
    return centers, radii


def _evaluate(centers, w, h, radii):
    """Evaluate validity and objective."""
    n = len(centers)
    
    if w + h > 2.0 + 1e-6:
        return False, 0.0
    
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > w + 1e-6 or y - r < -1e-6 or y + r > h + 1e-6:
            return False, 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - 1e-6:
                return False, 0.0
    
    return True, np.sum(radii)


def _penalized_objective(params, n, w, h):
    """Penalized objective for scipy optimization."""
    positions = params[:2 * n].reshape(n, 2)
    radii = params[2 * n:]
    
    penalty = 0.0
    
    for i in range(n):
        x, y = positions[i]
        r = radii[i]
        if x - r < 0:
            penalty += (x - r) ** 2
        if x + r > w:
            penalty += (x + r - w) ** 2
        if y - r < 0:
            penalty += (y - r) ** 2
        if y + r > h:
            penalty += (y + r - h) ** 2
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    
    return -np.sum(radii) + 100.0 * penalty


def construct():
    """
    Pack 21 circles with disjoint interiors in a variable rectangle of perimeter 4
    and maximize the sum of their radii.
    """
    n = N
    best_solution = None
    best_score = -np.inf
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Try multiple aspect ratios and initial configurations
    aspect_ratios = [
        (1.0, 1.0), (0.9, 1.1), (1.1, 0.9), (0.85, 1.15), (1.15, 0.85),
        (0.8, 1.2), (1.2, 0.8), (0.95, 1.05), (1.05, 0.95), (1.0, 1.0)
    ]
    
    for trial in range(15):
        if time.time() - start_time > time_limit:
            break
        
        random.seed(trial * 98765 + 123)
        np.random.seed(trial * 98765 + 123)
        
        w, h = aspect_ratios[trial % len(aspect_ratios)]
        
        # Initial hexagonal layout
        centers = _initial_hex_layout(n, w, h)
        
        # Initial radii estimate
        cols = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
        rows = int(np.ceil(n / cols))
        base_r = min(w / cols, h / rows) * 0.45
        radii = np.full(n, base_r)
        
        # Topological relaxation
        centers, radii = _topological_relaxation(centers, radii, w, h, 
                                                  iterations=1500, lr=0.03)
        
        is_valid, score = _evaluate(centers, w, h, radii)
        
        if is_valid and score > best_score:
            best_score = score
            best_solution = np.column_stack([centers, radii])
        
        # Refinement pass with smaller learning rate
        if time.time() - start_time < time_limit - 15:
            centers2, radii2 = _topological_relaxation(centers, radii, w, h,
                                                        iterations=800, lr=0.01)
            is_valid2, score2 = _evaluate(centers2, w, h, radii2)
            if is_valid2 and score2 > best_score:
                best_score = score2
                best_solution = np.column_stack([centers2, radii2])
        
        # Try scipy optimization from current state
        if time.time() - start_time < time_limit - 20:
            params = np.concatenate([centers.flatten(), radii])
            
            def obj(p):
                return _penalized_objective(p, n, w, h)
            
            try:
                result = minimize(obj, params, method='L-BFGS-B',
                                  options={'maxiter': 150, 'ftol': 1e-8})
                
                if result.fun < 0:
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
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (j + 0.5) / side
                y = (i + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_solution
# EVOLVE-BLOCK-END
