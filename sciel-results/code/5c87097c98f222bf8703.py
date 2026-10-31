"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _compute_score(circles):
    """Compute the sum of radii if circles are valid, else 0."""
    n = len(circles)
    if n != N:
        return 0.0
    # Check bounds
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0:
            return 0.0
        if x - r < -1e-8 or x + r > 1 + 1e-8 or y - r < -1e-8 or y + r > 1 + 1e-8:
            return 0.0
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist < r1 + r2 - 1e-8:
                return 0.0
    return np.sum(circles[:, 2])

def _make_radial_initial():
    """
    Create a structured initial configuration:
    1 central circle, 8 inner ring, 17 outer ring.
    """
    circles = np.zeros((N, 3))
    
    # Central circle
    circles[0] = [0.5, 0.5, 0.08]
    
    # Inner ring (8 circles)
    r1_dist = 0.16
    for i in range(8):
        angle = 2 * np.pi * i / 8
        x = 0.5 + r1_dist * np.cos(angle)
        y = 0.5 + r1_dist * np.sin(angle)
        circles[1 + i] = [x, y, 0.07]
    
    # Outer ring (17 circles)
    r2_dist = 0.32
    for i in range(17):
        angle = 2 * np.pi * i / 17 + np.pi / 17 # offset
        x = 0.5 + r2_dist * np.cos(angle)
        y = 0.5 + r2_dist * np.sin(angle)
        circles[9 + i] = [x, y, 0.06]
        
    # Clamp positions and radii to be inside
    for i in range(N):
        circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
        circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
        
    return circles

def _objective_and_gradient(params):
    """
    Objective: minimize -sum(radii) + penalty for violations.
    params: (3N,) array [x0, y0, r0, x1, y1, r1, ...]
    """
    circles = params.reshape(-1, 3)
    n = circles.shape[0]
    
    # Penalty terms
    penalty = 0.0
    grad_penalty = np.zeros_like(params)
    
    # Bounds penalty
    for i in range(n):
        x, y, r = circles[i]
        # x - r < 0
        if x - r < 0:
            v = x - r
            penalty += 100 * v**2
            grad_penalty[3*i] += -200 * v
            grad_penalty[3*i+2] += 200 * v
        # x + r > 1
        if x + r > 1:
            v = x + r - 1
            penalty += 100 * v**2
            grad_penalty[3*i] += 200 * v
            grad_penalty[3*i+2] += 200 * v
        # y - r < 0
        if y - r < 0:
            v = y - r
            penalty += 100 * v**2
            grad_penalty[3*i+1] += -200 * v
            grad_penalty[3*i+2] += 200 * v
        # y + r > 1
        if y + r > 1:
            v = y + r - 1
            penalty += 100 * v**2
            grad_penalty[3*i+1] += 200 * v
            grad_penalty[3*i+2] += 200 * v
        
        # r > 0
        if r < 0:
            penalty += 100 * r**2
            grad_penalty[3*i+2] += -200 * r

    # Overlap penalty
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dx = x1 - x2
            dy = y1 - y2
            dist_sq = dx**2 + dy**2
            dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
            sum_r = r1 + r2
            if dist < sum_r:
                v = sum_r - dist
                penalty += 100 * v**2
                # Gradient of v wrt x1: -dx/dist
                # d(v^2)/dx1 = 2v * (-dx/dist)
                grad_penalty[3*i] += -200 * v * dx / dist
                grad_penalty[3*i+1] += -200 * v * dy / dist
                grad_penalty[3*j] += 200 * v * dx / dist
                grad_penalty[3*j+1] += 200 * v * dy / dist
                # Gradient of v wrt r1: 1
                grad_penalty[3*i+2] += 200 * v
                grad_penalty[3*j+2] += 200 * v

    # Objective: -sum(r) + penalty
    obj = -np.sum(circles[:, 2]) + penalty
    grad = -np.ones_like(params) * np.tile([0, 0, 1], N)
    grad += grad_penalty
    
    return obj, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial guess
    best_circles = _make_radial_initial()
    best_score = _compute_score(best_circles)
    
    # Local search using L-BFGS-B
    # We want to maximize sum(r), so minimize -sum(r).
    # We use a penalty method to handle constraints.
    
    params0 = best_circles.flatten()
    
    # Run optimization
    try:
        # L-BFGS-B with analytical gradient
        res = minimize(
            _objective_and_gradient,
            params0,
            method='L-BFGS-B',
            jac=True, # We return gradient
            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        
        if time.time() > deadline - 10:
             pass # Just stop using result if too late? No, check validity.
        
        new_circles = res.x.reshape(-1, 3)
        # Ensure radii are positive
        new_circles[:, 2] = np.maximum(new_circles[:, 2], 1e-6)
        
        new_score = _compute_score(new_circles)
        if new_score > best_score:
            best_circles = new_circles
            best_score = new_score
            
    except Exception:
        pass
        
    # If time remains, try a second round from the best result or a different start
    if time.time() < deadline - 20:
        # Perturb and run again? Or just run from best again with higher precision?
        # Let's try a random restart with a simpler grid but optimized.
        # Actually, just refine the current best again.
        params1 = best_circles.flatten()
        try:
            res2 = minimize(
                _objective_and_gradient,
                params1,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
            )
            new_circles2 = res2.x.reshape(-1, 3)
            new_circles2[:, 2] = np.maximum(new_circles2[:, 2], 1e-6)
            
            new_score2 = _compute_score(new_circles2)
            if new_score2 > best_score:
                best_circles = new_circles2
                best_score = new_score2
        except Exception:
            pass

    # Final safety check and cleanup
    if _compute_score(best_circles) == 0:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
        
    return best_circles

# EVOLVE-BLOCK-END
