"""Physics-based soft-body simulation for circle packing in perimeter-4 rectangle."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def _energy(pos, radii, W, H):
    """
    Compute total energy: repulsion between circles, wall repulsion, and a small
    regularization to keep radii positive and bounded.
    We minimize energy, which includes -sum(radii) to maximize the sum.
    """
    n = len(radii)
    energy = 0.0
    
    # Inter-circle repulsion (soft constraint for non-overlapping)
    # Using a quadratic penalty for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist * min_dist:
                dist = np.sqrt(dist_sq)
                overlap = min_dist - dist
                # Strong repulsion
                energy += 500.0 * overlap * overlap
    
    # Wall repulsion: circles must stay inside [0, W] x [0, H]
    for i in range(n):
        r = radii[i]
        x, y = pos[i]
        # Left wall
        if x < r:
            energy += 500.0 * (r - x) ** 2
        # Right wall
        if x > W - r:
            energy += 500.0 * (x - (W - r)) ** 2
        # Bottom wall
        if y < r:
            energy += 500.0 * (r - y) ** 2
        # Top wall
        if y > H - r:
            energy += 500.0 * (y - (H - r)) ** 2
    
    # Objective: Maximize sum of radii -> Minimize -sum(radii)
    # We scale this to balance with the penalty terms.
    # If penalties are ~500 * overlap^2, and we want to encourage growth,
    # the gradient from -sum(r) should be competitive.
    energy -= 1.0 * np.sum(radii)
    
    # Keep radii in reasonable bounds (prevent explosion or zero)
    # Soft penalty for radii deviating from a target range? 
    # Actually, just bounding is handled by the optimizer or post-processing.
    # Let's add a small penalty for very small radii to avoid degeneracy.
    energy += 0.1 * np.sum(np.maximum(0, 0.05 - radii) ** 2)

    return energy

def _energy_grad(pos, radii, W, H):
    """Gradient of the energy function."""
    n = len(radii)
    grad_pos = np.zeros_like(pos)
    grad_radii = np.zeros_like(radii)
    
    # Inter-circle gradients
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist * min_dist:
                dist = np.sqrt(dist_sq) + 1e-12
                overlap = min_dist - dist
                # d/dx (500 * (min_dist - dist)^2) = 500 * 2 * overlap * (-dx/dist)
                factor = 1000.0 * overlap / dist
                grad_pos[i, 0] += factor * dx
                grad_pos[i, 1] += factor * dy
                grad_pos[j, 0] -= factor * dx
                grad_pos[j, 1] -= factor * dy
                # d/dr_i (500 * (min_dist - dist)^2) = 500 * 2 * overlap
                grad_radii[i] += 1000.0 * overlap
                grad_radii[j] += 1000.0 * overlap
    
    # Wall gradients
    for i in range(n):
        r = radii[i]
        x, y = pos[i]
        if x < r:
            g = 1000.0 * (r - x)
            grad_pos[i, 0] += -g
            grad_radii[i] += g
        if x > W - r:
            g = 1000.0 * (x - (W - r))
            grad_pos[i, 0] += g
            grad_radii[i] += -g
        if y < r:
            g = 1000.0 * (r - y)
            grad_pos[i, 1] += -g
            grad_radii[i] += g
        if y > H - r:
            g = 1000.0 * (y - (H - r))
            grad_pos[i, 1] += g
            grad_radii[i] += -g
    
    # Gradient of -sum(radii)
    grad_radii -= 1.0
    
    # Gradient of small radius penalty
    for i in range(n):
        if radii[i] < 0.05:
            grad_radii[i] += 0.2 * (0.05 - radii[i])

    return grad_pos, grad_radii

def construct():
    # Perimeter = 2*(W + H) = 4 => W + H = 2
    best_result = None
    best_sum = 0.0
    
    # Aspect ratios to try: W/H
    # W = 2 * a / (1+a), H = 2 / (1+a)
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.66, 2.0, 0.5, 1.33, 0.75, 1.25]
    
    n_restarts = 5
    rng = np.random.default_rng(42)
    
    for aspect in aspect_ratios:
        W = 2.0 * aspect / (1.0 + aspect)
        H = 2.0 / (1.0 + aspect)
        
        for restart in range(n_restarts):
            # Random initialization
            # Spread points roughly to avoid immediate high energy
            # Use a grid with jitter for better start than pure random
            side = int(np.ceil(np.sqrt(N)))
            pos = np.zeros((N, 2))
            for i in range(N):
                row = i // side
                col = i % side
                # Add jitter to break symmetry
                jitter_x = rng.uniform(-0.1, 0.1) * (W / side)
                jitter_y = rng.uniform(-0.1, 0.1) * (H / side)
                pos[i, 0] = (col + 0.5) * W / side + jitter_x
                pos[i, 1] = (row + 0.5) * H / side + jitter_y
            
            # Clamp positions to be strictly inside, away from walls initially
            eps = 0.01
            pos[:, 0] = np.clip(pos[:, 0], eps, W - eps)
            pos[:, 1] = np.clip(pos[:, 1], eps, H - eps)
            
            # Initial radii: small
            # Estimate max possible radius
            area = W * H
            avg_area_per_circle = area / N
            r_init = np.sqrt(avg_area_per_circle / np.pi) * 0.5
            r_init = min(r_init, min(W, H) / 10)
            radii = np.full(N, r_init)
            
            x0 = np.concatenate([pos.flatten(), radii])
            
            def objective(x):
                p = x[:2 * N].reshape(N, 2)
                r = x[2 * N:]
                return _energy(p, r, W, H)
            
            def grad(x):
                p = x[:2 * N].reshape(N, 2)
                r = x[2 * N:]
                gp, gr = _energy_grad(p, r, W, H)
                return np.concatenate([gp.flatten(), gr])
            
            # Optimize
            # Use L-BFGS-B with bounds? 
            # Bounds: positions in [0, W]x[0, H], radii in [0, min(W,H)/2]
            bounds_pos = [(0, W)] * N + [(0, H)] * N
            bounds_rad = [(0, min(W, H) / 2)] * N
            bounds = bounds_pos + bounds_rad
            
            try:
                result = minimize(objective, x0, jac=grad, method='L-BFGS-B',
                                  bounds=bounds,
                                  options={'maxiter': 1000, 'gtol': 1e-8})
                
                pos_opt = result.x[:2 * N].reshape(N, 2)
                radii_opt = result.x[2 * N:]
                
                # Clamp radii to be positive
                radii_opt = np.maximum(radii_opt, 1e-6)
                
                # Check validity: no overlaps, all inside
                valid = True
                for i in range(N):
                    if pos_opt[i, 0] < radii_opt[i] - 1e-9 or pos_opt[i, 0] > W - radii_opt[i] + 1e-9:
                        valid = False
                        break
                    if pos_opt[i, 1] < radii_opt[i] - 1e-9 or pos_opt[i, 1] > H - radii_opt[i] + 1e-9:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i + 1, N):
                            dist = np.linalg.norm(pos_opt[i] - pos_opt[j])
                            if dist < radii_opt[i] + radii_opt[j] - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    # Final safety clamp: ensure circles are strictly inside
                    for i in range(N):
                        radii_opt[i] = min(radii_opt[i], 
                                           pos_opt[i, 0], 
                                           W - pos_opt[i, 0], 
                                           pos_opt[i, 1], 
                                           H - pos_opt[i, 1])
                        radii_opt[i] = max(radii_opt[i], 1e-6)
                    
                    # Resolve any remaining overlaps by shrinking
                    for _ in range(50):
                        resolved = True
                        for i in range(N):
                            for j in range(i + 1, N):
                                dist = np.linalg.norm(pos_opt[i] - pos_opt[j])
                                min_dist = radii_opt[i] + radii_opt[j]
                                if dist < min_dist - 1e-10:
                                    resolved = False
                                    shrink = (min_dist - dist) / 2.0 + 1e-4
                                    radii_opt[i] = max(radii_opt[i] - shrink, 1e-6)
                                    radii_opt[j] = max(radii_opt[j] - shrink, 1e-6)
                        if resolved:
                            break
                    
                    # Ensure inside bounds after shrinking
                    for i in range(N):
                        radii_opt[i] = min(radii_opt[i], 
                                           pos_opt[i, 0], 
                                           W - pos_opt[i, 0], 
                                           pos_opt[i, 1], 
                                           H - pos_opt[i, 1])
                        radii_opt[i] = max(radii_opt[i], 1e-6)
                    
                    total = np.sum(radii_opt)
                    if total > best_sum:
                        best_sum = total
                        best_result = (pos_opt.copy(), radii_opt.copy(), W, H)
                        
            except Exception:
                pass

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        W, H = 1.0, 1.0
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) * W / side
            y = (row + 0.5) * H / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
        return np.array(cells)
    
    pos_opt, radii_opt, W, H = best_result
    result = np.column_stack([pos_opt, radii_opt])
    return result
# EVOLVE-BLOCK-END
