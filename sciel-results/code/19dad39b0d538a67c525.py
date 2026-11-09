"""Physics-based soft-body simulation with Lennard-Jones potentials for circle packing."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _lj_energy(pos, radii, W, H, sigma_scale=1.0, epsilon=50.0):
    """
    Lennard-Jones-like energy for circle packing.
    Uses 12-6 potential for repulsion and attractive boundary forces.
    """
    n = len(radii)
    energy = 0.0
    
    # Inter-circle Lennard-Jones potential
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy) + 1e-10
            sigma = sigma_scale * (radii[i] + radii[j])
            # LJ potential: 4*epsilon*((sigma/dist)^12 - (sigma/dist)^6)
            ratio = sigma / dist
            energy += epsilon * (ratio**12 - ratio**6)
    
    # Wall constraints: strong penalty for going outside
    for i in range(n):
        r = radii[i]
        x, y = pos[i]
        # Left wall
        if x < r:
            energy += 1000.0 * (r - x) ** 2
        # Right wall
        if x > W - r:
            energy += 1000.0 * (x - (W - r)) ** 2
        # Bottom wall
        if y < r:
            energy += 1000.0 * (r - y) ** 2
        # Top wall
        if y > H - r:
            energy += 1000.0 * (y - (H - r)) ** 2
    
    # Objective: maximize sum of radii, so minimize negative sum
    energy -= 50.0 * np.sum(radii)
    
    # Regularization to keep radii reasonable
    energy += 0.1 * np.sum((radii - 0.08) ** 2)
    
    return energy


def _lj_energy_grad(pos, radii, W, H, sigma_scale=1.0, epsilon=50.0):
    """Gradient of Lennard-Jones energy."""
    n = len(radii)
    grad_pos = np.zeros_like(pos)
    grad_radii = np.zeros_like(radii)
    
    # Inter-circle gradients
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy) + 1e-10
            sigma = sigma_scale * (radii[i] + radii[j])
            ratio = sigma / dist
            
            # d/d(dist) of 4*eps*((sigma/dist)^12 - (sigma/dist)^6)
            # = 4*eps*(-12*sigma^12/dist^13 + 6*sigma^6/dist^7)
            # = 4*eps/dist * (-12*ratio^12 + 6*ratio^6)
            dE_ddist = epsilon * (12 * ratio**12 - 6 * ratio**6) / dist
            
            # Gradient w.r.t. position: dE/dx_i = dE_ddist * dx/dist
            grad_pos[i, 0] += dE_ddist * dx / dist
            grad_pos[i, 1] += dE_ddist * dy / dist
            grad_pos[j, 0] -= dE_ddist * dx / dist
            grad_pos[j, 1] -= dE_ddist * dy / dist
            
            # Gradient w.r.t. radii: dE/dsigma * dsigma/dr_i
            # dE/dsigma = 4*eps*(12*sigma^11/dist^12 - 6*sigma^5/dist^6)
            # = 4*eps*sigma/dist * (12*ratio^11 - 6*ratio^5)
            dE_dsigma = epsilon * (12 * ratio**11 - 6 * ratio**5) * sigma / dist
            grad_radii[i] += dE_dsigma * sigma_scale
            grad_radii[j] += dE_dsigma * sigma_scale
    
    # Wall gradients
    for i in range(n):
        r = radii[i]
        x, y = pos[i]
        if x < r:
            g = 2000.0 * (r - x)
            grad_pos[i, 0] += -g
            grad_radii[i] += g
        if x > W - r:
            g = 2000.0 * (x - (W - r))
            grad_pos[i, 0] += g
            grad_radii[i] += -g
        if y < r:
            g = 2000.0 * (r - y)
            grad_pos[i, 1] += -g
            grad_radii[i] += g
        if y > H - r:
            g = 2000.0 * (y - (H - r))
            grad_pos[i, 1] += g
            grad_radii[i] += -g
    
    # Regularization gradients
    grad_radii -= 50.0
    grad_radii += 0.2 * (radii - 0.08)
    
    return grad_pos, grad_radii


def _simulate(pos, radii, W, H, n_steps=2000, dt=0.001, damping=0.95, 
              sigma_scale=1.0, epsilon=50.0):
    """
    Physics-based simulation using velocity Verlet integration.
    Returns optimized positions and radii.
    """
    n = len(radii)
    pos = pos.copy()
    radii = radii.copy()
    vel_pos = np.zeros_like(pos)
    vel_radii = np.zeros_like(radii)
    
    for step in range(n_steps):
        # Compute forces (negative gradient)
        gp, gr = _lj_energy_grad(pos, radii, W, H, sigma_scale, epsilon)
        force_pos = -gp
        force_radii = -gr
        
        # Update velocities with damping
        vel_pos = vel_pos * damping + force_pos * dt
        vel_radii = vel_radii * damping + force_radii * dt
        
        # Update positions
        pos = pos + vel_pos * dt
        radii = radii + vel_radii * dt
        
        # Clamp radii to positive
        radii = np.maximum(radii, 0.005)
        
        # Clamp positions to stay roughly inside
        for i in range(n):
            r = radii[i]
            pos[i, 0] = np.clip(pos[i, 0], r * 0.5, W - r * 0.5)
            pos[i, 1] = np.clip(pos[i, 1], r * 0.5, H - r * 0.5)
    
    return pos, radii


def _polish(pos, radii, W, H):
    """
    Polish the solution using local optimization.
    """
    x0 = np.concatenate([pos.flatten(), radii])
    
    def objective(x):
        p = x[:2 * N].reshape(N, 2)
        r = x[2 * N:]
        return _lj_energy(p, r, W, H)
    
    def grad(x):
        p = x[:2 * N].reshape(N, 2)
        r = x[2 * N:]
        gp, gr = _lj_energy_grad(p, r, W, H)
        return np.concatenate([gp.flatten(), gr])
    
    result = minimize(objective, x0, jac=grad, method='L-BFGS-B',
                      options={'maxiter': 300, 'gtol': 1e-10})
    
    pos_opt = result.x[:2 * N].reshape(N, 2)
    radii_opt = result.x[2 * N:]
    radii_opt = np.maximum(radii_opt, 0.005)
    
    return pos_opt, radii_opt


def _check_validity(pos, radii, W, H, tol=1e-9):
    """Check if circles are valid (non-overlapping and inside bounds)."""
    n = len(radii)
    
    # Check bounds
    for i in range(n):
        r = radii[i]
        if pos[i, 0] < r - tol or pos[i, 0] > W - r + tol:
            return False
        if pos[i, 1] < r - tol or pos[i, 1] > H - r + tol:
            return False
    
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist < radii[i] + radii[j] - tol:
                return False
    
    return True


def _resolve_overlaps(pos, radii, W, H, max_iter=1000):
    """
    Resolve overlaps by shrinking radii.
    """
    for _ in range(max_iter):
        resolved = True
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(pos[i] - pos[j])
                min_dist = radii[i] + radii[j]
                if dist < min_dist - 1e-10:
                    resolved = False
                    overlap = min_dist - dist
                    # Shrink both radii proportionally
                    shrink_i = overlap * radii[i] / (radii[i] + radii[j]) + 1e-6
                    shrink_j = overlap * radii[j] / (radii[i] + radii[j]) + 1e-6
                    radii[i] = max(radii[i] - shrink_i, 0.005)
                    radii[j] = max(radii[j] - shrink_j, 0.005)
        if resolved:
            break
    
    # Final clamp to bounds
    for i in range(N):
        radii[i] = min(radii[i], 
                       pos[i, 0], 
                       W - pos[i, 0], 
                       pos[i, 1], 
                       H - pos[i, 1])
        radii[i] = max(radii[i], 0.005)
    
    return radii


def construct():
    """
    Main function: pack 21 circles in a perimeter-4 rectangle.
    Returns array of shape (21, 3) with [x, y, r] for each circle.
    """
    best_result = None
    best_sum = 0.0
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.5]
    
    for aspect in aspect_ratios:
        W = 2.0 * aspect / (1.0 + aspect)
        H = 2.0 / (1.0 + aspect)
        
        # Multiple initialization strategies
        inits = []
        
        # Strategy 1: Grid with jitter
        side = int(np.ceil(np.sqrt(N)))
        pos_grid = np.zeros((N, 2))
        for i in range(N):
            row = i // side
            col = i % side
            pos_grid[i, 0] = (col + 0.5) * W / side
            pos_grid[i, 1] = (row + 0.5) * H / side
        pos_grid += np.random.uniform(-0.02, 0.02, size=(N, 2))
        r_max_grid = min(W, H) / (2 * side)
        radii_grid = np.full(N, r_max_grid * 0.6)
        inits.append((pos_grid, radii_grid))
        
        # Strategy 2: Random placement
        pos_rand = np.random.uniform(0.1, 0.9, size=(N, 2)) * np.array([W, H])
        radii_rand = np.full(N, min(W, H) * 0.05)
        inits.append((pos_rand, radii_rand))
        
        # Strategy 3: Hexagonal-like packing
        pos_hex = np.zeros((N, 2))
        hex_spacing = min(W, H) / 4.5
        for i in range(N):
            row = i // 4
            col = i % 4
            x_offset = (col + (row % 2) * 0.5) * hex_spacing
            y_offset = row * hex_spacing * 0.866
            pos_hex[i, 0] = x_offset + W * 0.1
            pos_hex[i, 1] = y_offset + H * 0.1
        pos_hex = np.clip(pos_hex, 0.05, np.array([W - 0.05, H - 0.05]))
        radii_hex = np.full(N, hex_spacing * 0.45)
        inits.append((pos_hex, radii_hex))
        
        for pos_init, radii_init in inits:
            # Run physics simulation
            pos_sim, radii_sim = _simulate(pos_init, radii_init, W, H, 
                                           n_steps=1500, dt=0.0008, damping=0.92)
            
            # Polish with local optimization
            pos_opt, radii_opt = _polish(pos_sim, radii_sim, W, H)
            
            # Check validity
            if _check_validity(pos_opt, radii_opt, W, H):
                total = np.sum(radii_opt)
                if total > best_sum:
                    best_sum = total
                    best_result = (pos_opt, radii_opt, W, H)
            else:
                # Try to resolve overlaps
                radii_fixed = _resolve_overlaps(pos_opt, radii_opt, W, H)
                if _check_validity(pos_opt, radii_fixed, W, H):
                    total = np.sum(radii_fixed)
                    if total > best_sum:
                        best_sum = total
                        best_result = (pos_opt, radii_fixed, W, H)
    
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
    
    # Final safety: ensure strict validity
    radii_opt = _resolve_overlaps(pos_opt, radii_opt, W, H, max_iter=500)
    
    result = np.column_stack([pos_opt, radii_opt])
    return result
# EVOLVE-BLOCK-END
