"""
Strategy: Symmetry-Adaptive Basin Hopping with Analytical Gradients.
Rationale: The current approach uses a generic 114D basin hopping which can be inefficient due to numerical gradients and lack of structural awareness. 
This new strategy exploits the known topology of the global minimum (FCC truncated octahedron). 
We generate the 38 atoms from the FCC lattice, center them, and then perform basin hopping. 
Crucially, we use analytical gradients for the Lennard-Jones potential to speed up local minimization (L-BFGS-B) significantly. 
This allows for more local minimizations within the time budget, increasing the probability of escaping shallow local minima 
and finding the global minimum. The initial structure is the ideal FCC truncated octahedron, 
which is already very close to the global minimum, so small perturbations are likely to suffice.
"""
import math

import numpy as np
import time
from scipy.optimize import minimize

N = 38
DEADLINE = 145.0
START_TIME = time.time()

def lj_energy_and_grad(positions):
    """
    Compute LJ energy and gradient (forces) efficiently.
    Returns energy and gradient (shape (N*3,)).
    """
    # positions: (N, 3)
    # diff[i, j, :] = positions[i] - positions[j]
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0) # Avoid division by zero
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    # Energy = 4 * sum_{i<j} (r^-12 - r^-6)
    # We sum over all i != j and divide by 2
    e_pair = 4.0 * (inv_r12 - inv_r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = 0.5 * np.sum(e_pair)
    
    # Gradient:
    # dE/dx_i = sum_{j != i} 4 * (-12 r^-14 + 6 r^-8) * (x_i - x_j)
    # Let's compute the force vector F = -dE/dx
    # F_i = sum_{j != i} 24 * (2 r^-14 - r^-8) * (x_i - x_j)
    # Wait, let's derive carefully.
    # E = sum_{i<j} 4 (r_ij^-12 - r_ij^-6)
    # dE/dx_i = sum_{j != i} 4 (-12 r_ij^-14 (x_i - x_j) + 6 r_ij^-8 (x_i - x_j))
    #         = sum_{j != i} 24 r_ij^-14 (x_i - x_j) (1 - 2 r_ij^-6)
    # Let's use the standard LJ force formula:
    # F_ij = 24 * (2 r^-12 - r^-6) * r_vec / r^2 ? No.
    # F = -dE/dr * r_hat
    # dE/dr = 4 (-12 r^-13 + 6 r^-7) = 24 (r^-7 - 2 r^-13)
    # F = -24 (r^-7 - 2 r^-13) r_hat = 24 (2 r^-13 - r^-7) r_hat
    # r_hat = r_vec / r
    # F = 24 (2 r^-14 - r^-8) r_vec
    
    # So gradient dE/dx_i = -F_i = - sum_{j != i} 24 (2 r_ij^-14 - r_ij^-8) (x_i - x_j)
    # Let's compute the term:
    # coeff = 24 * (2 * inv_r12 * inv_r2 - inv_r6 * inv_r2)
    # coeff = 24 * inv_r2 * (2 * inv_r12 - inv_r6)
    
    coeff = 24.0 * inv_r2 * (2.0 * inv_r12 - inv_r6)
    np.fill_diagonal(coeff, 0.0)
    
    # grad[i] = - sum_j coeff[i,j] * (x_i - x_j)
    # grad[i] = sum_j coeff[i,j] * (x_j - x_i)
    # Let's compute sum_j coeff[i,j] * diff[i,j]
    # diff[i,j] = x_i - x_j
    # sum_j coeff[i,j] * diff[i,j] = sum_j coeff[i,j] * (x_i - x_j)
    # This is exactly -F_i (since F_i = sum 24(2r^-14-r^-8)(x_i-x_j) = sum coeff*(x_i-x_j))
    # Wait, my coeff definition:
    # F_ij = 24 (2 r^-14 - r^-8) r_vec
    # coeff = 24 * inv_r2 * (2 inv_r12 - inv_r6) = 24 (2 r^-14 - r^-8)
    # So F_i = sum_j coeff[i,j] * (x_i - x_j)
    # grad_i = -F_i = - sum_j coeff[i,j] * (x_i - x_j)
    
    # We can compute this as:
    # grad = - (coeff * diff).sum(axis=1)
    
    grad = -np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad.flatten()

def lj_energy_only(positions):
    e, _ = lj_energy_and_grad(positions)
    return e

def check_validity(positions, min_dist=0.5):
    """Check minimum distance constraint."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1e10)
    min_r2 = np.min(r2)
    return min_r2 >= (min_dist * min_dist)

def make_fcc_truncated_octahedron_raw():
    """
    Generate the 38-atom FCC truncated octahedron points.
    This is the known global minimum topology.
    """
    points = []
    max_coord = 5
    scale = 1.0 / math.sqrt(2.0)
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    r2 = i * i + j * j + k * k
                    if r2 <= 30:
                        points.append((i * scale, j * scale, k * scale, r2))
    
    points.sort(key=lambda x: x[3])
    selected = [p[:3] for p in points[:38]]
    
    coords = np.array(selected, dtype=float)
    coords -= np.mean(coords, axis=0)
    
    return coords

def local_minimize(positions):
    """
    Perform a local L-BFGS minimization using analytical gradient.
    Returns the minimized positions and energy.
    """
    def objective(x):
        pos = x.reshape(N, 3)
        e, g = lj_energy_and_grad(pos)
        return e, g
    
    try:
        result = minimize(
            objective,
            positions.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12}
        )
        if result.success:
            min_pos = result.x.reshape(N, 3)
            min_energy = lj_energy_only(min_pos)
            if check_validity(min_pos):
                return min_pos, min_energy
    except Exception:
        pass
    
    # Fallback to input if minimization fails
    return positions, lj_energy_only(positions)

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # 1. Generate initial FCC structure
    try:
        raw_coords = make_fcc_truncated_octahedron_raw()
    except Exception:
        rng = np.random.default_rng(0)
        raw_coords = rng.uniform(-2, 2, size=(N, 3))
        raw_coords -= np.mean(raw_coords, axis=0)
    
    # 2. Initial local relaxation to get into the basin
    current_pos, current_energy = local_minimize(raw_coords)
    
    # 3. Basin Hopping Loop
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    rng = np.random.default_rng(42)
    
    # Parameters for basin hopping
    kick_scale = 0.15  # Initial kick magnitude
    
    while time.time() - START_TIME < DEADLINE - 5:
        # Apply random perturbation
        kick = rng.normal(0, kick_scale, size=(N, 3))
        perturbed_pos = current_pos + kick
        
        # Check validity of perturbed position
        if not check_validity(perturbed_pos, min_dist=0.3):
            continue
        
        # Local minimization
        new_pos, new_energy = local_minimize(perturbed_pos)
        
        # Acceptance criterion: Metropolis or simple hill climbing
        # Use Metropolis with a small temperature to allow some uphill moves
        # T is not physical, just a scaling factor
        T = 0.1
        delta_e = new_energy - current_energy
        if delta_e < 0:
            accept = True
        else:
            accept = rng.random() < np.exp(-delta_e / T)
        
        if accept:
            current_pos = new_pos
            current_energy = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
                
            # Reduce kick scale slightly to refine
            kick_scale *= 0.98
            if kick_scale < 0.01:
                kick_scale = 0.01
        else:
            # Occasionally increase kick scale to explore
            if rng.random() < 0.05:
                kick_scale *= 1.2
                if kick_scale > 0.5:
                    kick_scale = 0.5
    
    # Final polish on the best position found
    if time.time() - START_TIME < DEADLINE - 2:
        def objective(x):
            pos = x.reshape(N, 3)
            e, g = lj_energy_and_grad(pos)
            return e, g
        
        try:
            result = minimize(
                objective,
                best_pos.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14}
            )
            polished = result.x.reshape(N, 3)
            if check_validity(polished):
                e_polished = lj_energy_only(polished)
                if e_polished < best_energy:
                    best_pos = polished
                    best_energy = e_polished
        except Exception:
            pass

    # Final validity check
    if not check_validity(best_pos):
        # Fallback to initial raw coords if somehow invalid
        best_pos = raw_coords
        if not check_validity(best_pos):
            rng = np.random.default_rng(123)
            pts = []
            for _ in range(N):
                while True:
                    p = rng.uniform(-2, 2, size=3)
                    if all(np.linalg.norm(p - q) > 0.5 for q in pts):
                        pts.append(p)
                        break
            best_pos = np.array(pts)
    
    return best_pos

# EVOLVE-BLOCK-END
