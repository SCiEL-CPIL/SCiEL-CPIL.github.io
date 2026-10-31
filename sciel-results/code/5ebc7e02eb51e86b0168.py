"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _lj_energy(positions):
    """Compute total Lennard-Jones energy for a set of positions."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    energy = 4.0 * np.sum(r12 - r6)
    return energy

def _lj_energy_grad(positions):
    """Compute total Lennard-Jones energy and gradient."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    # Avoid division by zero in gradient if any r2 is 0 (shouldn't happen with min dist check)
    # r^2 = r2
    # E = 4 * sum(r^-12 - r^-6)
    # dE/dr = 4 * (-12 r^-13 + 6 r^-7)
    # dE/dx_i = sum_j dE/dr_ij * (x_i - x_j) / r_ij
    # Note: diff[i,j] = pos[i] - pos[j]
    
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    energy = 4.0 * np.sum(r12 - r6)
    
    # Gradient calculation
    # factor = 4 * (-12 * r^-14 + 6 * r^-8) * (pos[i] - pos[j])
    # r^-14 = r2^-7
    # r^-8 = r2^-4
    r8 = r2 ** (-4)
    r14 = r2 ** (-7)
    
    # 4 * (-12 r^-14 + 6 r^-8) = -48 r^-14 + 24 r^-8
    coeff = -48.0 * r14 + 24.0 * r8
    
    # Zero out diagonal
    np.fill_diagonal(coeff, 0.0)
    
    # grad[i] = sum_j coeff[i,j] * (pos[i] - pos[j])
    # diff is (N, N, 3)
    # coeff is (N, N)
    # We need to multiply coeff[i,j] with diff[i,j] and sum over j
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _relax(positions):
    """Relax positions to a local minimum using L-BFGS-B."""
    x0 = positions.flatten()
    
    def fun(x):
        pos = x.reshape(-1, 3)
        # Simple constraint check for min distance to prevent numerical blowup
        # If min dist < 0.5, return high energy
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(r2, 1e10)
        if np.min(r2) < 0.25:
            return 1e10, np.zeros_like(x)
        e, g = _lj_energy_grad(pos)
        return e, g

    result = minimize(fun, x0, method='L-BFGS-B', jac=True, 
                      options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
    return result.x.reshape(-1, 3)

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Initial configuration: Random points in a sphere
    # Radius chosen such that density is reasonable. 
    # For 38 atoms, a sphere of radius ~4-5 is good.
    radius = 4.0
    rng = np.random.default_rng(42)
    
    best_positions = None
    best_energy = float('inf')
    
    # Multi-start / Basin Hopping loop
    # We will try several random starts and then do basin hopping on the best one.
    
    num_starts = 5
    max_kicks = 200
    
    # 1. Try a few random starts
    for i in range(num_starts):
        if time.time() - start_time > time_limit:
            break
            
        # Generate random positions in a sphere
        positions = rng.normal(0, radius/np.sqrt(3), size=(N, 3))
        
        # Relax
        try:
            relaxed_pos = _relax(positions)
            energy = _lj_energy(relaxed_pos)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
        except Exception:
            continue

    # 2. Basin Hopping on the best solution found
    if best_positions is not None:
        current_pos = best_positions.copy()
        current_energy = best_energy
        T = 5.0 # Initial temperature
        T_min = 0.01
        cooling = 0.95
        kick_strength = 0.5
        
        for i in range(max_kicks):
            if time.time() - start_time > time_limit:
                break
            if T < T_min:
                break
                
            # Apply random kick
            kick = rng.normal(0, kick_strength, size=(N, 3))
            new_pos = current_pos + kick
            
            # Relax
            try:
                relaxed_pos = _relax(new_pos)
                new_energy = _lj_energy(relaxed_pos)
            except Exception:
                continue
            
            # Metropolis criterion
            delta_e = new_energy - current_energy
            if delta_e < 0 or np.exp(-delta_e / T) > rng.random():
                current_pos = relaxed_pos
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_positions = current_pos
            
            T *= cooling
            
            # Increase kick strength slightly as we cool? Or keep constant?
            # Usually kick strength can be tuned. Let's keep it moderate.
            # If we get stuck, we might need larger kicks, but cooling helps escape.
            
    # Final check: ensure minimum distance >= 0.5
    if best_positions is not None:
        diff = best_positions[:, np.newaxis, :] - best_positions[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(r2, 1e10)
        min_r = np.sqrt(np.min(r2))
        if min_r < 0.5:
            best_positions = best_positions * (0.5 / min_r)
            
    return best_positions
# EVOLVE-BLOCK-END
