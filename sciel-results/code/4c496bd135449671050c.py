"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    # Leave 10s buffer for overhead/termination
    deadline = start_time + 110.0 

    # 1. Construct initial guess: FCC truncated octahedron
    # Generate FCC lattice points (i+j+k even)
    # Range [-8, 8] is sufficient for 38 atoms
    coords = []
    for i in range(-8, 9):
        for j in range(-8, 9):
            for k in range(-8, 9):
                if (i + j + k) % 2 == 0:
                    coords.append([i, j, k])
    
    coords = np.array(coords)
    # Sort by distance to origin to get the central cluster
    dists = np.sum(coords**2, axis=1)
    sorted_idx = np.argsort(dists)
    points = coords[sorted_idx][:N]
    
    # Scale to approximate LJ minimum distance
    # Nearest neighbor distance in this integer lattice is sqrt(2)
    # Target distance ~ 1.12
    scale = 1.12 / np.sqrt(2)
    points = points * scale

    # 2. Define Energy and Gradient
    def energy_and_grad(pos_flat):
        x = pos_flat.reshape(N, 3)
        
        # Pairwise differences
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        r2 = np.sum(diff**2, axis=2)
        
        # Mask diagonal
        mask = np.eye(N, dtype=bool)
        r2_masked = r2.copy()
        r2_masked[mask] = 1.0 # Avoid div by zero in calculations, will be masked later
        
        r6 = r2_masked**3
        r12 = r6**2
        
        # Energy: 0.5 * sum 4(r^-12 - r^-6)
        e_pairs = 4.0 * (r12 - r6)
        e_pairs[mask] = 0.0
        total_e = 0.5 * np.sum(e_pairs)
        
        # Gradient
        # dE/dx_i = sum_{j!=i} 4 * (-12 r^-14 + 6 r^-8) * (x_i - x_j)
        inv_r2 = 1.0 / r2_masked
        inv_r2[mask] = 0.0
        
        # Coefficient for each pair (i,j)
        # Note: r^-14 = r^-12 * r^-2, r^-8 = r^-6 * r^-2
        coeff = 4.0 * (-12.0 * (r12 * inv_r2) + 6.0 * (r6 * inv_r2))
        coeff[mask] = 0.0
        
        # grad_i = sum_j coeff[i,j] * (x_i - x_j)
        grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        return total_e, grad.flatten()

    # 3. Relaxation
    # Check if we have enough time for relaxation
    if time.time() < deadline - 30.0:
        try:
            res = minimize(
                energy_and_grad,
                points.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            if res.success:
                points = res.x.reshape(N, 3)
        except Exception:
            pass

    # 4. Optional: Basin Hopping if energy is not close to expected global minimum
    # Expected global min ~ -173.9
    # Check time again
    if time.time() < deadline - 10.0:
        # Calculate current energy
        try:
            current_e, _ = energy_and_grad(points.flatten())
        except:
            current_e = 0.0
            
        if current_e > -170.0: # If not close to global min
             # Simple basin hopping with small kicks
             best_e = current_e
             best_pos = points.copy()
             
             for _ in range(5): # Few iterations
                 if time.time() > deadline - 5.0: break
                 
                 # Perturb
                 noise = np.random.normal(0, 0.5, (N, 3))
                 trial_pos = best_pos + noise
                 
                 # Relax
                 try:
                     res = minimize(
                         energy_and_grad,
                         trial_pos.flatten(),
                         jac=True,
                         method='L-BFGS-B',
                         options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
                     )
                     if res.success:
                         trial_pos = res.x.reshape(N, 3)
                         trial_e, _ = energy_and_grad(trial_pos.flatten())
                         if trial_e < best_e:
                             best_e = trial_e
                             best_pos = trial_pos
                 except Exception:
                     pass
             
             points = best_pos

    return points
# EVOLVE-BLOCK-END
