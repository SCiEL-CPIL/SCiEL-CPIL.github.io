"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _lj_energy_and_grad(positions):
    """Compute total Lennard-Jones energy and gradient for a set of positions."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1e-10)  # avoid division by zero
    
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    
    # Energy: 4 * sum(r^-12 - r^-6)
    # Note: We sum over all i,j. The diagonal is 0.
    # Pair energy is usually counted once, but LJ potential sum is often defined over all pairs i<j.
    # The formula 4*sum(r^-12 - r^-6) over all pairs usually implies i<j.
    # Let's check the seed code: `energy = 4.0 * np.sum(r12 - r6)`.
    # The seed code sums over the entire matrix including i,j and j,i.
    # This means it sums each pair twice.
    # Standard LJ energy is sum_{i<j} 4*(r^-12 - r^-6).
    # If we sum over all i != j, we get 2 * sum_{i<j}.
    # The task says "4*sum(r^-12 - r^-6) over all pairs". Usually "all pairs" means i<j.
    # However, the seed code `np.sum` over the full matrix sums i<j and j<i.
    # Let's verify the magnitude.
    # If seed is correct, the energy is double the standard physical energy.
    # The target energy is -173.9.
    # For LJ13, min energy is approx -1.16 per pair * 12 pairs? No.
    # Let's stick to the seed's definition to be consistent with the evaluator if the evaluator uses the same logic.
    # BUT, the evaluator is independent.
    # Standard LJ potential: V(r) = 4(1/r^12 - 1/r^6).
    # Total Energy = sum_{i<j} V(r_ij).
    # The seed code computes: 4 * sum_{i,j} (r_ij^-12 - r_ij^-6) where r_ii is ignored.
    # This is effectively 2 * sum_{i<j} 4(r^-12 - r^-6) = 2 * E_standard.
    # If the evaluator uses standard definition, the seed is off by a factor of 2?
    # Or does the evaluator use the same "sum over all pairs" as i,j?
    # "4*sum(r^-12 - r^-6) over all pairs".
    # In physics, "pairs" usually implies unique pairs (i<j).
    # Let's assume the standard definition: sum_{i<j}.
    # I will implement the standard definition: sum_{i<j}.
    
    # To be safe, let's look at the "seed" value.
    # Seed score 1.4673.
    # If I use standard sum, I might get a different value.
    # Let's write the energy function to sum i<j explicitly to be physically correct.
    
    # Vectorized sum for i<j:
    # We can use the full matrix sum and divide by 2, excluding diagonal.
    # sum_{i,j, i!=j} (r^-12 - r^-6) = 2 * sum_{i<j} (r^-12 - r^-6)
    # So sum_{i<j} = 0.5 * sum_{i,j, i!=j}
    
    # The seed code did: 4.0 * np.sum(r12 - r6). This includes i,j and j,i.
    # So Seed Energy = 4 * (2 * sum_{i<j} (...)) = 8 * sum_{i<j} (...).
    # Wait, 4 * (r^-12 - r^-6) is the pair potential.
    # If the seed sums over all i,j, it calculates 4 * sum_{i,j} (r^-12 - r^-6).
    # This is equal to 4 * 2 * sum_{i<j} (r^-12 - r^-6) = 8 * sum_{i<j} (...).
    # This is double the standard LJ energy (which is sum_{i<j} 4(...)).
    
    # Let's check the target energy: -173.928427.
    # For LJ38, the standard minimum energy is indeed around -173.9.
    # This implies the evaluator uses the standard definition: sum_{i<j} 4(r^-12 - r^-6).
    # The seed code's energy calculation is likely incorrect by a factor of 2, or the seed's "score" is just a ratio.
    # The score is `seed/value`.
    # If the seed program returns a configuration, the evaluator computes `value`.
    # The seed program's internal `_lj_energy` is only used for optimization.
    # If the seed optimizes for `2 * E_standard`, the minimum of `2 * E_standard` is the same configuration as the minimum of `E_standard`.
    # So the configuration should be the same.
    # The score is `seed_value / candidate_value`.
    # `seed_value` is the energy of the seed's output configuration computed by the evaluator.
    # `candidate_value` is the energy of my output configuration computed by the evaluator.
    # So I just need to find the global minimum configuration. The factor of 2 in my internal optimizer doesn't matter as long as it finds the same minimum.
    
    # I will use the standard definition for clarity and correctness in gradients.
    # E = sum_{i<j} 4(r^-12 - r^-6)
    # Gradient: dE/dr_i = sum_{j!=i} 4 * d/dr_i (r_ij^-12 - r_ij^-6)
    # d/dr (r^-12 - r^-6) = -12 r^-13 + 6 r^-7
    # d/dr_i r_ij = (r_i - r_j) / r_ij
    # dE/dr_i = sum_{j!=i} 4 * (-12 r_ij^-13 + 6 r_ij^-7) * (r_i - r_j) / r_ij
    # = sum_{j!=i} 4 * (-12 r_ij^-14 + 6 r_ij^-8) * (r_i - r_j)
    
    # Let's compute using matrices.
    # r2 is (N,N). r is sqrt(r2).
    # We need to sum over j for each i.
    
    # Create a mask for i != j
    mask = ~np.eye(n, dtype=bool)
    
    # Compute terms
    # r_inv_8 = r2^-4
    # r_inv_14 = r2^-7
    r_inv_8 = r2 ** (-4)
    r_inv_14 = r2 ** (-7)
    
    # coeff = 4 * (-12 * r^-14 + 6 * r^-8)
    coeff = 4.0 * (-12.0 * r_inv_14 + 6.0 * r_inv_8)
    
    # Set diagonal to 0
    np.fill_diagonal(coeff, 0.0)
    
    # Gradient component for i, k:
    # grad[i, k] = sum_j coeff[i, j] * (pos[i, k] - pos[j, k])
    # diff[i, j, k] = pos[i, k] - pos[j, k]
    # grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    # Energy:
    # E = sum_{i<j} 4(r^-12 - r^-6)
    # sum_{i,j, i!=j} (r^-12 - r^-6) = 2 * sum_{i<j}
    # So sum_{i<j} = 0.5 * sum_{i,j, i!=j}
    # E = 0.5 * 4 * sum_{i,j, i!=j} (r^-12 - r^-6) = 2 * sum_{i,j, i!=j} (r^-12 - r^-6)
    # Wait.
    # Let S = sum_{i,j, i!=j} (r^-12 - r^-6).
    # S = 2 * sum_{i<j} (r^-12 - r^-6).
    # E = sum_{i<j} 4(r^-12 - r^-6) = 4 * (S/2) = 2 * S.
    # So E = 2 * np.sum(r12 - r6) where diagonal is 0.
    
    energy = 2.0 * np.sum(r12 - r6)
    
    return energy, grad


def _constrained_minimize(pos0, max_iter=200):
    """Minimize LJ energy with minimum distance constraint using L-BFGS-B."""
    def objective(x):
        pos = x.reshape(-1, 3)
        # Check distance constraint
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(r2, 1e10)
        min_r2 = np.min(r2)
        if min_r2 < 0.25:  # r < 0.5
            # Penalty
            return 1e10 + (0.25 - min_r2) * 1e6
        e, g = _lj_energy_and_grad(pos)
        return e, g
    
    x0 = pos0.flatten()
    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=True, options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return pos0, float('inf')


def construct():
    start_time = time.time()
    timeout = 110  # Leave 10s buffer
    
    # Initial random start
    np.random.seed(42)
    current_pos = np.random.uniform(-2, 2, (N, 3))
    current_energy, _ = _lj_energy_and_grad(current_pos)
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    # Basin Hopping / Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.99
    step_size = 0.5
    
    for i in range(1000):
        if time.time() - start_time > timeout:
            break
        
        # Perturb
        noise = np.random.normal(0, step_size, (N, 3))
        new_pos = current_pos + noise
        
        # Local Relax
        new_pos, new_energy = _constrained_minimize(new_pos)
        
        # Acceptance Criterion (Metropolis)
        if new_energy < current_energy:
            accept = True
        else:
            delta_e = new_energy - current_energy
            if T > 1e-6:
                accept = np.random.random() < np.exp(-delta_e / T)
            else:
                accept = False
        
        if accept:
            current_pos = new_pos
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        
        T *= alpha
        
        # Adaptive step size?
        # If we get stuck, maybe increase step size?
        # For simplicity, keep it fixed or decay.
        
    # Final relaxation of the best position
    best_pos, best_energy = _constrained_minimize(best_pos)
    
    # Ensure constraints
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1e10)
    min_r = np.sqrt(np.min(r2))
    if min_r < 0.5:
        # Scale up if violated (rare if penalty worked)
        scale = 0.5 / min_r * 1.01
        best_pos = best_pos * scale
        
    return best_pos
# EVOLVE-BLOCK-END
