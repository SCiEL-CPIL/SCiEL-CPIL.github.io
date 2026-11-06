# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy_and_grad_vec(positions, rho):
    """Compute Morse energy and gradient using vectorized NumPy operations."""
    n = positions.shape[0]
    if n < 2:
        return 0.0, np.zeros_like(positions)
        
    # Compute all pair differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    
    # Mask for upper triangle (i < j)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    r_valid = r[mask]
    
    if len(r_valid) == 0:
        return 0.0, np.zeros_like(positions)
        
    # Avoid division by zero in unit vectors
    r_safe = np.where(r > 1e-12, r, 1.0)
    
    # Calculate energy terms
    rho_one_minus_r = rho * (1.0 - r_valid)
    exp_val = np.exp(rho_one_minus_r)
    
    e_pair = exp_val * (exp_val - 2.0)
    de_dr = -rho * exp_val * (2.0 * exp_val - 1.0)
    
    energy = np.sum(e_pair)
    
    # Calculate gradient
    # unit_vecs[i, j] = (pos[i] - pos[j]) / r_ij
    unit_vecs = diff / r_safe[:, :, np.newaxis]
    
    # Create matrix of de_dr for all pairs, zero for invalid
    de_dr_matrix = np.zeros((n, n))
    de_dr_matrix[mask] = de_dr
    
    # Force on i from j is de_dr_ij * unit_vec_ij
    # Note: force on i is +de_dr * (r_i - r_j)/r
    # force on j is -de_dr * (r_i - r_j)/r
    # Our diff is pos[i] - pos[j].
    # So force_matrix[i, j] = de_dr_ij * unit_vecs[i, j]
    force_matrix = de_dr_matrix[:, :, np.newaxis] * unit_vecs
    
    # grad[i] = sum_j force_matrix[i, j]
    grad = np.sum(force_matrix, axis=1)
    
    return energy, grad


def _initial_guess(n, rho):
    """Generate a reasonable initial compact cluster using FCC lattice."""
    points = []
    a = 1.0
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    max_shell = int(np.ceil(2.0 * n ** (1.0 / 3.0))) + 2
    candidates = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + b
                    candidates.append(p)
    candidates = np.array(candidates)
    candidates -= np.mean(candidates, axis=0)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    points = candidates[order[:n]]
    return points


def _local_minimize(positions, rho, max_iter=50):
    """Perform local minimization using L-BFGS-B with vectorized gradient."""
    def fun_and_jac(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad_vec(pos, rho)
        return e, g.flatten()
    
    try:
        result = minimize(
            fun=lambda x: fun_and_jac(x)[0],
            x0=positions.flatten(),
            jac=lambda x: fun_and_jac(x)[1],
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        return result.x.reshape(-1, 3), result.fun
    except Exception:
        return positions, _morse_energy_and_grad_vec(positions, rho)[0]


def construct(instance):
    """Find low-energy Morse clusters using Basin Hopping global search."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    if potential != "morse":
        # Fallback
        spacing = 1.0
        rng = np.random.default_rng(size)
        radius = spacing * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                points.append(p)
        energy, positions = relax(np.array(points), potential, rho)
        return positions

    # Basin Hopping Parameters
    rng = np.random.default_rng(42)
    
    # Initialize
    current_pos = _initial_guess(size, rho)
    current_energy, current_pos = _local_minimize(current_pos, rho, max_iter=100)
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    # Temperature for Metropolis acceptance
    T_start = 1.0
    T_end = 0.05
    
    # Number of steps
    if size < 50:
        n_steps = 400
    else:
        n_steps = 250
        
    kick_scale = 0.1
    
    for step in range(n_steps):
        # Anneal temperature
        T = T_start * (T_end / T_start) ** (step / n_steps)
        
        # Kick
        kick = rng.normal(0, kick_scale, size=(size, 3))
        trial_pos = current_pos + kick
        
        # Local Minimize
        trial_energy, trial_pos = _local_minimize(trial_pos, rho, max_iter=30)
        
        # Metropolis Criterion
        delta_e = trial_energy - current_energy
        if delta_e < 0:
            current_pos = trial_pos
            current_energy = trial_energy
        else:
            # Accept with probability exp(-delta_e / T)
            if T > 1e-6:
                accept_prob = np.exp(-delta_e / T)
                if rng.random() < accept_prob:
                    current_pos = trial_pos
                    current_energy = trial_energy
        
        # Update Best
        if trial_energy < best_energy:
            best_energy = trial_energy
            best_pos = trial_pos.copy()
            
    # Final relaxation of the best found
    final_pos, final_energy = _local_minimize(best_pos, rho, max_iter=100)
    
    # Also try the provided relax on the final best to ensure no better local min is missed
    try:
        e_relax, pos_relax = relax(final_pos, 'morse', rho)
        if e_relax < final_energy:
            final_pos = pos_relax
            final_energy = e_relax
    except Exception:
        pass
        
    return final_pos
# EVOLVE-BLOCK-END
