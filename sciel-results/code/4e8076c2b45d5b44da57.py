"""Morse cluster global minima search using Simulated Annealing with local relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_TIME_LIMIT = 290.0  # 300s limit, keep 10s safety margin


def _morse_energy(positions, rho):
    """Compute exact Morse energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
    mask = ~np.eye(n, dtype=bool)
    exp_term = np.exp(rho * (1.0 - dist))
    energy_pairs = exp_term * (exp_term - 2.0)
    return 0.5 * np.sum(energy_pairs * mask)


def _relax_with_gradient(positions, rho, max_iter=200):
    """Relax positions using L-BFGS-B with exact gradient."""
    def objective(x):
        pos = x.reshape(-1, 3)
        e, g = morse_energy_gradient(pos, rho)
        return e, g.ravel()
    
    x0 = positions.ravel()
    try:
        result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': max_iter, 'gtol': 1e-5, 'maxls': 10})
        return result.fun, result.x.reshape(-1, 3)
    except Exception:
        # Fallback to library relax if minimize fails
        energy, pos = relax(positions, 'morse', rho)
        return energy, pos


def _is_valid(positions, min_dist=0.5):
    """Check if all pairwise distances are >= min_dist."""
    n = len(positions)
    if n < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    return np.min(dist) >= min_dist - 1e-6


def _generate_initial(n, seed=42):
    """Generate random initial positions within a sphere."""
    rng = np.random.default_rng(seed)
    # Estimate radius for n points with spacing ~1
    radius = (n / (4/3 * np.pi)) ** (1/3) * 1.5
    points = rng.uniform(-radius, radius, (n, 3))
    # Center
    points -= np.mean(points, axis=0)
    return points


def construct(instance):
    """Simulated Annealing with local relaxation."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    if time.time() - _START_TIME > _TIME_LIMIT - 5:
        return np.zeros((size, 3))
    
    # Initial guess
    current_pos = _generate_initial(size, seed=42)
    current_energy, current_pos = _relax_with_gradient(current_pos, rho)
    
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    # SA Parameters
    T_start = 1.0
    T_end = 0.01
    alpha = 0.95  # Cooling rate
    
    T = T_start
    rng = np.random.default_rng(42)
    
    # Estimate time per iteration to determine number of steps
    # We will run iterations until time limit is reached
    while time.time() - _START_TIME < _TIME_LIMIT - 10:
        # Perturbation: move a random subset of atoms
        n_perturb = max(1, size // 10)
        indices = rng.choice(size, n_perturb, replace=False)
        
        new_pos = current_pos.copy()
        # Gaussian perturbation
        scale = 0.1 * T + 0.01
        new_pos[indices] += rng.normal(0, scale, (n_perturb, 3))
        
        # Relax
        new_energy, new_pos_relaxed = _relax_with_gradient(new_pos, rho)
        
        # Check validity
        if not _is_valid(new_pos_relaxed):
            # If invalid, try to fix by nudging or reject
            # For simplicity, reject if invalid
            continue
            
        # Metropolis criterion
        delta_E = new_energy - current_energy
        if delta_E < 0 or rng.random() < np.exp(-delta_E / max(T, 1e-9)):
            current_pos = new_pos_relaxed
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        
        # Cool
        T *= alpha
        if T < T_end:
            break
            
    return best_pos
# EVOLVE-BLOCK-END
