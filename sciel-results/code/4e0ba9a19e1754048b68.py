"""Construct a 55-atom LJ cluster using basin-hopping global optimization."""
import numpy as np
from scipy.optimize import minimize

N = 55

def _lj_energy_and_grad(positions):
    """Compute LJ energy and gradient using vectorized operations."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    r2_safe = np.where(r2 < 1e-12, 1e-12, r2)
    inv_r6 = 1.0 / (r2_safe ** 3)
    inv_r12 = inv_r6 * inv_r6
    energy = 4.0 * np.sum(np.triu(inv_r12 - inv_r6, k=1))
    
    inv_r8 = inv_r6 * (1.0 / r2_safe)
    inv_r14 = inv_r12 * (1.0 / r2_safe)
    coeff = 48.0 * inv_r14 - 24.0 * inv_r8
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _relax(positions, maxiter=300):
    """Relax positions using L-BFGS-B with gradient."""
    def func_and_grad(x):
        pos = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos)
        return e, g.ravel()
    
    try:
        result = minimize(func_and_grad, positions.ravel(), method='L-BFGS-B', 
                         jac=True, options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
        return result.x.reshape(-1, 3)
    except Exception:
        return positions

def _check_separation(positions, min_sep=0.5):
    """Check if all pairs are at least min_sep apart."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, float('inf'))
    return np.min(dists) >= min_sep - 1e-9

def _fix_separation(positions, min_sep=0.5, max_iter=200):
    """Iteratively fix pairs that are too close."""
    for _ in range(max_iter):
        if _check_separation(positions, min_sep):
            break
        n = len(positions)
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, float('inf'))
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        if i > j:
            i, j = j, i
        d = dists[i, j]
        if d < 1e-12:
            positions[i] += np.array([1e-6, 0, 0])
        else:
            direction = (positions[i] - positions[j]) / d
            needed = min_sep - d
            positions[i] += direction * needed / 2
            positions[j] -= direction * needed / 2
    return positions

def _random_initial_config(n_atoms, scale=3.0):
    """Generate a random initial configuration within a sphere."""
    theta = np.random.uniform(0, 2 * np.pi, n_atoms)
    phi = np.random.uniform(0, np.pi, n_atoms)
    r = scale * np.random.uniform(0, 1, n_atoms) ** (1.0/3.0)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.column_stack([x, y, z])

def _fcc_fragment(n_atoms, spacing=1.6):
    """Generate an FCC lattice fragment as initial configuration."""
    positions = []
    a = spacing
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                positions.append(np.array([i*a, j*a, k*a]))
                if i < 5 and j < 5:
                    positions.append(np.array([(i+0.5)*a, (j+0.5)*a, k*a]))
                if i < 5 and k < 5:
                    positions.append(np.array([(i+0.5)*a, j*a, (k+0.5)*a]))
                if j < 5 and k < 5:
                    positions.append(np.array([i*a, (j+0.5)*a, (k+0.5)*a]))
    
    positions = np.array(positions)
    positions -= np.mean(positions, axis=0)
    
    dists = np.linalg.norm(positions, axis=1)
    indices = np.argsort(dists)[:n_atoms]
    return positions[indices]

def _basin_hopping(n_atoms, n_iterations=80, perturbation=0.4, temp_start=3.0, temp_end=0.05):
    """Basin-hopping algorithm for global optimization."""
    initial_configs = []
    fcc = _fcc_fragment(n_atoms)
    initial_configs.append(fcc)
    for _ in range(3):
        rand = _random_initial_config(n_atoms, scale=4.0)
        initial_configs.append(rand)
    
    best_energy = float('inf')
    best_pos = None
    
    for config in initial_configs:
        relaxed = _relax(config, maxiter=100)
        relaxed = _fix_separation(relaxed)
        energy, _ = _lj_energy_and_grad(relaxed)
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
    
    current_pos = best_pos.copy()
    current_energy = best_energy
    
    for it in range(n_iterations):
        temp = temp_start * (temp_end / temp_start) ** (it / max(1, n_iterations - 1))
        
        perturb_scale = perturbation * (1.0 + 0.5 * np.random.random())
        noise = np.random.normal(0, perturb_scale, size=current_pos.shape)
        perturbed = current_pos + noise
        perturbed = _fix_separation(perturbed)
        
        relaxed = _relax(perturbed, maxiter=200)
        relaxed = _fix_separation(relaxed)
        new_energy, _ = _lj_energy_and_grad(relaxed)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / max(temp, 1e-10)):
            current_pos = relaxed
            current_energy = new_energy
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
    
    return best_pos, best_energy

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """Construct 55-atom LJ cluster using basin-hopping global optimization."""
    np.random.seed(42)
    
    n_atoms = N
    best_pos, best_energy = _basin_hopping(n_atoms, n_iterations=80, perturbation=0.4, temp_start=3.0, temp_end=0.05)
    
    final_pos = _relax(best_pos, maxiter=500)
    final_pos = _fix_separation(final_pos)
    
    if not _check_separation(final_pos, 0.5):
        final_pos = _fix_separation(final_pos, max_iter=500)
    
    return final_pos
# EVOLVE-BLOCK-END
