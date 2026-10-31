"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _lj_energy(positions):
    """Compute total Lennard-Jones energy for a set of positions."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    return 4.0 * np.sum(r12 - r6)


def _is_valid(positions):
    """Check if minimum distance is >= 0.5."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    return np.sqrt(np.min(r2)) >= 0.5


def _local_minimize(positions):
    """Perform local minimization using L-BFGS-B."""
    def energy_func(x):
        pos = x.reshape(-1, 3)
        if not _is_valid(pos):
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            r2 = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(r2, 1.0)
            min_r2 = np.min(r2)
            if min_r2 < 0.25:
                return 1e10 + (0.25 - min_r2) * 1e6
            return _lj_energy(pos)
        return _lj_energy(pos)
    
    x0 = positions.flatten()
    result = minimize(energy_func, x0, method='L-BFGS-B', 
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
    
    pos = result.x.reshape(-1, 3)
    if _is_valid(pos):
        return pos, _lj_energy(pos)
    else:
        return pos, 1e10


def _basin_hopping(initial_pos, n_kicks=30, kick_scale=0.1, T=1.0):
    """Run a basin hopping loop."""
    current_pos, current_energy = _local_minimize(initial_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    T_current = T
    for i in range(n_kicks):
        noise = np.random.normal(0, kick_scale, initial_pos.shape)
        perturbed_pos = current_pos + noise
        
        new_pos, new_energy = _local_minimize(perturbed_pos)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T_current):
            current_pos = new_pos
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        
        T_current *= 0.95
        
    return best_pos, best_energy


def _generate_fcc_truncated_octahedron():
    """Generate 38 atoms in FCC truncated octahedron geometry."""
    a = 1.0
    v1 = np.array([a/2, a/2, 0.0])
    v2 = np.array([a/2, 0.0, a/2])
    v3 = np.array([0.0, a/2, a/2])
    
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                p = i * v1 + j * v2 + k * v3
                points.append(p)
    
    points = np.array(points)
    dists = np.sum(points ** 2, axis=1)
    order = np.argsort(dists)
    points = points[order]
    points = points[:N]
    
    scale = 1.122 / 0.707
    return points * scale


def _generate_random_cluster():
    """Generate a random cluster."""
    return np.random.uniform(-4, 4, (N, 3))


def construct():
    starts = [
        _generate_fcc_truncated_octahedron(),
        _generate_random_cluster(),
        _generate_random_cluster()
    ]
    
    best_pos = starts[0]
    best_energy = np.inf
    
    for start in starts:
        try:
            pos, energy = _basin_hopping(start, n_kicks=30, kick_scale=0.1, T=1.0)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos
        except Exception:
            pass
            
    return best_pos
# EVOLVE-BLOCK-END
