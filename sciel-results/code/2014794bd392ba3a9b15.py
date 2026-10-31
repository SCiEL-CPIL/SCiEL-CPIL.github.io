"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution

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

def _penalized_energy_flat(x):
    """Energy function for DE, flattened, with penalty for overlap."""
    pos = x.reshape(-1, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    min_r2 = np.min(r2)
    
    # Penalty for overlap
    if min_r2 < 0.25:
        return 1e10 + (0.25 - min_r2) * 1e6
    
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    return 4.0 * np.sum(r12 - r6)

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
    try:
        result = minimize(energy_func, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        pos = result.x.reshape(-1, 3)
        if _is_valid(pos):
            return pos, _lj_energy(pos)
    except Exception:
        pass
    return positions, _lj_energy(positions)

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

def construct():
    # Bounds for Differential Evolution
    bounds = [(-5, 5)] * (N * 3)
    
    best_pos = _generate_fcc_truncated_octahedron()
    best_energy = _lj_energy(best_pos)
    
    # Run Differential Evolution
    try:
        de_result = differential_evolution(
            _penalized_energy_flat, 
            bounds, 
            maxiter=1000, 
            popsize=20, 
            tol=1e-8, 
            mutation=(0.5, 1.0),
            recombination=0.9,
            seed=42
        )
        
        if de_result.fun < best_energy:
            pos = de_result.x.reshape(-1, 3)
            # Polish with local minimizer
            polished_pos, polished_energy = _local_minimize(pos)
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_pos = polished_pos
    except Exception:
        pass
    
    # Fallback: Random restarts with local minimization
    for _ in range(10):
        random_pos = np.random.uniform(-4, 4, (N, 3))
        pos, energy = _local_minimize(random_pos)
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
            
    return best_pos
# EVOLVE-BLOCK-END
