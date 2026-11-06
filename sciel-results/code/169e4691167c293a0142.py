"""Morse cluster global minima search with structured initialization and multi-start local optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import morse_energy_gradient, relax


def _morse_energy(pos, rho):
    """Compute Morse potential energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    x = rho * (1.0 - dists)
    ex = np.exp(x)
    energy = np.sum(ex * (ex - 2.0)) / 2.0
    return float(energy)


def _morse_energy_grad(pos, rho):
    """Compute Morse potential energy and gradient."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    x = rho * (1.0 - dists)
    ex = np.exp(x)
    # energy = sum_{i<j} ex*(ex-2)/2
    energy = np.sum(ex * (ex - 2.0)) / 2.0
    # gradient: dE/dpos_i = sum_{j!=i} ex*(ex-1)*rho/dists_ij * (pos_i - pos_j)
    # Actually: d/dr [exp(rho(1-r))*(exp(rho(1-r))-2)] = -rho*exp(rho(1-r))*(exp(rho(1-r))-1)
    # dE/dpos_i = sum_{j!=i} d/dr_ij * (pos_i - pos_j)/r_ij
    # d/dr_ij of [exp(x)*(exp(x)-2)] where x=rho(1-r) is -rho*exp(x)*(exp(x)-1)
    grad_factor = -rho * ex * (ex - 1.0) / dists
    np.fill_diagonal(grad_factor, 0.0)
    # grad[i] = sum_j grad_factor[i,j] * (pos[i] - pos[j])
    # = sum_j grad_factor[i,j] * pos[i] - sum_j grad_factor[i,j] * pos[j]
    # = pos[i] * sum_j grad_factor[i,j] - sum_j grad_factor[i,j] * pos[j]
    row_sum = np.sum(grad_factor, axis=1, keepdims=True)
    grad = row_sum * pos - grad_factor @ pos
    return float(energy), grad


def _lattice_init(n, lattice_type='fcc'):
    """Generate initial positions on a lattice structure."""
    if lattice_type == 'fcc':
        # FCC lattice: atoms at corners and face centers of cubic cells
        positions = []
        # Generate FCC points in a cube
        a = 1.0  # lattice constant
        # For FCC, conventional cell has 4 atoms
        # We'll generate enough and take first n
        max_idx = int(np.ceil(n / 4.0)) + 2
        for i in range(max_idx):
            for j in range(max_idx):
                for k in range(max_idx):
                    # Corner
                    positions.append([i*a, j*a, k*a])
                    # Face centers
                    if i < max_idx - 1:
                        positions.append([(i+0.5)*a, j*a, k*a])
                    if j < max_idx - 1:
                        positions.append([i*a, (j+0.5)*a, k*a])
                    if k < max_idx - 1:
                        positions.append([i*a, j*a, (k+0.5)*a])
        positions = np.array(positions[:n])
    elif lattice_type == 'bcc':
        positions = []
        a = 1.0
        max_idx = int(np.ceil(n / 2.0)) + 2
        for i in range(max_idx):
            for j in range(max_idx):
                for k in range(max_idx):
                    # Corner
                    positions.append([i*a, j*a, k*a])
                    # Body center
                    positions.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
        positions = np.array(positions[:n])
    else:  # simple cubic
        positions = []
        a = 1.0
        max_idx = int(np.ceil(n ** (1.0/3.0))) + 2
        for i in range(max_idx):
            for j in range(max_idx):
                for k in range(max_idx):
                    positions.append([i*a, j*a, k*a])
        positions = np.array(positions[:n])
    
    # Center the cluster
    positions -= np.mean(positions, axis=0)
    return positions


def _random_packed_init(n, spacing=1.0, seed=None):
    """Generate random packed initial positions."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    # If we couldn't get enough, just fill randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _shrink_to_min_dist(pos, min_dist=0.5):
    """Ensure all pairwise distances are at least min_dist by shrinking."""
    n = len(pos)
    if n < 2:
        return pos
    # Find minimum distance
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1e10)
    min_d = np.min(dists)
    if min_d < min_dist:
        scale = (min_dist / min_d) * 1.01
        pos = pos * scale
    return pos


def construct(instance):
    """Find low-energy Morse cluster configurations using multi-start local optimization."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter if isinstance(parameter, (int, float)) else 3
    
    start_time = time.time()
    deadline = start_time + 280  # 50s safety margin
    
    # Generate diverse initial configurations
    inits = []
    
    # Lattice-based inits (physics-informed)
    for lattice in ['fcc', 'bcc', 'sc']:
        pos = _lattice_init(size, lattice)
        pos = _shrink_to_min_dist(pos, 0.5)
        inits.append(pos)
    
    # Random packed inits with different seeds
    for seed in [42, 123, 777, 2024]:
        pos = _random_packed_init(size, spacing=1.0, seed=seed)
        pos = _shrink_to_min_dist(pos, 0.5)
        inits.append(pos)
    
    # Scaled lattice inits (slightly perturbed)
    for lattice in ['fcc', 'bcc']:
        pos = _lattice_init(size, lattice)
        rng = np.random.default_rng(hash(lattice) % 2**32)
        pos += rng.normal(0, 0.1, pos.shape)
        pos = _shrink_to_min_dist(pos, 0.5)
        inits.append(pos)
    
    best_energy = np.inf
    best_positions = inits[0].copy()
    
    for idx, init_pos in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Try local optimization with L-BFGS-B
        try:
            result = minimize(
                _morse_energy_grad,
                init_pos,
                args=(rho,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            energy = result.fun
            if energy < best_energy:
                best_energy = energy
                best_positions = result.x.copy()
        except Exception:
            pass
        
        # Also try the provided relax function
        if time.time() > deadline:
            break
        try:
            energy2, pos2 = relax(init_pos.copy(), potential, parameter)
            if energy2 < best_energy:
                best_energy = energy2
                best_positions = pos2.copy()
        except Exception:
            pass
    
    # Final check: ensure minimum distance constraint
    best_positions = _shrink_to_min_dist(best_positions, 0.5)
    
    return best_positions
# EVOLVE-BLOCK-END
