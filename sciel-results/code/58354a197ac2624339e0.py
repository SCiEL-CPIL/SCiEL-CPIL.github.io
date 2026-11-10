"""Morse cluster global minima: crystalline seeds + L-BFGS-B with deadline guard."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Deadline guard
TOTAL_DEADLINE = 290.0
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _remaining():
    return max(0.0, TOTAL_DEADLINE - _elapsed())


def _morse_energy(positions, rho):
    """Compute Morse energy."""
    n = positions.shape[0]
    d = squareform(pdist(positions))
    np.fill_diagonal(d, 1.0)
    r = np.where(d > 1e-10, d, 1.0)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    return 0.5 * np.sum(v)


def _morse_energy_grad(positions, rho):
    """Compute Morse energy and gradient."""
    n = positions.shape[0]
    d = squareform(pdist(positions))
    np.fill_diagonal(d, 1.0)
    r = np.where(d > 1e-10, d, 1.0)
    
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    energy = 0.5 * np.sum(v)
    
    # Gradient: dV/dr = -rho * ex * (ex - 1)
    dvdr = -rho * ex * (ex - 1.0)
    
    # Build gradient vector
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            ri = r[i, j]
            if ri > 1e-10:
                grad[i] += dvdr[i, j] * rij / ri
                grad[j] -= dvdr[i, j] * rij / ri
    
    return energy, grad


def _is_valid(positions, min_dist=0.5):
    """Check if positions satisfy minimum distance constraint."""
    n = positions.shape[0]
    d = squareform(pdist(positions))
    np.fill_diagonal(d, np.inf)
    return np.min(d) >= min_dist


def _generate_fcc(n, spacing=1.0):
    """Generate FCC-like initial configuration."""
    # FCC unit cell vectors
    a = spacing
    # Generate points in a cubic box and keep those in FCC positions
    points = []
    # FCC lattice points in unit cell: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    fcc_offsets = [
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ]
    
    # Estimate box size needed
    # Volume per atom in FCC = a^3/4
    # Box side ~ (4*n)^(1/3) * a
    box_side = int(np.ceil((4 * n) ** (1.0/3.0))) + 2
    
    for ix in range(box_side):
        for iy in range(box_side):
            for iz in range(box_side):
                for off in fcc_offsets:
                    p = np.array([ix + off[0], iy + off[1], iz + off[2]]) * a
                    points.append(p)
                    if len(points) >= n:
                        break
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    points = np.array(points[:n])
    # Center the cluster
    points -= points.mean(axis=0)
    return points


def _generate_hcp(n, spacing=1.0):
    """Generate HCP-like initial configuration."""
    # HCP lattice
    a = spacing
    c = np.sqrt(8.0/3.0) * a  # ideal c/a ratio
    
    points = []
    # AB stacking
    # Layer A: triangular lattice
    # Layer B: shifted by (a/3, a/(3*sqrt(3)), c/2)
    
    # Generate triangular lattice points
    a_vec1 = np.array([a, 0, 0])
    a_vec2 = np.array([a/2, a*np.sqrt(3)/2, 0])
    
    # Estimate number of layers needed
    # Each layer has ~ n/2 points (ABAB...)
    points_per_layer = max(1, int(np.ceil(np.sqrt(n * 2 / 3.0))))
    num_layers = int(np.ceil(n / points_per_layer)) + 2
    
    for layer in range(num_layers):
        z = layer * c / 2.0
        # Determine offset for this layer
        if layer % 2 == 0:
            offset = np.array([0, 0, 0])
        else:
            offset = np.array([a/3.0, a/(3.0*np.sqrt(3.0)), 0])
        
        for ix in range(-points_per_layer//2, points_per_layer//2 + 1):
            for iy in range(-points_per_layer//2, points_per_layer//2 + 1):
                p = ix * a_vec1 + iy * a_vec2 + offset + np.array([0, 0, z])
                points.append(p)
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _generate_bcc(n, spacing=1.0):
    """Generate BCC-like initial configuration."""
    a = spacing * (3.0 / 4.0) ** (1.0/3.0)  # Adjust for packing
    points = []
    
    bcc_offsets = [
        [0, 0, 0],
        [0.5, 0.5, 0.5]
    ]
    
    box_side = int(np.ceil((2 * n) ** (1.0/3.0))) + 2
    
    for ix in range(box_side):
        for iy in range(box_side):
            for iz in range(box_side):
                for off in bcc_offsets:
                    p = np.array([ix + off[0], iy + off[1], iz + off[2]]) * a
                    points.append(p)
                    if len(points) >= n:
                        break
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _generate_random_compact(n, spacing=1.0, seed=None):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.7 * spacing for q in points):
            points.append(p)
    return np.array(points)


def _relax_lbfgs(positions, rho, max_time=10.0):
    """Relax using L-BFGS-B with time limit."""
    start = time.time()
    
    def objective(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_grad(pos, rho)
        return e, g.flatten()
    
    x0 = positions.flatten()
    
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
    )
    
    elapsed = time.time() - start
    if elapsed > max_time:
        return positions, _morse_energy(positions, rho), False
    
    pos = result.x.reshape(-1, 3)
    energy = _morse_energy(pos, rho)
    
    # Check validity
    if not _is_valid(pos, 0.5):
        return positions, _morse_energy(positions, rho), False
    
    return pos, energy, True


def _relax_lib(positions, rho, max_time=5.0):
    """Relax using the library's relax function with time limit."""
    start = time.time()
    energy, pos = relax(positions, 'morse', rho)
    elapsed = time.time() - start
    return pos, energy, elapsed <= max_time


def construct(instance):
    """Find global minimum of Morse cluster."""
    global START_TIME
    START_TIME = time.time()
    
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    best_energy = np.inf
    best_positions = None
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # Crystalline structures
    initial_configs.append(_generate_fcc(n, 1.0))
    initial_configs.append(_generate_hcp(n, 1.0))
    initial_configs.append(_generate_bcc(n, 1.0))
    
    # Random compact configurations with different seeds
    for seed in [n, n + 1, n + 2, n + 3, n + 4]:
        initial_configs.append(_generate_random_compact(n, 1.0, seed=seed))
    
    # Slightly perturbed versions
    for i, config in enumerate(initial_configs[:3]):
        perturbed = config + 0.05 * np.random.default_rng(i).normal(size=config.shape)
        initial_configs.append(perturbed)
    
    # Try each configuration
    for config in initial_configs:
        if _remaining() < 15.0:
            break
        
        # First try library relax (fast)
        pos_lib, energy_lib, ok_lib = _relax_lib(config, rho, max_time=8.0)
        if ok_lib and energy_lib < best_energy:
            best_energy = energy_lib
            best_positions = pos_lib
        
        # Then try L-BFGS-B for better convergence
        if _remaining() < 10.0:
            continue
        
        pos_lbfgs, energy_lbfgs, ok_lbfgs = _relax_lbfgs(config, rho, max_time=12.0)
        if ok_lbfgs and energy_lbfgs < best_energy:
            best_energy = energy_lbfgs
            best_positions = pos_lbfgs
    
    # If no valid result found, return a simple valid configuration
    if best_positions is None or not _is_valid(best_positions, 0.5):
        # Generate a valid configuration as fallback
        rng = np.random.default_rng(42)
        radius = 1.0 * n ** (1.0/3.0) * 1.5
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        best_positions = np.array(points)
        best_energy = _morse_energy(best_positions, rho)
    
    # Final check
    if not _is_valid(best_positions, 0.5):
        # Last resort: spread points evenly
        best_positions = np.zeros((n, 3))
        for i in range(n):
            best_positions[i] = [i * 0.6, 0, 0]
    
    return best_positions
