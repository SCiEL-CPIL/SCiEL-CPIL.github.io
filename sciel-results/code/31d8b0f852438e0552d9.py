"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(positions, rho):
    """Compute exact Morse energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    iu = np.triu_indices(n, k=1)
    return float(np.sum(e[iu]))


def _morse_energy_fast(positions, rho):
    """Faster energy using upper triangle only."""
    n = len(positions)
    if n < 2:
        return 0.0
    iu = np.triu_indices(n, k=1)
    diff = positions[iu[0]] - positions[iu[1]]
    r = np.linalg.norm(diff, axis=1)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    return float(np.sum(ex * (ex - 2.0)))


def _morse_gradient(positions, rho):
    """Compute Morse energy gradient."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    iu = np.triu_indices(n, k=1)
    diff = positions[iu[0]] - positions[iu[1]]
    r = np.linalg.norm(diff, axis=1)
    r = np.maximum(r, 1e-12)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    # dE/dr = ex * (ex - 2) * (-rho)
    de_dr = -rho * ex * (ex - 2.0)
    # gradient contribution
    grad = np.zeros_like(positions)
    # dE/dpos_i = de_dr * (r_i - r_j)/r
    unit = diff / r[:, np.newaxis]
    contrib = de_dr[:, np.newaxis] * unit
    np.add.at(grad, iu[0], contrib)
    np.add.at(grad, iu[1], -contrib)
    return grad


def _is_valid(positions, min_dist=0.5):
    """Check if all pairs are at least min_dist apart."""
    n = len(positions)
    if n < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    return np.min(r) >= min_dist


def _make_fcc_cluster(n):
    """Generate an FCC-like cluster of n atoms."""
    # FCC lattice points
    a = 1.0
    lattice_points = []
    # Generate FCC points in a cube
    for i in range(8):
        for j in range(8):
            for k in range(8):
                # Corner
                lattice_points.append(np.array([i, j, k]) * a)
                # Face centers
                lattice_points.append(np.array([i + 0.5, j + 0.5, k]) * a)
                lattice_points.append(np.array([i + 0.5, j, k + 0.5]) * a)
                lattice_points.append(np.array([i, j + 0.5, k + 0.5]) * a)
    
    # Sort by distance from center and take n closest
    lattice_points = np.array(lattice_points)
    center = lattice_points.mean(axis=0)
    dists = np.linalg.norm(lattice_points - center, axis=1)
    idx = np.argsort(dists)[:n]
    points = lattice_points[idx] - center
    return points


def _make_bcc_cluster(n):
    """Generate a BCC-like cluster of n atoms."""
    lattice_points = []
    a = 1.0
    for i in range(8):
        for j in range(8):
            for k in range(8):
                lattice_points.append(np.array([i, j, k]) * a)
                lattice_points.append((np.array([i, j, k]) + 0.5) * a)
    
    lattice_points = np.array(lattice_points)
    center = lattice_points.mean(axis=0)
    dists = np.linalg.norm(lattice_points - center, axis=1)
    idx = np.argsort(dists)[:n]
    points = lattice_points[idx] - center
    return points


def _make_random_compact(n, rng, spacing=1.0):
    """Generate random compact cluster."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 1000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.7 * spacing for q in points):
                points.append(p)
        attempts += 1
    # If we didn't get enough, add random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _bfgs_relax(positions, rho, max_iter=200):
    """Relax using BFGS with the provided gradient."""
    def objective(x):
        pos = x.reshape(-1, 3)
        return _morse_energy_fast(pos, rho)
    
    def gradient(x):
        pos = x.reshape(-1, 3)
        g = _morse_gradient(pos, rho)
        return g.flatten()
    
    try:
        result = minimize(objective, positions.flatten(), jac=gradient, method='BFGS',
                         options={'maxiter': max_iter, 'gtol': 1e-6})
        return result.fun, result.x.reshape(-1, 3)
    except Exception:
        return _morse_energy_fast(positions, rho), positions


def _simulate_annealing(positions, rho, rng, n_steps=500, t_start=1.0, t_end=0.01):
    """Simple simulated annealing."""
    current_energy = _morse_energy_fast(positions, rho)
    best_energy = current_energy
    best_positions = positions.copy()
    
    for step in range(n_steps):
        t = t_start * (t_end / t_start) ** (step / n_steps)
        # Random perturbation
        perturbation = rng.normal(0, 0.1, positions.shape)
        new_positions = positions + perturbation
        
        new_energy = _morse_energy_fast(new_positions, rho)
        
        if new_energy < current_energy or rng.random() < np.exp(-(new_energy - current_energy) / max(t, 1e-10)):
            positions = new_positions
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = positions.copy()
    
    return best_energy, best_positions


def construct(instance):
    """Find low-energy arrangements using multiple strategies."""
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 6))
    
    rng = np.random.default_rng(42)
    
    best_energy = np.inf
    best_positions = None
    
    # Strategy 1: FCC lattice start
    fcc_points = _make_fcc_cluster(size)
    energy, positions = _bfgs_relax(fcc_points, rho)
    if _is_valid(positions):
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
    
    # Check time
    if time.time() > deadline:
        return best_positions if best_positions is not None else fcc_points
    
    # Strategy 2: BCC lattice start
    bcc_points = _make_bcc_cluster(size)
    energy, positions = _bfgs_relax(bcc_points, rho)
    if _is_valid(positions):
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
    
    if time.time() > deadline:
        return best_positions if best_positions is not None else bcc_points
    
    # Strategy 3: Random compact starts with relaxation
    for i in range(8):
        if time.time() > deadline:
            break
        random_points = _make_random_compact(size, rng)
        energy, positions = _bfgs_relax(random_points, rho)
        if _is_valid(positions):
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
    
    if time.time() > deadline:
        return best_positions if best_positions is not None else random_points
    
    # Strategy 4: Simulated annealing from best position
    if best_positions is not None:
        sa_energy, sa_positions = _simulate_annealing(best_positions, rho, rng, n_steps=300)
        if _is_valid(sa_positions):
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_positions = sa_positions.copy()
    
    # Final relaxation
    if best_positions is not None:
        final_energy, final_positions = _bfgs_relax(best_positions, rho, max_iter=300)
        if _is_valid(final_positions):
            best_positions = final_positions
    
    return best_positions if best_positions is not None else _make_fcc_cluster(size)
# EVOLVE-BLOCK-END
