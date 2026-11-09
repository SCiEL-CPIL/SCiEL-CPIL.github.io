"""Morse cluster global minima search with multi-start parallel global search."""
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax

_START_TIME = time.time()
_TOTAL_BUDGET = 300.0
_DEADLINE = _START_TIME + 285.0


def _time_remaining():
    return _DEADLINE - time.time()


def _check_validity(positions, min_dist=0.5):
    """Check if positions are valid (no overlaps)."""
    if len(positions) < 2:
        return True
    dists = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_dist


def _morse_energy(positions, rho):
    """Compute Morse energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    x = rho * (1.0 - r)
    e = np.exp(x)
    energy = np.sum(e * (e - 2.0))
    return float(energy)


def _morse_gradient(positions, rho):
    """Compute Morse gradient."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, 1e-10)
    x = rho * (1.0 - r)
    e = np.exp(x)
    # dE/dr = 2*rho*exp(rho(1-r))*(exp(rho(1-r))-1)
    dE_dr = 2.0 * rho * e * (e - 1.0)
    # grad = -dE_dr * diff / r
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = diff[i, j]
            ri = r[i, j]
            g = -dE_dr[i, j] * rij / ri
            grad[i] += g
            grad[j] -= g
    return grad


def _lbfgs_minimize(positions, rho, maxiter=200):
    """Local minimization using L-BFGS-B."""
    from scipy.optimize import minimize
    
    def fun_flat(x):
        pos = x.reshape(-1, 3)
        return _morse_energy(pos, rho)
    
    def grad_flat(x):
        pos = x.reshape(-1, 3)
        return _morse_gradient(pos, rho).flatten()
    
    x0 = positions.flatten()
    try:
        result = minimize(fun_flat, x0, jac=grad_flat, method='L-BFGS-B', 
                         options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
        if result.success or result.fun < _morse_energy(positions, rho):
            new_pos = result.x.reshape(-1, 3)
            if _check_validity(new_pos):
                return result.fun, new_pos
    except:
        pass
    return _morse_energy(positions, rho), positions


def _relax_and_refine(positions, potential, parameter, size):
    """First use chem_clusters.relax, then refine with L-BFGS-B."""
    try:
        energy, relaxed = relax(positions, potential, parameter)
        if _check_validity(relaxed):
            # Refine with L-BFGS-B
            e_refined, pos_refined = _lbfgs_minimize(relaxed, parameter, maxiter=100)
            if e_refined < energy:
                return e_refined, pos_refined
            return energy, relaxed
    except:
        pass
    # Fallback to L-BFGS-B only
    return _lbfgs_minimize(positions, parameter, maxiter=200)


def _generate_fcc_cluster(n, rng):
    """Generate FCC-like cluster."""
    lattice = []
    a = 1.0
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                # FCC: add face-centered points
                lattice.append([i*a, j*a, k*a])
                lattice.append([(i+0.5)*a, (j+0.5)*a, k*a])
                lattice.append([(i+0.5)*a, j*a, (k+0.5)*a])
                lattice.append([i*a, (j+0.5)*a, (k+0.5)*a])
    
    lattice = np.array(lattice)
    # Remove duplicates and center
    center = lattice.mean(axis=0)
    lattice -= center
    
    # Sort by distance from center and take first n
    dists = np.linalg.norm(lattice, axis=1)
    indices = np.argsort(dists)[:n]
    positions = lattice[indices]
    
    # Add small random perturbation
    positions += rng.normal(0, 0.02, size=(n, 3))
    return positions


def _generate_hcp_cluster(n, rng):
    """Generate HCP-like cluster."""
    lattice = []
    a = 1.0
    c = 1.633 * a  # ideal HCP c/a ratio
    
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                # A layer
                lattice.append([i*a, j*a*1.732/2, k*c])
                # B layer (offset)
                lattice.append([(i+0.5)*a, (j+0.5)*a*1.732/2, k*c + c/3])
    
    lattice = np.array(lattice)
    center = lattice.mean(axis=0)
    lattice -= center
    
    dists = np.linalg.norm(lattice, axis=1)
    indices = np.argsort(dists)[:n]
    positions = lattice[indices]
    positions += rng.normal(0, 0.02, size=(n, 3))
    return positions


def _generate_icosahedral_cluster(n, rng):
    """Generate icosahedral-like cluster (concentric shells)."""
    positions = []
    
    # Inner shell: icosahedron vertices (12)
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico /= np.linalg.norm(ico[0])
    ico *= 1.0  # First shell radius
    positions.extend(ico.tolist())
    
    # Second shell: icosidodecahedron-like (30 vertices)
    shell2_radius = 1.7
    # Use points from icosahedron rotated and scaled
    for angle in [0, np.pi/5, 2*np.pi/5]:
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        for p in ico:
            new_p = R @ p * shell2_radius
            positions.append(new_p.tolist())
    
    # Third shell if needed
    shell3_radius = 2.4
    if len(positions) < n:
        for angle in [0, np.pi/10, np.pi/5, 3*np.pi/10]:
            R = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            for p in ico:
                new_p = R @ p * shell3_radius
                positions.append(new_p.tolist())
    
    positions = np.array(positions[:n])
    # If we don't have enough, add random points on sphere
    if len(positions) < n:
        remaining = n - len(positions)
        extra = rng.normal(0, 1, size=(remaining, 3))
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        extra *= shell3_radius * 1.1
        positions = np.vstack([positions, extra])
    
    positions += rng.normal(0, 0.02, size=(n, 3))
    return positions


def _generate_shell_cluster(n, rng):
    """Generate concentric spherical shell cluster."""
    positions = []
    shells = [1.0, 1.8, 2.5, 3.2]
    
    # Distribute atoms among shells
    shell_counts = []
    remaining = n
    for i, r in enumerate(shells):
        if i == len(shells) - 1:
            shell_counts.append(remaining)
        else:
            count = min(12 + i * 10, remaining)
            shell_counts.append(count)
            remaining -= count
    
    for r, count in zip(shells, shell_counts):
        if count <= 0:
            continue
        # Fibonacci sphere for each shell
        phi_gold = np.pi * (3.0 - np.sqrt(5.0))
        i = np.arange(count) + 0.5
        cos_theta = 1.0 - (i / count)
        sin_theta = np.sqrt(1.0 - cos_theta ** 2)
        theta = i * phi_gold
        
        pts = np.column_stack([
            sin_theta * np.cos(theta),
            sin_theta * np.sin(theta),
            cos_theta
        ])
        pts *= r
        positions.append(pts)
    
    positions = np.vstack(positions)
    if len(positions) < n:
        extra = rng.normal(0, 1, size=(n - len(positions), 3))
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        extra *= 3.5
        positions = np.vstack([positions, extra])
    
    positions = positions[:n]
    positions += rng.normal(0, 0.02, size=(n, 3))
    return positions


def _generate_random_sphere(n, radius, rng):
    """Generate random points in a sphere."""
    pts = rng.normal(0, 1, size=(n, 3))
    norms = np.linalg.norm(pts, axis=1)
    # Scale to be within radius
    pts *= radius / np.max(norms) * 0.9
    return pts


def _worker(seed_type, size, potential, parameter, rng_seed):
    """Worker function for parallel search."""
    rng = np.random.default_rng(rng_seed)
    
    if seed_type == 'fcc':
        positions = _generate_fcc_cluster(size, rng)
    elif seed_type == 'hcp':
        positions = _generate_hcp_cluster(size, rng)
    elif seed_type == 'ico':
        positions = _generate_icosahedral_cluster(size, rng)
    elif seed_type == 'shell':
        positions = _generate_shell_cluster(size, rng)
    else:  # random
        radius = 1.0 * size ** (1.0 / 3.0) * 1.2
        positions = _generate_random_sphere(size, radius, rng)
    
    # Check initial validity, if invalid add more spread
    if not _check_validity(positions):
        positions *= 1.5
        if not _check_validity(positions):
            positions = _generate_random_sphere(size, 1.0 * size ** (1.0/3.0) * 1.5, rng)
    
    # Relax and refine
    energy, relaxed = _relax_and_refine(positions, potential, parameter, size)
    return energy, relaxed


def construct(instance):
    """
    Find the lowest-energy arrangement of n atoms bound by Morse potentials.
    Uses multi-start parallel global search with physically motivated prototypes.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Define seed types for parallel workers
    seed_types = ['fcc', 'hcp', 'ico', 'shell', 'random', 'random', 'random']
    
    # Use up to 4 threads
    n_threads = min(4, len(seed_types))
    
    # Time budget per worker
    worker_budget = _time_remaining() / n_threads
    
    results = []
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for i in range(n_threads):
            seed_type = seed_types[i]
            rng_seed = 42 + i * 1000
            future = executor.submit(_worker, seed_type, size, potential, parameter, rng_seed)
            futures.append(future)
        
        for future in as_completed(futures):
            if _time_remaining() < 5.0:
                break
            try:
                energy, positions = future.result(timeout=_time_remaining() - 2.0)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except:
                continue
    
    # If we still have time, do additional basin-hopping-like refinement
    if best_positions is not None and _time_remaining() > 20.0:
        # Run a few more random restarts with L-BFGS
        for i in range(3):
            if _time_remaining() < 10.0:
                break
            rng = np.random.default_rng(42 + 5000 + i)
            radius = 1.0 * size ** (1.0 / 3.0) * 1.2
            positions = _generate_random_sphere(size, radius, rng)
            
            if not _check_validity(positions):
                positions *= 1.3
            
            energy, relaxed = _relax_and_refine(positions, potential, parameter, size)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
    
    # Fallback
    if best_positions is None:
        best_positions = np.zeros((size, 3))
        best_positions[:, 0] = np.arange(size) * 0.6
        if _time_remaining() > 2.0:
            try:
                energy, relaxed = relax(best_positions, potential, parameter)
                if _check_validity(relaxed):
                    best_positions = relaxed
            except:
                pass
    
    return best_positions
