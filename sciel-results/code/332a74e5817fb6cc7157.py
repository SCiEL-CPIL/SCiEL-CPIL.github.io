"""Morse cluster global minima finder with structured seeds and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
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
    energy = np.sum(ex * (ex - 2.0)) / 2.0
    grad_factor = -rho * ex * (ex - 1.0) / dists
    np.fill_diagonal(grad_factor, 0.0)
    row_sum = np.sum(grad_factor, axis=1, keepdims=True)
    grad = row_sum * pos - grad_factor @ pos
    return float(energy), grad


def _fcc_points(n, a=1.0):
    """Generate FCC lattice points."""
    positions = []
    max_idx = int(np.ceil(n / 4.0)) + 2
    for i in range(max_idx):
        for j in range(max_idx):
            for k in range(max_idx):
                positions.append([i * a, j * a, k * a])
                if i < max_idx - 1:
                    positions.append([(i + 0.5) * a, j * a, k * a])
                if j < max_idx - 1:
                    positions.append([i * a, (j + 0.5) * a, k * a])
                if k < max_idx - 1:
                    positions.append([i * a, j * a, (k + 0.5) * a])
    positions = np.array(positions[:n])
    positions -= np.mean(positions, axis=0)
    return positions


def _hcp_points(n, a=1.0):
    """Generate HCP lattice points."""
    positions = []
    c = a * np.sqrt(2.0 / 3.0)
    max_idx = int(np.ceil(n / 6.0)) + 2
    for i in range(max_idx):
        for j in range(max_idx):
            for k in range(max_idx):
                # A layer
                positions.append([i * a + j * a * 0.5, j * a * np.sqrt(3.0) / 2.0, k * c])
                # B layer
                positions.append([i * a + j * a * 0.5 + a / 3.0, j * a * np.sqrt(3.0) / 2.0 + a * np.sqrt(3.0) / 6.0, k * c + c / 3.0])
                # C layer
                positions.append([i * a + j * a * 0.5 + 2 * a / 3.0, j * a * np.sqrt(3.0) / 2.0 - a * np.sqrt(3.0) / 6.0, k * c + 2 * c / 3.0])
    positions = np.array(positions[:n])
    positions -= np.mean(positions, axis=0)
    return positions


def _bcc_points(n, a=1.0):
    """Generate BCC lattice points."""
    positions = []
    max_idx = int(np.ceil(n / 2.0)) + 2
    for i in range(max_idx):
        for j in range(max_idx):
            for k in range(max_idx):
                positions.append([i * a, j * a, k * a])
                positions.append([(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a])
    positions = np.array(positions[:n])
    positions -= np.mean(positions, axis=0)
    return positions


def _icosahedral_cluster(n, scale=1.0):
    """Generate icosahedral cluster."""
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    positions = [np.zeros(3)]  # center
    positions.extend([v * scale for v in verts])  # 12 vertices
    
    # Add next shell if needed
    if n > 13:
        # Add edge midpoints and face centers
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 1.5:  # close to edge length
                    edges.append((verts[i] + verts[j]) / 2.0 * scale)
        positions.extend(edges)
    
    # Fill remaining with random points on surface
    while len(positions) < n:
        idx = np.random.randint(0, 12)
        p = verts[idx] * scale * (1.0 + 0.1 * np.random.randn())
        positions.append(p)
    
    positions = np.array(positions[:n])
    positions -= np.mean(positions, axis=0)
    return positions


def _random_dense(n, radius=None, seed=None):
    """Generate random dense configuration."""
    rng = np.random.default_rng(seed)
    if radius is None:
        radius = n ** (1.0 / 3.0) * 0.6
    points = []
    for _ in range(n):
        while True:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                points.append(p)
                break
    return np.array(points)


def _ensure_min_dist(pos, min_dist=0.5, max_iter=1000):
    """Ensure minimum distance constraint."""
    n = len(pos)
    for _ in range(max_iter):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, 1e10)
        min_d = np.min(dists)
        if min_d >= min_dist:
            break
        # Push apart closest pairs
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        if i == j:
            break
        direction = pos[i] - pos[j]
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            direction = np.random.randn(3)
            norm = np.linalg.norm(direction)
        direction /= norm
        shift = (min_dist - min_d) / 2.0 + 0.01
        pos[i] += direction * shift
        pos[j] -= direction * shift
    return pos


def _relax_lbfgs(pos, rho, maxiter=300):
    """Relax using L-BFGS-B."""
    try:
        result = minimize(
            _morse_energy_grad,
            pos,
            args=(rho,),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        return result.x, result.fun
    except Exception:
        return pos, _morse_energy(pos, rho)


def _basin_hopping(pos, rho, n_iter=20, step_scale=0.3, time_limit=None, start_time=None):
    """Simple basin hopping algorithm."""
    if start_time is None:
        start_time = time.time()
    best_pos = pos.copy()
    best_energy = _morse_energy(pos, rho)
    
    for i in range(n_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Perturb
        rng = np.random.default_rng(i + 1000)
        perturbed = pos + rng.normal(0, step_scale, pos.shape)
        perturbed = _ensure_min_dist(perturbed, 0.5)
        
        # Relax
        new_pos, new_energy = _relax_lbfgs(perturbed, rho)
        
        # Accept if better
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos
            pos = new_pos
        else:
            # Metropolis acceptance
            delta_e = new_energy - best_energy
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / 0.1):
                pos = new_pos
    
    return best_pos, best_energy


def construct(instance):
    """Find low-energy Morse cluster configurations."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter if isinstance(parameter, (int, float)) else 3
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Generate diverse initial configurations based on rho
    inits = []
    
    # For long-range (low rho), FCC/HCP are good
    if rho <= 6:
        inits.append(_fcc_points(size))
        inits.append(_hcp_points(size))
        inits.append(_bcc_points(size))
    else:
        # For short-range (high rho), icosahedral structures are better
        inits.append(_icosahedral_cluster(size))
        inits.append(_fcc_points(size))
        inits.append(_hcp_points(size))
    
    # Add random configurations
    for seed in [42, 123, 777]:
        inits.append(_random_dense(size, seed=seed))
    
    # Relax all initial configurations
    best_energy = np.inf
    best_positions = inits[0].copy()
    
    for idx, init_pos in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Ensure minimum distance
        init_pos = _ensure_min_dist(init_pos, 0.5)
        
        # Relax
        relaxed_pos, energy = _relax_lbfgs(init_pos, rho)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed_pos.copy()
    
    # Basin hopping from the best configuration
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        bh_pos, bh_energy = _basin_hopping(
            best_positions, rho, 
            n_iter=min(50, int(remaining_time / 5)),
            step_scale=0.2,
            time_limit=remaining_time - 5,
            start_time=start_time
        )
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_pos
    
    # Final check: ensure minimum distance constraint
    best_positions = _ensure_min_dist(best_positions, 0.5)
    
    return best_positions
# EVOLVE-BLOCK-END
