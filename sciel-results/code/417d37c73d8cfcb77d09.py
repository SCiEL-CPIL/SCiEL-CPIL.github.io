"""Morse cluster global minima: crystalline initializations + L-BFGS-B with deadline guard."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Deadline guard: evaluator kills at 330s, use 295s as hard limit with safety margin
TOTAL_DEADLINE = 295.0
START_TIME = None


def _morse_energy_and_grad(positions, rho):
    """Compute Morse energy and gradient for a cluster.
    
    V(r) = exp(rho*(1-r)) * (exp(rho*(1-r)) - 2)
    """
    n = positions.shape[0]
    d = squareform(pdist(positions))
    np.fill_diagonal(d, 1.0)
    r = np.where(d > 1e-10, d, 1.0)
    
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    energy = 0.5 * np.sum(v)
    
    # Gradient: dV/dr = -rho * ex * (2*ex - 1)
    # dV/drij = dV/dr * (ri - rj)/r
    dvdr = -rho * ex * (2.0 * ex - 1.0)
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    rij = positions[iu[0]] - positions[iu[1]]
    r_ij = np.linalg.norm(rij, axis=1)
    r_ij_safe = np.where(r_ij > 1e-10, r_ij, 1.0)
    
    # Gradient contribution to each atom
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij_vec = positions[i] - positions[j]
            r_ij = np.linalg.norm(rij_vec)
            if r_ij < 1e-10:
                continue
            x_ij = rho * (1.0 - r_ij)
            ex_ij = np.exp(x_ij)
            dvdr_ij = -rho * ex_ij * (2.0 * ex_ij - 1.0)
            grad[i] += dvdr_ij * rij_vec / r_ij
            grad[j] -= dvdr_ij * rij_vec / r_ij
    
    return energy, grad


def _morse_energy_only(positions, rho):
    """Compute only Morse energy."""
    n = positions.shape[0]
    d = squareform(pdist(positions))
    np.fill_diagonal(d, 1.0)
    r = np.where(d > 1e-10, d, 1.0)
    
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    energy = 0.5 * np.sum(v)
    return energy


def _is_valid(positions, min_dist=0.5):
    """Check if all pairs are at least min_dist apart."""
    d = pdist(positions)
    return np.min(d) >= min_dist - 1e-6


def _generate_fcc(n, spacing=1.0):
    """Generate FCC-like cluster with n atoms."""
    # FCC lattice vectors
    a = spacing
    fcc_basis = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ]) * a
    
    # Generate FCC lattice points
    points = []
    # Approximate number of unit cells needed
    n_cells = int(np.ceil(n / 4.0)) + 1
    # Use a cubic arrangement of unit cells
    side = int(np.ceil(n_cells ** (1.0 / 3.0))) + 1
    
    for ix in range(-side, side + 1):
        for iy in range(-side, side + 1):
            for iz in range(-side, side + 1):
                for basis in fcc_basis:
                    p = np.array([ix, iy, iz]) * a + basis
                    points.append(p)
    
    points = np.array(points)
    
    # Sort by distance from center and take n closest
    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _generate_bcc(n, spacing=1.0):
    """Generate BCC-like cluster with n atoms."""
    a = spacing
    bcc_basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0.5]
    ]) * a
    
    points = []
    n_cells = int(np.ceil(n / 2.0)) + 1
    side = int(np.ceil(n_cells ** (1.0 / 3.0))) + 1
    
    for ix in range(-side, side + 1):
        for iy in range(-side, side + 1):
            for iz in range(-side, side + 1):
                for basis in bcc_basis:
                    p = np.array([ix, iy, iz]) * a + basis
                    points.append(p)
    
    points = np.array(points)
    
    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _generate_hcp(n, spacing=1.0):
    """Generate HCP-like cluster with n atoms."""
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    
    # HCP basis in one unit cell (2 atoms)
    # Layer A: hexagonal arrangement
    hcp_basis = []
    # A layer
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            x = a * (dx + 0.5 * dy)
            y = a * np.sqrt(3.0) / 2.0 * dy
            hcp_basis.append([x, y, 0])
    # B layer
    hcp_basis.append([a / 2.0, a * np.sqrt(3.0) / 6.0, c / 2.0])
    hcp_basis.append([-a / 2.0, a * np.sqrt(3.0) / 6.0, c / 2.0])
    hcp_basis.append([0, -a * np.sqrt(3.0) / 3.0, c / 2.0])
    
    hcp_basis = np.array(hcp_basis)
    
    points = []
    n_cells = int(np.ceil(n / 3.0)) + 1
    side = int(np.ceil(n_cells ** (1.0 / 3.0))) + 1
    
    for ix in range(-side, side + 1):
        for iy in range(-side, side + 1):
            for iz in range(-side, side + 1):
                origin = np.array([
                    a * (ix + 0.5 * iy),
                    a * np.sqrt(3.0) / 2.0 * iy,
                    c * iz
                ])
                for basis in hcp_basis:
                    p = origin + basis
                    points.append(p)
    
    points = np.array(points)
    
    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _generate_random_compact(n, spacing=1.0, seed=42):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points.append(p)
    return np.array(points)


def _generate_deformed_crystal(n, base_generator, spacing=1.0, noise_scale=0.1, seed=42):
    """Generate crystal with random deformation."""
    rng = np.random.default_rng(seed)
    positions = base_generator(n, spacing)
    # Add small random noise
    positions = positions + rng.normal(0, noise_scale, positions.shape)
    return positions


def _elapsed():
    return time.time() - START_TIME


def _remaining():
    return TOTAL_DEADLINE - _elapsed()


def construct(instance):
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter
    
    # Check deadline immediately
    if _remaining() < 10.0:
        # Return a quick valid result
        positions = _generate_fcc(size, spacing=1.0)
        return positions
    
    # Generate diverse initial configurations
    generators = [
        lambda: _generate_fcc(size, spacing=1.0),
        lambda: _generate_bcc(size, spacing=1.0),
        lambda: _generate_hcp(size, spacing=1.0),
        lambda: _generate_deformed_crystal(size, _generate_fcc, spacing=1.0, noise_scale=0.1, seed=1),
        lambda: _generate_deformed_crystal(size, _generate_fcc, spacing=1.0, noise_scale=0.2, seed=2),
        lambda: _generate_deformed_crystal(size, _generate_bcc, spacing=1.0, noise_scale=0.1, seed=3),
        lambda: _generate_deformed_crystal(size, _generate_hcp, spacing=1.0, noise_scale=0.1, seed=4),
        lambda: _generate_random_compact(size, spacing=1.0, seed=42),
        lambda: _generate_random_compact(size, spacing=1.0, seed=123),
        lambda: _generate_random_compact(size, spacing=1.0, seed=456),
    ]
    
    best_energy = float('inf')
    best_positions = None
    
    # Try each generator
    for gen_idx, gen in enumerate(generators):
        if _remaining() < 15.0:
            break
        
        try:
            positions = gen()
            
            # Validate
            if not _is_valid(positions, min_dist=0.5):
                continue
            
            # Optimize using L-BFGS-B
            result = minimize(
                _morse_energy_and_grad,
                positions,
                args=(rho,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 10000}
            )
            
            if _remaining() < 10.0:
                break
            
            energy = result.fun
            pos = result.x
            
            # Validate result
            if _is_valid(pos, min_dist=0.5):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos
        except Exception:
            continue
    
    # If no valid result found, use a fallback
    if best_positions is None:
        positions = _generate_fcc(size, spacing=1.0)
        return positions
    
    return best_positions
