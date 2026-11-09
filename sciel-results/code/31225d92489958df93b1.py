"""Morse cluster global minima search with basin hopping and structured initializations."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_TOTAL_BUDGET = 290.0  # Leave 10s safety margin before 300s kill
_DEADLINE = _START_TIME + _TOTAL_BUDGET


def _time_remaining():
    return max(0.0, _DEADLINE - time.time())


def _morse_energy(rho, coords):
    """Compute Morse energy with penalty for distances < 0.5."""
    n = len(coords)
    if n < 2:
        return 0.0
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    min_d = np.min(d)
    if min_d < 0.5:
        return 1e10 + (0.5 - min_d) ** 2
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = d[i, j]
            x = rho * (1.0 - r)
            exp_x = np.exp(x)
            E += exp_x * (exp_x - 2.0)
    return E


def _morse_gradient(rho, coords):
    """Compute gradient of Morse energy."""
    n = len(coords)
    grad = np.zeros_like(coords)
    d = cdist(coords, coords)
    for i in range(n):
        for j in range(i + 1, n):
            r = d[i, j]
            if r < 1e-10:
                continue
            x = rho * (1.0 - r)
            exp_x = np.exp(x)
            # dE/dr = rho * exp(2x) * (2*exp_x - 2)
            dE_dr = rho * exp_x * exp_x * (2.0 * exp_x - 2.0)
            # Chain rule: dE/dxi = dE/dr * (r_i - r_j)/r
            factor = dE_dr / r
            diff = coords[i] - coords[j]
            grad[i] += factor * diff
            grad[j] -= factor * diff
    return grad


def _local_minimize(coords, rho, maxiter=200):
    """Local minimization using L-BFGS-B."""
    result = minimize(
        lambda x: _morse_energy(rho, x.reshape(-1, 3)),
        coords.reshape(-1),
        jac=lambda x: _morse_gradient(rho, x.reshape(-1, 3)).reshape(-1),
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'gtol': 1e-8}
    )
    return result.fun, result.x.reshape(-1, 3)


def _build_icosahedral_shell(n, rng, rho):
    """Build a cluster with icosahedral symmetry."""
    # Start with icosahedron (12 vertices + center)
    phi = (1 + np.sqrt(5)) / 2
    icosahedron = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    icosahedron = icosahedron / np.linalg.norm(icosahedron[0])  # normalize to unit radius
    
    points = [np.zeros(3)]  # center
    # Add icosahedron vertices
    for v in icosahedron:
        points.append(v)
    
    # Add more atoms on shells if needed
    shell_radius = 1.0
    while len(points) < n:
        shell_radius += 0.8
        # Generate random points on sphere
        theta = rng.uniform(0, 2 * np.pi)
        phi_angle = rng.uniform(-np.pi/2, np.pi/2)
        p = shell_radius * np.array([
            np.cos(phi_angle) * np.cos(theta),
            np.cos(phi_angle) * np.sin(theta),
            np.sin(phi_angle)
        ])
        # Check minimum distance
        if all(np.linalg.norm(p - q) >= 0.55 for q in points):
            points.append(p)
    
    return np.array(points[:n])


def _build_fcc_like(n, rng, rho):
    """Build an FCC-like cluster."""
    points = []
    # FCC lattice vectors
    a = 1.0  # lattice parameter
    lattice_vectors = [
        np.array([0, 0, 0]),
        np.array([0, a/2, a/2]),
        np.array([a/2, 0, a/2]),
        np.array([a/2, a/2, 0]),
    ]
    
    # Generate points in a cubic region
    max_coord = int(np.ceil(n ** (1/3))) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for lv in lattice_vectors:
                    p = np.array([i, j, k]) * a + lv
                    # Check if within radius and minimum distance
                    r = np.linalg.norm(p)
                    if r <= 2.5 * n ** (1/3):
                        if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                            points.append(p)
                            if len(points) >= n:
                                break
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    # If we don't have enough, add random points
    while len(points) < n:
        p = rng.uniform(-3, 3, 3)
        if all(np.linalg.norm(p - q) >= 0.55 for q in points):
            points.append(p)
    
    return np.array(points[:n])


def _build_random_compact(n, rng, rho):
    """Build a random compact cluster."""
    points = []
    radius = 2.0 * n ** (1/3)
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.55 for q in points):
            points.append(p)
    return np.array(points)


def _ensure_min_distance(coords, min_dist=0.51):
    """Ensure all pairs are at least min_dist apart."""
    coords = coords.copy()
    n = len(coords)
    for _ in range(100):  # max iterations
        d = cdist(coords, coords)
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        if min_d >= min_dist:
            break
        # Find the closest pair
        i, j = np.unravel_index(np.argmin(d), d.shape)
        # Move one atom away
        direction = coords[i] - coords[j]
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            direction = np.random.randn(3)
            norm = np.linalg.norm(direction)
        direction /= norm
        coords[i] += (min_dist - min_d + 0.01) * direction
    return coords


def _basin_hopping(coords, rho, n_iter, rng):
    """Basin hopping algorithm."""
    best_coords = coords.copy()
    best_energy = _morse_energy(rho, coords)
    
    for it in range(n_iter):
        if _time_remaining() < 5.0:
            break
        
        # Perturb
        perturbation_size = 0.3 + 0.2 * np.exp(-it / 20.0)
        perturbed = coords + perturbation_size * rng.standard_normal(coords.shape)
        perturbed = _ensure_min_distance(perturbed)
        
        # Local minimize
        e_new, coords_new = _local_minimize(perturbed, rho, maxiter=100)
        
        # Accept/reject (Metropolis)
        if e_new < best_energy:
            best_coords = coords_new
            best_energy = e_new
        else:
            # Metropolis criterion
            delta_e = e_new - best_energy
            T = 0.5 * np.exp(-it / 10.0)
            if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-10)):
                best_coords = coords_new
                best_energy = e_new
        
        coords = best_coords
    
    return best_coords, best_energy


def construct(instance):
    """Find global minima of Morse clusters using basin hopping with structured starts."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(n)
    
    # Generate diverse initial configurations
    initializers = [
        lambda: _build_icosahedral_shell(n, rng, rho),
        lambda: _build_fcc_like(n, rng, rho),
        lambda: _build_random_compact(n, rng, rho),
        lambda: _build_random_compact(n, rng, rho),
        lambda: _build_random_compact(n, rng, rho),
    ]
    
    best_coords = None
    best_energy = np.inf
    
    # Try each initializer
    for init_func in initializers:
        if _time_remaining() < 10.0:
            break
        
        try:
            coords = init_func()
            coords = _ensure_min_distance(coords)
            
            # Local minimize first
            e_local, coords_local = _local_minimize(coords, rho, maxiter=150)
            
            if e_local < best_energy:
                best_energy = e_local
                best_coords = coords_local
            
            # Basin hopping from this start
            coords_bh, e_bh = _basin_hopping(coords_local, rho, n_iter=30, rng=rng)
            
            if e_bh < best_energy:
                best_energy = e_bh
                best_coords = coords_bh
                
        except Exception:
            continue
    
    # Final local polish
    if best_coords is not None:
        if _time_remaining() > 5.0:
            e_final, coords_final = _local_minimize(best_coords, rho, maxiter=300)
            if e_final < best_energy:
                best_energy = e_final
                best_coords = coords_final
    
    # Fallback if something went wrong
    if best_coords is None:
        coords = _build_random_compact(n, rng, rho)
        coords = _ensure_min_distance(coords)
        _, best_coords = _local_minimize(coords, rho, maxiter=100)
    
    return best_coords
# EVOLVE-BLOCK-END
