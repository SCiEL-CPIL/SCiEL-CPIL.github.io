"""Morse cluster global minima search: best-of-many with diverse starts and L-BFGS-B refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy_vec(pos, rho):
    """Compute Morse energy vectorized."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dist, np.inf)
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    r = dist[iu]
    exp_term = np.exp(rho * (1.0 - r))
    return np.sum(exp_term * (exp_term - 2.0))


def _morse_gradient_vec(pos, rho):
    """Compute Morse gradient vectorized."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
    np.fill_diagonal(dist, np.inf)
    
    # dE/dr = -2*rho*exp(rho*(1-r))*(exp(rho*(1-r)) - 1)
    # For each pair (i,j), gradient contribution to atom i is:
    # dE/dri = dE/dr * (ri - rj) / r
    # and to atom j is: -dE/dr * (ri - rj) / r
    
    exp_term = np.exp(rho * (1.0 - dist))
    dEdr = -2.0 * rho * exp_term * (exp_term - 1.0)
    
    # Gradient: for each pair (i,j), add dEdr_ij * (pos_i - pos_j) / r_ij to grad_i
    # and -dEdr_ij * (pos_i - pos_j) / r_ij to grad_j
    # Which is: grad_i += dEdr_ij * (pos_i - pos_j) / r_ij
    #          grad_j -= dEdr_ij * (pos_i - pos_j) / r_ij
    
    # dEdr / dist gives the scalar factor
    factor = dEdr / dist
    np.fill_diagonal(factor, 0.0)
    
    # grad[i] = sum_j factor[i,j] * (pos[i] - pos[j])
    # = sum_j factor[i,j] * pos[i] - sum_j factor[i,j] * pos[j]
    # = pos[i] * sum_j factor[i,j] - (factor @ pos)[i]
    
    row_sum = factor.sum(axis=1)
    grad = pos * row_sum[:, np.newaxis] - factor @ pos
    
    return grad


def _make_fcc_start(n):
    """Generate FCC-like lattice points, take first n."""
    points = []
    # FCC lattice vectors
    a = 1.0  # equilibrium distance
    # Generate points in a cube
    extent = int(math.ceil(n ** (1.0/3.0))) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                # FCC: corners and face centers
                # Corner
                points.append(np.array([i, j, k]) * a)
                # Face centers
                if i < extent:
                    points.append(np.array([i + 0.5, j + 0.5, k]) * a)
                if j < extent:
                    points.append(np.array([i + 0.5, j, k + 0.5]) * a)
                if k < extent:
                    points.append(np.array([i, j + 0.5, k + 0.5]) * a)
    
    # Center the points
    points = np.array(points)
    points -= points.mean(axis=0)
    
    # Sort by distance from center and take closest n
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _make_icosahedral_shell(n):
    """Generate icosahedral-like shell structure."""
    if n <= 1:
        return np.zeros((n, 3))
    
    points = [np.zeros(3)]
    
    # Icosahedron vertices
    phi = (1 + math.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    ico = ico / np.linalg.norm(ico[0])
    
    # Add icosahedron vertices scaled
    for v in ico:
        if len(points) < n:
            points.append(v * 1.0)
    
    # Add second shell (dodecahedron-like or just more ico vertices at larger radius)
    if len(points) < n:
        # Add vertices at ~2.0 radius
        for v in ico:
            if len(points) < n:
                points.append(v * 2.0)
    
    # Add more if needed with jitter
    rng = np.random.default_rng(42)
    while len(points) < n:
        # Random direction
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v)
        r = 1.0 + 0.5 * rng.random()
        points.append(v * r)
    
    points = np.array(points[:n])
    return points


def _make_random_compact(n, seed=0):
    """Random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    points = [np.zeros(3)]
    radius = n ** (1.0/3.0) * 0.8
    
    attempts = 0
    while len(points) < n and attempts < 100000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            min_dist = min(np.linalg.norm(p - q) for q in points) if points else np.inf
            if min_dist >= 0.55:
                points.append(p)
        attempts += 1
    
    # If we couldn't place all, add random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _make_jittered_fcc(n, jitter=0.1, seed=0):
    """FCC with random jitter."""
    base = _make_fcc_start(n)
    rng = np.random.default_rng(seed)
    base = base + rng.normal(0, jitter, size=base.shape)
    return base


def _make_jittered_ico(n, jitter=0.1, seed=0):
    """Icosahedral with random jitter."""
    base = _make_icosahedral_shell(n)
    rng = np.random.default_rng(seed)
    base = base + rng.normal(0, jitter, size=base.shape)
    return base


def _lbfgs_refine(pos, rho, max_iter=500):
    """Refine using L-BFGS-B with our own gradient."""
    def fun_and_grad(x):
        p = x.reshape(-1, 3)
        e = _morse_energy_vec(p, rho)
        g = _morse_gradient_vec(p, rho).ravel()
        return e, g
    
    result = minimize(fun_and_grad, pos.ravel(), method='L-BFGS-B',
                      jac=True, options={'maxiter': max_iter, 'gtol': 1e-6})
    return result.fun, result.x.reshape(-1, 3)


def construct(instance):
    """Find low-energy Morse cluster configurations."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s safety margin
    
    # Generate diverse starting configurations
    starts = []
    
    # FCC-based
    starts.append(_make_fcc_start(size))
    starts.append(_make_jittered_fcc(size, jitter=0.05, seed=1))
    starts.append(_make_jittered_fcc(size, jitter=0.1, seed=2))
    starts.append(_make_jittered_fcc(size, jitter=0.2, seed=3))
    
    # Icosahedral
    starts.append(_make_icosahedral_shell(size))
    starts.append(_make_jittered_ico(size, jitter=0.05, seed=10))
    starts.append(_make_jittered_ico(size, jitter=0.1, seed=11))
    
    # Random compact
    for seed in range(5):
        starts.append(_make_random_compact(size, seed=seed + 100))
    
    # Remove duplicates (approximately)
    unique_starts = []
    for s in starts:
        is_dup = False
        for u in unique_starts:
            if np.allclose(s, u, atol=0.01):
                is_dup = True
                break
        if not is_dup:
            unique_starts.append(s)
    
    starts = unique_starts
    
    best_energy = np.inf
    best_pos = starts[0]
    
    # Try each start with L-BFGS refinement
    for i, start in enumerate(starts):
        if time.time() > deadline:
            break
        
        # First do a quick local relaxation using the provided relax function
        try:
            energy, pos = relax(start.copy(), potential, rho)
        except Exception:
            continue
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
        
        # Then refine with L-BFGS for more precision
        if time.time() > deadline - 30:
            break
        
        try:
            e2, p2 = _lbfgs_refine(pos, rho, max_iter=300)
            if e2 < best_energy:
                best_energy = e2
                best_pos = p2.copy()
        except Exception:
            pass
    
    # If we have time, do a few more random perturbations of the best
    if time.time() < deadline - 20:
        rng = np.random.default_rng(12345)
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            perturbed = best_pos + rng.normal(0, 0.05, size=best_pos.shape)
            try:
                energy, pos = relax(perturbed, potential, rho)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos.copy()
                
                if time.time() < deadline - 5:
                    e2, p2 = _lbfgs_refine(pos, rho, max_iter=200)
                    if e2 < best_energy:
                        best_energy = e2
                        best_pos = p2.copy()
            except Exception:
                pass
    
    return best_pos
# EVOLVE-BLOCK-END
