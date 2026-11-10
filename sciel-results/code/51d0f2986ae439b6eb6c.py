"""Morse cluster global minima: basin hopping with crystalline starts and deadline guard."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Deadline guard
TOTAL_DEADLINE = 290.0  # safety margin before 330s kill
_START_TIME = None


def _morse_energy_and_grad(positions, rho):
    """Compute Morse energy and gradient vectorized."""
    n = positions.shape[0]
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(d, 1.0)
    r = np.maximum(d, 1e-10)
    
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    energy = 0.5 * np.sum(v)
    
    # Gradient: dV/dr = -rho * exp(rho*(1-r)) * (2*exp(rho*(1-r)) - 1)
    # Actually: V(r) = exp(x)(exp(x) - 2), dV/dr = exp(x)(exp(x)-1) * dx/dr
    # dx/dr = -rho, so dV/dr = -rho * exp(x) * (exp(x) - 1)
    dvdr = -rho * ex * (ex - 1.0)
    
    # grad_i = sum_{j != i} dvdr_ij * (r_i - r_j) / r_ij
    # diff[i,j] = r_i - r_j
    # Contribution to grad_i from pair (i,j): dvdr_ij * (r_i - r_j) / r_ij
    # = dvdr_ij * diff[i,j] / d[i,j]
    
    # Build weight matrix: w[i,j] = dvdr_ij / d_ij for i != j
    with np.errstate(divide='ignore', invalid='ignore'):
        w = dvdr / d
    np.fill_diagonal(w, 0.0)
    
    grad = np.sum(w[:, :, np.newaxis] * diff, axis=1)
    return energy, grad


def _morse_energy_only(positions, rho):
    """Compute Morse energy only."""
    energy, _ = _morse_energy_and_grad(positions, rho)
    return energy


def _is_valid(positions, min_dist=0.5):
    """Check if all pair distances are at least min_dist."""
    d = pdist(positions)
    return np.min(d) >= min_dist


def _generate_fcc(n, spacing=1.0):
    """Generate an FCC-like compact cluster."""
    # Generate points in FCC lattice, take n closest to center
    # FCC lattice vectors
    a = spacing
    # Generate a grid of FCC points
    max_coord = int(np.ceil(n ** (1.0/3.0)) * 2) + 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in units of a
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    points.append(p)
    points = np.array(points)
    # Sort by distance from center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]].copy()


def _generate_bcc(n, spacing=1.0):
    """Generate a BCC-like compact cluster."""
    a = spacing
    max_coord = int(np.ceil(n ** (1.0/3.0)) * 2) + 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]].copy()


def _generate_hcp(n, spacing=1.0):
    """Generate an HCP-like compact cluster."""
    a = spacing
    # HCP: two interpenetrating FCC-like layers
    max_coord = int(np.ceil(n ** (1.0/3.0)) * 2) + 4
    points = []
    # Layer A (triangular)
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            x = i * a
            y = j * a * np.sqrt(3) / 2
            # Alternate rows offset
            if i % 2 == 1:
                y += a * np.sqrt(3) / 4
            points.append(np.array([x, y, 0]))
    # Layer B at z = a*sqrt(2/3)
    z_b = a * np.sqrt(2.0/3.0)
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            x = i * a
            y = j * a * np.sqrt(3) / 2
            if i % 2 == 1:
                y += a * np.sqrt(3) / 4
            points.append(np.array([x, y, z_b]))
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]].copy()


def _generate_random_compact(n, spacing=1.0, seed=None):
    """Generate a random compact cluster with minimum spacing."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    min_dist = 0.55 * spacing
    points = []
    attempts = 0
    max_attempts = n * 200
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    # If we couldn't place all, fill remaining with small perturbations
    while len(points) < n:
        p = np.array(points[len(points) % max(1, len(points))]) + rng.normal(0, 0.1, 3)
        points.append(p)
    return np.array(points)


def _local_minimize(positions, rho, max_iter=500):
    """Local minimization using L-BFGS-B."""
    def fun(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(pos, rho)
        return e, g.flatten()
    
    x0 = positions.flatten()
    result = minimize(fun, x0, method='L-BFGS-B', jac=True, 
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    return result.x.reshape(-1, 3), result.fun


def construct(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    # Generate diverse starting configurations
    starts = []
    
    # Crystalline starts
    try:
        starts.append(_generate_fcc(size, 1.0))
    except Exception:
        pass
    try:
        starts.append(_generate_bcc(size, 1.0))
    except Exception:
        pass
    try:
        starts.append(_generate_hcp(size, 1.0))
    except Exception:
        pass
    
    # Random compact starts
    for seed in [size, size+1, size+2, size+3]:
        try:
            starts.append(_generate_random_compact(size, 1.0, seed=seed))
        except Exception:
            pass
    
    if not starts:
        # Fallback
        starts.append(_generate_random_compact(size, 1.0, seed=42))
    
    best_energy = np.inf
    best_positions = starts[0]
    
    # Basin hopping for each start
    for s_idx, start_pos in enumerate(starts):
        # Check deadline
        if time.time() - _START_TIME > TOTAL_DEADLINE - 30:
            break
        
        current_pos = start_pos.copy()
        # Initial local minimization
        try:
            current_pos, current_energy = _local_minimize(current_pos, rho, max_iter=300)
        except Exception:
            current_energy = _morse_energy_only(current_pos, rho)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_positions = current_pos.copy()
        
        # Basin hopping iterations
        rng = np.random.default_rng(s_idx * 1000 + size)
        bh_iters = 20  # Number of basin hopping steps
        
        for bh_i in range(bh_iters):
            # Check deadline
            if time.time() - _START_TIME > TOTAL_DEADLINE - 30:
                break
            
            # Perturbation: Gaussian noise scaled by temperature
            # Temperature annealing: start high, decrease
            temp = 0.3 * (1.0 - bh_i / bh_iters) + 0.05
            noise = rng.normal(0, temp, current_pos.shape)
            perturbed = current_pos + noise
            
            # Local minimization of perturbed
            try:
                new_pos, new_energy = _local_minimize(perturbed, rho, max_iter=200)
            except Exception:
                new_pos = perturbed
                new_energy = _morse_energy_only(new_pos, rho)
            
            # Metropolis accept
            if new_energy < current_energy:
                current_pos = new_pos
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_positions = current_pos.copy()
            else:
                # Accept with probability exp(-(E_new - E_old)/T)
                delta_e = new_energy - current_energy
                if delta_e < 50:  # Avoid overflow
                    accept_prob = np.exp(-delta_e / max(temp, 1e-6))
                    if rng.random() < accept_prob:
                        current_pos = new_pos
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_positions = current_pos.copy()
    
    # Final check: ensure validity
    if not _is_valid(best_positions, 0.5):
        # Try to fix by small local minimization
        try:
            best_positions, best_energy = _local_minimize(best_positions, rho, max_iter=100)
        except Exception:
            pass
    
    return best_positions
