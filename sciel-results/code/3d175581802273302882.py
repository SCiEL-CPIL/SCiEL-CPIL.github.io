# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(positions, rho):
    """Compute exact Morse energy efficiently using upper triangle only."""
    n = len(positions)
    if n < 2:
        return 0.0
    iu = np.triu_indices(n, k=1)
    diff = positions[iu[0]] - positions[iu[1]]
    r = np.sqrt(np.sum(diff * diff, axis=1))
    # Clip to avoid overflow in exp
    x = rho * (1.0 - r)
    x = np.clip(x, -50, 50)  # Prevent overflow
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    return float(np.sum(e))


def _morse_gradient(positions, rho):
    """Compute Morse energy gradient."""
    n = len(positions)
    grad = np.zeros_like(positions)
    if n < 2:
        return grad
    iu = np.triu_indices(n, k=1)
    diff = positions[iu[0]] - positions[iu[1]]
    r = np.sqrt(np.sum(diff * diff, axis=1))
    r = np.maximum(r, 1e-10)  # Avoid division by zero
    x = rho * (1.0 - r)
    x = np.clip(x, -50, 50)
    ex = np.exp(x)
    # dE/dr = 2*rho*ex*(ex-1) for each pair
    de_dr = 2.0 * rho * ex * (ex - 1.0)
    # dE/dx_i = de_dr * (x_i - x_j) / r
    for i, j in zip(iu[0], iu[1]):
        factor = de_dr[i] / r[i]
        grad[i] += factor * diff[i]
        grad[j] -= factor * diff[i]
    return grad


def _generate_fcc(n, spacing=1.0):
    """Generate an FCC-like cluster of n atoms."""
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * spacing
                    points.append(p)
    
    points = np.array(points)
    center = np.mean(points, axis=0)
    points -= center
    
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    
    return points


def _generate_bcc(n, spacing=1.0):
    """Generate a BCC-like cluster of n atoms."""
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * spacing
                    points.append(p)
    
    points = np.array(points)
    center = np.mean(points, axis=0)
    points -= center
    
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    
    return points


def _generate_hcp(n, spacing=1.0):
    """Generate an HCP-like cluster of n atoms."""
    points = []
    a = spacing
    c = spacing * np.sqrt(8.0 / 3.0)
    
    for layer in range(-5, 6):
        z = layer * c / 2.0
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = (i + 0.5 * (j % 2)) * a
                y = j * a * np.sqrt(3.0) / 2.0
                points.append(np.array([x, y, z]))
                points.append(np.array([x + a * np.sqrt(3.0) / 6.0, 
                                        y + a / 2.0, z + c / 6.0]))
    
    points = np.array(points)
    center = np.mean(points, axis=0)
    points -= center
    
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    
    return points


def _generate_shell(n, spacing=1.0):
    """Generate a spherical shell-like cluster."""
    if n <= 1:
        return np.zeros((n, 3))
    
    points = [np.zeros(3)]
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    radius = spacing * (n ** (1/3.0) * 0.5)
    for i in range(1, n):
        z = 1.0 - 2.0 * i / (n - 1)
        r = np.sqrt(max(0, 1.0 - z * z))
        theta = golden_angle * i
        x = np.cos(theta) * r
        y = np.sin(theta) * r
        points.append(np.array([x, y, z]) * radius)
    
    return np.array(points)


def _generate_random_cluster(n, seed=42, min_dist=0.5):
    """Generate a random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = 1.0 * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    max_attempts = 10000
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms, add remaining randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _relax_with_gradient(positions, rho, max_iter=500, tol=1e-8):
    """Relax using L-BFGS-B with exact Morse gradient."""
    def fun(x):
        pos = x.reshape(-1, 3)
        return _morse_energy(pos, rho)
    
    def jac(x):
        pos = x.reshape(-1, 3)
        return _morse_gradient(pos, rho).flatten()
    
    x0 = positions.flatten()
    try:
        result = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'gtol': tol, 'maxls': 50})
        final_pos = result.x.reshape(-1, 3)
        final_energy = _morse_energy(final_pos, rho)
        return final_energy, final_pos
    except Exception:
        return _morse_energy(positions, rho), positions


def construct(instance):
    """Find low-energy Morse cluster configurations."""
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill
    
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 3))
    
    # Generate diverse starting configurations
    generators = [
        lambda: _generate_fcc(n, spacing=1.0),
        lambda: _generate_bcc(n, spacing=1.0),
        lambda: _generate_hcp(n, spacing=1.0),
        lambda: _generate_shell(n, spacing=1.0),
        lambda: _generate_random_cluster(n, seed=42),
        lambda: _generate_random_cluster(n, seed=123),
        lambda: _generate_random_cluster(n, seed=777),
        lambda: _generate_random_cluster(n, seed=999),
    ]
    
    best_energy = np.inf
    best_positions = None
    
    # Try each generator with relaxation
    for gen_func in generators:
        if time.time() > deadline:
            break
        
        try:
            positions = gen_func()
            # Check minimum distance constraint
            if len(positions) >= 2:
                iu = np.triu_indices(n, k=1)
                diffs = positions[iu[0]] - positions[iu[1]]
                dists = np.sqrt(np.sum(diffs * diffs, axis=1))
                if np.min(dists) < 0.4:
                    continue
            
            energy, relaxed = _relax_with_gradient(positions, rho, max_iter=300)
            
            # Verify minimum distance after relaxation
            if len(relaxed) >= 2:
                iu = np.triu_indices(n, k=1)
                diffs = relaxed[iu[0]] - relaxed[iu[1]]
                dists = np.sqrt(np.sum(diffs * diffs, axis=1))
                if np.min(dists) < 0.5:
                    continue
            
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except Exception:
            continue
    
    # Fallback if no valid configuration found
    if best_positions is None:
        rng = np.random.default_rng(n)
        radius = 1.0 * n ** (1.0 / 3.0) * 0.75
        positions = []
        while len(positions) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(
                np.linalg.norm(p - q) >= 0.8 for q in positions
            ):
                positions.append(p)
        energy, relaxed = relax(np.array(positions), "morse", rho)
        best_energy = energy
        best_positions = relaxed
    
    # If time remains, try perturbations of the best solution
    if best_positions is not None and time.time() < deadline - 30:
        rng = np.random.default_rng(n + 1)
        for _ in range(5):
            if time.time() > deadline:
                break
            
            noise = rng.normal(0, 0.3, size=best_positions.shape)
            perturbed = best_positions + noise
            
            try:
                energy, relaxed = _relax_with_gradient(perturbed, rho, max_iter=200)
                if len(relaxed) >= 2:
                    iu = np.triu_indices(n, k=1)
                    diffs = relaxed[iu[0]] - relaxed[iu[1]]
                    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
                    if np.min(dists) >= 0.5 and energy < best_energy:
                        best_energy = energy
                        best_positions = relaxed
            except Exception:
                continue
    
    return best_positions
# EVOLVE-BLOCK-END
