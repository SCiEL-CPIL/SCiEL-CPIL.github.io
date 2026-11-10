# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _morse_energy(positions, rho):
    """Compute exact Morse energy efficiently using upper triangle only."""
    n = len(positions)
    if n < 2:
        return 0.0
    iu = np.triu_indices(n, k=1)
    diff = positions[iu[0]] - positions[iu[1]]
    r = np.sqrt(np.sum(diff * diff, axis=1))
    x = rho * (1.0 - r)
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    return float(np.sum(e))


def _morse_gradient(positions, rho):
    """Compute Morse gradient efficiently."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    grad = np.zeros_like(positions)
    iu = np.triu_indices(n, k=1)
    diff = positions[iu[0]] - positions[iu[1]]
    r = np.sqrt(np.sum(diff * diff, axis=1))
    r_safe = np.where(r < 1e-10, 1e-10, r)
    x = rho * (1.0 - r_safe)
    ex = np.exp(x)
    dE_dr = -2.0 * rho * ex * (ex - 1.0)
    factor = dE_dr / r_safe
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        grad[i] += factor[idx] * diff[idx]
        grad[j] -= factor[idx] * diff[idx]
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
    for i in range(1, n):
        z = 1.0 - 2.0 * i / (n - 1)
        radius = np.sqrt(max(0, 1.0 - z * z))
        theta = golden_angle * i
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        points.append(np.array([x, y, z]) * spacing * (n ** (1/3.0) * 0.5))
    
    return np.array(points)


def _generate_random(n, seed, min_dist=0.6):
    """Generate a random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = 1.2 * n ** (1.0 / 3.0)
    positions = []
    max_attempts = 10000
    attempts = 0
    while len(positions) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if len(positions) == 0:
            positions.append(p)
        else:
            dists = np.linalg.norm(np.array(positions) - p, axis=1)
            if np.min(dists) >= min_dist:
                positions.append(p)
    
    while len(positions) < n:
        p = rng.uniform(-radius, radius, 3)
        positions.append(p)
    
    return np.array(positions)


def _relax_lbfgs(positions, rho, max_iter=500, tol=1e-10):
    """Relax using L-BFGS-B with analytic gradient."""
    def fun(x):
        pos = x.reshape(-1, 3)
        return _morse_energy(pos, rho)
    
    def jac(x):
        pos = x.reshape(-1, 3)
        return _morse_gradient(pos, rho).flatten()
    
    x0 = positions.flatten()
    try:
        result = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'gtol': tol, 'maxls': 100})
        final_pos = result.x.reshape(-1, 3)
        final_energy = _morse_energy(final_pos, rho)
        return final_energy, final_pos
    except Exception:
        return np.inf, positions


def _check_minimum_distance(positions, min_dist=0.5):
    """Check if all pairs are at least min_dist apart."""
    n = len(positions)
    if n < 2:
        return True
    iu = np.triu_indices(n, k=1)
    diff = positions[iu[0]] - positions[iu[1]]
    r = np.sqrt(np.sum(diff * diff, axis=1))
    return bool(np.all(r >= min_dist))


def _fix_minimum_distance(positions, min_dist=0.5, max_iter=200):
    """Iteratively fix minimum distance violations."""
    n = len(positions)
    if n < 2:
        return positions
    
    for _ in range(max_iter):
        iu = np.triu_indices(n, k=1)
        diff = positions[iu[0]] - positions[iu[1]]
        r = np.sqrt(np.sum(diff * diff, axis=1))
        violations = r < min_dist
        if not np.any(violations):
            break
        
        for idx in np.where(violations)[0]:
            i, j = iu[0][idx], iu[1][idx]
            d = positions[i] - positions[j]
            norm = np.linalg.norm(d)
            if norm < 1e-10:
                d = np.array([1.0, 0.0, 0.0])
                norm = 1.0
            else:
                d = d / norm
            separation = (min_dist - norm) / 2.0 + 0.01
            positions[i] += separation * d
            positions[j] -= separation * d
    
    return positions


def construct(instance):
    """Find low-energy Morse cluster configurations."""
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin before 330s kill
    
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 3))
    
    # Generate diverse starting configurations
    generators = [
        _generate_fcc,
        _generate_bcc,
        _generate_hcp,
        _generate_shell,
    ]
    
    best_energy = np.inf
    best_positions = None
    
    # Try each generator with relaxation
    for gen_func in generators:
        if time.time() > deadline:
            break
        
        try:
            positions = gen_func(n, spacing=1.0)
            positions = _fix_minimum_distance(positions, 0.5, max_iter=50)
            energy, relaxed = _relax_lbfgs(positions, rho, max_iter=500)
            
            if energy < best_energy and _check_minimum_distance(relaxed, 0.5):
                best_energy = energy
                best_positions = relaxed
        except Exception:
            continue
    
    # Add random restarts with more seeds
    rng_seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, 333]
    for seed in rng_seeds:
        if time.time() > deadline - 30:
            break
        
        try:
            positions = _generate_random(n, seed, min_dist=0.6)
            positions = _fix_minimum_distance(positions, 0.5, max_iter=50)
            energy, relaxed = _relax_lbfgs(positions, rho, max_iter=400)
            
            if energy < best_energy and _check_minimum_distance(relaxed, 0.5):
                best_energy = energy
                best_positions = relaxed
        except Exception:
            continue
    
    # If still no valid solution, use fallback
    if best_positions is None:
        rng = np.random.default_rng(n)
        radius = 1.2 * n ** (1.0 / 3.0)
        positions = rng.uniform(-radius, radius, (n, 3))
        positions = _fix_minimum_distance(positions, 0.5, max_iter=100)
        energy, relaxed = _relax_lbfgs(positions, rho, max_iter=200)
        best_positions = relaxed
        if not _check_minimum_distance(best_positions, 0.5):
            best_positions = _fix_minimum_distance(best_positions, 0.5, max_iter=100)
    
    # If time remains, try perturbations of the best solution
    if best_positions is not None and time.time() < deadline - 25:
        rng = np.random.default_rng(n + 100)
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            
            noise = rng.normal(0, 0.15, size=best_positions.shape)
            perturbed = best_positions + noise
            perturbed = _fix_minimum_distance(perturbed, 0.5, max_iter=50)
            
            try:
                energy, relaxed = _relax_lbfgs(perturbed, rho, max_iter=300)
                if energy < best_energy and _check_minimum_distance(relaxed, 0.5):
                    best_energy = energy
                    best_positions = relaxed
            except Exception:
                continue
    
    # Final high-precision polish pass
    if best_positions is not None and time.time() < deadline - 10:
        try:
            energy, polished = _relax_lbfgs(best_positions, rho, max_iter=1000, tol=1e-12)
            if energy < best_energy and _check_minimum_distance(polished, 0.5):
                best_energy = energy
                best_positions = polished
        except Exception:
            pass
    
    # Final check: ensure minimum distance
    if not _check_minimum_distance(best_positions, 0.5):
        best_positions = _fix_minimum_distance(best_positions, 0.5, max_iter=200)
    
    return best_positions
# EVOLVE-BLOCK-END
