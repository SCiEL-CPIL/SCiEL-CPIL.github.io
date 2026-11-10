"""Morse cluster global minima search with crystalline starting points and gradient-based relaxation."""
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
    iu = np.triu_indices(n, k=1)
    diff = positions[iu[0]] - positions[iu[1]]
    r = np.sqrt(np.sum(diff * diff, axis=1))
    x = rho * (1.0 - r)
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    return float(np.sum(e))


def _morse_gradient(positions, rho):
    """Compute Morse gradient."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    grad = np.zeros_like(positions)
    iu = np.triu_indices(n, k=1)
    diff = positions[iu[0]] - positions[iu[1]]
    r = np.sqrt(np.sum(diff * diff, axis=1))
    # Avoid division by zero
    r_safe = np.where(r > 1e-10, r, 1.0)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    # dE/dr = -rho * exp(x) * (2*exp(x) - 1)
    dE_dr = -rho * ex * (2.0 * ex - 1.0)
    # grad_i = -dE_dr / r * diff_i, grad_j = +dE_dr / r * diff_j
    factor = dE_dr / r_safe
    np.add.at(grad, iu[0], -factor[:, None] * diff)
    np.add.at(grad, iu[1], factor[:, None] * diff)
    return grad


def _generate_fcc(n):
    """Generate FCC-like cluster."""
    points = []
    a = 1.0
    # FCC lattice points
    fcc_offsets = [
        (0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5),
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (1, 1, 0), (1, 0, 1), (0, 1, 1),
        (1, 1, 1), (1.5, 0.5, 0), (1.5, 0, 0.5), (1, 0.5, 0.5),
        (0.5, 1.5, 0), (0, 1.5, 0.5), (0.5, 0, 1.5),
        (0.5, 1.5, 0.5), (1.5, 0.5, 0.5), (0.5, 0.5, 1.5),
        (2, 0, 0), (0, 2, 0), (0, 0, 2),
        (2, 1, 0), (2, 0, 1), (1, 2, 0), (0, 2, 1), (1, 0, 2), (0, 1, 2),
        (2, 1, 1), (1, 2, 1), (1, 1, 2),
        (2, 2, 0), (2, 0, 2), (0, 2, 2),
        (2, 2, 1), (2, 1, 2), (1, 2, 2),
        (3, 0, 0), (0, 3, 0), (0, 0, 3),
        (3, 1, 0), (3, 0, 1), (1, 3, 0), (0, 3, 1), (1, 0, 3), (0, 1, 3),
        (3, 2, 0), (3, 0, 2), (2, 3, 0), (0, 3, 2), (2, 0, 3), (0, 2, 3),
        (3, 1, 1), (1, 3, 1), (1, 1, 3),
        (3, 2, 1), (3, 1, 2), (2, 3, 1), (1, 3, 2), (2, 1, 3), (1, 2, 3),
        (3, 2, 2), (2, 3, 2), (2, 2, 3),
        (4, 0, 0), (0, 4, 0), (0, 0, 4),
        (4, 1, 0), (4, 0, 1), (1, 4, 0), (0, 4, 1), (1, 0, 4), (0, 1, 4),
        (4, 2, 0), (4, 0, 2), (2, 4, 0), (0, 4, 2), (2, 0, 4), (0, 2, 4),
        (4, 3, 0), (4, 0, 3), (3, 4, 0), (0, 4, 3), (3, 0, 4), (0, 3, 4),
        (4, 1, 1), (1, 4, 1), (1, 1, 4),
        (4, 2, 1), (4, 1, 2), (2, 4, 1), (1, 4, 2), (2, 1, 4), (1, 2, 4),
        (4, 3, 1), (4, 1, 3), (3, 4, 1), (1, 4, 3), (3, 1, 4), (1, 3, 4),
        (4, 2, 2), (2, 4, 2), (2, 2, 4),
        (4, 3, 2), (4, 2, 3), (3, 4, 2), (2, 4, 3), (3, 2, 4), (2, 3, 4),
        (4, 3, 3), (3, 4, 3), (3, 3, 4),
        (5, 0, 0), (0, 5, 0), (0, 0, 5),
        (5, 1, 0), (5, 0, 1), (1, 5, 0), (0, 5, 1), (1, 0, 5), (0, 1, 5),
        (5, 2, 0), (5, 0, 2), (2, 5, 0), (0, 5, 2), (2, 0, 5), (0, 2, 5),
        (5, 3, 0), (5, 0, 3), (3, 5, 0), (0, 5, 3), (3, 0, 5), (0, 3, 5),
        (5, 4, 0), (5, 0, 4), (4, 5, 0), (0, 5, 4), (4, 0, 5), (0, 4, 5),
        (5, 1, 1), (1, 5, 1), (1, 1, 5),
        (5, 2, 1), (5, 1, 2), (2, 5, 1), (1, 5, 2), (2, 1, 5), (1, 2, 5),
        (5, 3, 1), (5, 1, 3), (3, 5, 1), (1, 5, 3), (3, 1, 5), (1, 3, 5),
        (5, 4, 1), (5, 1, 4), (4, 5, 1), (1, 5, 4), (4, 1, 5), (1, 4, 5),
        (5, 2, 2), (2, 5, 2), (2, 2, 5),
        (5, 3, 2), (5, 2, 3), (3, 5, 2), (2, 5, 3), (3, 2, 5), (2, 3, 5),
        (5, 4, 2), (5, 2, 4), (4, 5, 2), (2, 5, 4), (4, 2, 5), (2, 4, 5),
        (5, 3, 3), (3, 5, 3), (3, 3, 5),
        (5, 4, 3), (5, 3, 4), (4, 5, 3), (3, 5, 4), (4, 3, 5), (3, 4, 5),
        (5, 4, 4), (4, 5, 4), (4, 4, 5),
    ]
    for off in fcc_offsets:
        if len(points) >= n:
            break
        points.append(np.array(off, dtype=float) * a)
    
    # If we still need more points, add them in a shell
    if len(points) < n:
        radius = np.max(np.linalg.norm(points, axis=1)) if points else 1.0
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
    
    points = np.array(points[:n])
    # Center the cluster
    points -= np.mean(points, axis=0)
    return points


def _generate_bcc(n):
    """Generate BCC-like cluster."""
    points = []
    a = 1.0
    # BCC lattice points
    bcc_offsets = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                bcc_offsets.append((i, j, k))
                if (i + j + k) % 2 == 1:
                    bcc_offsets.append((i + 0.5, j + 0.5, k + 0.5))
    
    # Sort by distance from center
    bcc_offsets.sort(key=lambda x: x[0]**2 + x[1]**2 + x[2]**2)
    
    for off in bcc_offsets:
        if len(points) >= n:
            break
        points.append(np.array(off, dtype=float) * a)
    
    if len(points) < n:
        radius = np.max(np.linalg.norm(points, axis=1)) if points else 1.0
        rng = np.random.default_rng(43)
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points


def _generate_hcp(n):
    """Generate HCP-like cluster."""
    points = []
    a = 1.0
    c = a * np.sqrt(8.0 / 3.0)
    
    # HCP layers
    for layer in range(-3, 4):
        z = layer * c
        if layer % 2 == 0:
            # A layer
            for i in range(-3, 4):
                for j in range(-3, 4):
                    if i + j % 2 == 0 or True:  # Simple hexagonal
                        x = (i - j / 2) * a
                        y = j * a * np.sqrt(3) / 2
                        points.append(np.array([x, y, z]))
        else:
            # B layer (offset)
            for i in range(-3, 4):
                for j in range(-3, 4):
                    x = (i - j / 2 + 0.5) * a
                    y = (j + 0.5) * a * np.sqrt(3) / 2
                    points.append(np.array([x, y, z]))
    
    # Sort by distance from center
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order]
    
    if len(points) < n:
        radius = np.max(dists) if len(dists) > 0 else 1.0
        rng = np.random.default_rng(44)
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points = np.vstack([points, p])
    
    points = points[:n]
    points -= np.mean(points, axis=0)
    return points


def _generate_shell(n):
    """Generate concentric shell structure."""
    points = [np.zeros(3)]
    shells = [
        (1.0, 6),    # octahedron
        (1.0, 12),   # cube
        (1.0, 8),    # tetrahedron
        (1.4, 6),
        (1.4, 12),
        (1.7, 8),
        (2.0, 6),
        (2.0, 12),
        (2.4, 8),
        (2.7, 6),
        (2.7, 12),
        (3.0, 8),
        (3.4, 6),
        (3.4, 12),
    ]
    
    rng = np.random.default_rng(45)
    for radius, count in shells:
        if len(points) >= n:
            break
        for _ in range(count):
            if len(points) >= n:
                break
            # Random point on sphere
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            p = v * radius
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-3.5, 3.5, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points


def _relax_with_gradient(positions, potential, parameter, maxiter=500):
    """Relax using gradient-based optimization."""
    def objective(x):
        pos = x.reshape(-1, 3)
        return _morse_energy(pos, parameter)
    
    def gradient(x):
        pos = x.reshape(-1, 3)
        return _morse_gradient(pos, parameter).flatten()
    
    x0 = positions.flatten()
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': 2000}
        )
        if result.success or result.fun < _morse_energy(positions, parameter):
            return result.fun, result.x.reshape(-1, 3)
    except Exception:
        pass
    return _morse_energy(positions, parameter), positions


def _is_valid(positions, min_dist=0.5):
    """Check if all pairs are at least min_dist apart."""
    n = len(positions)
    if n < 2:
        return True
    iu = np.triu_indices(n, k=1)
    diff = positions[iu[0]] - positions[iu[1]]
    r = np.sqrt(np.sum(diff * diff, axis=1))
    return np.min(r) >= min_dist - 1e-6


def construct(instance):
    """Find low-energy Morse cluster configuration."""
    start_time = time.time()
    deadline = start_time + 280  # 280s deadline, 50s safety margin
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting points
    generators = [
        _generate_fcc,
        _generate_bcc,
        _generate_hcp,
        _generate_shell,
    ]
    
    # Add random starts
    for seed in [42, 123, 456, 789]:
        def random_start(n=size, s=seed):
            rng = np.random.default_rng(s)
            points = []
            radius = n ** (1.0 / 3.0) * 0.8
            while len(points) < n:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.55 for q in points):
                    points.append(p)
            points = np.array(points)
            points -= np.mean(points, axis=0)
            return points
        generators.append(random_start)
    
    # Try each generator
    for gen in generators:
        if time.time() > deadline:
            break
        
        try:
            positions = gen(size)
            if not _is_valid(positions):
                continue
            
            # Relax with gradient
            energy, relaxed = _relax_with_gradient(positions, potential, parameter, maxiter=300)
            
            if _is_valid(relaxed) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed.copy()
                
        except Exception:
            continue
    
    # If we have time, try more random restarts
    if best_positions is not None and time.time() < deadline - 30:
        rng = np.random.default_rng(999)
        for _ in range(10):
            if time.time() > deadline:
                break
            try:
                radius = size ** (1.0 / 3.0) * 0.8
                points = []
                while len(points) < size:
                    p = rng.uniform(-radius, radius, 3)
                    if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.55 for q in points):
                        points.append(p)
                points = np.array(points)
                points -= np.mean(points, axis=0)
                
                if not _is_valid(points):
                    continue
                
                energy, relaxed = _relax_with_gradient(points, potential, parameter, maxiter=200)
                
                if _is_valid(relaxed) and energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed.copy()
                    
            except Exception:
                continue
    
    # Fallback: use the library relax
    if best_positions is None:
        try:
            rng = np.random.default_rng(42)
            radius = size ** (1.0 / 3.0) * 0.8
            points = []
            while len(points) < size:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.55 for q in points):
                    points.append(p)
            energy, positions = relax(np.array(points), potential, parameter)
            best_positions = positions
        except Exception:
            # Last resort: simple grid
            best_positions = np.zeros((size, 3))
            for i in range(size):
                best_positions[i] = [i * 0.6, 0, 0]
    
    return best_positions
# EVOLVE-BLOCK-END
