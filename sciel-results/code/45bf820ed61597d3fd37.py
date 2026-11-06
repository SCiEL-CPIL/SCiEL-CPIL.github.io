# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy_and_grad(positions, rho):
    """Compute Morse energy and gradient directly for speed."""
    n = positions.shape[0]
    energy = 0.0
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            diff = positions[i] - positions[j]
            r = np.linalg.norm(diff)
            if r < 1e-12:
                continue
            rho_one_minus_r = rho * (1.0 - r)
            exp_val = np.exp(rho_one_minus_r)
            e_pair = exp_val * (exp_val - 2.0)
            de_dr = -rho * exp_val * (2.0 * exp_val - 1.0)
            unit = diff / r
            force = de_dr * unit
            grad[i] += force
            grad[j] -= force
            energy += e_pair
    return energy, grad


def _build_fibonacci_sphere(n, scale=1.0):
    """Fibonacci sphere with scaling."""
    points = []
    for i in range(n):
        z = 1.0 - 2.0 * i / (n - 1) if n > 1 else 1.0
        radius_xy = np.sqrt(max(0.0, 1.0 - z * z))
        theta = np.pi * (3.0 - np.sqrt(5.0)) * i
        x = np.cos(theta) * radius_xy
        y = np.sin(theta) * radius_xy
        points.append([x, y, z])
    points = np.array(points)
    target_radius = scale * (0.5 * n ** (1.0 / 3.0) + 0.5)
    current_radius = np.max(np.linalg.norm(points, axis=1))
    if current_radius > 1e-10:
        points *= target_radius / current_radius
    return points


def _build_fcc_cluster(n, lattice_constant=1.0):
    """Build FCC lattice cluster."""
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    max_shell = int(np.ceil(2.5 * n ** (1.0 / 3.0))) + 2
    candidates = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k]) * lattice_constant + b
                    candidates.append(p)
    candidates = np.array(candidates)
    candidates -= np.mean(candidates, axis=0)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    return candidates[order[:n]]


def _build_bcc_cluster(n, lattice_constant=1.0):
    """Build BCC lattice cluster."""
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    max_shell = int(np.ceil(2.5 * n ** (1.0 / 3.0))) + 2
    candidates = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k]) * lattice_constant + b
                    candidates.append(p)
    candidates = np.array(candidates)
    candidates -= np.mean(candidates, axis=0)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    return candidates[order[:n]]


def _build_hcp_cluster(n, lattice_constant=1.0):
    """Build HCP-like cluster."""
    a = lattice_constant
    c = a * np.sqrt(2.0 / 3.0) * 2.0
    basis = np.array([
        [0, 0, 0],
        [a, 0, 0],
        [a/2, a*np.sqrt(3)/2, 0],
        [a/2, a*np.sqrt(3)/6, c/2],
        [a, a*np.sqrt(3)/3, c/2],
        [a/2, 2*a*np.sqrt(3)/3, c/2],
    ])
    max_shell = int(np.ceil(2.5 * n ** (1.0 / 3.0))) + 2
    candidates = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + b
                    candidates.append(p)
    candidates = np.array(candidates)
    candidates -= np.mean(candidates, axis=0)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    return candidates[order[:n]]


def _build_random_compact(n, seed, min_dist=0.55):
    """Generate random points with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    points = []
    radius = 0.6 * n ** (1.0 / 3.0) + 0.3
    
    attempts = 0
    max_attempts = n * 800
    
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                valid = False
                break
        if valid:
            points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _build_deformed_fcc(n, seed, deform=0.1):
    """Build FCC cluster with random deformation."""
    base = _build_fcc_cluster(n)
    rng = np.random.default_rng(seed)
    deformed = base + rng.normal(0, deform, size=base.shape)
    return deformed


def construct(instance):
    """Find low-energy Morse clusters using multiple starting configurations and local optimization."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    if potential != "morse":
        spacing = {"morse": 1.0, "sutton_chen": 0.72, "dzugutov": 1.1}.get(potential, 1.0)
        rng = np.random.default_rng(size)
        radius = spacing * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points.append(p)
        energy, positions = relax(np.array(points), potential, rho)
        return positions
    
    # Generate multiple diverse starting configurations
    start_configs = []
    
    # 1. Fibonacci sphere variants
    for scale in [0.9, 1.0, 1.1]:
        start_configs.append(_build_fibonacci_sphere(size, scale))
    
    # 2. FCC lattice
    start_configs.append(_build_fcc_cluster(size))
    
    # 3. BCC lattice
    start_configs.append(_build_bcc_cluster(size))
    
    # 4. HCP-like lattice
    start_configs.append(_build_hcp_cluster(size))
    
    # 5. Deformed FCC variants
    for seed in [42, 123, 456]:
        for deform in [0.05, 0.1, 0.15]:
            start_configs.append(_build_deformed_fcc(size, seed + deform*100, deform))
    
    # 6. Random compact clusters
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        start_configs.append(_build_random_compact(size, seed))
    
    best_energy = np.inf
    best_positions = None
    
    # First pass: L-BFGS-B optimization
    for idx, x0 in enumerate(start_configs):
        try:
            result = minimize(
                fun=lambda x: _morse_energy_and_grad(x.reshape(-1, 3), rho)[0],
                x0=x0.flatten(),
                jac=lambda x: _morse_energy_and_grad(x.reshape(-1, 3), rho)[1].flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_positions = result.x.reshape(-1, 3)
        except Exception:
            continue
    
    # Second pass: refine top candidates with different method
    if best_positions is not None:
        # Try Nelder-Mead on best few
        try:
            result = minimize(
                fun=lambda x: _morse_energy_and_grad(x.reshape(-1, 3), rho)[0],
                x0=best_positions.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_positions = result.x.reshape(-1, 3)
        except Exception:
            pass
    
    # Third pass: use the provided relax function from multiple starting points
    if best_positions is not None:
        try:
            energy, positions = relax(best_positions, 'morse', rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
    
    # Also try relax from a few other good starting points
    for idx, x0 in enumerate(start_configs[:5]):
        try:
            energy, positions = relax(x0, 'morse', rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
    
    # If all else fails, return a simple configuration
    if best_positions is None:
        best_positions = _build_fibonacci_sphere(size)
    
    return best_positions
# EVOLVE-BLOCK-END
