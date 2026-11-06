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
            # E_pair = exp(rho(1-r)) * (exp(rho(1-r)) - 2)
            e_pair = exp_val * (exp_val - 2.0)
            # dE/dr = -rho * exp(rho(1-r)) * (2*exp(rho(1-r)) - 1)
            de_dr = -rho * exp_val * (2.0 * exp_val - 1.0)
            # grad_i += de_dr * (diff / r), grad_j -= de_dr * (diff / r)
            unit = diff / r
            force = de_dr * unit
            grad[i] += force
            grad[j] -= force
            energy += e_pair
    return energy, grad


def _build_icosahedral_shell(n):
    """Generate a roughly spherical cluster using golden spiral points on a sphere, then scale."""
    # Use Fibonacci sphere for uniform distribution
    points = []
    for i in range(n):
        z = 1.0 - 2.0 * i / (n - 1) if n > 1 else 1.0
        radius_xy = np.sqrt(max(0.0, 1.0 - z * z))
        theta = np.pi * (3.0 - np.sqrt(5.0)) * i  # golden angle
        x = np.cos(theta) * radius_xy
        y = np.sin(theta) * radius_xy
        points.append([x, y, z])
    points = np.array(points)
    # Scale to approximate equilibrium spacing of 1.0
    # For a cluster of n atoms, typical radius ~ n^(1/3)
    target_radius = 0.5 * n ** (1.0 / 3.0) + 0.5
    current_radius = np.max(np.linalg.norm(points, axis=1))
    if current_radius > 1e-10:
        points *= target_radius / current_radius
    return points


def _build_lattice_cluster(n):
    """Build a compact FCC-like lattice cluster."""
    points = []
    # Generate FCC lattice points within a sphere
    a = 1.0  # lattice constant, equilibrium distance
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    
    # Estimate max shell index needed
    max_shell = int(np.ceil(2.0 * n ** (1.0 / 3.0))) + 2
    
    candidates = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + b
                    candidates.append(p)
    
    candidates = np.array(candidates)
    # Center at origin
    candidates -= np.mean(candidates, axis=0)
    
    # Sort by distance from center, take closest n
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    points = candidates[order[:n]]
    return points


def _build_random_compact(n, seed):
    """Generate random points with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    points = []
    radius = 0.6 * n ** (1.0 / 3.0) + 0.3
    min_dist = 0.55
    
    attempts = 0
    max_attempts = n * 500
    
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
    
    # If we didn't get enough points, fill with random points (may violate min dist slightly)
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def construct(instance):
    """Find low-energy Morse clusters using multiple starting configurations and local optimization."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    if potential != "morse":
        # Fallback for other potentials
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
    
    # 1. Fibonacci sphere (spherical shell)
    start_configs.append(_build_icosahedral_shell(size))
    
    # 2. FCC lattice cluster
    start_configs.append(_build_lattice_cluster(size))
    
    # 3. Random compact clusters with different seeds
    for seed in [42, 123, 456, 789, 101]:
        start_configs.append(_build_random_compact(size, seed))
    
    best_energy = np.inf
    best_positions = None
    
    for idx, x0 in enumerate(start_configs):
        # Ensure minimum distance constraint is roughly satisfied
        # Run L-BFGS-B with custom energy and gradient
        try:
            result = minimize(
                fun=lambda x: _morse_energy_and_grad(x.reshape(-1, 3), rho)[0],
                x0=x0.flatten(),
                jac=lambda x: _morse_energy_and_grad(x.reshape(-1, 3), rho)[1].flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < best_energy:
                pos = result.x.reshape(-1, 3)
                e = result.fun
                if e < best_energy:
                    best_energy = e
                    best_positions = pos
        except Exception:
            continue
    
    # Also try the relax function on the best configuration found
    if best_positions is not None:
        try:
            energy, positions = relax(best_positions, 'morse', rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
    
    # If all else fails, return a simple configuration
    if best_positions is None:
        best_positions = _build_icosahedral_shell(size)
    
    return best_positions
# EVOLVE-BLOCK-END
