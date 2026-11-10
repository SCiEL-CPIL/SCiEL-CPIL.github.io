# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(positions, rho):
    """Compute exact Morse energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(r, np.inf)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    iu = np.triu_indices(n, k=1)
    return float(np.sum(e[iu]))


def _generate_fcc(n, spacing=1.0):
    """Generate an FCC-like cluster of n atoms."""
    # FCC lattice points in a cube
    points = []
    # Generate FCC lattice points
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * spacing
                    points.append(p)
    
    points = np.array(points)
    # Center the cluster
    center = np.mean(points, axis=0)
    points -= center
    
    # Sort by distance from center and take the n closest
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    
    return points


def _generate_bcc(n, spacing=1.0):
    """Generate a BCC-like cluster of n atoms."""
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                # BCC basis: (0,0,0), (0.5,0.5,0.5)
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
    
    # Generate HCP lattice points
    for layer in range(-4, 5):
        z = layer * c / 2.0
        # A and B layers
        offset = 0.0 if layer % 2 == 0 else 1.0
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = (i + 0.5 * (j % 2)) * a
                y = j * a * np.sqrt(3.0) / 2.0
                # A layer
                points.append(np.array([x, y, z]))
                # B layer (offset)
                if offset == 0:
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
    # Use Fibonacci sphere for even distribution
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(1, n):
        z = 1.0 - 2.0 * i / (n - 1)
        radius = np.sqrt(max(0, 1.0 - z * z))
        theta = golden_angle * i
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        points.append(np.array([x, y, z]) * spacing * (n ** (1/3.0) * 0.5))
    
    return np.array(points)


def _relax_with_gradient(positions, potential, rho, max_iter=500, tol=1e-8):
    """Custom relaxation using L-BFGS-B with Morse gradient."""
    from scipy.optimize import minimize
    
    def fun(x):
        pos = x.reshape(-1, 3)
        return _morse_energy(pos, rho)
    
    def jac(x):
        pos = x.reshape(-1, 3)
        e, g = morse_energy_gradient(pos, rho)
        return g.flatten()
    
    x0 = positions.flatten()
    result = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'gtol': tol, 'maxls': 50})
    final_pos = result.x.reshape(-1, 3)
    final_energy = _morse_energy(final_pos, rho)
    return final_energy, final_pos


def construct(instance):
    """Find low-energy Morse cluster configurations."""
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
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
            energy, relaxed = _relax_with_gradient(positions, "morse", rho, max_iter=300)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except Exception:
            continue
    
    # Additional random restarts if time permits
    if best_positions is None:
        # Fallback: use the original approach
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
    
    # If time remains, try a few more random perturbations of the best solution
    if best_positions is not None and time.time() < deadline - 30:
        rng = np.random.default_rng(n + 1)
        for _ in range(5):
            if time.time() > deadline:
                break
            
            # Perturb the best solution
            noise = rng.normal(0, 0.3, size=best_positions.shape)
            perturbed = best_positions + noise
            
            try:
                energy, relaxed = _relax_with_gradient(perturbed, "morse", rho, max_iter=200)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
            except Exception:
                continue
    
    return best_positions
# EVOLVE-BLOCK-END
