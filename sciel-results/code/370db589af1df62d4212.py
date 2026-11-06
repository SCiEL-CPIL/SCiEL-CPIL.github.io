# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time tracking to ensure we don't exceed the 300s limit
_GLOBAL_START_TIME = None
_GLOBAL_BUDGET = 290.0  # Safety margin for 300s limit

def _get_global_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    elapsed = time.time() - _GLOBAL_START_TIME
    return max(0.0, _GLOBAL_BUDGET - elapsed)

def _morse_energy(pos, rho):
    """Compute Morse energy for a configuration."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    upper = np.triu_indices(n, 1)
    r = dists[upper]
    val = rho * (1.0 - r)
    exp_val = np.exp(val)
    return np.sum(exp_val * (exp_val - 2.0))

def _morse_energy_and_grad(pos, rho):
    """Compute Morse energy and gradient for a configuration."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Avoid division by zero in gradient calculation
    dists_safe = np.where(dists < 1e-10, 1e-10, dists)
    
    upper = np.triu_indices(n, 1)
    r = dists_safe[upper]
    val = rho * (1.0 - r)
    exp_val = np.exp(val)
    energy = np.sum(exp_val * (exp_val - 2.0))
    
    # Gradient calculation
    # dE/dr_ij = 2 * rho * exp(2*rho(1-r)) - 2*rho*exp(rho(1-r))
    # dE/dx_i = sum_j (dE/dr_ij) * (x_i - x_j) / r_ij
    dE_dr = 2.0 * rho * (np.exp(2.0 * val) - np.exp(val))
    
    # Vectorized gradient accumulation
    # We need to sum contributions from all pairs
    # diff[upper] gives vectors r_i - r_j
    vecs = diff[upper]
    # dE/dx_i = sum (dE_dr * vec / r)
    # dE/dx_j = - sum (dE_dr * vec / r)
    
    # Create a matrix of contributions
    # contrib = dE_dr[:, None] * vecs / r[:, None]
    contrib = (dE_dr / r_safe)[:, None] * vecs
    
    grad = np.zeros_like(pos)
    # Scatter add
    # This is tricky to vectorize perfectly without loops for arbitrary n, 
    # but for n<100, a simple loop over pairs is fast enough in numpy if vectorized per pair?
    # Actually, let's use the provided morse_energy_gradient if it's fast, 
    # or implement a fast vectorized version.
    
    # Let's use the provided function for gradient to be safe and fast, 
    # as it's likely optimized in C or Cython in the suite.
    # The function signature is morse_energy_gradient(positions, rho) -> (energy, gradient)
    # Let's check if it returns both. The docstring says "may be called", implying it's a utility.
    # Usually, these utilities return (energy, gradient).
    
    try:
        e, g = morse_energy_gradient(pos, rho)
        return float(e), g
    except Exception:
        # Fallback to manual calculation if the function fails or returns differently
        # Re-calculate energy manually to be sure
        energy_manual = _morse_energy(pos, rho)
        
        # Manual gradient
        grad = np.zeros_like(pos)
        for i in range(n):
            for j in range(i+1, n):
                rij = pos[i] - pos[j]
                r = np.linalg.norm(rij)
                if r < 1e-10: continue
                val = rho * (1.0 - r)
                exp_val = np.exp(val)
                dE_dr = 2.0 * rho * (np.exp(2.0 * val) - exp_val)
                force_vec = (dE_dr / r) * rij
                grad[i] += force_vec
                grad[j] -= force_vec
        return energy_manual, grad

def _min_dist_check(pos, min_d=0.5):
    n = len(pos)
    if n < 2: return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_d

def _fix_min_dist(pos, min_d=0.5, max_iter=100):
    """Try to increase minimum distance by scaling or shifting."""
    for _ in range(max_iter):
        if _min_dist_check(pos, min_d):
            return pos
        # Scale up slightly
        pos = pos * 1.01
    return pos

def _fcc_cluster(n, rng):
    """Generate FCC lattice points."""
    points = []
    a = 1.0
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    max_coord = int(np.ceil(n ** (1.0 / 3.0)) + 3)
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + b
                    points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order]
    if len(points) >= n:
        return points[:n].copy()
    else:
        extra = rng.uniform(-1, 1, (n - len(points), 3))
        return np.vstack([points, extra])

def _hcp_cluster(n, rng):
    """Generate HCP lattice points."""
    points = []
    a = 1.0
    c = 1.633 * a
    # Hexagonal lattice vectors
    # v1 = (1, 0, 0), v2 = (0.5, sqrt(3)/2, 0), v3 = (0, 0, c)
    # Basis: (0,0,0), (1/3, 2/3, 0.5) in fractional coords of the hex prism?
    # Simpler: Generate layers
    max_coord = int(np.ceil(n ** (1.0 / 3.0)) + 3)
    for layer in range(-max_coord, max_coord + 1):
        z = layer * c
        # Hexagonal 2D lattice
        for i in range(-max_coord, max_coord + 1):
            for j in range(-max_coord, max_coord + 1):
                # Cartesian coords for hex lattice
                x = i * a + 0.5 * j * a
                y = j * a * np.sqrt(3) / 2
                # Shift for B layer
                if layer % 2 != 0:
                    x += a / 3
                    y += a * np.sqrt(3) / 3
                points.append([x, y, z])
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order]
    if len(points) >= n:
        return points[:n].copy()
    else:
        extra = rng.uniform(-1, 1, (n - len(points), 3))
        return np.vstack([points, extra])

def _icosahedron_shell(n, rng):
    """Generate points on icosahedral shells."""
    points = []
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0] # Duplicates in standard def, let's use unique
    ])
    # Standard 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Add center
    points.append([0, 0, 0])
    
    # Shell 1: 12 vertices
    points.extend(verts)
    
    if n <= 13:
        return np.array(points[:n])
    
    # Shell 2: 30 vertices (midpoints of edges)
    # Edges of icosahedron... this is complex. 
    # Alternative: Use a simpler "magic number" shell approach or just perturb FCC.
    # For n > 13, icosahedral structures are less common as global minima for Morse 
    # unless n is specific magic numbers. 
    # Let's just return a perturbed FCC for larger n to keep it simple and fast,
    # or generate more points on the sphere.
    
    # Generate points on sphere using Fibonacci spiral
    def fibonacci_sphere(num_points, radius=2.0):
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(num_points):
            y = 1 - (i / float(num_points - 1)) * 2
            radius_at_y = np.sqrt(1 - y*y)
            theta = phi * i
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            points.append([x * radius, y * radius, z * radius])
        return np.array(points)
    
    remaining = n - len(points)
    shell_pts = fibonacci_sphere(remaining, radius=2.0)
    points.extend(shell_pts)
    
    return np.array(points[:n])

def _random_compact(n, rng, min_dist=0.7):
    """Generate n random points in a sphere with minimum distance constraint."""
    radius = n ** (1.0 / 3.0) * 0.8
    points = []
    max_attempts = n * 300
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
            attempts += 1
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _optimize(pos, rho, time_budget):
    """Run L-BFGS-B optimization."""
    if time_budget <= 0:
        return pos, _morse_energy(pos, rho)
    
    start_time = time.time()
    
    def obj(x):
        e, g = _morse_energy_and_grad(x.reshape(-1, 3), rho)
        return e, g
    
    try:
        res = minimize(obj, pos.reshape(-1), method='L-BFGS-B', jac=True, 
                       options={'maxiter': 2000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        # Check time
        if time.time() - start_time > time_budget:
            return pos, _morse_energy(pos, rho)
            
        new_pos = res.x.reshape(-1, 3)
        new_energy = res.fun
        
        # Ensure min distance
        if not _min_dist_check(new_pos, 0.5):
             new_pos = _fix_min_dist(new_pos, 0.5)
             new_energy = _morse_energy(new_pos, rho)
             
        return new_pos, new_energy
    except Exception:
        return pos, _morse_energy(pos, rho)

def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    
    remaining_time = _get_global_remaining_time()
    if remaining_time <= 1.0:
        # Not enough time, return a valid lattice
        return _fcc_cluster(size, np.random.default_rng(42))
    
    # Allocate time for this instance
    # We have 6 instances. Let's give each ~40s, but adjust dynamically.
    # If it's the last instance, give it all remaining.
    # Simple heuristic: 1/3 of remaining time, min 10s, max 60s.
    time_budget = max(10.0, min(60.0, remaining_time * 0.4))
    
    rng = np.random.default_rng(size * 7 + int(rho))
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial configurations
    init_configs = []
    
    # 1. FCC
    init_configs.append(_fcc_cluster(size, rng))
    # 2. HCP
    init_configs.append(_hcp_cluster(size, rng))
    # 3. Icosahedral/Shell
    if size <= 30:
        init_configs.append(_icosahedron_shell(size, rng))
    else:
        # For larger n, use perturbed FCC
        init_configs.append(_fcc_cluster(size, rng) + rng.normal(0, 0.1, size=(size, 3)))
        
    # 4-6. Random compact
    for md in [0.8, 0.7, 0.6]:
        init_configs.append(_random_compact(size, rng, min_dist=md))
        
    # 7. Perturbed FCC
    init_configs.append(_fcc_cluster(size, rng) + rng.normal(0, 0.2, size=(size, 3)))
    
    # Relax each initial configuration
    for idx, init_pos in enumerate(init_configs):
        if _get_global_remaining_time() < 2.0:
            break
            
        # Ensure valid minimum distance before optimization
        if not _min_dist_ok(init_pos, 0.5):
            init_pos = _fix_min_dist(init_pos, 0.5)
            
        # Run optimization
        opt_pos, opt_energy = _optimize(init_pos.copy(), rho, time_budget / len(init_configs))
        
        if opt_energy < best_energy and _min_dist_check(opt_pos, 0.5):
            best_energy = opt_energy
            best_positions = opt_pos.copy()
            
    # If no valid result, try a few more random starts or use fallback
    if best_positions is None:
        best_positions = _fcc_cluster(size, rng)
        best_positions = _fix_min_dist(best_positions, 0.5)
        best_energy = _morse_energy(best_positions, rho)
        
    # Final Polish: Use remaining time to polish the best candidate
    remaining = _get_global_remaining_time()
    if remaining > 5.0:
        polished_pos, polished_energy = _optimize(best_positions.copy(), rho, remaining - 2.0)
        if polished_energy < best_energy and _min_dist_check(polished_pos, 0.5):
            best_energy = polished_energy
            best_positions = polished_pos.copy()
            
    return best_positions
# EVOLVE-BLOCK-END
