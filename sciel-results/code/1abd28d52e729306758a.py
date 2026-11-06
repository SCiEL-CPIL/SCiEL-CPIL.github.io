"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time
from scipy.optimize import minimize
import threading

# Global deadline management
_START_TIME = None
_DEADLINE = 330.0  # seconds

def _get_remaining_time():
    """Return remaining time in seconds."""
    if _START_TIME is None:
        return _DEADLINE
    return max(0.0, _DEADLINE - (time.time() - _START_TIME))

def _dzugutov_energy(positions, A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16):
    """Compute Dzugutov potential energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    # Compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r = dist[i, j]
            if r < a:
                energy += A * (r**(-m) - B) * np.exp(c / (r - a))
            elif r < b:
                energy += B * np.exp(d / (r - b))
    return energy

def _generate_mackay_icosahedron(n, spacing=1.1):
    """Generate a Mackay icosahedral cluster with n atoms."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Icosahedron vertices (12)
    phi = (1 + np.sqrt(5)) / 2
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices[0]) * spacing
    
    # Second shell: octahedron (6) at distance ~2*spacing
    oct_vertices = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ])
    oct_vertices = oct_vertices * spacing * 1.9
    
    # Third shell: 30 atoms (edges of icosahedron)
    third_shell = []
    for i in range(12):
        for j in range(i+1, 12):
            v1 = ico_vertices[i]
            v2 = ico_vertices[j]
            # Check if these are adjacent (distance ~ spacing * sqrt(5))
            d = np.linalg.norm(v1 - v2)
            if 1.5 * spacing < d < 2.5 * spacing:
                # Midpoint, pushed outward
                mid = (v1 + v2) / 2
                mid = mid / np.linalg.norm(mid) * spacing * 2.8
                third_shell.append(mid)
    
    third_shell = np.array(third_shell) if third_shell else np.array([])
    
    # Fourth shell: 20 atoms (faces of icosahedron)
    fourth_shell = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                v1, v2, v3 = ico_vertices[i], ico_vertices[j], ico_vertices[k]
                # Check if these form a face (all pairwise distances ~ spacing*sqrt(5))
                d12 = np.linalg.norm(v1 - v2)
                d13 = np.linalg.norm(v1 - v3)
                d23 = np.linalg.norm(v2 - v3)
                if (1.5 * spacing < d12 < 2.5 * spacing and
                    1.5 * spacing < d13 < 2.5 * spacing and
                    1.5 * spacing < d23 < 2.5 * spacing):
                    centroid = (v1 + v2 + v3) / 3
                    centroid = centroid / np.linalg.norm(centroid) * spacing * 3.5
                    fourth_shell.append(centroid)
    
    fourth_shell = np.array(fourth_shell) if fourth_shell else np.array([])
    
    # Assemble cluster
    atoms = [np.array([0.0, 0.0, 0.0])]  # center
    atoms.extend(ico_vertices.tolist())  # 12
    atoms.extend(oct_vertices.tolist())  # 6
    atoms.extend(third_shell.tolist())   # up to 30
    atoms.extend(fourth_shell.tolist())  # up to 20
    
    atoms = np.array(atoms[:n])
    return atoms

def _generate_random_compact(n, spacing=1.1, rng=None):
    """Generate a random compact cluster."""
    if rng is None:
        rng = np.random.default_rng()
    
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    attempts = 0
    max_attempts = 10000
    
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 * spacing for q in points):
                points.append(p)
    
    # If we didn't get enough, add more randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)

def _relax_with_l_bfgs(positions, potential, parameter, max_iter=2000):
    """Relax using L-BFGS-B with gradient."""
    def objective(x):
        pos = x.reshape(-1, 3)
        # Check minimum distance constraint
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        if np.min(dist) < 0.5:
            return 1e6
        return _dzugutov_energy(pos)
    
    def gradient(x):
        pos = x.reshape(-1, 3)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        if np.min(dist) < 0.5:
            return np.zeros_like(x)
        
        # Compute gradient numerically or use provided function
        try:
            grad = dzugutov_energy_gradient(pos)
            return grad.flatten()
        except:
            # Numerical gradient
            eps = 1e-7
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
            return grad
    
    x0 = positions.flatten()
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    
    if result.success:
        return result.fun, result.x.reshape(-1, 3)
    else:
        return objective(x0), positions

def _basin_hopping(positions, potential, parameter, n_iterations=50, step_size=0.5):
    """Simple basin hopping algorithm."""
    best_energy, best_positions = _relax_with_l_bfgs(positions, potential, parameter, max_iter=1000)
    
    current_energy = best_energy
    current_positions = best_positions.copy()
    rng = np.random.default_rng(hash(tuple(positions[0])) % 2**32)
    
    for _ in range(n_iterations):
        if _get_remaining_time() < 5.0:
            break
        
        # Perturb
        perturbed = current_positions + rng.normal(0, step_size, current_positions.shape)
        
        # Relax
        new_energy, new_positions = _relax_with_l_bfgs(perturbed, potential, parameter, max_iter=500)
        
        # Metropolis acceptance
        if new_energy < current_energy:
            current_energy = new_energy
            current_positions = new_positions
            if new_energy < best_energy:
                best_energy = new_energy
                best_positions = new_positions.copy()
    
    return best_energy, best_positions

def construct(instance):
    """Find low-energy structure for Dzugutov cluster."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Generate multiple starting structures
    starts = []
    
    # 1. Mackay icosahedral
    try:
        ico = _generate_mackay_icosahedron(size, spacing=1.1)
        if len(ico) == size:
            starts.append(ico)
    except:
        pass
    
    # 2. Random compact (multiple)
    rng = np.random.default_rng(size)
    for _ in range(3):
        starts.append(_generate_random_compact(size, spacing=1.1, rng=rng))
    
    # 3. Slightly perturbed icosahedral
    if len(starts) > 0:
        perturbed_ico = starts[0] + rng.normal(0, 0.1, starts[0].shape)
        starts.append(perturbed_ico)
    
    # Relax each starting structure
    for i, start in enumerate(starts):
        if _get_remaining_time() < 10.0:
            break
        
        # First try direct relaxation
        try:
            energy, positions = _relax_with_l_bfgs(start, potential, parameter, max_iter=1500)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception as e:
            pass
        
        # If time permits, try basin hopping on the best so far
        if i == 0 and _get_remaining_time() > 30.0 and best_positions is not None:
            try:
                bh_energy, bh_positions = _basin_hopping(best_positions, potential, parameter, 
                                                         n_iterations=30, step_size=0.4)
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_positions.copy()
            except:
                pass
    
    # Final fallback: use the relax function from the library
    if best_positions is None or _get_remaining_time() > 5.0:
        try:
            energy, positions = relax(starts[0], potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            pass
    
    if best_positions is None:
        # Absolute fallback
        best_positions = _generate_random_compact(size, spacing=1.1)
    
    # Ensure minimum distance constraint
    for _ in range(100):
        diff = best_positions[:, np.newaxis, :] - best_positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        if np.min(dist) >= 0.5:
            break
        # Push apart atoms that are too close
        for i in range(size):
            for j in range(i+1, size):
                if dist[i, j] < 0.5:
                    direction = best_positions[i] - best_positions[j]
                    direction = direction / (np.linalg.norm(direction) + 1e-10)
                    best_positions[i] += 0.01 * direction
                    best_positions[j] -= 0.01 * direction
    
    return best_positions
# EVOLVE-BLOCK-END
