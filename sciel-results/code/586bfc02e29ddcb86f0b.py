"""Morse cluster global minima finder using structured initializers and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time management
TOTAL_TIME_BUDGET = 290.0  # Leave 10s margin for evaluator overhead
INSTANCE_TIME_BUDGET = TOTAL_TIME_BUDGET / 6.0
_start_time_global = time.time()

def _morse_energy(positions, rho):
    """Compute Morse energy for a cluster."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            x = rho * (1.0 - r)
            e = np.exp(x)
            energy += e * (e - 2.0)
    return energy

def _morse_gradient(positions, rho):
    """Compute Morse energy gradient."""
    n = len(positions)
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            r_vec = positions[i] - positions[j]
            r = np.linalg.norm(r_vec)
            if r < 1e-12:
                continue
            x = rho * (1.0 - r)
            e = np.exp(x)
            # dE/dr = -rho * e * (2e - 1)
            dEdr = -rho * e * (2.0 * e - 1.0)
            # Force on i is -dE/dr * (r_vec / r), on j is opposite
            f = dEdr * r_vec / r
            grad[i] += f
            grad[j] -= f
    return grad

def _local_minimize(positions, rho, max_steps=2000, tol=1e-10):
    """Local minimization using L-BFGS-B."""
    from scipy.optimize import minimize
    
    def objective(x):
        pos = x.reshape(-1, 3)
        return _morse_energy(pos, rho)
    
    def gradient(x):
        pos = x.reshape(-1, 3)
        return _morse_gradient(pos, rho).flatten()
    
    x0 = positions.flatten()
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                      options={'maxiter': max_steps, 'ftol': tol, 'gtol': 1e-8})
    best_pos = result.x.reshape(-1, 3)
    best_energy = result.fun
    
    # Verify with the provided relax function
    energy_relax, positions_relax = relax(positions.copy(), 'morse', rho)
    if energy_relax < best_energy:
        best_energy = energy_relax
        best_pos = positions_relax
    
    return best_energy, best_pos

def _generate_fcc_shells(n, spacing=1.0):
    """Generate n atoms in FCC shell structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # FCC lattice points
    points = []
    # Start with origin
    points.append(np.array([0.0, 0.0, 0.0]))
    
    # Generate FCC lattice points in shells
    max_coord = int(np.ceil(n ** (1/3.0)) * 1.5) + 3
    fcc_points = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC: points where i+j+k is even, scaled by spacing/2
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (spacing / 2.0)
                    fcc_points.append(p)
    
    # Sort by distance from origin
    fcc_points = np.array(fcc_points)
    distances = np.linalg.norm(fcc_points, axis=1)
    sorted_indices = np.argsort(distances)
    
    # Select n closest points
    selected = fcc_points[sorted_indices[:n]]
    return selected

def _generate_hcp_shells(n, spacing=1.0):
    """Generate n atoms in HCP shell structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = []
    # HCP layers
    layer_spacing = spacing * np.sqrt(2.0 / 3.0)
    in_layer_spacing = spacing
    
    # Generate layers
    max_layer = int(np.ceil(n ** (1/3.0))) + 3
    for layer_idx in range(-max_layer, max_layer + 1):
        z = layer_idx * layer_spacing
        # Hexagonal grid in this layer
        max_r = int(np.ceil(n ** (1/3.0))) + 3
        for i in range(-max_r, max_r + 1):
            for j in range(-max_r, max_r + 1):
                # Hexagonal coordinates
                x = (i + 0.5 * (j % 2)) * in_layer_spacing
                y = j * in_layer_spacing * np.sqrt(3.0) / 2.0
                p = np.array([x, y, z])
                points.append(p)
    
    points = np.array(points)
    distances = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(distances)
    selected = points[sorted_indices[:n]]
    return selected

def _generate_icosahedral(n, spacing=1.0):
    """Generate n atoms in icosahedral shell structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])  # Normalize to unit sphere
    
    # Generate shells
    points = [np.array([0.0, 0.0, 0.0])]  # Center
    shell_radii = [1.0, 1.6, 2.2, 2.8, 3.4, 4.0]  # Approximate icosahedral shell radii
    
    for radius in shell_radii:
        if len(points) >= n:
            break
        # Distribute points on sphere using icosahedron vertices and subdivisions
        for v in ico_verts:
            if len(points) >= n:
                break
            p = v * radius * spacing
            points.append(p)
            # Add antipodal point
            if len(points) < n:
                points.append(-p)
    
    points = np.array(points[:n])
    return points

def _generate_random_compact(n, spacing=1.0, seed=None):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms, use a fallback
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)

def _basin_hopping(positions, rho, n_restarts=5, max_steps_per_restart=500, 
                   delta_scale=0.5, time_limit=None):
    """Basin hopping with multiple restarts."""
    best_energy = _morse_energy(positions, rho)
    best_positions = positions.copy()
    
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Generate perturbation
        delta = delta_scale * rng.normal(size=positions.shape)
        perturbed = positions + delta
        
        # Local minimize
        energy, pos = _local_minimize(perturbed, rho, max_steps=max_steps_per_restart)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = pos
            positions = pos.copy()
    
    return best_energy, best_positions

def construct(instance):
    """Find global minimum of Morse cluster."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Time budget for this instance
    elapsed_global = time.time() - _start_time_global
    remaining_time = max(1.0, INSTANCE_TIME_BUDGET - (elapsed_global % INSTANCE_TIME_BUDGET))
    
    # Generate multiple initial structures
    initial_structures = []
    
    # 1. FCC shells
    try:
        fcc = _generate_fcc_shells(size)
        initial_structures.append(fcc)
    except Exception:
        pass
    
    # 2. HCP shells
    try:
        hcp = _generate_hcp_shells(size)
        initial_structures.append(hcp)
    except Exception:
        pass
    
    # 3. Icosahedral
    try:
        ico = _generate_icosahedral(size)
        initial_structures.append(ico)
    except Exception:
        pass
    
    # 4. Random compact structures
    for seed in [0, 1, 2, 3]:
        try:
            rand = _generate_random_compact(size, seed=seed)
            initial_structures.append(rand)
        except Exception:
            pass
    
    # Ensure we have at least one structure
    if not initial_structures:
        initial_structures.append(_generate_random_compact(size, seed=0))
    
    # Local minimize all initial structures
    best_energy = np.inf
    best_positions = None
    
    start_time = time.time()
    for i, struct in enumerate(initial_structures):
        if (time.time() - start_time) > remaining_time * 0.5:
            break
        
        try:
            energy, pos = _local_minimize(struct, parameter, max_steps=500)
            if energy < best_energy:
                best_energy = energy
                best_positions = pos
        except Exception:
            continue
    
    # If we have time, do basin hopping from the best structure
    if best_positions is not None and (time.time() - start_time) < remaining_time * 0.8:
        try:
            bh_time = remaining_time * 0.15
            bh_energy, bh_pos = _basin_hopping(best_positions, parameter, 
                                               n_restarts=3, max_steps_per_restart=300,
                                               time_limit=bh_time)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos
        except Exception:
            pass
    
    # Final check with relax
    if best_positions is not None:
        try:
            final_energy, final_pos = relax(best_positions, potential, parameter)
            if final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_pos
        except Exception:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
