"""Dzugutov cluster global minima search using polytetrahedral/icosahedral initial guesses."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16.0


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy exactly."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-12:
                continue
            if r < a:
                energy += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                energy += B * np.exp(d / (r - b))
    return energy


def _dzugutov_energy_and_grad(positions):
    """Compute Dzugutov potential energy and gradient efficiently."""
    n = len(positions)
    energy = 0.0
    grad = np.zeros_like(positions)
    
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            r = np.linalg.norm(rij)
            if r < 1e-12:
                continue
            
            if r < a:
                r_inv_m = r ** (-m)
                r_inv_m1 = r ** (-(m + 1))
                V = A * (r_inv_m - B) * np.exp(c / (r - a))
                dV_dr = A * (-m * r_inv_m1 - B * r_inv_m1) * np.exp(c / (r - a)) + \
                        A * (r_inv_m - B) * np.exp(c / (r - a)) * c / (r - a) ** 2
            elif r < b:
                V = B * np.exp(d / (r - b))
                dV_dr = B * np.exp(d / (r - b)) * (-d / (r - b) ** 2)
            else:
                continue
            
            energy += V
            factor = dV_dr / r
            grad[i] += factor * rij
            grad[j] -= factor * rij
    
    return energy, grad


def _icosahedron_vertices(radius=1.0):
    """Generate 12 vertices of a regular icosahedron with given radius."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * radius
    return verts


def _build_icosahedral_cluster(n, rng, scale=1.1):
    """Build an icosahedral cluster with central atom and shells."""
    # Start with icosahedral core
    positions = _icosahedron_vertices(scale).tolist()
    positions.append([0.0, 0.0, 0.0])  # Central atom (13 total)
    
    if n <= 13:
        return np.array(positions[:n])
    
    # Add second shell atoms at edge midpoints
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts_unit = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts_unit, axis=1, keepdims=True)
    verts_unit = verts_unit / norms
    
    # Compute edge midpoints (30 edges)
    edges = [
        (0,1),(0,2),(0,4),(0,5),(0,8),(0,11),
        (1,3),(1,4),(1,5),(1,9),(1,10),
        (2,3),(2,6),(2,7),(2,8),(2,11),
        (3,6),(3,7),(3,9),(3,10),
        (4,5),(4,6),(4,7),(4,9),(4,11),
        (5,6),(5,7),(5,8),(5,9),
        (6,8),(6,10),
        (7,8),(7,10),
        (8,11),(9,11)
    ]
    
    second_shell = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        mid = mid / np.linalg.norm(mid) * scale * 1.4
        second_shell.append(mid)
    
    for pos in second_shell:
        if len(positions) >= n:
            break
        positions.append(pos)
    
    # If still not enough, add more in third shell
    while len(positions) < n:
        r = scale * (1.8 + 0.3 * rng.random())
        theta = rng.uniform(0, 2 * np.pi)
        cos_phi = rng.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi ** 2)
        p = np.array([r * sin_phi * np.cos(theta), r * sin_phi * np.sin(theta), r * cos_phi])
        positions.append(p)
    
    return np.array(positions[:n])


def _build_polytetrahedral_cluster(n, rng, scale=1.1):
    """Build a cluster by growing polytetrahedral structure."""
    if n <= 12:
        verts = _icosahedron_vertices(scale)
        return verts[:n].copy()
    
    # Start with icosahedral core
    positions = _icosahedron_vertices(scale).tolist()
    
    # Add atoms in positions that favor polytetrahedral coordination
    target_count = n
    min_dist = 0.55
    
    attempts = 0
    max_attempts = 10000
    
    while len(positions) < target_count and attempts < max_attempts:
        attempts += 1
        r = scale * (1.0 + 0.3 * np.sqrt(attempts / max_attempts))
        theta = rng.uniform(0, 2 * np.pi)
        cos_phi = rng.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi ** 2)
        p = np.array([r * sin_phi * np.cos(theta), r * sin_phi * np.sin(theta), r * cos_phi])
        
        if all(np.linalg.norm(np.array(pos) - p) >= min_dist for pos in positions):
            positions.append(p)
    
    while len(positions) < target_count:
        r = scale * (1.0 + 0.2 * rng.random())
        theta = rng.uniform(0, 2 * np.pi)
        cos_phi = rng.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi ** 2)
        p = np.array([r * sin_phi * np.cos(theta), r * sin_phi * np.sin(theta), r * cos_phi])
        positions.append(p)
    
    return np.array(positions)


def _build_random_compact(n, rng, scale=1.1):
    """Generate a random compact cluster."""
    positions = []
    radius = scale * n ** (1.0 / 3.0) * 0.8
    min_dist = 0.55
    
    attempts = 0
    max_attempts = 50000
    
    while len(positions) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - np.array(pos)) >= min_dist for pos in positions):
                positions.append(p)
    
    while len(positions) < n:
        p = rng.uniform(-radius, radius, 3)
        positions.append(p)
    
    return np.array(positions)


def _ensure_min_distance(positions, rng, min_dist=0.5, max_iter=100):
    """Ensure minimum distance between all atom pairs."""
    for _ in range(max_iter):
        min_dist_violation = False
        for k in range(len(positions)):
            for l in range(k + 1, len(positions)):
                dist = np.linalg.norm(positions[k] - positions[l])
                if dist < min_dist:
                    if dist < 1e-6:
                        positions[k] += 0.1 * rng.standard_normal(3)
                    else:
                        direction = (positions[k] - positions[l]) / dist
                        positions[k] += 0.05 * direction
                        positions[l] -= 0.05 * direction
                    min_dist_violation = True
        if not min_dist_violation:
            break
    return positions


def _simulated_annealing(positions, rng, time_budget=15.0):
    """Simulated annealing with gradient-based moves."""
    start_time = time.time()
    T = 1.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.995
    step_size = 0.1
    
    current_energy, _ = _dzugutov_energy_and_grad(positions)
    best_positions = positions.copy()
    best_energy = current_energy
    
    n_steps = 0
    while time.time() - start_time < time_budget:
        # Generate a gradient-based move with noise
        _, grad = _dzugutov_energy_and_grad(positions)
        
        # Gradient descent step with noise
        noise = step_size * rng.standard_normal(positions.shape)
        proposed = positions - 0.01 * grad + noise
        
        # Ensure minimum distance
        proposed = _ensure_min_distance(proposed, rng, min_dist=0.5, max_iter=20)
        
        proposed_energy, _ = _dzugutov_energy_and_grad(proposed)
        
        # Metropolis criterion
        delta_E = proposed_energy - current_energy
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            positions = proposed
            current_energy = proposed_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = positions.copy()
        
        # Cool down
        T = max(T_min, T * cooling_rate)
        n_steps += 1
    
    return best_positions, best_energy, n_steps


def _refine_with_bfgs(positions, time_budget=10.0):
    """Refine positions using BFGS with gradient."""
    start_time = time.time()
    
    def objective(x):
        if time.time() - start_time > time_budget:
            return float('inf')
        pos = x.reshape(-1, 3)
        e, g = _dzugutov_energy_and_grad(pos)
        return e
    
    def gradient(x):
        if time.time() - start_time > time_budget:
            return np.zeros_like(x)
        pos = x.reshape(-1, 3)
        _, g = _dzugutov_energy_and_grad(pos)
        return g.flatten()
    
    try:
        x0 = positions.flatten()
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                         options={'maxiter': 100, 'maxfun': 500})
        if result.fun < float('inf'):
            refined = result.x.reshape(-1, 3)
            refined = _ensure_min_distance(refined, np.random.default_rng(42), min_dist=0.5)
            return refined, result.fun
    except Exception:
        pass
    
    return positions, _dzugutov_energy(positions)


def construct(instance):
    """
    Find low-energy structures for Dzugutov clusters using multiple
    polytetrahedral/icosahedral initial guesses, simulated annealing,
    and gradient-based refinement.
    """
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    rng = np.random.default_rng(n * 7919)
    
    best_energy = float('inf')
    best_positions = None
    
    # Phase 1: Generate diverse initial guesses and relax
    n_restarts = 8
    
    for i in range(n_restarts):
        if time.time() > deadline - 40:
            break
        
        # Choose initialization strategy
        strategy = i % 4
        
        if strategy == 0:
            positions = _build_icosahedral_cluster(n, rng, scale=1.1 + 0.05 * rng.random())
        elif strategy == 1:
            positions = _build_polytetrahedral_cluster(n, rng, scale=1.1 + 0.05 * rng.random())
        elif strategy == 2:
            positions = _build_random_compact(n, rng, scale=1.1 + 0.1 * rng.random())
        else:
            positions = _build_icosahedral_cluster(n, rng, scale=1.1)
            noise = 0.1 * rng.standard_normal(positions.shape)
            positions = positions + noise
        
        # Ensure minimum distance
        positions = _ensure_min_distance(positions, rng, min_dist=0.5)
        
        # Relax
        try:
            energy, relaxed = relax(positions, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed.copy()
        except Exception:
            pass
    
    # Phase 2: Simulated annealing on best candidates
    if best_positions is not None and time.time() < deadline - 30:
        # Run SA on the best position with different temperatures
        for sa_run in range(3):
            if time.time() > deadline - 20:
                break
            
            noise = 0.2 * rng.standard_normal(best_positions.shape)
            sa_start = best_positions + noise
            sa_start = _ensure_min_distance(sa_start, rng, min_dist=0.5)
            
            time_per_sa = min(15.0, (deadline - 20 - time.time()) / (3 - sa_run))
            sa_result, sa_energy, _ = _simulated_annealing(sa_start, rng, time_budget=time_per_sa)
            
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_positions = sa_result.copy()
    
    # Phase 3: BFGS refinement on the best candidate
    if best_positions is not None and time.time() < deadline - 15:
        refined, refined_energy = _refine_with_bfgs(best_positions, time_budget=min(10.0, deadline - 15 - time.time()))
        
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_positions = refined.copy()
    
    # Final relaxation
    if best_positions is not None and time.time() < deadline - 5:
        try:
            energy, relaxed = relax(best_positions, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed.copy()
        except Exception:
            pass
    
    if best_positions is None:
        # Fallback: return a simple compact arrangement
        positions = _build_random_compact(n, rng, scale=1.1)
        positions = _ensure_min_distance(positions, rng, min_dist=0.5)
        try:
            energy, relaxed = relax(positions, potential, parameter)
            best_positions = relaxed
        except Exception:
            best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END
