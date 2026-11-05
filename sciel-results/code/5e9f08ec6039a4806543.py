"""Topological optimization for Dzugutov clusters using polytetrahedral connectivity constraints."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = None
_DEADLINE_MARGIN = 10.0  # seconds before the 330s limit

def _get_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME + 330.0 - _DEADLINE_MARGIN

def _time_remaining():
    return _get_deadline() - time.time()

def _dzugutov_params():
    return {'A': 5.82, 'B': 1.28, 'a': 1.87, 'b': 1.94, 'c': 1.10, 'd': 0.27, 'm': 16}

def _dzugutov_energy(positions, params=None):
    """Compute Dzugutov potential energy."""
    if params is None:
        params = _dzugutov_params()
    A, B, a, b, c, d, m = params['A'], params['B'], params['a'], params['b'], params['c'], params['d'], params['m']
    n = len(positions)
    if n < 2:
        return 0.0
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    total_energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r = dist[i, j]
            if r < a:
                total_energy += A * (r**(-m) - B) * np.exp(c / (r - a))
            elif r < b:
                total_energy += B * np.exp(d / (r - b))
    return total_energy

def _find_rmin(params=None):
    """Find the minimum of the Dzugutov potential."""
    if params is None:
        params = _dzugutov_params()
    A, B, a, b, c, d, m = params['A'], params['B'], params['a'], params['b'], params['c'], params['d'], params['m']
    
    def V(r):
        if r < a:
            return A * (r**(-m) - B) * np.exp(c / (r - a))
        elif r < b:
            return B * np.exp(d / (r - b))
        else:
            return 0.0
    
    # Scan for minimum
    rs = np.linspace(0.5, b, 1000)
    Vs = np.array([V(r) for r in rs])
    idx = np.argmin(Vs)
    rmin = rs[idx]
    
    # Refine with golden section
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(V, bounds=(rs[max(0,idx-5)], rs[min(len(rs)-1, idx+5)]), method='bounded')
    return res.x

_RMIN = None
def get_rmin():
    global _RMIN
    if _RMIN is None:
        _RMIN = _find_rmin()
    return _RMIN

def _generate_icosahedron(n, rmin, seed=42):
    """Generate an icosahedral cluster of n atoms."""
    rng = np.random.default_rng(seed)
    
    # Icosahedron vertices (12 atoms)
    phi = (1 + np.sqrt(5)) / 2
    icosahedron = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    icosahedron /= np.linalg.norm(icosahedron[0])
    icosahedron *= rmin
    
    if n <= 12:
        return icosahedron[:n]
    
    # Second shell: 30 atoms (octahedral + other positions)
    # For simplicity, add atoms in a spherical shell
    positions = [icosahedron]
    
    # Add second shell (30 atoms at distance ~2*rmin)
    if n > 12:
        # Generate points on sphere of radius 2*rmin
        second_shell_radius = 2.0 * rmin
        # Use fibonacci sphere for uniform distribution
        def fibonacci_sphere(n_points, radius):
            points = []
            golden_angle = np.pi * (3 - np.sqrt(5))
            for i in range(n_points):
                y = 1 - (i / (n_points - 1)) * 2
                radius_at_y = np.sqrt(1 - y**2)
                theta = golden_angle * i
                x = np.cos(theta) * radius_at_y
                z = np.sin(theta) * radius_at_y
                points.append([x * radius, y * radius, z * radius])
            return np.array(points)
        
        n_second = min(30, n - 12)
        second_shell = fibonacci_sphere(n_second, second_shell_radius)
        positions.append(second_shell)
    
    # Add third shell if needed
    if n > 42:
        third_shell_radius = 3.0 * rmin
        n_third = n - 42
        if n_third > 0:
            third_shell = fibonacci_sphere(n_third, third_shell_radius)
            positions.append(third_shell)
    
    all_positions = np.vstack(positions)
    return all_positions[:n]

def _generate_mackay_icosahedron(n, rmin):
    """Generate Mackay icosahedral cluster."""
    if n == 13:
        # Central atom + 12 icosahedron vertices
        center = np.array([[0, 0, 0]])
        phi = (1 + np.sqrt(5)) / 2
        icosahedron = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        icosahedron /= np.linalg.norm(icosahedron[0])
        icosahedron *= rmin
        return np.vstack([center, icosahedron])
    elif n == 55:
        # Mackay 55: 1 + 12 + 30 + 12
        positions = []
        # Center
        positions.append(np.array([[0, 0, 0]]))
        
        phi = (1 + np.sqrt(5)) / 2
        icosahedron = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        icosahedron /= np.linalg.norm(icosahedron[0])
        
        # First shell: 12 atoms
        positions.append(icosahedron * rmin)
        
        # Second shell: 30 atoms
        # These are at the midpoints of edges and face centers
        # For simplicity, use fibonacci sphere
        def fibonacci_sphere(n_points, radius):
            points = []
            golden_angle = np.pi * (3 - np.sqrt(5))
            for i in range(n_points):
                y = 1 - (i / (n_points - 1)) * 2
                radius_at_y = np.sqrt(1 - y**2)
                theta = golden_angle * i
                x = np.cos(theta) * radius_at_y
                z = np.sin(theta) * radius_at_y
                points.append([x * radius, y * radius, z * radius])
            return np.array(points)
        
        positions.append(fibonacci_sphere(30, 2.0 * rmin))
        positions.append(fibonacci_sphere(12, 3.0 * rmin))
        
        all_pos = np.vstack(positions)
        return all_pos[:55]
    else:
        # Fallback to generic icosahedron
        return _generate_icosahedron(n, rmin)

def _build_connectivity_graph(positions, rmin, n_neighbors=12):
    """Build a k-nearest neighbor graph based on distance to rmin."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # For each atom, find n_neighbors closest neighbors
    adjacency = np.zeros((n, n), dtype=int)
    for i in range(n):
        neighbors = np.argsort(dist[i])[:n_neighbors]
        for j in neighbors:
            adjacency[i, j] = 1
            adjacency[j, i] = 1
    
    return adjacency

def _topology_objective(x, n, rmin, adjacency, params=None):
    """Objective: Dzugutov energy + barrier for bond length deviations."""
    if params is None:
        params = _dzugutov_params()
    
    positions = x.reshape(-1, 3)
    
    # Check minimum distance constraint
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    if np.min(dist) < 0.5:
        return 1e10
    
    # Compute Dzugutov energy
    energy = _dzugutov_energy(positions, params)
    
    # Barrier term: penalize deviations from rmin for bonded pairs
    barrier = 0.0
    barrier_weight = 50.0
    for i in range(n):
        for j in range(i+1, n):
            if adjacency[i, j] > 0:
                r = dist[i, j]
                deviation = abs(r - rmin) / rmin
                barrier += barrier_weight * deviation**2
    
    return energy + barrier

def _optimize_topology(positions, n, rmin, adjacency, params=None, max_iter=200):
    """Optimize positions with topology constraints using L-BFGS."""
    if params is None:
        params = _dzugutov_params()
    
    x0 = positions.flatten()
    
    def obj(x):
        return _topology_objective(x, n, rmin, adjacency, params)
    
    def grad(x):
        # Numerical gradient
        eps = 1e-6
        g = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            g[i] = (obj(x_plus) - obj(x_minus)) / (2 * eps)
        return g
    
    try:
        result = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                         options={'maxiter': max_iter, 'gtol': 1e-6})
        if result.success or result.fun < obj(x0):
            return result.x.reshape(-1, 3), result.fun
        else:
            return positions, obj(x0)
    except Exception:
        return positions, obj(x0)

def _local_relax(positions, potential='dzugutov', parameter=None):
    """Perform local relaxation using the provided relax function."""
    try:
        energy, relaxed_positions = relax(positions, potential, parameter)
        return energy, relaxed_positions
    except Exception:
        return _dzugutov_energy(positions), positions

def _basin_hopping(positions, n, rmin, potential='dzugutov', parameter=None, 
                   n_steps=50, step_size=0.1, rng=None):
    """Basin hopping with topology-aware perturbations."""
    if rng is None:
        rng = np.random.default_rng()
    
    current_energy, current_pos = _local_relax(positions, potential, parameter)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    adjacency = _build_connectivity_graph(current_pos, rmin)
    
    for step in range(n_steps):
        if _time_remaining() < 5.0:
            break
        
        # Perturb: random displacement with topology-aware bias
        delta = rng.normal(0, step_size, (n, 3))
        proposed_pos = current_pos + delta
        
        # Ensure minimum distance
        diff = proposed_pos[:, np.newaxis, :] - proposed_pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        if np.min(dist) < 0.5:
            continue
        
        proposed_energy, proposed_pos_relaxed = _local_relax(proposed_pos, potential, parameter)
        
        # Metropolis acceptance
        delta_e = proposed_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / 0.1):
            current_energy = proposed_energy
            current_pos = proposed_pos_relaxed
            adjacency = _build_connectivity_graph(current_pos, rmin)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
    
    return best_energy, best_pos

def _generate_initial_structures(n, rmin, n_structures=6, seed=42):
    """Generate diverse initial structures."""
    rng = np.random.default_rng(seed)
    structures = []
    
    # 1. Icosahedral
    structures.append(_generate_icosahedron(n, rmin, seed=seed))
    
    # 2. Mackay icosahedral
    structures.append(_generate_mackay_icosahedron(n, rmin))
    
    # 3. Random compact
    radius = rmin * n**(1/3) * 0.8
    for _ in range(n_structures - 2):
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.6 * rmin for q in points):
                points.append(p)
        structures.append(np.array(points))
    
    return structures

def construct(instance):
    """Construct the lowest-energy Dzugutov cluster."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = instance['n']
    potential = instance['potential']
    parameter = instance.get('rho', instance.get('kind'))
    
    rmin = get_rmin()
    
    # Generate initial structures
    structures = _generate_initial_structures(n, rmin, n_structures=8, seed=n)
    
    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Local relaxation of each structure
    for i, struct in enumerate(structures):
        if _time_remaining() < 10.0:
            break
        
        energy, relaxed = _local_relax(struct, potential, parameter)
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed.copy()
    
    if best_positions is None:
        # Fallback: return first structure
        return structures[0]
    
    # Phase 2: Basin hopping from best structure
    if _time_remaining() > 20.0:
        bh_energy, bh_positions = _basin_hopping(
            best_positions, n, rmin, potential, parameter,
            n_steps=min(100, int(_time_remaining() / 0.5)),
            step_size=0.15
        )
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_positions
    
    # Phase 3: Topology-constrained optimization
    if _time_remaining() > 15.0:
        adjacency = _build_connectivity_graph(best_positions, rmin)
        opt_positions, opt_energy = _optimize_topology(
            best_positions, n, rmin, adjacency,
            max_iter=100
        )
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_positions = opt_positions
        
        # Final local relaxation
        final_energy, final_positions = _local_relax(best_positions, potential, parameter)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_positions
    
    # Phase 4: Try a few more basin hopping runs from different starts
    if _time_remaining() > 10.0:
        for struct in structures[:3]:
            if _time_remaining() < 5.0:
                break
            energy, relaxed = _local_relax(struct, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed.copy()
            
            if _time_remaining() > 8.0:
                bh_e, bh_p = _basin_hopping(
                    relaxed, n, rmin, potential, parameter,
                    n_steps=30, step_size=0.12
                )
                if bh_e < best_energy:
                    best_energy = bh_e
                    best_positions = bh_p
    
    return best_positions
# EVOLVE-BLOCK-END
