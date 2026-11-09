"""
Morse Cluster Global Minima Finder
Strategy: Multi-start basin hopping with diverse structural seeds and adaptive step sizes
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Time management
TOTAL_TIME_BUDGET = 285.0  # Leave margin before 300s limit
_INSTANCE_DEADLINE = None
_instance_start = None


def _time_remaining():
    if _INSTANCE_DEADLINE is None:
        return float('inf')
    return _INSTANCE_DEADLINE - time.time()


def _morse_energy(positions, rho):
    """Compute Morse energy exactly."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Avoid division by zero / self-interaction
    np.fill_diagonal(dist, np.inf)
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    r = dist[iu]
    x = rho * (1.0 - r)
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    return float(np.sum(e))


def _local_minimize(positions, rho, max_iter=300, tol=1e-9):
    """Local minimization using L-BFGS-B."""
    from scipy.optimize import minimize
    
    def energy_func(pos):
        return _morse_energy(pos.reshape(-1, 3), rho)
    
    def gradient_func(pos):
        p = pos.reshape(-1, 3)
        # Use provided gradient function if available, else numerical
        try:
            _, grad = morse_energy_gradient(p, rho)
            return grad.ravel()
        except:
            # Numerical gradient fallback
            eps = 1e-7
            g = np.zeros_like(pos)
            for i in range(len(pos)):
                for j in range(3):
                    pos_plus = pos.copy()
                    pos_minus = pos.copy()
                    pos_plus[i, j] += eps
                    pos_minus[i, j] -= eps
                    g[i, j] = (energy_func(pos_plus) - energy_func(pos_minus)) / (2 * eps)
            return g
    
    x0 = positions.ravel()
    result = minimize(
        energy_func, x0,
        jac=gradient_func,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-7, 'maxfun': 10000}
    )
    return result.x.reshape(-1, 3), result.fun


def _generate_fcc(n, spacing=1.0):
    """Generate n atoms in FCC-like shell structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # FCC lattice points
    points = []
    max_coord = int(np.ceil(n ** (1/3.0)) * 2) + 3
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC basis
                for offset in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i + offset[0], j + offset[1], k + offset[2]]) * spacing
                    if len(points) < n * 2:  # Generate extra
                        points.append(p)
    
    points = np.array(points)
    # Sort by distance from origin and take n closest
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _generate_hcp(n, spacing=1.0):
    """Generate n atoms in HCP-like structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = []
    a = spacing
    c = spacing * np.sqrt(8.0/3.0)
    
    # Hexagonal layers
    layer_spacing = c / 3.0
    max_layer = int(np.ceil(n ** (1/3.0)) * 2) + 3
    
    for layer in range(-max_layer, max_layer + 1):
        z = layer * layer_spacing
        # Hexagonal arrangement in xy plane
        for i in range(-max_layer, max_layer + 1):
            for j in range(-max_layer, max_layer + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0) / 2.0 * j
                # Alternate layers offset
                if layer % 2 != 0:
                    x += a * 0.5
                p = np.array([x, y, z])
                if len(points) < n * 2:
                    points.append(p)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _generate_icosahedral(n, spacing=1.0):
    """Generate n atoms in icosahedral shell structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Icosahedral vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True) * spacing
    
    # Add icosahedral shells
    shell_radii = [spacing, spacing * 1.5, spacing * 2.0, spacing * 2.5, spacing * 3.0]
    for r in shell_radii:
        if len(points) >= n:
            break
        for v in ico_verts:
            if len(points) >= n:
                break
            p = v * r
            # Check minimum distance
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
    
    # Fill remaining with random points near surface
    rng = np.random.default_rng(12345)
    while len(points) < n:
        theta = rng.uniform(0, 2 * np.pi)
        phi_angle = rng.uniform(0, np.pi)
        r = spacing * (1.0 + rng.uniform(0, 2.0))
        p = r * np.array([
            np.sin(phi_angle) * np.cos(theta),
            np.sin(phi_angle) * np.sin(theta),
            np.cos(phi_angle)
        ])
        if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
            points.append(p)
    
    return np.array(points[:n])


def _generate_random_cluster(n, spacing=1.0, rng=None):
    """Generate random compact cluster."""
    if rng is None:
        rng = np.random.default_rng()
    
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
            points.append(p)
        attempts += 1
    
    # If not enough points, relax constraints
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _generate_bcc(n, spacing=1.0):
    """Generate n atoms in BCC-like structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = []
    max_coord = int(np.ceil(n ** (1/3.0)) * 2) + 3
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # BCC basis
                for offset in [(0,0,0), (0.5,0.5,0.5)]:
                    p = np.array([i + offset[0], j + offset[1], k + offset[2]]) * spacing
                    if len(points) < n * 2:
                        points.append(p)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _basin_hopping(positions, rho, n_steps=50, step_size=0.3, rng=None):
    """Adaptive basin hopping algorithm."""
    if rng is None:
        rng = np.random.default_rng()
    
    current_pos = positions.copy()
    current_energy = _morse_energy(current_pos, rho)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    # Adaptive step size
    base_step = step_size
    success_count = 0
    fail_count = 0
    
    for step in range(n_steps):
        if _time_remaining() < 3.0:
            break
        
        # Adaptive step size
        if success_count > 5 and fail_count < 2:
            step_size = base_step * 0.7  # Shrink
        elif fail_count > 5 and success_count < 2:
            step_size = base_step * 1.3  # Expand
        step_size = np.clip(step_size, 0.05, 1.0)
        
        # Perturbation
        noise = rng.normal(0, step_size, current_pos.shape)
        perturbed = current_pos + noise
        
        # Local minimize
        minimized, min_energy = _local_minimize(perturbed, rho, max_iter=150, tol=1e-8)
        
        # Metropolis acceptance with adaptive temperature
        delta_e = min_energy - current_energy
        temp = 0.05 + 0.05 * (step / max(1, n_steps))
        if delta_e < 0 or rng.uniform() < np.exp(-delta_e / temp):
            current_pos = minimized
            current_energy = min_energy
            success_count += 1
            fail_count = max(0, fail_count - 1)
        else:
            fail_count += 1
            success_count = max(0, success_count - 1)
        
        # Track best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
    
    return best_pos, best_energy


def construct(instance):
    """Main entry point: construct cluster for given instance."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))
    potential = instance["potential"]
    
    # Set up time budget for this instance
    global _INSTANCE_DEADLINE, _instance_start
    _instance_start = time.time()
    _INSTANCE_DEADLINE = _instance_start + (TOTAL_TIME_BUDGET / 6.0 - 1.0)
    
    # Generate diverse initial structures
    initial_structures = []
    
    # FCC
    fcc = _generate_fcc(n, spacing=1.0)
    initial_structures.append(fcc)
    
    # HCP
    hcp = _generate_hcp(n, spacing=1.0)
    initial_structures.append(hcp)
    
    # BCC
    bcc = _generate_bcc(n, spacing=1.0)
    initial_structures.append(bcc)
    
    # Icosahedral
    ico = _generate_icosahedral(n, spacing=1.0)
    initial_structures.append(ico)
    
    # Random clusters with different seeds
    for i in range(8):
        rng = np.random.default_rng(n * 100 + i * 7)
        random_cluster = _generate_random_cluster(n, spacing=1.0, rng=rng)
        initial_structures.append(random_cluster)
    
    # Run basin hopping in parallel with time budget
    best_pos = None
    best_energy = float('inf')
    
    # Use ThreadPoolExecutor for parallelism
    n_workers = min(4, len(initial_structures))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i, struct in enumerate(initial_structures):
            if _time_remaining() < 5.0:
                break
            rng = np.random.default_rng(n * 1000 + i * 17)
            # Use more steps for structured seeds, fewer for random
            n_steps = 40 if i < 4 else 25
            step_size = 0.25 if i < 4 else 0.35
            future = executor.submit(_basin_hopping, struct, rho, n_steps=n_steps, step_size=step_size, rng=rng)
            futures.append((future, i))
        
        for future, idx in futures:
            if _time_remaining() < 3.0:
                break
            try:
                pos, energy = future.result(timeout=15)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos
            except Exception as e:
                pass
    
    # Final polish with more iterations on best
    if best_pos is not None and _time_remaining() > 8.0:
        try:
            polished, polished_energy = _local_minimize(best_pos, rho, max_iter=800, tol=1e-10)
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_pos = polished
        except:
            pass
    
    # Additional basin hopping from the best found
    if best_pos is not None and _time_remaining() > 15.0:
        try:
            rng = np.random.default_rng(n * 9999)
            pos2, energy2 = _basin_hopping(best_pos, rho, n_steps=30, step_size=0.2, rng=rng)
            if energy2 < best_energy:
                best_energy = energy2
                best_pos = pos2
                # Polish again
                if _time_remaining() > 5.0:
                    polished, polished_energy = _local_minimize(best_pos, rho, max_iter=500, tol=1e-10)
                    if polished_energy < best_energy:
                        best_energy = polished_energy
                        best_pos = polished
        except:
            pass
    
    # Ensure minimum distance constraint (0.5 apart)
    if best_pos is not None:
        for _ in range(300):
            violated = False
            for i in range(len(best_pos)):
                for j in range(i+1, len(best_pos)):
                    d = np.linalg.norm(best_pos[i] - best_pos[j])
                    if d < 0.5:
                        violated = True
                        direction = best_pos[i] - best_pos[j]
                        if np.linalg.norm(direction) < 1e-10:
                            direction = np.array([1.0, 0.0, 0.0])
                        direction = direction / np.linalg.norm(direction)
                        shift = direction * (0.5 - d) / 2 * 1.2
                        best_pos[i] += shift
                        best_pos[j] -= shift
            if not violated:
                break
    
    if best_pos is None:
        # Fallback: simple random
        rng = np.random.default_rng(n)
        best_pos = rng.uniform(-1, 1, (n, 3))
    
    return best_pos
# EVOLVE-BLOCK-END
