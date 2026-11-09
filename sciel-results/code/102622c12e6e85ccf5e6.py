"""Morse cluster global minima finder with basin hopping and structured initializers."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time management
TOTAL_TIME_BUDGET = 290.0  # Leave 10s margin for overhead
start_time_global = time.time()


def _morse_energy(positions, rho):
    """Compute Morse potential energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    d = pdist(positions)
    x = rho * (1.0 - d)
    e = np.exp(x)
    return float(np.sum(e * (e - 2.0)))


def _morse_gradient(positions, rho):
    """Compute Morse potential gradient."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Avoid division by zero
    mask = dist > 1e-10
    dist_safe = np.where(mask, dist, 1.0)
    
    x = rho * (1.0 - dist_safe)
    e = np.exp(x)
    # dE/dr = 2 * e * (e - 1) * rho
    dE_dr = 2.0 * e * (e - 1.0) * rho
    dE_dr = np.where(mask, dE_dr, 0.0)
    
    # Gradient: sum over j of dE/dr_ij * (r_i - r_j)/r_ij
    # dE/dr_ij * (r_i - r_j)/r_ij = dE_dr * diff / dist
    grad = np.sum(dE_dr[:, :, np.newaxis] * diff / dist_safe[:, :, np.newaxis], axis=1)
    return grad


def _local_minimize(positions, rho, maxiter=1000):
    """Local minimization using L-BFGS-B."""
    def objective(x):
        return _morse_energy(x.reshape(-1, 3), rho)
    
    def gradient(x):
        return _morse_gradient(x.reshape(-1, 3), rho).flatten()
    
    result = minimize(
        objective,
        positions.flatten(),
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    return result.fun, result.x.reshape(-1, 3)


def _basin_hopping(positions, rho, time_budget, n_iter_max=50):
    """Basin hopping with time budget control."""
    best_energy, best_positions = _local_minimize(positions, rho)
    current_energy, current_positions = best_energy, best_positions
    temp = 0.5  # Initial temperature
    cooling = 0.95
    
    n = len(positions)
    rng = np.random.default_rng(hash((n, int(rho))) % (2**32))
    
    for i in range(n_iter_max):
        if time.time() - start_time_global > TOTAL_TIME_BUDGET - 5:
            break
        if time.time() > time_budget:
            break
        
        # Perturb
        step_size = 0.3 * (1.0 + 0.5 * np.cos(i / max(1, n_iter_max / 3)))
        perturbed = current_positions + step_size * rng.normal(size=current_positions.shape)
        
        # Local minimize
        try:
            new_energy, new_positions = _local_minimize(perturbed, rho, maxiter=500)
        except Exception:
            continue
        
        # Metropolis acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / temp):
            current_energy, current_positions = new_energy, new_positions
        
        # Update best
        if current_energy < best_energy:
            best_energy, best_positions = current_energy, current_positions
        
        # Cool
        temp *= cooling
    
    return best_energy, best_positions


def _generate_fcc_cluster(n, spacing=1.0):
    """Generate an FCC-like cluster."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # FCC lattice points
    points = []
    max_coord = int(np.ceil(n ** (1/3.0))) + 2
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * spacing
                    points.append(p)
    
    points = np.array(points)
    
    # Select n closest to center
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)[:n]
    selected = points[idx]
    
    # Center
    selected = selected - np.mean(selected, axis=0)
    
    return selected


def _generate_icosahedral(n, spacing=1.0):
    """Generate an icosahedral cluster (approximate)."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Start with icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True) * spacing
    
    points = [ico]
    current_n = 12
    
    # Add shells
    shell_radius = spacing * 1.5
    while current_n < n:
        # Add points on a sphere
        n_add = min(n - current_n, 20)
        rng = np.random.default_rng(current_n)
        for _ in range(n_add):
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            p = shell_radius * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
            # Check minimum distance
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points.flat):
                points.append(p)
                current_n += 1
        shell_radius += spacing * 0.8
    
    all_points = np.vstack(points)
    dists = np.sum(all_points**2, axis=1)
    idx = np.argsort(dists)[:n]
    selected = all_points[idx]
    selected = selected - np.mean(selected, axis=0)
    
    return selected


def _generate_random_compact(n, spacing=1.0, seed=None):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 50:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
            points.append(p)
        attempts += 1
    
    if len(points) < n:
        # Fill remaining randomly
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    
    points = np.array(points)
    points = points - np.mean(points, axis=0)
    return points


def construct(instance):
    """Find low-energy Morse cluster configurations."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    # Per-instance time budget
    instance_budget = min(
        (TOTAL_TIME_BUDGET - (time.time() - start_time_global)) / max(1, 6),
        45.0
    )
    instance_deadline = time.time() + instance_budget
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # FCC-like
    try:
        initial_configs.append(_generate_fcc_cluster(size))
    except Exception:
        pass
    
    # Icosahedral
    try:
        initial_configs.append(_generate_icosahedral(size))
    except Exception:
        pass
    
    # Random compact with different seeds
    for seed in [size, size * 7 + 1, size * 13 + 2, size * 17 + 3]:
        try:
            initial_configs.append(_generate_random_compact(size, seed=seed))
        except Exception:
            pass
    
    # Remove duplicates and keep best candidates
    best_energy = np.inf
    best_positions = None
    
    # First, do quick local minimization on all to rank them
    ranked_configs = []
    for config in initial_configs:
        if time.time() > instance_deadline - 10:
            break
        try:
            e, pos = _local_minimize(config, rho, maxiter=200)
            ranked_configs.append((e, pos))
        except Exception:
            pass
    
    # Sort by energy
    ranked_configs.sort(key=lambda x: x[0])
    
    # Basin hopping on top candidates
    n_bh = min(4, len(ranked_configs))
    bh_time_per = max(2.0, (instance_deadline - time.time() - 5) / max(1, n_bh))
    
    for i in range(n_bh):
        if time.time() > instance_deadline - 3:
            break
        e, pos = ranked_configs[i]
        try:
            bh_e, bh_pos = _basin_hopping(pos, rho, time_budget=instance_deadline - 3, n_iter_max=30)
            if bh_e < best_energy:
                best_energy = bh_e
                best_positions = bh_pos
        except Exception:
            continue
    
    # If basin hopping didn't improve, use best local min
    if best_positions is None or best_energy > ranked_configs[0][0]:
        best_energy = ranked_configs[0][0]
        best_positions = ranked_configs[0][1]
    
    # Final safety: ensure minimum distance constraint
    # If any pair is too close, do a quick fix
    d = pdist(best_positions)
    if np.min(d) < 0.5:
        # Nudge overlapping atoms apart
        for _ in range(50):
            d = pdist(best_positions)
            if np.min(d) >= 0.5:
                break
            # Find closest pair
            dist_matrix = squareform(d)
            np.fill_diagonal(dist_matrix, np.inf)
            i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            direction = best_positions[i] - best_positions[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.random.randn(3)
                norm = np.linalg.norm(direction)
            direction = direction / norm
            shift = (0.5 - dist_matrix[i, j]) / 2 + 0.01
            best_positions[i] += direction * shift
            best_positions[j] -= direction * shift
    
    return best_positions
# EVOLVE-BLOCK-END
