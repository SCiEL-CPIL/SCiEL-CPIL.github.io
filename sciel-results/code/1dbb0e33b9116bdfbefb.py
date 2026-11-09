"""Morse cluster global minima search: multi-start parallel with diverse prototypes and basin hopping."""
import numpy as np
import time
import concurrent.futures
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline management
_START_TIME = time.time()
_TOTAL_BUDGET = 300.0
_DEADLINE = _START_TIME + 285.0  # 15s safety margin


def _time_remaining():
    return _DEADLINE - time.time()


def _morse_energy(rho, coords):
    """Compute Morse energy. Returns huge value if any pair is closer than 0.5."""
    n = len(coords)
    if n < 2:
        return 0.0
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    min_d = d.min()
    if min_d < 0.5:
        return 1e10
    x = rho * (1.0 - d)
    # Clip x to avoid overflow: x_max = rho * (1 - 0.5) = 0.5*rho, x_min = rho*(1-max_dist)
    # For reasonable clusters, distances < 20, so x_min > rho*(1-20) = -19*rho
    # exp(-19*10) = exp(-190) ~ 0, fine. exp(5) ~ 148, fine.
    ex = np.exp(np.clip(x, -50, 50))
    e_ij = ex * (ex - 2.0)
    return 0.5 * e_ij.sum()


def _morse_energy_grad(rho, coords):
    """Compute Morse energy and gradient."""
    n = len(coords)
    if n < 2:
        return 0.0, np.zeros_like(coords)
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    min_d = d.min()
    if min_d < 0.5:
        return 1e10, np.zeros_like(coords)
    x = rho * (1.0 - d)
    ex = np.exp(np.clip(x, -50, 50))
    e_ij = ex * (ex - 2.0)
    energy = 0.5 * e_ij.sum()

    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    d_safe = np.where(d > 1e-8, d, 1.0)
    de_dr = -2.0 * rho * ex * (ex - 1.0)
    coeff = de_dr / d_safe
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijc->ic', coeff, diff)
    return energy, grad


def _local_minimize(coords, rho, maxiter=300):
    """Minimize energy from given starting point using L-BFGS-B."""
    n = len(coords)

    def obj_grad(x):
        c = x.reshape(-1, 3)
        e, g = _morse_energy_grad(rho, c)
        if e > 1e9:
            # If invalid, return a large gradient pointing outward
            return e, g * 0.0 + 1.0
        return e, g

    x0 = coords.flatten()
    try:
        res = minimize(obj_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxls': 40, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < 1e9:
            return res.fun, res.x.reshape(-1, 3)
        else:
            e0, _ = _morse_energy_grad(rho, coords)
            return e0, coords
    except Exception:
        e0, _ = _morse_energy_grad(rho, coords)
        return e0, coords


def _fibonacci_sphere(n_points, radius, rng):
    """Generate points on a sphere using Fibonacci spiral."""
    if n_points <= 0:
        return np.empty((0, 3))
    if n_points == 1:
        return np.array([[0.0, 0.0, radius]])
    phi_angle = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n_points) + 0.5
    cos_theta = 1.0 - (i / n_points)
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
    theta = i * phi_angle
    x = sin_theta * np.cos(theta)
    y = sin_theta * np.sin(theta)
    z = cos_theta
    pts = np.column_stack((x, y, z)) * radius
    # Add small random perturbation to break symmetry
    pts += rng.normal(0, 0.02, pts.shape)
    return pts


def _generate_fcc_cluster(n, rng):
    """Generate FCC-like cluster."""
    a = 1.0
    points = []
    basis = np.array([[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]])
    extent = int(np.ceil(n ** (1/3))) + 4
    for i in range(-extent, extent+1):
        for j in range(-extent, extent+1):
            for k in range(-extent, extent+1):
                origin = np.array([i*a, j*a, k*a])
                for b in basis:
                    p = origin + b * a
                    points.append(p)
    points = np.array(points)
    if len(points) <= n:
        remaining = n - len(points)
        rand_pts = rng.normal(0, 2, (remaining, 3))
        points = np.vstack([points, rand_pts])
    center = points.mean(axis=0)
    points -= center
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx] + rng.normal(0, 0.05, (n, 3))


def _generate_hcp_cluster(n, rng):
    """Generate HCP-like cluster."""
    a = 1.0
    c = a * np.sqrt(8.0/3.0)
    points = []
    extent = int(np.ceil(n ** (1/3))) + 4
    for layer in range(-extent, extent+1):
        z = layer * c
        offset = (0.5, 0.0, 0.0) if layer % 2 == 0 else (0.0, 0.0, 0.0)
        for i in range(-extent, extent+1):
            for j in range(-extent, extent+1):
                x = i * a + offset[0] * a
                y = j * a * np.sqrt(3.0)/2.0 + offset[1] * a
                points.append([x, y, z])
    points = np.array(points)
    if len(points) <= n:
        remaining = n - len(points)
        rand_pts = rng.normal(0, 2, (remaining, 3))
        points = np.vstack([points, rand_pts])
    center = points.mean(axis=0)
    points -= center
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx] + rng.normal(0, 0.05, (n, 3))


def _generate_icosahedral_cluster(n, rng):
    """Generate icosahedral shell cluster."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    points = [np.array([0.0, 0.0, 0.0])]
    for v in verts:
        points.append(v * 1.0)
    shells_radii = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    for r in shells_radii:
        if len(points) >= n:
            break
        m = min(n - len(points), max(12, int(4 * np.pi * r**2)))
        if m <= 0:
            continue
        shell_pts = _fibonacci_sphere(m, r, rng)
        points.extend(shell_pts)
    points = np.array(points)
    if len(points) < n:
        remaining = n - len(points)
        rand_pts = rng.normal(0, 2, (remaining, 3))
        points = np.vstack([points, rand_pts])
    center = points.mean(axis=0)
    points -= center
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx] + rng.normal(0, 0.05, (n, 3))


def _generate_shell_cluster(n, rng):
    """Generate multi-shell cluster."""
    points = [np.array([0.0, 0.0, 0.0])]
    shells = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    for r in shells:
        if len(points) >= n:
            break
        m = min(n - len(points), max(12, int(4 * np.pi * r**2)))
        if m <= 0:
            continue
        shell_pts = _fibonacci_sphere(m, r, rng)
        points.extend(shell_pts)
    points = np.array(points)
    if len(points) < n:
        remaining = n - len(points)
        rand_pts = rng.normal(0, 2, (remaining, 3))
        points = np.vstack([points, rand_pts])
    center = points.mean(axis=0)
    points -= center
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx] + rng.normal(0, 0.05, (n, 3))


def _generate_random_sphere(n, rng, radius=None):
    """Generate random points in a sphere."""
    if radius is None:
        radius = (n / 8.0) ** (1.0 / 3.0) * 1.8
    coords = rng.normal(0, radius, (n, 3))
    return coords


def _make_initial_config(n, rng, rho, method):
    """Create a reasonable initial configuration."""
    if method == 'fcc':
        coords = _generate_fcc_cluster(n, rng)
    elif method == 'hcp':
        coords = _generate_hcp_cluster(n, rng)
    elif method == 'ico':
        coords = _generate_icosahedral_cluster(n, rng)
    elif method == 'shell':
        coords = _generate_shell_cluster(n, rng)
    else:
        coords = _generate_random_sphere(n, rng)
    return coords


def _basin_hopping_step(coords, rho, displacement, rng):
    """One basin hopping step: displace + minimize."""
    displaced = coords + rng.normal(0, displacement, coords.shape)
    e, new_coords = _local_minimize(displaced, rho)
    return e, new_coords


def _worker(args):
    """Worker function for parallel search."""
    n, rho, method, seed, time_budget = args
    rng = np.random.default_rng(seed)
    start_time = time.time()
    deadline = start_time + time_budget

    best_energy = float('inf')
    best_coords = None

    # Generate initial config
    coords = _make_initial_config(n, rng, rho, method)
    e, coords = _local_minimize(coords, rho)
    if e < best_energy:
        best_energy = e
        best_coords = coords.copy()

    if best_coords is None:
        coords = _make_initial_config(n, rng, rho, 'random')
        best_energy, best_coords = _local_minimize(coords, rho)

    # Basin hopping
    current_energy = best_energy
    current_coords = best_coords.copy()
    base_disp = 0.4

    while time.time() < deadline - 2.0:
        disp = base_disp * (1.0 + 0.3 * rng.random())
        e_new, c_new = _basin_hopping_step(current_coords, rho, disp, rng)

        if e_new < current_energy:
            current_energy = e_new
            current_coords = c_new
            if e_new < best_energy:
                best_energy = e_new
                best_coords = c_new.copy()
        else:
            delta = e_new - current_energy
            if delta < 3.0:
                temp = max(0.05, abs(current_energy) * 0.03)
                if rng.random() < np.exp(-delta / temp):
                    current_coords = c_new

        # Occasionally try a fresh start
        if rng.random() < 0.05:
            fresh = _make_initial_config(n, rng, rho, method)
            e_fresh, c_fresh = _local_minimize(fresh, rho)
            if e_fresh < best_energy:
                best_energy = e_fresh
                best_coords = c_fresh.copy()
                current_coords = c_fresh.copy()
                current_energy = e_fresh

    return best_energy, best_coords


def _solve_one_instance(instance, time_budget):
    """Solve one instance within time budget using parallel workers."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))

    start_time = time.time()
    deadline = start_time + time_budget

    methods = ['fcc', 'hcp', 'ico', 'shell', 'random', 'fcc', 'hcp', 'ico']
    # Use up to 4 threads
    num_workers = min(4, len(methods))
    
    # Allocate time per worker
    worker_time = max(2.0, (time_budget - 5.0) / num_workers)

    tasks = []
    for i in range(num_workers):
        method = methods[i]
        seed = hash((n, rho, method, i)) % (2**32)
        tasks.append((n, rho, method, seed, worker_time))

    best_energy = float('inf')
    best_coords = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_worker, task): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            if time.time() > deadline - 3.0:
                break
            try:
                e, coords = future.result(timeout=5.0)
                if e < best_energy:
                    best_energy = e
                    best_coords = coords.copy()
            except Exception:
                pass

    # If no result from workers, do a quick local minimization
    if best_coords is None:
        rng = np.random.default_rng(n)
        coords = _make_initial_config(n, rng, rho, 'fcc')
        best_energy, best_coords = _local_minimize(coords, rho)

    # Final refinement: do a few more basin hopping steps from the best
    if time.time() < deadline - 3.0 and best_coords is not None:
        rng = np.random.default_rng(n + rho)
        current_energy = best_energy
        current_coords = best_coords.copy()
        for _ in range(20):
            if time.time() > deadline - 3.0:
                break
            disp = 0.3 * (1.0 + 0.3 * rng.random())
            e_new, c_new = _basin_hopping_step(current_coords, rho, disp, rng)
            if e_new < current_energy:
                current_energy = e_new
                current_coords = c_new
                if e_new < best_energy:
                    best_energy = e_new
                    best_coords = c_new.copy()
            else:
                delta = e_new - current_energy
                if delta < 2.0:
                    temp = max(0.05, abs(current_energy) * 0.02)
                    if rng.random() < np.exp(-delta / temp):
                        current_coords = c_new

    return best_coords


def construct(instance):
    """Find the lowest-energy arrangement of n atoms bound by Morse potentials."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))

    time_budget = min(45.0, _time_remaining() - 5.0)
    if time_budget < 3.0:
        # Not enough time, just do a quick local minimization
        rng = np.random.default_rng(n)
        coords = _make_initial_config(n, rng, rho, 'fcc')
        e, coords = _local_minimize(coords, rho)
        return coords

    result = _solve_one_instance(instance, time_budget)

    # Final validation
    if result is not None and len(result) >= 2:
        d = cdist(result, result)
        np.fill_diagonal(d, np.inf)
        if d.min() < 0.5:
            center = result.mean(axis=0)
            result = (result - center) * 1.3 + center
            d = cdist(result, result)
            np.fill_diagonal(d, np.inf)
            if d.min() < 0.5:
                rng = np.random.default_rng(n + 1)
                coords = _make_initial_config(n, rng, rho, 'random')
                e, coords = _local_minimize(coords, rho)
                result = coords

    return result
