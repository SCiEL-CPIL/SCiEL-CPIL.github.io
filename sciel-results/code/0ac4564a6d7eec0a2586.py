"""Basin-hopping + parallel L-BFGS-B with structural bias for SC cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _time_left(start_time, deadline=290.0):
    return deadline - (time.time() - start_time)


def _fcc_shells(n, spacing=1.0):
    """Generate n points from FCC lattice shells, centered."""
    if n == 0:
        return np.zeros((0, 3))
    
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 2.5)) + 2
    radius_sq = (n ** (1/3) * spacing * 1.3) ** 2
    
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                for bx, by, bz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    x = (ix + bx) * spacing
                    y = (iy + by) * spacing
                    z = (iz + bz) * spacing
                    if x*x + y*y + z*z <= radius_sq:
                        points.append([x, y, z])
    
    points = np.array(points)
    if len(points) > n:
        dists = np.sum(points**2, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.uniform(-n**(1/3)*spacing, n**(1/3)*spacing, 3)
            points = np.vstack([points, [p]])
    
    points -= np.mean(points, axis=0)
    return points


def _hcp_shells(n, spacing=1.0):
    """Generate n points from HCP lattice shells."""
    if n == 0:
        return np.zeros((0, 3))
    
    points = []
    a = spacing
    c = np.sqrt(8/3) * a
    max_coord = int(np.ceil(n ** (1/3) * 2.5)) + 2
    radius_sq = (n ** (1/3) * spacing * 1.3) ** 2
    
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                if iz % 2 == 0:
                    x = ix * a + 0.25 * iy * a
                    y = iy * a * np.sqrt(3)/2
                else:
                    x = ix * a + 0.5*a + 0.25 * iy * a
                    y = iy * a * np.sqrt(3)/2
                z = iz * c / 2
                if x*x + y*y + z*z <= radius_sq:
                    points.append([x, y, z])
    
    points = np.array(points)
    if len(points) > n:
        dists = np.sum(points**2, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.uniform(-n**(1/3)*spacing, n**(1/3)*spacing, 3)
            points = np.vstack([points, [p]])
    
    points -= np.mean(points, axis=0)
    return points


def _icosahedron_based(n, spacing=1.0):
    """Generate points based on icosahedral symmetry with nested shells."""
    phi = (1 + 5 ** 0.5) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    vertices = vertices / np.linalg.norm(vertices[0]) * spacing
    
    points = [[0.0, 0.0, 0.0]]
    points.extend([v.tolist() for v in vertices])
    
    if n > 13:
        dodec = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
            [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
            [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
            [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0]
        ])
        dodec = dodec / np.linalg.norm(dodec[0]) * spacing * 1.8
        for v in dodec:
            if len(points) >= n:
                break
            points.append(v.tolist())
    
    rng = np.random.default_rng(42)
    while len(points) < n:
        base = points[rng.integers(0, len(points))]
        p = np.array(base) + rng.normal(0, 0.3 * spacing, 3)
        points.append(p.tolist())
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points


def _random_compact(n, spacing=1.0, seed=None):
    """Generate a random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.95
    min_dist = 0.55 * spacing
    
    points = []
    attempts = 0
    max_attempts = n * 300
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - np.array(q)) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    return points


def _check_validity(positions, min_dist=0.35):
    """Check if positions are valid (finite and no atoms too close)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    mask = ~np.eye(n, dtype=bool)
    if np.any(dists[mask] < min_dist):
        return False
    return True


def _sc_energy_and_grad(x_flat, kind):
    """Energy and gradient for scipy minimize from flat array."""
    pos = x_flat.reshape(-1, 3)
    e, g = sutton_chen_energy_gradient(pos, kind)
    return e, g.flatten()


def _refine(positions, kind, start_time, deadline=290.0, maxiter=200):
    """Refine a structure using L-BFGS-B with the SC gradient."""
    if _time_left(start_time, deadline) < 5.0:
        e, _ = sutton_chen_energy_gradient(positions, kind)
        return e, positions
    try:
        res = minimize(_sc_energy_and_grad, positions.flatten(), jac=True,
                       method='L-BFGS-B', options={'maxiter': maxiter, 'gtol': 1e-8,
                                                     'maxfun': maxiter * 15})
        if np.isfinite(res.fun):
            return res.fun, res.x.reshape(-1, 3)
        return sutton_chen_energy_gradient(positions, kind)
    except Exception:
        try:
            return relax(positions, 'sutton_chen', kind)
        except Exception:
            return sutton_chen_energy_gradient(positions, kind)


def _basin_hop(positions, kind, start_time, deadline, rng, max_steps=80, step_init=0.5, step_min=0.05):
    """Basin-hopping: perturb + local minimization, accept if lower energy."""
    current_energy, current_pos = sutton_chen_energy_gradient(positions, kind)
    best_energy, best_pos = current_energy, positions.copy()
    step_size = step_init
    
    for step in range(max_steps):
        if _time_left(start_time, deadline) < 10.0:
            break
        
        # Perturb
        perturbed = current_pos + rng.normal(0, step_size, current_pos.shape)
        
        # Local minimization
        try:
            new_energy, new_pos = _refine(perturbed, kind, start_time, deadline, maxiter=60)
        except Exception:
            continue
        
        if not np.isfinite(new_energy):
            continue
        
        # Accept if lower energy
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos.copy()
            step_size = max(step_min, step_size * 0.9)
        else:
            step_size = max(step_min, step_size * 0.95)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
    
    return best_energy, best_pos


def _structural_bias_penalty(positions, kind):
    """
    Compute a penalty that encourages FCC/HCP-like local environments.
    This guides the search toward known structural motifs.
    """
    n = len(positions)
    if n < 4:
        return 0.0
    
    # Compute nearest neighbor distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    
    # For each atom, get its 12 nearest neighbors
    penalty = 0.0
    for i in range(n):
        nn_dists = np.sort(dists[i, :])[:12]
        if len(nn_dists) < 12:
            continue
        
        # Ideal FCC: 12 neighbors at equal distance
        # Ideal HCP: similar
        # Penalize variance in neighbor distances
        mean_dist = np.mean(nn_dists)
        if mean_dist < 0.5:
            continue
        variance = np.var(nn_dists) / (mean_dist ** 2)
        penalty += variance
    
    return penalty / n


def _combined_objective(x_flat, kind, bias_weight=0.1):
    """Combined energy + structural bias for basin hopping guidance."""
    pos = x_flat.reshape(-1, 3)
    e, g = sutton_chen_energy_gradient(pos, kind)
    
    # Add structural bias (approximate gradient via finite difference)
    bias = _structural_bias_penalty(pos, kind)
    total_e = e + bias_weight * bias
    
    # Approximate gradient of bias term
    grad_bias = np.zeros_like(x_flat)
    eps = 1e-5
    for idx in range(min(len(x_flat), 30)):  # Only first 30 coords for speed
        x_plus = x_flat.copy()
        x_plus[idx] += eps
        pos_plus = x_plus.reshape(-1, 3)
        bias_plus = _structural_bias_penalty(pos_plus, kind)
        grad_bias[idx] = (bias_plus - bias) / eps
    
    total_g = g + bias_weight * grad_bias
    return total_e, total_g.flatten()


def _parallel_refine(starters, kind, start_time, deadline, max_workers=4):
    """Refine multiple starting structures in parallel."""
    results = []
    
    def _refine_one(args):
        idx, starter = args
        try:
            energy, positions = _refine(starter, kind, start_time, deadline, maxiter=150)
            return (energy, positions.copy())
        except Exception:
            return None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_refine_one, (i, s)): i for i, s in enumerate(starters)}
        for future in as_completed(futures):
            if _time_left(start_time, deadline) < 10.0:
                break
            result = future.result()
            if result is not None and np.isfinite(result[0]):
                results.append(result)
    
    return results


def construct(instance):
    """
    Basin-hopping + parallel L-BFGS-B with structural bias for SC cluster global minima.
    Uses diverse structural templates, parallel local minimization, and basin-hopping
    with FCC/HCP bias to escape local minima.
    """
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    start_time = time.time()
    deadline = 290.0
    
    # Generate diverse starting structures
    starters = [
        _fcc_shells(size, spacing=1.0),
        _hcp_shells(size, spacing=1.0),
        _icosahedron_based(size, spacing=1.0),
        _random_compact(size, spacing=1.0, seed=size),
        _random_compact(size, spacing=1.0, seed=size + 7),
        _random_compact(size, spacing=1.0, seed=size + 13),
    ]
    
    # Phase 1: Parallel relaxation of all starters
    population = _parallel_refine(starters, kind, start_time, deadline, max_workers=4)
    
    if not population:
        # Emergency fallback
        best_positions = _random_compact(size, spacing=1.0, seed=size)
        return best_positions
    
    # Sort by energy
    population.sort(key=lambda x: x[0])
    best_energy, best_positions = population[0]
    
    # Phase 2: Basin-hopping from best structures
    rng = np.random.default_rng(size + 100)
    max_bh_steps = 60
    bh_results = []
    
    # Run basin-hopping from top 3 structures
    for i, (energy, pos) in enumerate(population[:3]):
        if _time_left(start_time, deadline) < 30.0:
            break
        try:
            bh_energy, bh_pos = _basin_hop(pos, kind, start_time, deadline, rng, 
                                            max_steps=max_bh_steps, step_init=0.4, step_min=0.05)
            if np.isfinite(bh_energy):
                bh_results.append((bh_energy, bh_pos.copy()))
        except Exception:
            continue
    
    # Combine all results
    all_results = population + bh_results
    all_results.sort(key=lambda x: x[0])
    
    if all_results[0][0] < best_energy:
        best_energy = all_results[0][0]
        best_positions = all_results[0][1].copy()
    
    # Phase 3: Additional basin-hopping from best with larger steps
    if _time_left(start_time, deadline) > 40.0:
        rng2 = np.random.default_rng(size + 200)
        try:
            bh_energy2, bh_pos2 = _basin_hop(best_positions, kind, start_time, deadline, rng2,
                                              max_steps=40, step_init=0.6, step_min=0.1)
            if np.isfinite(bh_energy2) and bh_energy2 < best_energy:
                best_energy = bh_energy2
                best_positions = bh_pos2.copy()
        except Exception:
            pass
    
    # Phase 4: Final fine-tuning with more iterations
    if _time_left(start_time, deadline) > 15.0:
        try:
            final_energy, final_pos = _refine(best_positions, kind, start_time, deadline, maxiter=300)
            if np.isfinite(final_energy) and final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_pos.copy()
        except Exception:
            pass
    
    # Final validity check
    if not _check_validity(best_positions):
        rng_fix = np.random.default_rng(size + 500)
        for _ in range(30):
            fixed = best_positions + rng_fix.normal(0, 0.15, best_positions.shape)
            if _check_validity(fixed):
                best_positions = fixed
                break
    
    return best_positions
# EVOLVE-BLOCK-END
