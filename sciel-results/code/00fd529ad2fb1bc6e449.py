"""Global minima search for Sutton-Chen TM clusters via multi-start basin-hopping with diverse templates."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _time_left(start_time, deadline=285.0):
    return deadline - (time.time() - start_time)


def _fcc_template(n, spacing=1.0):
    """Generate n points from FCC lattice, centered."""
    if n == 0:
        return np.zeros((0, 3))
    
    # FCC lattice points
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 2.0)) + 2
    radius_sq = (n ** (1/3) * spacing * 1.2) ** 2
    
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


def _hcp_template(n, spacing=1.0):
    """Generate n points from HCP lattice, centered."""
    if n == 0:
        return np.zeros((0, 3))
    
    points = []
    a = spacing
    c = np.sqrt(8/3) * a
    max_coord = int(np.ceil(n ** (1/3) * 2.0)) + 2
    radius_sq = (n ** (1/3) * spacing * 1.2) ** 2
    
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


def _bcc_template(n, spacing=1.0):
    """Generate n points from BCC lattice, centered."""
    if n == 0:
        return np.zeros((0, 3))
    
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 2.0)) + 2
    radius_sq = (n ** (1/3) * spacing * 1.2) ** 2
    
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                for bx, by, bz in [(0,0,0), (0.5,0.5,0.5)]:
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


def _icosahedral_template(n, spacing=1.0):
    """Generate points with icosahedral symmetry."""
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
        # Add more shells
        rng = np.random.default_rng(42)
        while len(points) < n:
            # Add points near existing ones
            base = points[rng.integers(0, len(points))]
            p = np.array(base) + rng.normal(0, 0.3 * spacing, 3)
            points.append(p.tolist())
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points


def _random_packed(n, spacing=1.0, seed=None):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    min_dist = 0.5 * spacing
    
    points = []
    attempts = 0
    max_attempts = n * 200
    
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


def _enforce_min_dist(positions, min_dist=0.36):
    """Ensure no two atoms are closer than min_dist."""
    n = len(positions)
    for _ in range(50):
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        mask = ~np.eye(n, dtype=bool)
        too_close = np.any(dists[mask] < min_dist)
        if not too_close:
            break
        
        # Find closest pair and separate them
        dists_copy = dists.copy()
        np.fill_diagonal(dists_copy, np.inf)
        min_idx = np.unravel_index(np.argmin(dists_copy), dists_copy.shape)
        i, j = min_idx
        diff_vec = positions[i] - positions[j]
        current_dist = np.linalg.norm(diff_vec)
        if current_dist < 1e-10:
            diff_vec = np.array([1.0, 0, 0])
        diff_vec = diff_vec / np.linalg.norm(diff_vec)
        sep = (min_dist - current_dist) / 2 + 0.01
        positions[i] += diff_vec * sep
        positions[j] -= diff_vec * sep
    
    return positions


def _sc_energy_and_grad(x_flat, kind):
    """Energy and gradient from flat array."""
    pos = x_flat.reshape(-1, 3)
    e, g = sutton_chen_energy_gradient(pos, kind)
    return e, g.flatten()


def _polish(positions, kind, start_time, deadline=285.0, maxiter=100):
    """Local minimization with L-BFGS-B."""
    if _time_left(start_time, deadline) < 3.0:
        e, _ = sutton_chen_energy_gradient(positions, kind)
        return e, positions
    try:
        res = minimize(_sc_energy_and_grad, positions.flatten(), jac=True,
                       method='L-BFGS-B', options={'maxiter': maxiter, 'gtol': 1e-8,
                                                     'maxfun': maxiter * 10})
        if np.isfinite(res.fun):
            return res.fun, res.x.reshape(-1, 3)
        return sutton_chen_energy_gradient(positions, kind)
    except Exception:
        try:
            return relax(positions, 'sutton_chen', kind)
        except Exception:
            return sutton_chen_energy_gradient(positions, kind)


def _bh_loop(positions, kind, start_time, deadline, rng, max_steps=50, step_init=0.4, step_min=0.05):
    """Basin-hopping loop."""
    current_energy, current_pos = sutton_chen_energy_gradient(positions, kind)
    best_energy, best_pos = current_energy, positions.copy()
    step_size = step_init
    
    for step in range(max_steps):
        if _time_left(start_time, deadline) < 8.0:
            break
        
        perturbed = current_pos + rng.normal(0, step_size, current_pos.shape)
        perturbed = _enforce_min_dist(perturbed)
        
        try:
            new_energy, new_pos = _polish(perturbed, kind, start_time, deadline, maxiter=40)
        except Exception:
            continue
        
        if not np.isfinite(new_energy):
            continue
        
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos.copy()
            step_size = max(step_min, step_size * 0.92)
        else:
            step_size = max(step_min, step_size * 0.96)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
    
    return best_energy, best_pos


def _parallel_refine(starters, kind, start_time, deadline, max_workers=4):
    """Refine multiple starting structures in parallel."""
    results = []
    
    def _refine_one(args):
        idx, starter = args
        try:
            energy, positions = _polish(starter, kind, start_time, deadline, maxiter=80)
            if np.isfinite(energy):
                return (energy, positions.copy())
        except Exception:
            pass
        return None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_refine_one, (i, s)): i for i, s in enumerate(starters)}
        for future in as_completed(futures):
            if _time_left(start_time, deadline) < 10.0:
                break
            result = future.result()
            if result is not None:
                results.append(result)
    
    return results


def construct(instance):
    """
    Multi-start basin-hopping with diverse structural templates for SC cluster global minima.
    """
    size = instance["n"]
    kind = instance.get("kind", instance.get("rho"))
    
    start_time = time.time()
    deadline = 285.0
    
    # Generate diverse starting structures
    starters = [
        _fcc_template(size, spacing=1.0),
        _hcp_template(size, spacing=1.0),
        _bcc_template(size, spacing=1.0),
        _icosahedral_template(size, spacing=1.0),
        _random_packed(size, spacing=1.0, seed=size),
        _random_packed(size, spacing=1.0, seed=size + 7),
        _random_packed(size, spacing=1.0, seed=size + 13),
        _random_packed(size, spacing=1.0, seed=size + 23),
    ]
    
    # Phase 1: Parallel relaxation
    population = _parallel_refine(starters, kind, start_time, deadline, max_workers=4)
    
    if not population:
        best_positions = _random_packed(size, spacing=1.0, seed=size)
        best_positions = _enforce_min_dist(best_positions)
        return best_positions
    
    population.sort(key=lambda x: x[0])
    best_energy, best_positions = population[0]
    
    # Phase 2: Basin-hopping from best structures
    rng = np.random.default_rng(size + 100)
    bh_results = []
    
    # Run basin-hopping from top 4 structures
    for i, (energy, pos) in enumerate(population[:4]):
        if _time_left(start_time, deadline) < 25.0:
            break
        try:
            bh_energy, bh_pos = _bh_loop(pos, kind, start_time, deadline, rng,
                                          max_steps=40, step_init=0.35, step_min=0.04)
            if np.isfinite(bh_energy):
                bh_results.append((bh_energy, bh_pos.copy()))
        except Exception:
            continue
    
    all_results = population + bh_results
    all_results.sort(key=lambda x: x[0])
    
    if all_results[0][0] < best_energy:
        best_energy = all_results[0][0]
        best_positions = all_results[0][1].copy()
    
    # Phase 3: Additional basin-hopping with different step sizes
    if _time_left(start_time, deadline) > 35.0:
        rng2 = np.random.default_rng(size + 200)
        try:
            bh_energy2, bh_pos2 = _bh_loop(best_positions, kind, start_time, deadline, rng2,
                                            max_steps=30, step_init=0.5, step_min=0.08)
            if np.isfinite(bh_energy2) and bh_energy2 < best_energy:
                best_energy = bh_energy2
                best_positions = bh_pos2.copy()
        except Exception:
            pass
    
    # Phase 4: Final polish
    if _time_left(start_time, deadline) > 12.0:
        try:
            final_energy, final_pos = _polish(best_positions, kind, start_time, deadline, maxiter=200)
            if np.isfinite(final_energy) and final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_pos.copy()
        except Exception:
            pass
    
    # Ensure validity
    best_positions = _enforce_min_dist(best_positions, min_dist=0.36)
    
    return best_positions
# EVOLVE-BLOCK-END
