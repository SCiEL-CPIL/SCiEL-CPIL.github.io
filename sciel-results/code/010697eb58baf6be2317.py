"""Optimized basin hopping with parallel multi-start for SC cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _time_left(start_time, deadline=310.0):
    return deadline - (time.time() - start_time)


def _fcc_shells(n, spacing=1.0):
    """Generate n points from FCC lattice shells, centered."""
    if n == 0:
        return np.zeros((0, 3))
    
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 1.5)) + 3
    
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                for bx, by, bz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    x = (ix + bx) * spacing
                    y = (iy + by) * spacing
                    z = (iz + bz) * spacing
                    if x*x + y*y + z*z <= (n ** (1/3) * spacing * 1.1) ** 2:
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
    max_coord = int(np.ceil(n ** (1/3) * 1.5)) + 3
    
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                if iz % 2 == 0:
                    x = ix * a + 0.25 * iy * a
                    y = iy * a * np.sqrt(3)/2
                else:
                    x = ix * a + 0.5 * a + 0.25 * iy * a
                    y = iy * a * np.sqrt(3)/2
                z = iz * c / 2
                if x*x + y*y + z*z <= (n ** (1/3) * spacing * 1.1) ** 2:
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
        # Second shell: 30 points (edges of icosahedron)
        phi_inv = 1/phi
        second_shell = []
        for i in range(12):
            for j in range(i+1, 12):
                v1, v2 = vertices[i], vertices[j]
                dist = np.linalg.norm(v1 - v2)
                if dist < 2.2 * spacing:  # nearest neighbors
                    mid = (v1 + v2) / 2
                    second_shell.append(mid.tolist())
        
        for v in second_shell:
            if len(points) >= n:
                break
            points.append(v)
    
    if n > len(points):
        # Third shell: more points
        rng = np.random.default_rng(42)
        for _ in range(n - len(points)):
            base = points[rng.integers(0, len(points))]
            p = np.array(base) + rng.normal(0, 0.3 * spacing, 3)
            points.append(p.tolist())
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points


def _random_compact(n, spacing=1.0, seed=None):
    """Generate a random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    min_dist = 0.55 * spacing
    
    points = []
    attempts = 0
    max_attempts = n * 150
    
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
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(positions[i] - positions[j]) < min_dist:
                return False
    return True


def _refine(positions, potential, kind, start_time, deadline=310.0):
    """Refine positions using L-BFGS-B then BFGS."""
    if _time_left(start_time, deadline) < 5.0:
        return positions, sutton_chen_energy_gradient(positions, kind)[0]
    
    # First: L-BFGS-B via relax
    try:
        energy, positions = relax(positions, potential, kind)
    except Exception:
        energy, positions = sutton_chen_energy_gradient(positions, kind)
    
    if not np.isfinite(energy):
        return positions, energy
    
    # Second: BFGS refinement
    if _time_left(start_time, deadline) > 10.0:
        try:
            def objective_and_grad(x):
                pos = x.reshape(-1, 3)
                e, g = sutton_chen_energy_gradient(pos, kind)
                return e, g.flatten()
            
            res = minimize(objective_and_grad, positions.flatten(), jac=True,
                          method='BFGS', options={'maxiter': 300, 'gtol': 1e-9})
            if res.success or res.fun < energy:
                return res.x.reshape(-1, 3), res.fun
        except Exception:
            pass
    
    return positions, energy


def _basin_hop(positions, potential, kind, start_time, deadline=310.0,
               n_iter=40, step_size=0.3, seed=42):
    """
    Basin-hopping with adaptive step size and Metropolis acceptance.
    """
    current_energy, current_pos = sutton_chen_energy_gradient(positions, kind)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    rng = np.random.default_rng(seed)
    
    for i in range(n_iter):
        if _time_left(start_time, deadline) < 8.0:
            break
        
        # Adaptive step size: larger early, smaller late
        progress = i / max(n_iter - 1, 1)
        sigma = step_size * (1.0 - 0.5 * progress)
        
        # Perturb
        perturbed = current_pos + rng.normal(0, sigma, current_pos.shape)
        
        # Relax
        try:
            energy, relaxed_pos = _refine(perturbed, potential, kind, start_time, deadline)
        except Exception:
            continue
        
        if not np.isfinite(energy):
            continue
        
        # Metropolis acceptance
        accept = False
        if energy < current_energy:
            accept = True
        else:
            T = max(0.05, 1.5 * (1 - progress))
            if rng.random() < np.exp(-(energy - current_energy) / T):
                accept = True
        
        if accept:
            current_energy = energy
            current_pos = relaxed_pos
            
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos.copy()
    
    return best_energy, best_pos


def _parallel_refine(starters, potential, kind, start_time, deadline=310.0, max_workers=3):
    """Refine multiple starting structures in parallel."""
    results = []
    
    def refine_one(starter):
        try:
            energy, positions = _refine(starter, potential, kind, start_time, deadline)
            return (energy, positions)
        except Exception:
            return (np.inf, starter)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(refine_one, s): i for i, s in enumerate(starters)}
        for future in as_completed(futures):
            if _time_left(start_time, deadline) < 10.0:
                break
            try:
                result = future.result(timeout=5.0)
                if np.isfinite(result[0]):
                    results.append(result)
            except Exception:
                continue
    
    return results


def construct(instance):
    """Multi-start basin hopping with parallel refinement for SC clusters."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    start_time = time.time()
    deadline = 310.0
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting structures
    starters = [
        _fcc_shells(size, spacing=1.0),
        _hcp_shells(size, spacing=1.0),
        _icosahedron_based(size, spacing=1.0),
        _random_compact(size, spacing=1.0, seed=size),
        _random_compact(size, spacing=1.0, seed=size + 1),
        _random_compact(size, spacing=1.0, seed=size + 2),
    ]
    
    # Parallel refinement of all starters
    refined_results = _parallel_refine(starters, potential, kind, start_time, deadline)
    
    # Sort by energy
    refined_results.sort(key=lambda x: x[0])
    
    for energy, positions in refined_results:
        if np.isfinite(energy) and energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
    
    # Basin-hopping from the best solution
    if best_positions is not None:
        # Primary basin hopping
        if _time_left(start_time, deadline) > 25.0:
            try:
                bh_energy, bh_positions = _basin_hop(
                    best_positions, potential, kind, start_time, deadline,
                    n_iter=35, step_size=0.28, seed=size
                )
                if np.isfinite(bh_energy) and bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_positions.copy()
            except Exception:
                pass
        
        # Basin hopping from second-best if time remains
        if _time_left(start_time, deadline) > 35.0 and len(refined_results) > 1:
            try:
                second_best = refined_results[1][1]
                bh_energy, bh_positions = _basin_hop(
                    second_best, potential, kind, start_time, deadline,
                    n_iter=25, step_size=0.32, seed=size + 100
                )
                if np.isfinite(bh_energy) and bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_positions.copy()
            except Exception:
                pass
        
        # Small perturbations of best solution
        for attempt in range(3):
            if _time_left(start_time, deadline) < 15.0:
                break
            
            perturbed = best_positions + np.random.default_rng(size + 200 + attempt).normal(0, 0.04, best_positions.shape)
            try:
                energy, positions = _refine(perturbed, potential, kind, start_time, deadline)
                if np.isfinite(energy) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except Exception:
                continue
        
        # Additional random starts if time permits
        for attempt in range(4):
            if _time_left(start_time, deadline) < 12.0:
                break
            
            starter = _random_compact(size, spacing=1.0, seed=size + 300 + attempt)
            try:
                energy, positions = _refine(starter, potential, kind, start_time, deadline)
                if np.isfinite(energy) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except Exception:
                continue
    
    # Fallback: return a simple valid configuration if nothing worked
    if best_positions is None:
        best_positions = _random_compact(size, spacing=1.0, seed=size)
    
    # Final validity check
    if not _check_validity(best_positions):
        rng = np.random.default_rng(size + 500)
        for _ in range(15):
            fixed = best_positions + rng.normal(0, 0.08, best_positions.shape)
            if _check_validity(fixed):
                best_positions = fixed
                break
    
    return best_positions
# EVOLVE-BLOCK-END
