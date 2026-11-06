"""Rewritten program using symmetry-constrained variational optimization for Dzugutov clusters."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from scipy.optimize import minimize, differential_evolution
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 290.0  # Safety margin before 300s limit


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _dzugutov_energy(pos):
    """Compute Dzugutov potential energy."""
    n = len(pos)
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            if r < 1e-10:
                continue
            if r < a:
                total += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                total += B * np.exp(d / (r - b))
    return total


def _dzugutov_gradient(pos):
    """Compute gradient of Dzugutov potential energy."""
    n = len(pos)
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                continue
            if r < a:
                exp_term = np.exp(c / (r - a))
                dV_dr = A * (-m * r ** (-m - 1)) * exp_term + A * (r ** (-m) - B) * exp_term * (-c / (r - a) ** 2)
            elif r < b:
                exp_term = np.exp(d / (r - b))
                dV_dr = B * exp_term * (-d / (r - b) ** 2)
            else:
                continue
            factor = dV_dr / r
            grad[i] += factor * rij
            grad[j] -= factor * rij
    return grad


def _get_icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron on the unit sphere."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts


def _icosahedral_shells(n, r1=1.0, r2=1.7, r3=2.3, twist1=0.0, twist2=0.0, twist3=0.0):
    """Generate icosahedral cluster with parameterized shell radii and twists.
    
    Parameters:
    - r1, r2, r3: radii of shells 1, 2, 3
    - twist1, twist2, twist3: twist angles for each shell
    """
    verts = _get_icosahedron_vertices()
    points = [np.array([0.0, 0.0, 0.0])]  # center
    
    # Shell 1: 12 icosahedron vertices
    if n > 1:
        shell1 = verts * r1
        # Apply twist1
        if abs(twist1) > 1e-10:
            axis = np.array([0.0, 0.0, 1.0])
            c, s = np.cos(twist1), np.sin(twist1)
            R = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
            shell1 = shell1 @ R.T
        points.extend(shell1)
    
    if len(points) >= n:
        return np.array(points[:n])
    
    # Shell 2: 42 atoms (30 edge midpoints + 12 face centers)
    if n > 13:
        # Find edges
        edge_mids = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if 0.95 < d < 1.15:
                    mid = (verts[i] + verts[j]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    edge_mids.append(mid)
        
        # Find faces
        face_centers = []
        for i in range(12):
            for j in range(i + 1, 12):
                for k in range(j + 1, 12):
                    d1 = np.linalg.norm(verts[i] - verts[j])
                    d2 = np.linalg.norm(verts[j] - verts[k])
                    d3 = np.linalg.norm(verts[k] - verts[i])
                    if 0.95 < d1 < 1.15 and 0.95 < d2 < 1.15 and 0.95 < d3 < 1.15:
                        center = (verts[i] + verts[j] + verts[k]) / 3.0
                        center = center / np.linalg.norm(center)
                        face_centers.append(center)
        
        shell2_dirs = np.array(edge_mids + face_centers)
        shell2 = shell2_dirs * r2
        # Apply twist2
        if abs(twist2) > 1e-10:
            axis = np.array([0.0, 0.0, 1.0])
            c, s = np.cos(twist2), np.sin(twist2)
            R = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
            shell2 = shell2 @ R.T
        points.extend(shell2)
    
    if len(points) >= n:
        return np.array(points[:n])
    
    # Shell 3: 120 atoms - use Fibonacci sphere distribution
    if n > 55:
        shell3 = []
        for i in range(120):
            y = 1 - 2 * (i + 0.5) / 120
            r = np.sqrt(max(0, 1 - y ** 2))
            theta = 2 * np.pi * i * 0.381966  # golden angle
            p = np.array([r * np.cos(theta), y, r * np.sin(theta)])
            shell3.append(p)
        shell3 = np.array(shell3) * r3
        # Apply twist3
        if abs(twist3) > 1e-10:
            axis = np.array([0.0, 0.0, 1.0])
            c, s = np.cos(twist3), np.sin(twist3)
            R = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
            shell3 = shell3 @ R.T
        points.extend(shell3)
    
    if len(points) >= n:
        return np.array(points[:n])
    
    # Fill remaining with perturbed points
    rng = np.random.default_rng(42)
    while len(points) < n:
        idx = rng.integers(0, len(points))
        p = points[idx] + rng.normal(0, 0.05, 3)
        points.append(p)
    
    return np.array(points[:n])


def _variational_objective(params, n, potential, parameter):
    """Objective function for variational optimization over symmetry parameters.
    
    params: [r1, r2, r3, twist1, twist2, twist3]
    """
    if _time_remaining() < 15:
        return 1e10
    
    r1, r2, r3, twist1, twist2, twist3 = params
    pos = _icosahedral_shells(n, r1, r2, r3, twist1, twist2, twist3)
    
    if pos.shape[0] != n:
        return 1e10
    
    # Check minimum distance constraint
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < 0.5:
                return 1e10
    
    try:
        energy, relaxed_pos = relax(pos, potential, parameter)
        return energy
    except Exception:
        return _dzugutov_energy(pos)


def _optimize_symmetry_params(n, potential, parameter, initial_params=None, max_iter=50):
    """Optimize symmetry parameters using gradient-based method."""
    if _time_remaining() < 20:
        return None, None
    
    # Define bounds for parameters
    base_scale = n ** (1.0 / 3.0) * 0.38
    bounds = [
        (base_scale * 0.7, base_scale * 1.3),  # r1
        (base_scale * 1.2, base_scale * 2.2),  # r2
        (base_scale * 1.8, base_scale * 2.8),  # r3
        (-0.3, 0.3),  # twist1
        (-0.3, 0.3),  # twist2
        (-0.3, 0.3),  # twist3
    ]
    
    if initial_params is None:
        initial_params = np.array([
            base_scale,
            base_scale * 1.7,
            base_scale * 2.3,
            0.0,
            0.0,
            0.0
        ])
    
    def objective(params):
        return _variational_objective(params, n, potential, parameter)
    
    try:
        result = minimize(
            objective, initial_params,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4}
        )
        
        if result.fun < 1e9:
            pos = _icosahedral_shells(n, *result.x)
            if pos.shape[0] == n:
                # Verify minimum distance
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        if np.linalg.norm(pos[i] - pos[j]) < 0.5:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return result.fun, pos
    except Exception:
        pass
    
    return None, None


def _basin_hopping(pos, n, potential, parameter, n_steps=10, step_size=0.15):
    """Perform basin hopping starting from given positions."""
    if _time_remaining() < 20:
        return pos, _dzugutov_energy(pos)
    
    current_energy = _dzugutov_energy(pos)
    current_pos = pos.copy()
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    rng = np.random.default_rng(int(time.time() * 1000) % 10000)
    
    for step in range(n_steps):
        if _time_remaining() < 10:
            break
        
        # Perturb
        perturbed = current_pos + rng.normal(0, step_size, current_pos.shape)
        
        # Relax
        try:
            energy, relaxed_pos = relax(perturbed, potential, parameter)
        except Exception:
            energy = _dzugutov_energy(perturbed)
            relaxed_pos = perturbed
        
        # Check minimum distance
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(relaxed_pos[i] - relaxed_pos[j]) < 0.5:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Metropolis acceptance
        if energy < current_energy:
            current_energy = energy
            current_pos = relaxed_pos
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos.copy()
        else:
            delta = energy - current_energy
            if delta < 0.5:  # Accept uphill moves with some probability
                accept_prob = np.exp(-delta / 0.1)
                if rng.random() < accept_prob:
                    current_energy = energy
                    current_pos = relaxed_pos
    
    return best_pos, best_energy


def _random_compact(n, scale=1.0, seed=42):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = scale * 1.2
    points = []
    min_dist = 0.6
    attempts = 0
    max_attempts = n * 500
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    if len(points) < n:
        min_dist = 0.5
        attempts = 0
        while len(points) < n and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= min_dist for q in points):
                    points.append(p)
            attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius * 0.5, radius * 0.5, 3)
        points.append(p)
    
    return np.array(points[:n])


def _generate_seeds(n, base_scale):
    """Generate diverse initial seeds for optimization."""
    seeds = []
    
    # Seed 1: Ideal Mackay icosahedron
    seeds.append(_icosahedral_shells(n, r1=base_scale, r2=base_scale*1.7, r3=base_scale*2.3))
    
    # Seed 2: Compressed
    seeds.append(_icosahedral_shells(n, r1=base_scale*0.9, r2=base_scale*1.6, r3=base_scale*2.2))
    
    # Seed 3: Expanded
    seeds.append(_icosahedral_shells(n, r1=base_scale*1.1, r2=base_scale*1.8, r3=base_scale*2.4))
    
    # Seed 4: Twisted
    seeds.append(_icosahedral_shells(n, r1=base_scale, r2=base_scale*1.7, r3=base_scale*2.3,
                                      twist1=0.1, twist2=0.05, twist3=0.03))
    
    # Seed 5: Random compact
    seeds.append(_random_compact(n, scale=base_scale, seed=n))
    
    # Seed 6: Another random
    seeds.append(_random_compact(n, scale=base_scale*1.05, seed=n+100))
    
    return seeds


def construct(instance):
    """Construct low-energy Dzugutov cluster using symmetry-constrained variational optimization."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    def _update_best(energy, positions):
        nonlocal best_energy, best_positions
        if positions is None or len(positions) != size:
            return
        # Check minimum distance
        for i in range(size):
            for j in range(i + 1, size):
                if np.linalg.norm(positions[i] - positions[j]) < 0.5:
                    return
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
    
    base_scale = size ** (1.0 / 3.0) * 0.38
    
    # Strategy 1: Variational optimization over symmetry parameters
    if _time_remaining() > 60:
        energy, pos = _optimize_symmetry_params(size, potential, parameter, max_iter=40)
        if pos is not None:
            _update_best(energy, pos)
    
    # Strategy 2: Basin hopping from multiple seeds
    seeds = _generate_seeds(size, base_scale)
    
    for seed_idx, seed in enumerate(seeds):
        if _time_remaining() < 30:
            break
        
        # Relax the seed
        try:
            energy, relaxed_pos = relax(seed, potential, parameter)
            _update_best(energy, relaxed_pos)
        except Exception:
            energy = _dzugutov_energy(seed)
            _update_best(energy, seed)
        
        # Basin hopping from this seed
        if _time_remaining() > 40:
            bh_pos, bh_energy = _basin_hopping(relaxed_pos if 'relaxed_pos' in locals() else seed,
                                                 size, potential, parameter,
                                                 n_steps=8, step_size=0.12)
            _update_best(bh_energy, bh_pos)
    
    # Strategy 3: If we have time, do more basin hopping from best found
    if best_positions is not None and _time_remaining() > 40:
        for _ in range(3):
            if _time_remaining() < 20:
                break
            bh_pos, bh_energy = _basin_hopping(best_positions, size, potential, parameter,
                                                 n_steps=10, step_size=0.10)
            _update_best(bh_energy, bh_pos)
    
    # Strategy 4: Final relaxation of best
    if best_positions is not None and _time_remaining() > 10:
        try:
            energy, relaxed_pos = relax(best_positions, potential, parameter)
            _update_best(energy, relaxed_pos)
        except Exception:
            pass
    
    if best_positions is None:
        # Fallback: return ideal icosahedron
        best_positions = _icosahedral_shells(size, r1=base_scale, r2=base_scale*1.7, r3=base_scale*2.3)
    
    return best_positions
# EVOLVE-BLOCK-END
