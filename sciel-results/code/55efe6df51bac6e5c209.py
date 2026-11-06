"""Morse cluster global minima search with multi-start basin hopping and gradient descent."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Deadline guard: total budget is 330s, use 325s for safety
_TOTAL_DEADLINE = 325.0


def _morse_energy(positions, rho):
    """Compute Morse potential energy: sum_{i<j} exp(rho*(1-r_ij))*(exp(rho*(1-r_ij))-2)"""
    n = len(positions)
    # Compute pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    # Only upper triangle
    iu = np.triu_indices(n, 1)
    d = dists[iu]
    x = rho * (1.0 - d)
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    return np.sum(e)


def _morse_energy_gradient(positions, rho):
    """Compute gradient of Morse potential energy."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-30)
    # Avoid division by zero
    np.fill_diagonal(dists, 1.0)
    
    # x_ij = rho*(1 - r_ij)
    x = rho * (1.0 - dists)
    ex = np.exp(x)
    # dE/dr_ij = 2*rho*exp(x)*(exp(x)-1)
    de_dr = 2.0 * rho * ex * (ex - 1.0)
    # grad_{r_ij} for atom i: -de_dr * (r_i - r_j)/r_ij
    # For each pair (i,j), contribution to grad_i is de_dr*(r_j - r_i)/r_ij
    # and to grad_j is de_dr*(r_i - r_j)/r_ij
    
    # Vectorized: grad[i] = sum_{j != i} de_dr[i,j] * (positions[j] - positions[i]) / dists[i,j]
    # diff[i,j] = positions[i] - positions[j]
    # So positions[j] - positions[i] = -diff[i,j]
    # grad[i] = -sum_j de_dr[i,j] * diff[i,j] / dists[i,j]
    
    # de_dr has shape (n,n), diff has shape (n,n,3), dists has shape (n,n)
    # factor[i,j] = de_dr[i,j] / dists[i,j]
    factor = de_dr / dists
    np.fill_diagonal(factor, 0.0)
    
    # grad[i] = -sum_j factor[i,j] * diff[i,j]
    # = -factor @ diff (where factor is (n,n) and diff is (n,n,3))
    # Using einsum: grad[i,k] = -sum_j factor[i,j] * diff[i,j,k]
    grad = -np.einsum('ij,ijk->ik', factor, diff)
    return grad


def _check_min_distance(positions, min_dist=0.5):
    """Check if all pairwise distances are at least min_dist."""
    n = len(positions)
    if n < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_dist


def _generate_random_packed(n, rng, min_spacing=0.8):
    """Generate n points in a sphere with minimum spacing."""
    radius = min_spacing * n ** (1.0 / 3.0) * 1.2
    points = []
    max_attempts = 100000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_spacing for q in points):
            points.append(p)
    if len(points) < n:
        # Fallback: place remaining points at larger radius
        radius2 = radius * 1.5
        while len(points) < n:
            p = rng.uniform(-radius2, radius2, 3)
            if np.linalg.norm(p) <= radius2:
                points.append(p)
    return np.array(points)


def _generate_lattice(n, rng):
    """Generate points on a lattice structure."""
    # Simple cubic lattice
    side = int(np.ceil(n ** (1.0 / 3.0)))
    points = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if len(points) < n:
                    points.append([i, j, k])
    points = np.array(points, dtype=float)
    # Center and add small random perturbation
    points -= np.mean(points, axis=0)
    points += rng.normal(0, 0.05, size=points.shape)
    return points[:n]


def _generate_fcc_like(n, rng):
    """Generate points in FCC-like arrangement."""
    side = int(np.ceil(n ** (1.0 / 3.0)))
    points = []
    a = 1.0  # lattice constant
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if len(points) < n:
                    points.append([i * a, j * a, k * a])
                    if len(points) < n and i + 0.5 < side:
                        points.append([(i + 0.5) * a, (j + 0.5) * a, k * a])
                        if len(points) < n and j + 0.5 < side:
                            points.append([(i + 0.5) * a, i * a, (k + 0.5) * a])
    points = np.array(points, dtype=float)
    points -= np.mean(points, axis=0)
    points += rng.normal(0, 0.05, size=points.shape)
    return points[:n]


def _gradient_descent(positions, rho, max_iter=500, tol=1e-10, time_budget=None, start_time=None):
    """Simple gradient descent with adaptive step size."""
    pos = positions.copy()
    n = len(pos)
    step = 0.01
    best_energy = _morse_energy(pos, rho)
    
    for it in range(max_iter):
        if time_budget is not None and start_time is not None:
            if time.time() - start_time > time_budget:
                break
        
        grad = _morse_energy_gradient(pos, rho)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Adaptive step: reduce if energy increases
        new_pos = pos - step * grad
        new_energy = _morse_energy(new_pos, rho)
        
        if new_energy < best_energy:
            # Accept and try to increase step
            pos = new_pos
            best_energy = new_energy
            step = min(step * 1.05, 0.1)
        else:
            # Reject and decrease step
            step *= 0.7
            if step < 1e-6:
                break
    
    return pos, best_energy


def _basin_hopping_step(pos, rho, rng, kick_size=0.1, time_budget=None, start_time=None):
    """Single basin hopping step: kick + local relax."""
    kicked = pos + rng.normal(0, kick_size, size=pos.shape)
    # Local relax using gradient descent
    relaxed, energy = _gradient_descent(kicked, rho, max_iter=200, time_budget=time_budget, start_time=start_time)
    return relaxed, energy


def construct(instance):
    """Find low-energy Morse cluster configurations using multi-start basin hopping."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 6.0))
    
    start_time = time.time()
    remaining = _TOTAL_DEADLINE - (time.time() - start_time)
    if remaining <= 5:
        # Emergency fallback
        rng = np.random.default_rng(size)
        pos = _generate_random_packed(size, rng)
        return pos
    
    rng = np.random.default_rng(size * 7 + int(rho))
    
    best_energy = np.inf
    best_pos = None
    
    # Phase 1: Generate diverse initial configurations and do local relax
    n_initials = min(12, max(4, int(remaining / 30)))
    
    init_generators = [
        lambda: _generate_random_packed(size, rng, 0.8),
        lambda: _generate_random_packed(size, rng, 1.0),
        lambda: _generate_lattice(size, rng),
        lambda: _generate_fcc_like(size, rng),
    ]
    
    for i in range(n_initials):
        if time.time() - start_time > _TOTAL_DEADLINE - 10:
            break
        gen = init_generators[i % len(init_generators)]
        try:
            pos = gen()
            if len(pos) != size:
                continue
            # Local relax
            relaxed, energy = _gradient_descent(pos, rho, max_iter=300, 
                                                time_budget=remaining - (time.time() - start_time) - 5,
                                                start_time=start_time)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
        except Exception:
            continue
    
    if best_pos is None:
        # Fallback
        pos = _generate_random_packed(size, rng)
        relaxed, energy = _gradient_descent(pos, rho, max_iter=200)
        return relaxed
    
    # Phase 2: Basin hopping from best configuration
    time_left = _TOTAL_DEADLINE - (time.time() - start_time)
    if time_left > 15:
        kick_sizes = [0.2, 0.1, 0.05, 0.02]
        for ks in kick_sizes:
            if time.time() - start_time > _TOTAL_DEADLINE - 10:
                break
            n_bh = int(min(20, time_left / 5))
            for _ in range(n_bh):
                if time.time() - start_time > _TOTAL_DEADLINE - 10:
                    break
                try:
                    new_pos, new_energy = _basin_hopping_step(
                        best_pos, rho, rng, kick_size=ks,
                        time_budget=_TOTAL_DEADLINE - (time.time() - start_time) - 5,
                        start_time=start_time
                    )
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_pos = new_pos.copy()
                except Exception:
                    continue
    
    # Phase 3: Final polish with more gradient steps
    time_left = _TOTAL_DEADLINE - (time.time() - start_time)
    if time_left > 5:
        best_pos, best_energy = _gradient_descent(
            best_pos, rho, max_iter=1000,
            time_budget=time_left - 3, start_time=start_time
        )
    
    # Verify minimum distance constraint
    if not _check_min_distance(best_pos, 0.5):
        # Nudge atoms apart if needed
        n = len(best_pos)
        for _ in range(100):
            diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=-1))
            np.fill_diagonal(dists, 1.0)
            # Find pairs too close
            close = dists < 0.5
            if not np.any(close):
                break
            # Push apart
            iu = np.triu_indices(n, 1)
            for idx in range(len(iu[0])):
                i, j = iu[0][idx], iu[1][idx]
                if dists[i, j] < 0.5:
                    d = best_pos[i] - best_pos[j]
                    norm = np.linalg.norm(d)
                    if norm < 1e-10:
                        d = rng.normal(0, 0.1, 3)
                        norm = np.linalg.norm(d)
                    d = d / norm * 0.01
                    best_pos[i] += d
                    best_pos[j] -= d
    
    return best_pos
# EVOLVE-BLOCK-END
