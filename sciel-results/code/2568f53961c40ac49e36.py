"""Morse cluster global minima search using vectorized energy/grad, BFGS, and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
from scipy.optimize import minimize as scipy_minimize

_DEADLINE_MARGIN = 10.0  # seconds to spare before 330s kill


def _morse_energy_and_grad(positions, rho):
    """Vectorized Morse energy and gradient."""
    n = positions.shape[0]
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-15)
    iu = np.triu_indices(n, k=1)
    r = dists[iu]
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    energy = np.sum(v)
    # dV/dr = -rho * ex * (ex - 1)
    dv_dr = -rho * ex * (ex - 1.0)
    # gradient: for each pair (i,j), grad_i += (dv_dr/r) * (p_j - p_i), grad_j += (dv_dr/r) * (p_i - p_j)
    # factor = dv_dr / r
    factor = dv_dr / r
    grad = np.zeros_like(positions)
    i_idx, j_idx = iu
    diff_ij = positions[i_idx] - positions[j_idx]  # p_i - p_j
    # dE/dp_i = -factor * diff_ij / r ... let me be careful
    # V depends on r = |p_i - p_j|
    # dV/dp_i = dv_dr * d r / d p_i = dv_dr * (p_i - p_j)/r
    # dV/dp_j = dv_dr * d r / d p_j = dv_dr * (p_j - p_i)/r = -dv_dr * (p_i - p_j)/r
    # So grad[i] += dv_dr/r * (p_i - p_j), grad[j] += -dv_dr/r * (p_i - p_j)
    # factor = dv_dr / r
    # grad[i] += factor * diff_ij, grad[j] -= factor * diff_ij
    np.add.at(grad, i_idx, factor[:, np.newaxis] * diff_ij)
    np.add.at(grad, j_idx, -factor[:, np.newaxis] * diff_ij)
    return energy, grad


def _icosahedron_shell(n_atoms, rho):
    """Generate an icosahedral-like cluster with shells."""
    phi = (1 + np.sqrt(5)) / 2
    base = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    base = base / 2.0
    points = [np.zeros(3)]
    points.extend(base.tolist())
    if n_atoms > 13:
        remaining = n_atoms - len(points)
        if remaining > 0:
            n_shell = min(remaining, 30)
            shell_radius = np.sqrt(2.0)
            for k in range(n_shell):
                t = (k + 0.5) / n_shell
                z = 1 - 2 * t
                r = np.sqrt(max(0.0, 1 - z * z))
                theta = 2 * np.pi * k / n_shell
                p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * shell_radius
                points.append(p)
            remaining -= n_shell
            if remaining > 0:
                shell_radius2 = np.sqrt(3.0)
                for k in range(remaining):
                    t = (k + 0.5) / remaining
                    z = 1 - 2 * t
                    r = np.sqrt(max(0.0, 1 - z * z))
                    theta = 2 * np.pi * k / remaining
                    p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * shell_radius2
                    points.append(p)
    return np.array(points[:n_atoms])


def _fcc_cluster(n_atoms, rho):
    """Generate an FCC-like cluster."""
    points = []
    a = 1.0
    max_coord = int(np.ceil(n_atoms ** (1/3)) + 3)
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for offset in [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]:
                    p = np.array([i, j, k], dtype=float) + np.array(offset, dtype=float)
                    if np.linalg.norm(p) < 1.5 * n_atoms ** (1/3):
                        points.append(p)
    if len(points) > 0:
        points = np.array(points)
        norms = np.linalg.norm(points, axis=1)
        idx = np.argsort(norms)
        points = points[idx[:n_atoms]]
    else:
        points = np.zeros((n_atoms, 3))
    return points


def _decahedral_cluster(n_atoms, rho):
    """Generate a decahedral-like cluster (5-fold symmetric)."""
    points = [np.zeros(3)]
    for layer in range(1, 8):
        r = layer * 0.8
        n_in_ring = 10
        for k in range(n_in_ring):
            theta = 2 * np.pi * k / n_in_ring + (np.pi / 10 if layer % 2 else 0)
            p = np.array([r * np.cos(theta), r * np.sin(theta), 0])
            points.append(p)
            if len(points) >= n_atoms:
                break
        if len(points) >= n_atoms:
            break
    return np.array(points[:n_atoms])


def _random_compact(n_atoms, rng, spacing=1.0):
    """Generate a random compact cluster."""
    radius = spacing * n_atoms ** (1/3) * 0.8
    points = []
    attempts = 0
    while len(points) < n_atoms and attempts < n_atoms * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0 or np.min(np.linalg.norm(p - np.array(points), axis=1)) >= 0.55 * spacing:
                points.append(p)
        attempts += 1
    while len(points) < n_atoms:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _bfgs_local(pos_flat, rho, n):
    """L-BFGS-B local minimization using vectorized gradient."""
    def fun_and_grad(x):
        pos = x.reshape(n, 3)
        e, g = _morse_energy_and_grad(pos, rho)
        return e, g.reshape(-1)
    res = scipy_minimize(fun_and_grad, pos_flat, method='L-BFGS-B', jac=True,
                         options={'maxiter': 200, 'gtol': 1e-6, 'maxls': 20})
    return res.x.reshape(n, 3), res.fun


def _basin_hop_step(positions, rho, rng, step_size, n):
    """Perform one basin hopping step: perturb, then BFGS relax."""
    perturbed = positions + rng.normal(0, step_size, (n, 3))
    try:
        relaxed, energy = _bfgs_local(perturbed.ravel(), rho, n)
        # Validate minimum separation
        if np.min(np.linalg.norm(relaxed[:, np.newaxis, :] - relaxed[np.newaxis, :, :], axis=2)[np.triu_indices(n, 1)]) < 0.49:
            return np.inf, positions
        return energy, relaxed
    except:
        return np.inf, positions


def construct(instance):
    """Find global minimum of Morse cluster using vectorized BFGS + basin hopping."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))
    start_time = time.time()
    deadline = 330.0 - _DEADLINE_MARGIN

    rng = np.random.default_rng(hash((n, rho)) % (2**32))

    # Generate diverse initial configurations
    seeds = []
    for gen_func in [_icosahedron_shell, _fcc_cluster, _decahedral_cluster]:
        try:
            seeds.append(gen_func(n, rho))
        except:
            pass
    for _ in range(6):
        seeds.append(_random_compact(n, rng))

    # Relax all seeds with BFGS
    best_energy = np.inf
    best_positions = None

    for seed in seeds:
        if time.time() > deadline:
            break
        try:
            relaxed, energy = _bfgs_local(seed.ravel(), rho, n)
            # Check minimum separation
            dmin = np.min(np.linalg.norm(relaxed[:, np.newaxis, :] - relaxed[np.newaxis, :, :], axis=2)[np.triu_indices(n, 1)])
            if dmin < 0.49:
                continue
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed.copy()
        except:
            pass

    if best_positions is None:
        best_positions = _random_compact(n, rng)
        try:
            best_positions, best_energy = _bfgs_local(best_positions.ravel(), rho, n)
        except:
            pass

    # Basin hopping with Metropolis acceptance
    step_size = 0.3
    T = 0.5  # temperature for Metropolis
    hop_count = 0
    max_hops = 100
    stagnant = 0

    while hop_count < max_hops:
        if time.time() > deadline:
            break
        try:
            energy, new_pos = _basin_hop_step(best_positions, rho, rng, step_size, n)
            if energy < best_energy:
                best_energy = energy
                best_positions = new_pos.copy()
                stagnant = 0
                step_size = max(0.1, step_size * 0.95)
            else:
                # Metropolis acceptance
                delta_e = energy - best_energy
                if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                    best_positions = new_pos.copy()
                stagnant += 1
                if stagnant > 10:
                    step_size = min(0.6, step_size * 1.15)
                    stagnant = 0
        except:
            pass
        hop_count += 1

    # Final polish with tight BFGS
    if best_positions is not None and time.time() < deadline:
        try:
            polished, final_e = _bfgs_local(best_positions.ravel(), rho, n)
            dmin = np.min(np.linalg.norm(polished[:, np.newaxis, :] - polished[np.newaxis, :, :], axis=2)[np.triu_indices(n, 1)])
            if dmin >= 0.49 and final_e < best_energy:
                best_positions = polished
                best_energy = final_e
        except:
            pass

    # Ensure minimum separation
    if best_positions is not None:
        dmin = np.min(np.linalg.norm(best_positions[:, np.newaxis, :] - best_positions[np.newaxis, :, :], axis=2)[np.triu_indices(n, 1)])
        if dmin < 0.5:
            for _ in range(5):
                if time.time() > deadline:
                    break
                for i in range(n):
                    for j in range(i + 1, n):
                        d = np.linalg.norm(best_positions[i] - best_positions[j])
                        if d < 0.5 and d > 1e-10:
                            direction = (best_positions[i] - best_positions[j]) / d
                            shift = (0.5 - d) / 2 + 0.01
                            best_positions[i] += direction * shift
                            best_positions[j] -= direction * shift
                try:
                    relaxed, e = _bfgs_local(best_positions.ravel(), rho, n)
                    dmin2 = np.min(np.linalg.norm(relaxed[:, np.newaxis, :] - relaxed[np.newaxis, :, :], axis=2)[np.triu_indices(n, 1)])
                    if dmin2 >= 0.49:
                        best_positions = relaxed
                        best_energy = e
                        break
                except:
                    pass

    return best_positions
# EVOLVE-BLOCK-END
