"""Lennard-Jones 55-atom global minimum: Radial manifold optimization with Cartesian polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE_SECONDS = 140.0

# Precompute icosahedral directions
_PHI = (1.0 + np.sqrt(5.0)) / 2.0

def _get_icosahedral_directions():
    """Return 54 unit direction vectors: 12 vertex dirs, 30 edge-mid dirs, 12 vertex dirs again."""
    verts = np.array([
        [0, 1, _PHI], [0, -1, _PHI], [0, 1, -_PHI], [0, -1, -_PHI],
        [1, _PHI, 0], [-1, _PHI, 0], [1, -_PHI, 0], [-1, -_PHI, 0],
        [_PHI, 0, 1], [-_PHI, 0, 1], [_PHI, 0, -1], [-_PHI, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms

    # Find edges
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            dists[i, j] = d
            dists[j, i] = d
    min_dist = np.min(dists[dists > 0])
    edge_mask = dists < min_dist * 1.001

    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            if edge_mask[i, j]:
                mid = (verts_unit[i] + verts_unit[j]) / 2.0
                mid_unit = mid / np.linalg.norm(mid)
                edge_mids.append(mid_unit)
    edge_mids = np.array(edge_mids)  # (30, 3)

    # Order: 12 inner-vertex dirs, 30 edge-mid dirs, 12 outer-vertex dirs
    dirs = np.vstack([verts_unit, edge_mids, verts_unit])
    return dirs  # shape (54, 3)

_DIRS = _get_icosahedral_directions()  # precomputed at module load


def _lj_energy(pos):
    """Compute total LJ energy for a set of positions (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    np.fill_diagonal(e, 0.0)
    return np.sum(e) / 2.0


def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized, numerically stable)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)

    r6 = r2 * r2 * r2
    r12 = r6 * r6

    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0

    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(factor, 0.0)

    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos

    return energy, grad


def _radial_to_cartesian(radii):
    """Convert 54 radial distances to 54 Cartesian positions (excluding center at origin)."""
    return radii[:, np.newaxis] * _DIRS  # shape (54, 3)


def _cartesian_to_full(pos54):
    """Add center atom at origin to make full 55-atom array."""
    return np.vstack([np.zeros(3), pos54])


def _radial_energy(radii):
    """Energy as function of 54 radial distances (center fixed at origin)."""
    pos54 = _radial_to_cartesian(radii)
    pos_full = _cartesian_to_full(pos54)
    return _lj_energy(pos_full)


def _radial_grad(radii):
    """Gradient of energy w.r.t. radial distances via chain rule."""
    pos54 = _radial_to_cartesian(radii)
    pos_full = _cartesian_to_full(pos54)
    _, cart_grad = _lj_energy_grad(pos_full)  # shape (55, 3)
    # grad w.r.t. radii[i] = dot(cart_grad[i+1], dir[i])
    grad = np.sum(cart_grad[1:] * _DIRS, axis=1)
    return grad


def _radial_optimize(r0, deadline=None, maxiter=3000):
    """Optimize all 54 radial distances on the icosahedral direction manifold."""
    lb = np.concatenate([np.full(12, 0.5), np.full(30, 0.5), np.full(12, 0.5)])
    ub = np.concatenate([np.full(12, 3.0), np.full(30, 3.5), np.full(12, 3.5)])

    stages = [
        {'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-10, 'maxls': 50},
        {'maxiter': maxiter * 2, 'gtol': 1e-10, 'ftol': 1e-12, 'maxls': 100},
        {'maxiter': maxiter * 3, 'gtol': 1e-12, 'ftol': 1e-14, 'maxls': 200},
    ]

    best_energy = _radial_energy(r0)
    best_params = r0.copy()
    x0 = r0.copy()

    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 10.0:
            break
        try:
            result = minimize(
                _radial_energy, x0,
                jac=_radial_grad,
                bounds=list(zip(lb, ub)),
                method='L-BFGS-B',
                options=stage_opts
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x.copy()
                x0 = result.x
        except Exception:
            pass

    return best_params, best_energy


def _cartesian_polish(pos0, deadline=None, maxiter=5000):
    """High-precision L-BFGS polish in full 165D Cartesian space."""
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()

    best_energy = _lj_energy(pos0)
    best_pos = pos0.copy()
    x0 = pos0.flatten()

    stages = [
        {'maxiter': maxiter, 'gtol': 1e-10, 'maxls': 100, 'ftol': 1e-12},
        {'maxiter': maxiter * 2, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-14},
        {'maxiter': maxiter * 3, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': maxiter * 5, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-18},
    ]

    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 8.0:
            break
        try:
            result = minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                options=stage_opts
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass

    return best_pos, best_energy


def construct():
    """
    Construct 55-atom LJ cluster by optimizing the 54D radial manifold,
    then performing full 3D Cartesian relaxation and high-precision polish.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    # Standard Mackay radii for initial guess
    r_inner = 1.054
    r_outer_mid = 1.755
    r_outer_vert = 2.0
    r_base = np.concatenate([
        np.full(12, r_inner),
        np.full(30, r_outer_mid),
        np.full(12, r_outer_vert)
    ])

    # Radial perturbation variants for multi-start
    radial_variants = [
        np.zeros(54),  # base
        np.array([0.02] * 12 + [0.03] * 30 + [0.02] * 12),  # expand slightly
        np.array([-0.02] * 12 + [-0.03] * 30 + [-0.02] * 12),  # compress slightly
        np.array([0.01] * 12 + [0.0] * 30 + [0.01] * 12),  # expand vertices only
        np.array([0.0] * 12 + [0.02] * 30 + [0.0] * 12),  # expand edges only
        np.array([0.03] * 12 + [0.0] * 30 + [0.0] * 12),  # expand inner shell more
        np.array([0.0] * 12 + [0.0] * 30 + [0.03] * 12),  # expand outer vertices more
    ]

    best_radial_params = r_base.copy()
    best_radial_energy = _radial_energy(r_base)

    # Multi-start over radial variants
    for rad_var in radial_variants:
        if time.time() > deadline - 40.0:
            break
        try:
            r_perturbed = r_base + rad_var
            opt_params, opt_energy = _radial_optimize(r_perturbed, deadline=deadline, maxiter=2000)
            if opt_energy < best_radial_energy:
                best_radial_energy = opt_energy
                best_radial_params = opt_params.copy()
        except Exception:
            continue

    # Construct positions from best radial parameters
    pos_0 = _cartesian_to_full(_radial_to_cartesian(best_radial_params))
    e_0 = best_radial_energy

    # Full 3D L-BFGS relaxation for final refinement
    if time.time() < deadline - 30.0:
        pos_1, e_1 = _cartesian_polish(pos_0, deadline=deadline, maxiter=3000)
        if e_1 < e_0:
            pos_0 = pos_1
            e_0 = e_1

    # Project back to radial manifold and re-optimize
    if time.time() < deadline - 20.0:
        try:
            pos54 = pos_0[1:]  # exclude center
            radii_current = np.linalg.norm(pos54, axis=1)
            opt_params2, opt_energy2 = _radial_optimize(radii_current, deadline=deadline, maxiter=3000)
            if opt_energy2 < e_0:
                pos_0 = _cartesian_to_full(_radial_to_cartesian(opt_params2))
                e_0 = opt_energy2
        except Exception:
            pass

    # Final high-precision Cartesian polish
    if time.time() < deadline - 10.0:
        pos_final, e_final = _cartesian_polish(pos_0, deadline=deadline, maxiter=5000)
        if e_final < e_0:
            pos_0 = pos_final
            e_0 = e_final

    # Verify minimum distance constraint
    diff = pos_0[:, np.newaxis, :] - pos_0[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)

    if min_dist_found < 0.5:
        scale = 0.5 / min_dist_found * 1.01
        pos_0 = pos_0 * scale
        if time.time() < deadline - 5.0:
            pos_0, e_0 = _cartesian_polish(pos_0, deadline=deadline, maxiter=2000)

    return pos_0
# EVOLVE-BLOCK-END
