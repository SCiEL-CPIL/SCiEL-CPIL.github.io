import numpy as np
import time
from scipy.optimize import minimize

N = 55

def _lj_energy_grad(pos):
    """
    Compute total LJ energy and gradient for a set of positions.
    Vectorized for speed.
    """
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = -48.0 / (r2 ** 7) + 24.0 / (r2 ** 4)
    np.fill_diagonal(factor, 0.0)
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = factor @ pos
    grad = pos * sum_factor - sum_factor_pos
    return energy, grad

def _lj_energy(pos):
    e, _ = _lj_energy_grad(pos)
    return e

def _get_mackay_geometry():
    """
    Returns unit direction vectors and shell assignments for the 55-atom Mackay icosahedron.
    Shell 0: center (1 atom)
    Shell 1: 12 icosahedron vertices
    Shell 2: 30 edge midpoints
    Shell 3: 12 outer icosahedron vertices
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms

    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(verts_unit[i] - verts_unit[j])
    mask = dists > 1e-6
    min_dist = np.min(dists[mask])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))

    edge_mids = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)

    directions = np.zeros((55, 3))
    shell_ids = np.zeros(55, dtype=int)
    directions[0] = [0.0, 0.0, 0.0]
    shell_ids[0] = 0
    for i in range(12):
        directions[1 + i] = verts_unit[i]
        shell_ids[1 + i] = 1
    for i in range(30):
        directions[13 + i] = edge_mids[i]
        shell_ids[13 + i] = 2
    for i in range(12):
        directions[43 + i] = verts_unit[i]
        shell_ids[43 + i] = 3
    return directions, shell_ids

def _make_mackay(radii, rot=None, translate=None):
    """
    Build Mackay positions from 3 shell radii [r1, r2, r3], optional rotation matrix and translation.
    """
    directions, shell_ids = _get_mackay_geometry()
    pos = np.zeros((55, 3))
    for i in range(1, 55):
        s = shell_ids[i]
        if s == 1:
            pos[i] = directions[i] * radii[0]
        elif s == 2:
            pos[i] = directions[i] * radii[1]
        elif s == 3:
            pos[i] = directions[i] * radii[2]
    if rot is not None:
        pos = pos @ rot.T
    if translate is not None:
        pos = pos + translate
    return pos

def _relax_lbfgs(pos, maxiter=300, gtol=1e-8, ftol=1e-10):
    """
    Relax positions with L-BFGS-B using analytic gradient.
    Returns (positions, energy).
    """
    def obj_and_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_grad(p)
        return e, g.flatten()
    try:
        res = minimize(obj_and_grad, pos.flatten(), jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol, 'maxls': 100})
        e = res.fun
        p = res.x.reshape(-1, 3)
    except Exception:
        e = _lj_energy(pos)
        p = pos
    # Guard: if relaxation produced NaN or worse, fall back
    if not np.isfinite(e):
        e = _lj_energy(pos)
        p = pos
    return p, e

def _enforce_min_dist(pos, min_d=0.5):
    """Ensure all pairwise distances >= min_d by uniform scaling if needed."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    m = np.min(d)
    if m < min_d:
        pos = pos * (min_d / m) * 1.01
    return pos

def _optimize_radii(radii_init, deadline, nm_maxiter=80, lbfgs_maxiter=200):
    """
    Optimize the 3 shell radii via Nelder-Mead; each evaluation relaxes the full geometry with L-BFGS.
    Returns (best_pos, best_energy, best_radii).
    """
    def objective(radii):
        if time.time() > deadline:
            return 1e9
        radii = np.maximum(radii, 0.5)
        pos = _make_mackay(radii)
        pos, e = _relax_lbfgs(pos, maxiter=lbfgs_maxiter, gtol=1e-6, ftol=1e-8)
        return e
    best_pos = _make_mackay(radii_init)
    best_e = _lj_energy(best_pos)
    best_radii = radii_init.copy()
    try:
        res = minimize(objective, radii_init, method='Nelder-Mead',
                       options={'maxiter': nm_maxiter, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
        if res.fun < best_e:
            best_e = res.fun
            best_radii = res.x
            best_pos = _make_mackay(best_radii)
            best_pos, best_e = _relax_lbfgs(best_pos, maxiter=lbfgs_maxiter, gtol=1e-6, ftol=1e-8)
    except Exception:
        pass
    return best_pos, best_e, best_radii

def construct():
    start_time = time.time()
    # Evaluator kills at 150s; keep a hard safety margin.
    deadline = start_time + 135.0

    # 1. Parameterized manifold search: optimize shell radii via Nelder-Mead from several starts.
    radius_starts = [
        np.array([1.05, 1.85, 2.65]),
        np.array([1.10, 1.90, 2.70]),
        np.array([0.95, 1.75, 2.55]),
        np.array([1.12, 1.75, 1.95]),  # current program's guess
        np.array([1.00, 2.00, 2.80]),
    ]

    best_pos = _make_mackay(radius_starts[0])
    best_energy = _lj_energy(best_pos)

    for rs in radius_starts:
        if time.time() > deadline - 25.0:
            break
        try:
            pos, e, _ = _optimize_radii(rs, deadline, nm_maxiter=60, lbfgs_maxiter=150)
            if np.isfinite(e) and e < best_energy:
                best_energy = e
                best_pos = pos.copy()
        except Exception:
            continue

    # 2. Multi-start Cartesian perturbations from the best Mackay structure (breaks symmetry to escape manifold minima).
    rng = np.random.default_rng(42)
    for i in range(4):
        if time.time() > deadline - 20.0:
            break
        noise_scale = 0.05 + 0.03 * i
        pos_seed = best_pos + rng.normal(0, noise_scale, size=best_pos.shape)
        pos_seed = _enforce_min_dist(pos_seed)
        try:
            pos_relaxed, energy_relaxed = _relax_lbfgs(pos_seed, maxiter=300, gtol=1e-8, ftol=1e-10)
            if energy_relaxed < best_energy:
                best_energy = energy_relaxed
                best_pos = pos_relaxed.copy()
        except Exception:
            continue

    # 3. Final high-precision polish: tighter tolerance, more iterations.
    if time.time() < deadline - 8.0:
        try:
            pos_final, energy_final = _relax_lbfgs(best_pos, maxiter=1000, gtol=1e-12, ftol=1e-14)
            if energy_final < best_energy:
                best_energy = energy_final
                best_pos = pos_final
        except Exception:
            pass

    # 4. Final safety: enforce minimum distance constraint.
    best_pos = _enforce_min_dist(best_pos)

    return best_pos
