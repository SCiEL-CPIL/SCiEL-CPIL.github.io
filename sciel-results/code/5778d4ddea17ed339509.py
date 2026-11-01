"""Lennard-Jones 55-atom global minimum: Mackay icosahedron with enhanced optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START

def _lj_energy(pos):
    """Compute total LJ energy for a set of positions (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
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

def _build_mackay_geometry():
    """
    Construct the Mackay icosahedron geometry with exact icosahedral symmetry directions.
    Returns:
        dirs: array of shape (55, 3) with unit direction vectors for all 55 positions
        shell_ids: array of shape (55,) indicating which shell each atom belongs to
                   0=center, 1=inner 12, 2=edge 30, 3=outer 12
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    norm_val = np.sqrt(1.0 + phi * phi)
    
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float) / norm_val
    
    # Find edges (nearest neighbors)
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(verts[i] - verts[j])
    min_dist = np.min(dists[dists > 1e-6])
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    
    # Edge midpoints (30 directions)
    edge_dirs = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        edge_dirs.append(mid / np.linalg.norm(mid))
    edge_dirs = np.array(edge_dirs)
    
    # Assemble all directions
    center = np.array([[0.0, 0.0, 0.0]])
    dirs = np.vstack([center, verts, edge_dirs, verts])
    shell_ids = np.array([0] + [1]*12 + [2]*30 + [3]*12)
    
    return dirs, shell_ids

DIRS, SHELL_IDS = _build_mackay_geometry()

def _params_to_positions(params):
    """
    Convert 54 radial parameters to 55 positions.
    params: array of shape (54,)
    Order: [r_inner_12, r_edge_30, r_outer_12]
    """
    r_inner = params[0:12]
    r_edge = params[12:42]
    r_outer = params[42:54]
    
    # Center atom
    pos = np.zeros((55, 3))
    # Inner shell (12 atoms)
    pos[1:13] = DIRS[1:13] * r_inner[:, np.newaxis]
    # Edge shell (30 atoms)
    pos[13:43] = DIRS[13:43] * r_edge[:, np.newaxis]
    # Outer shell (12 atoms)
    pos[43:55] = DIRS[43:55] * r_outer[:, np.newaxis]
    
    return pos

def _energy_from_params(params):
    """Objective function for radial parameter optimization."""
    pos = _params_to_positions(params)
    # Check minimum distance constraint
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    if min_r2 < 0.25:  # 0.5^2
        return 1e10
    return _lj_energy(pos)

def _relax_positions(pos, deadline, maxiter=2000, gtol=1e-12):
    """L-BFGS relaxation with deadline guard."""
    best_pos = pos.copy()
    best_e = _lj_energy(pos)
    
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_grad(p)
        return e, g.flatten()
    
    x0 = pos.flatten()
    start = time.time()
    
    try:
        result = minimize(
            obj_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200, 'ftol': 1e-15}
        )
        if result.fun < best_e:
            best_e = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    # Deadline check
    if deadline is not None and time.time() > deadline - 3.0:
        return best_pos, best_e
    
    return best_pos, best_e

def _high_precision_polish(pos, deadline):
    """Multi-stage high-precision polish."""
    best_pos = pos.copy()
    best_e = _lj_energy(pos)
    
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_grad(p)
        return e, g.flatten()
    
    stages = [
        {'maxiter': 3000, 'gtol': 1e-14, 'maxls': 300, 'ftol': 1e-16},
        {'maxiter': 5000, 'gtol': 1e-15, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 8000, 'gtol': 1e-16, 'maxls': 800, 'ftol': 1e-17},
        {'maxiter': 12000, 'gtol': 1e-17, 'maxls': 1200, 'ftol': 1e-18},
    ]
    
    x0 = pos.flatten()
    for opts in stages:
        if deadline is not None and time.time() > deadline - 2.0:
            break
        try:
            result = minimize(
                obj_grad, x0, jac=True, method='L-BFGS-B',
                options=opts
            )
            if result.fun < best_e - 1e-15:
                best_e = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass
    
    return best_pos, best_e

def _optimize_radial(start_params, deadline):
    """Nelder-Mead optimization of radial parameters."""
    best_params = start_params.copy()
    best_e = _energy_from_params(best_params)
    
    start = time.time()
    try:
        res = minimize(
            _energy_from_params,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12, 
                     'adaptive': True, 'maxfev': 2000}
        )
        if res.fun < best_e:
            best_e = res.fun
            best_params = res.x
    except Exception:
        pass
    
    if deadline is not None and time.time() > deadline - 5.0:
        return best_params, best_e
    
    return best_params, best_e

def _try_perturbation(pos, deadline, num_trials=3, sigma=0.01):
    """Try small perturbations to escape potential shallow local minima."""
    best_pos = pos.copy()
    best_e = _lj_energy(pos)
    rng = np.random.RandomState(42)
    
    for _ in range(num_trials):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        perturbed = pos + rng.normal(0, sigma, pos.shape)
        p, e = _relax_positions(perturbed, deadline, maxiter=500, gtol=1e-10)
        if e < best_e - 1e-12:
            best_e = e
            best_pos = p
    
    return best_pos, best_e

def construct():
    """
    Construct 55-atom LJ cluster: Mackay icosahedron with optimized radials,
    full 3D relaxation, perturbation escape, and high-precision polish.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial radial parameters (typical Mackay values)
    r_inner_init = np.full(12, 1.04)
    r_edge_init = np.full(30, 1.76)
    r_outer_init = np.full(12, 1.98)
    init_params = np.concatenate([r_inner_init, r_edge_init, r_outer_init])
    
    # Step 1: Optimize radial parameters
    best_params, best_param_e = _optimize_radial(init_params, deadline)
    
    # Alternative starts
    if time.time() < deadline - 40.0:
        alt_configs = [
            (1.05, 1.75, 1.96),
            (1.03, 1.77, 1.97),
            (1.02, 1.78, 2.00),
            (1.06, 1.74, 1.95),
        ]
        for ri, re, ro in alt_configs:
            if time.time() > deadline - 15.0:
                break
            alt_params = np.concatenate([np.full(12, ri), np.full(30, re), np.full(12, ro)])
            ap, ae = _optimize_radial(alt_params, deadline)
            if ae < best_param_e:
                best_param_e = ae
                best_params = ap
    
    # Construct structure from optimized parameters
    pos_0 = _params_to_positions(best_params)
    
    # Step 2: Full 3D relaxation
    pos_1, e_1 = _relax_positions(pos_0, deadline, maxiter=1500, gtol=1e-11)
    
    # Step 3: Perturbation escape
    pos_2, e_2 = _try_perturbation(pos_1, deadline, num_trials=5, sigma=0.005)
    
    # Step 4: High-precision polish
    pos_final, e_final = _high_precision_polish(pos_2, deadline)
    
    # Verify distance constraint
    diff = pos_final[:, np.newaxis, :] - pos_final[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    
    if min_r < 0.5:
        scale = (0.5 / min_r) * 1.001
        pos_final = pos_final * scale
        if time.time() < deadline - 3.0:
            pos_final, e_final = _relax_positions(pos_final, deadline, maxiter=500, gtol=1e-12)
    
    return pos_final
# EVOLVE-BLOCK-END
