"""Lennard-Jones 55-atom global minimum: Parameterized Mackay icosahedron optimization."""
import numpy as np
import time
import sys

N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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

def _get_icosahedron_geometry():
    """
    Returns:
        v_inner: 12 unit vectors for the first shell (vertices).
        v_edge: 30 unit vectors for the second shell (edge midpoints).
        v_outer: 12 unit vectors for the second shell (vertices, same directions as inner).
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    v_inner = verts / norms
    
    # Find edges
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(v_inner[i] - v_inner[j])
    
    min_dist = np.min(dists[dists > 1e-6])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    
    v_edge = []
    for i, j in edges:
        mid = (v_inner[i] + v_inner[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            v_edge.append(mid / norm)
    v_edge = np.array(v_edge)
    
    # Outer vertices have same directions as inner vertices in the standard Mackay model
    v_outer = v_inner.copy()
    
    return v_inner, v_edge, v_outer

V_INNER, V_EDGE, V_OUTER = _get_icosahedron_geometry()

def _rotation_matrix_from_angles(theta, phi, psi):
    """Compute rotation matrix from Euler angles (XYZ convention)."""
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    cs, ss = np.cos(psi), np.sin(psi)
    
    R = np.array([
        [cs*cp, cs*sp*st - ss*ct, cs*sp*ct + ss*st],
        [ss*cp, ss*sp*st + cs*ct, ss*sp*ct - cs*st],
        [-sp,   cp*st,            cp*ct]
    ])
    return R

def _construct_from_manifold_params(params):
    """
    params: [r0, r1, ..., r54, theta, phi, psi, tx, ty, tz]
    r0: center atom radial (should be 0)
    r1-r12: inner shell radii
    r13-r42: edge midpoint radii  
    r43-r54: outer vertex radii
    theta, phi, psi: Euler rotation angles
    tx, ty, tz: translation
    """
    n_radial = 54
    r_vec = params[:n_radial]
    theta, phi, psi = params[n_radial:n_radial+3]
    t_vec = params[n_radial+3:n_radial+6]
    
    # Build base positions (54 atoms, excluding center)
    base_dirs = np.vstack([V_INNER, V_EDGE, V_OUTER])  # 54 unit vectors
    
    # Scale by radial parameters
    scaled = base_dirs * r_vec[:, np.newaxis]
    
    # Apply rotation
    R = _rotation_matrix_from_angles(theta, phi, psi)
    rotated = scaled @ R.T
    
    # Apply translation
    translated = rotated + t_vec[np.newaxis, :]
    
    # Prepend center atom at origin (with small offset if needed)
    center = np.array([[0.0, 0.0, 0.0]])
    pos = np.vstack([center, translated])
    
    return pos

def _manifold_objective(params):
    """Objective function for manifold parameter optimization."""
    pos = _construct_from_manifold_params(params)
    
    # Check minimum distance constraint
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    if min_dist < 0.5:
        return 1e10
    
    return _lj_energy(pos)

def _relax_lbfgs(pos, deadline=None, maxiter=500, gtol=1e-10):
    """Relax using L-BFGS-B with deadline guard."""
    start_time = time.time()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-14}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    if deadline is not None and time.time() > deadline - 5.0:
        return best_pos, best_energy
    
    return best_pos, best_energy

def _polish(pos, deadline=None):
    """High-precision polish pass."""
    start_time = time.time()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    stages = [
        {'maxiter': 1000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-16},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-18},
    ]
    
    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 5.0:
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

def _manifold_optimize(initial_params, deadline=None, maxiter=1000):
    """Optimize over the manifold parameter space."""
    start_time = time.time()
    best_energy = _manifold_objective(initial_params)
    best_params = initial_params.copy()
    
    def obj(x):
        return _manifold_objective(x)
    
    x0 = initial_params.copy()
    
    stages = [
        {'maxiter': maxiter, 'gtol': 1e-8, 'maxls': 100, 'ftol': 1e-10},
        {'maxiter': maxiter * 2, 'gtol': 1e-10, 'maxls': 200, 'ftol': 1e-12},
        {'maxiter': maxiter * 3, 'gtol': 1e-12, 'maxls': 300, 'ftol': 1e-14},
    ]
    
    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 10.0:
            break
        try:
            result = minimize(
                obj, x0, method='L-BFGS-B',
                options=stage_opts
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x.copy()
                x0 = result.x
        except Exception:
            pass
    
    return best_params, best_energy

def _refined_polish(pos, deadline=None):
    """
    Ultra-high-precision polish using multiple stages with progressively
    tighter tolerances. This aims to extract the last bits of energy.
    """
    start_time = time.time()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    # More aggressive stages
    stages = [
        {'maxiter': 2000, 'gtol': 1e-14, 'maxls': 300, 'ftol': 1e-16},
        {'maxiter': 5000, 'gtol': 1e-16, 'maxls': 500, 'ftol': 1e-18},
        {'maxiter': 10000, 'gtol': 1e-18, 'maxls': 1000, 'ftol': 1e-20},
        {'maxiter': 20000, 'gtol': 1e-20, 'maxls': 2000, 'ftol': 1e-22},
    ]
    
    for i, stage_opts in enumerate(stages):
        if deadline is not None and time.time() > deadline - 3.0:
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
    Construct 55-atom LJ cluster by optimizing the manifold parameters
    (54 radial + 3 rotation + 3 translation), then performing full 3D relaxation.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: evaluator kills at 150s

    # Initialize manifold parameters
    # Typical optimal radii for LJ55 Mackay icosahedron
    r1 = 1.04  # inner shell
    r2 = 1.76  # edge midpoints
    r3 = 1.98  # outer vertices
    
    # Build initial parameter vector
    r_vec = np.zeros(54)
    r_vec[0:12] = r1    # inner shell
    r_vec[12:42] = r2   # edge midpoints
    r_vec[42:54] = r3   # outer vertices
    
    # Initial rotation: identity (zeros)
    # Initial translation: zeros
    initial_params = np.concatenate([r_vec, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    
    # Multi-start over rotation angles to explore symmetry-breaking
    best_overall_params = initial_params.copy()
    best_overall_energy = _manifold_objective(initial_params)
    
    rotation_starts = [
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1],
        [0.05, 0.05, 0.05],
        [-0.05, 0.05, 0.0],
        [0.0, -0.05, 0.05],
    ]
    
    for rot_start in rotation_starts:
        if time.time() > deadline - 40.0:
            break
        try:
            params = np.concatenate([r_vec, rot_start, [0.0, 0.0, 0.0]])
            opt_params, opt_energy = _manifold_optimize(params, deadline=deadline, maxiter=500)
            if opt_energy < best_overall_energy:
                best_overall_energy = opt_energy
                best_overall_params = opt_params
        except Exception:
            continue
    
    # Construct positions from best manifold parameters
    pos_0 = _construct_from_manifold_params(best_overall_params)
    
    # Full 3D L-BFGS relaxation for final refinement
    pos_1, e_1 = _relax_lbfgs(pos_0, deadline=deadline, maxiter=500, gtol=1e-8)
    
    # High precision polish
    pos_2, e_2 = _polish(pos_1, deadline=deadline)
    
    # Ultra-high-precision polish for maximum accuracy
    pos_final, e_final = _refined_polish(pos_2, deadline=deadline)
    
    # Verify constraints
    diff = pos_final[:, np.newaxis, :] - pos_final[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < 0.5:
        scale = 0.5 / min_dist_found * 1.01
        pos_final = pos_final * scale
        pos_final, e_final = _relax_lbfgs(pos_final, deadline=deadline, maxiter=200, gtol=1e-10)
        # Re-polish after scaling
        pos_final, e_final = _refined_polish(pos_final, deadline=deadline)

    return pos_final
# EVOLVE-BLOCK-END
