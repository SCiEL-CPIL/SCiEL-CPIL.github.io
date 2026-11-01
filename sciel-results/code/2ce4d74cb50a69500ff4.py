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
        edges: list of 30 (i, j) index pairs for the icosahedron edges.
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
    
    return v_inner, edges

V_INNER, EDGES = _get_icosahedron_geometry()
# EDGES has 30 entries, each (i, j) with i < j
N_EDGES = len(EDGES)  # should be 30

# Precompute edge endpoint arrays for fast construction
EDGE_I = np.array([e[0] for e in EDGES], dtype=int)
EDGE_J = np.array([e[1] for e in EDGES], dtype=int)
# Unit vectors along each edge (from vertex i to vertex j)
EDGE_DIR = (V_INNER[EDGE_J] - V_INNER[EDGE_I])
EDGE_DIR_NORM = np.linalg.norm(EDGE_DIR, axis=1, keepdims=True)
EDGE_DIR_UNIT = EDGE_DIR / EDGE_DIR_NORM
# Midpoint unit directions for reference
EDGE_MID = (V_INNER[EDGE_I] + V_INNER[EDGE_J]) / 2.0
EDGE_MID_NORM = np.linalg.norm(EDGE_MID, axis=1, keepdims=True)
EDGE_MID_UNIT = EDGE_MID / EDGE_MID_NORM

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

def _construct_from_edge_params(params):
    """
    params layout:
      [0:3]   = [r_inner, r_edge_scale, r_outer_vertex_scale] (3 radial params)
      [3:33]  = 30 edge fractions t in [0,1]
      [33:36] = [theta, phi, psi] rotation
      [36:39] = [tx, ty, tz] translation
    Total: 39 parameters.
    
    Inner shell: 12 atoms at r_inner * v_inner[i]
    Edge shell: 30 atoms at r_edge_scale * ((1-t_k)*v_inner[EDGE_I[k]] + t_k*v_inner[EDGE_J[k]])
    Outer vertices: 12 atoms at r_outer_vertex_scale * v_inner[i] (same directions as inner)
    Center: 1 atom at origin
    Total: 55 atoms
    """
    r_inner = params[0]
    r_edge = params[1]
    r_outer = params[2]
    t_fracs = params[3:33]
    theta, phi, psi = params[33:36]
    t_vec = params[36:39]
    
    # Inner shell: 12 atoms
    inner_pos = V_INNER * r_inner  # (12, 3)
    
    # Edge shell: 30 atoms, each on an edge
    # Position = (1-t)*v_i + t*v_j, scaled by r_edge
    vi = V_INNER[EDGE_I]  # (30, 3)
    vj = V_INNER[EDGE_J]  # (30, 3)
    t_col = t_fracs[:, np.newaxis]  # (30, 1)
    edge_dirs_raw = (1.0 - t_col) * vi + t_col * vj  # (30, 3)
    edge_pos = edge_dirs_raw * r_edge  # (30, 3)
    
    # Outer vertices: 12 atoms at same directions as inner, scaled
    outer_pos = V_INNER * r_outer  # (12, 3)
    
    # Combine all 54 non-center atoms
    all_pos = np.vstack([inner_pos, edge_pos, outer_pos])  # (54, 3)
    
    # Apply rotation
    R = _rotation_matrix_from_angles(theta, phi, psi)
    rotated = all_pos @ R.T
    
    # Apply translation
    translated = rotated + t_vec[np.newaxis, :]
    
    # Prepend center atom
    center = np.array([[0.0, 0.0, 0.0]])
    pos = np.vstack([center, translated])
    
    return pos

def _manifold_objective(params):
    """Objective function for edge-fraction manifold parameter optimization."""
    pos = _construct_from_edge_params(params)
    
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
    """Optimize over the edge-fraction manifold parameter space with bounds."""
    start_time = time.time()
    best_energy = _manifold_objective(initial_params)
    best_params = initial_params.copy()
    
    def obj(x):
        return _manifold_objective(x)
    
    x0 = initial_params.copy()
    
    # Bounds: radii positive, edge fractions in [0,1], rotation/translation unbounded
    n_params = len(initial_params)
    lb = np.zeros(n_params)
    ub = np.full(n_params, np.inf)
    lb[0] = 0.8   # r_inner lower
    lb[1] = 1.2   # r_edge lower
    lb[2] = 1.5   # r_outer lower
    ub[0] = 2.0
    ub[1] = 2.5
    ub[2] = 3.0
    lb[3:33] = 0.0  # edge fractions
    ub[3:33] = 1.0
    lb[33:36] = -np.pi
    ub[33:36] = np.pi
    lb[36:39] = -5.0
    ub[36:39] = 5.0
    
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
                obj, x0, method='L-BFGS-B', bounds=(lb, ub),
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
    Construct 55-atom LJ cluster using edge-fraction manifold optimization,
    then full 3D relaxation and high-precision polish.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: evaluator kills at 150s

    # Initialize edge-fraction manifold parameters
    # params: [r_inner, r_edge, r_outer, t_0..t_29, theta, phi, psi, tx, ty, tz]
    # 3 + 30 + 3 + 3 = 39 parameters
    
    # Typical optimal radii for LJ55 Mackay icosahedron
    r1 = 1.04  # inner shell
    r2 = 1.76  # edge atoms
    r3 = 1.98  # outer vertices
    
    # Initial edge fractions: all 0.5 (midpoints, standard Mackay)
    t_init = np.full(N_EDGES, 0.5)
    
    initial_params = np.concatenate([
        [r1, r2, r3],
        t_init,
        [0.0, 0.0, 0.0],  # rotation
        [0.0, 0.0, 0.0]   # translation
    ])
    
    # Multi-start over rotation angles and edge fraction perturbations
    best_overall_params = initial_params.copy()
    best_overall_energy = _manifold_objective(initial_params)
    
    # Try several starting configurations
    starts = []
    
    # Standard Mackay (all midpoints)
    starts.append(initial_params.copy())
    
    # Slightly asymmetric edge fractions
    for perturb in [0.05, 0.1, 0.15]:
        t_pert = 0.5 + perturb * np.array([
            1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
            1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
            1, -1, 1, -1, 1, -1, 1, -1, 1, -1
        ])
        t_pert = np.clip(t_pert, 0.0, 1.0)
        p = np.concatenate([[r1, r2, r3], t_pert, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        starts.append(p)
    
    # Small rotation perturbations
    for rot in [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [0.05, 0.05, 0.05]]:
        p = np.concatenate([[r1, r2, r3], t_init, rot, [0.0, 0.0, 0.0]])
        starts.append(p)
    
    # Slightly different radii
    for r1v, r2v, r3v in [(1.0, 1.8, 2.0), (1.06, 1.72, 2.02), (1.02, 1.78, 1.95)]:
        p = np.concatenate([[r1v, r2v, r3v], t_init, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        starts.append(p)
    
    for s in starts:
        if time.time() > deadline - 30.0:
            break
        try:
            opt_params, opt_energy = _manifold_optimize(s, deadline=deadline, maxiter=800)
            if opt_energy < best_overall_energy:
                best_overall_energy = opt_energy
                best_overall_params = opt_params
        except Exception:
            continue
    
    # Construct positions from best manifold parameters
    pos_0 = _construct_from_edge_params(best_overall_params)
    
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
        pos_final, e_final = _refined_polish(pos_final, deadline=deadline)

    return pos_final
# EVOLVE-BLOCK-END
