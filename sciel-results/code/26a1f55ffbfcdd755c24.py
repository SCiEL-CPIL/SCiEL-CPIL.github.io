"""Lennard-Jones 55-atom global minimum: Variational potential field approach."""
import numpy as np
import time
import sys

N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _lj_energy(pos):
    """Compute total LJ energy for a set of positions (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0

def _generate_positions_from_field(coeffs):
    """
    Generate 55 positions from a variational potential field.
    coeffs: array of coefficients for radial basis functions.
    
    The potential field is: V(x) = sum_i c_i * exp(-||x - center_i||^2 / sigma_i^2)
    We place atoms at local minima of this field.
    """
    # centers and sigmas from coefficients
    n_centers = 12  # icosahedron vertices
    center_scale = coeffs[0]
    sigma = coeffs[1]
    amplitudes = coeffs[2:2+n_centers]
    
    # Use icosahedron geometry for centers
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norm_val = np.sqrt(1.0 + phi * phi)
    centers = verts / norm_val * center_scale
    
    # Generate a grid of candidate positions
    grid_size = 25
    extent = 2.5
    x = np.linspace(-extent, extent, grid_size)
    y = np.linspace(-extent, extent, grid_size)
    z = np.linspace(-extent, extent, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Compute potential field values
    V = np.zeros(len(grid_points))
    for i in range(n_centers):
        diff = grid_points - centers[i]
        r2 = np.sum(diff**2, axis=1)
        V += amplitudes[i] * np.exp(-r2 / (sigma * sigma))
    
    # Find local minima by sorting and selecting well-separated points
    order = np.argsort(V)
    selected = []
    min_sep = 0.8
    
    for idx in order:
        if len(selected) == 55:
            break
        pt = grid_points[idx]
        # Check if sufficiently separated from already selected
        if len(selected) == 0:
            selected.append(pt)
        else:
            dists = np.linalg.norm(np.array(selected) - pt, axis=1)
            if np.min(dists) > min_sep:
                selected.append(pt)
    
    # If we don't have 55, relax the separation requirement
    if len(selected) < 55:
        min_sep = 0.6
        selected = []
        for idx in order:
            if len(selected) == 55:
                break
            pt = grid_points[idx]
            if len(selected) == 0:
                selected.append(pt)
            else:
                dists = np.linalg.norm(np.array(selected) - pt, axis=1)
                if np.min(dists) > min_sep:
                    selected.append(pt)
    
    if len(selected) < 55:
        # Fallback: use Mackay icosahedron
        return _mackay_positions()
    
    return np.array(selected)

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

def _mackay_positions():
    """Generate standard Mackay icosahedron positions."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norm_val = np.sqrt(1.0 + phi * phi)
    v_inner = verts / norm_val
    
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
    
    r1, r2, r3 = 1.04, 1.76, 1.98
    center = np.array([[0.0, 0.0, 0.0]])
    shell1 = v_inner * r1
    shell2a = v_edge * r2
    shell2b = v_inner * r3
    
    return np.vstack([center, shell1, shell2a, shell2b])

def _field_objective(coeffs):
    """Objective: LJ energy of positions generated from the potential field."""
    pos = _generate_positions_from_field(coeffs)
    # Check minimum distance
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 0.5:
        return 1e10
    return _lj_energy(pos)

def _optimize_field(start_coeffs, deadline):
    """Optimize field coefficients using Nelder-Mead."""
    best_coeffs = start_coeffs.copy()
    best_energy = _field_objective(best_coeffs)
    
    try:
        res = minimize(
            _field_objective,
            best_coeffs,
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if res.fun < best_energy:
            best_energy = res.fun
            best_coeffs = res.x
    except Exception:
        pass
    
    if deadline is not None and time.time() > deadline - 5.0:
        return best_coeffs, best_energy
    
    return best_coeffs, best_energy

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
    """High-precision multi-stage polish pass."""
    start_time = time.time()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    stages = [
        {'maxiter': 2000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-16},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-15, 'maxls': 1000, 'ftol': 1e-17},
        {'maxiter': 20000, 'gtol': 1e-16, 'maxls': 2000, 'ftol': 1e-18},
    ]
    
    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 3.0:
            break
        try:
            result = minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                options=stage_opts
            )
            if result.fun < best_energy - 1e-15:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass
    
    return best_pos, best_energy



def construct():
    """
    Construct 55-atom LJ cluster using variational potential field approach,
    then perform full 3D relaxation and polish.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    # Initial field coefficients: [center_scale, sigma, 12 amplitudes]
    # Start with values that should produce a Mackay-like structure
    init_coeffs = np.array([1.0, 0.5] + [1.0]*12)
    
    # Step 1: Optimize field coefficients
    best_coeffs, best_field_energy = _optimize_field(init_coeffs, deadline)
    
    # Generate positions from optimized field
    pos_0 = _generate_positions_from_field(best_coeffs)
    
    # Also try Mackay as fallback/baseline
    pos_mackay = _mackay_positions()
    e_mackay = _lj_energy(pos_mackay)
    
    # Choose better starting point
    e_field = _lj_energy(pos_0)
    if e_field < e_mackay:
        pos_start = pos_0
    else:
        pos_start = pos_mackay
    
    # Step 2: Full 3D L-BFGS relaxation
    pos_1, e_1 = _relax_lbfgs(pos_start, deadline=deadline, maxiter=1000, gtol=1e-10)
    
    # Step 3: High precision multi-stage polish
    pos_final, e_final = _polish(pos_1, deadline=deadline)
    
    # Verify constraints
    diff = pos_final[:, np.newaxis, :] - pos_final[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < 0.5:
        scale = 0.5 / min_dist_found * 1.01
        pos_final = pos_final * scale
        if time.time() < deadline - 5.0:
            pos_final, e_final = _relax_lbfgs(pos_final, deadline=deadline, maxiter=500, gtol=1e-12)
    
    return pos_final
# EVOLVE-BLOCK-END
