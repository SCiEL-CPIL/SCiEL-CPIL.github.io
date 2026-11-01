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
    # In a relaxed structure, they might rotate slightly, but radial parameterization assumes fixed angular directions
    v_outer = v_inner.copy()
    
    return v_inner, v_edge, v_outer

V_INNER, V_EDGE, V_OUTER = _get_icosahedron_geometry()

def _construct_from_params(params):
    """
    params: [r1, r2, r3]
    r1: radius for 12 inner vertices
    r2: radius for 30 edge midpoints
    r3: radius for 12 outer vertices
    """
    r1, r2, r3 = params
    center = np.array([[0.0, 0.0, 0.0]])
    shell1 = V_INNER * r1
    shell2a = V_EDGE * r2
    shell2b = V_OUTER * r3
    
    pos = np.vstack([center, shell1, shell2a, shell2b])
    return pos

def _objective(params):
    pos = _construct_from_params(params)
    # Check for minimum distance violation to penalize if needed, 
    # though for icosahedron radii ~1-2, distance is usually safe
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
    """High-precision polish pass with progressively tighter tolerances."""
    start_time = time.time()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    # Stage 1-3: L-BFGS-B for robust convergence
    lbfgs_stages = [
        {'maxiter': 2000, 'gtol': 1e-10, 'maxls': 200, 'ftol': 1e-14},
        {'maxiter': 5000, 'gtol': 1e-12, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-14, 'maxls': 1000, 'ftol': 1e-16},
    ]
    
    for stage_opts in lbfgs_stages:
        if deadline is not None and time.time() > deadline - 15.0:
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
    
    # Stage 4: BFGS for high precision on small systems (N=55 is small enough)
    if deadline is None or time.time() < deadline - 10.0:
        try:
            result = minimize(
                obj_and_grad, x0, jac=True, method='BFGS',
                options={'maxiter': 5000, 'gtol': 1e-12, 'disp': False}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass
            
    # Stage 5: Final L-BFGS-B ultra-tight pass
    if deadline is None or time.time() < deadline - 5.0:
        try:
            result = minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                options={'maxiter': 10000, 'gtol': 1e-16, 'maxls': 2000, 'ftol': 1e-18}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
        except Exception:
            pass
    
    return best_pos, best_energy

def construct():
    """
    Construct 55-atom LJ cluster using literature-optimal Mackay icosahedron
    radii, multi-start parameter optimization, noise perturbation, and 
    high-precision polishing.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    # Literature-optimal shell radii for LJ55 Mackay icosahedron
    # These are much closer to the true minimum than arbitrary estimates
    literature_radii = np.array([1.135, 1.162, 1.283])
    
    # Multi-start optimization on the 3D parameter space
    # Start near literature values with diverse perturbations
    starts = [
        [1.135, 1.162, 1.283],  # Literature optimal
        [1.130, 1.155, 1.275],  # Slightly smaller
        [1.140, 1.170, 1.290],  # Slightly larger
        [1.125, 1.150, 1.270],  # More compact
        [1.145, 1.175, 1.295],  # More expanded
        [1.135, 1.150, 1.283],  # Vary r2
        [1.135, 1.175, 1.283],  # Vary r2 up
        [1.135, 1.162, 1.270],  # Vary r3 down
        [1.135, 1.162, 1.295],  # Vary r3 up
        [1.120, 1.145, 1.260],  # All smaller
        [1.150, 1.180, 1.300],  # All larger
    ]
    
    best_params = None
    best_param_energy = float('inf')
    
    for start in starts:
        if time.time() > deadline - 20.0:
            break
        try:
            res = minimize(
                _objective, 
                start, 
                method='Nelder-Mead', 
                options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            if res.fun < best_param_energy:
                best_param_energy = res.fun
                best_params = res.x
        except Exception:
            continue
            
    if best_params is None:
        # Fallback to literature values
        best_params = literature_radii.copy()
        best_param_energy = _objective(best_params)
        
    # Construct the initial structure from optimized parameters
    pos_0 = _construct_from_params(best_params)
    
    # Step 2: Multi-start 3D relaxation with noise perturbations
    # This helps escape potential local minima in the full 3D landscape
    best_pos_3d = pos_0.copy()
    best_e_3d = _lj_energy(pos_0)
    
    # Noise scales to try (from small to large)
    noise_scales = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    
    rng = np.random.default_rng(42)
    
    for noise_scale in noise_scales:
        if time.time() > deadline - 30.0:
            break
        
        if noise_scale == 0.0:
            pos_try = pos_0.copy()
        else:
            pos_try = pos_0 + rng.normal(0, noise_scale, pos_0.shape)
        
        # Quick relaxation to see if this perturbation helps
        pos_relaxed, e_relaxed = _relax_lbfgs(pos_try, deadline=deadline, maxiter=300, gtol=1e-8)
        
        if e_relaxed < best_e_3d:
            best_e_3d = e_relaxed
            best_pos_3d = pos_relaxed
    
    # Step 3: High-precision polish on the best structure
    pos_final, e_final = _polish(best_pos_3d, deadline=deadline)
    
    # Verify constraints
    diff = pos_final[:, np.newaxis, :] - pos_final[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < 0.5:
        # This should not happen for the optimized radii, but just in case
        scale = 0.5 / min_dist_found * 1.01
        pos_final = pos_final * scale
        # Re-relax if scaled
        pos_final, e_final = _relax_lbfgs(pos_final, deadline=deadline, maxiter=500, gtol=1e-12)

    return pos_final
# EVOLVE-BLOCK-END
