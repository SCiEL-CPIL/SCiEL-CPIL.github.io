"""Program for Lennard-Jones 55-atom cluster global minimum search.

Combines Mackay icosahedron geometry with:
1. Efficient vectorized energy/gradient computation
2. Multi-stage L-BFGS relaxation with tight tolerances
3. Radial manifold optimization (54 free radii on icosahedral directions)
4. Final high-precision Cartesian polish

The known global minimum is the Mackay icosahedron at E ≈ -279.248470.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 55
DEADLINE = 135.0  # Safety margin before 150s kill

# Precompute icosahedral geometry
def _get_icosahedron_vectors():
    """Returns 12 unit vectors (5-fold axes) and 30 unit vectors (2-fold axes)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    dists = np.linalg.norm(verts[:, np.newaxis, :] - verts[np.newaxis, :, :], axis=2)
    min_dist = np.min(dists[dists > 1e-8])
    
    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            if np.isclose(dists[i, j], min_dist, atol=1e-8):
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid)
                edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    return verts, edge_mids

_V5, _V2 = _get_icosahedron_vectors()
# 54 directions: 12 inner (5-fold), 30 middle (2-fold), 12 outer (5-fold)
_DIRS = np.vstack([_V5, _V2, _V5])  # shape (54, 3)

def _lj_energy(pos):
    """Compute total Lennard-Jones energy (vectorized)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 * r2 * r2
    e_pair = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    return np.sum(e_pair) / 2.0

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized)."""
    n = len(pos)
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

def _is_valid(pos):
    """Check if all inter-atomic distances are >= 0.5."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.5

def _relax_lbfgs(pos, maxiter=3000, gtol=1e-10, ftol=1e-14, maxls=100):
    """Relax positions using L-BFGS-B with analytical gradient."""
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    try:
        result = minimize(
            obj_and_grad, pos.flatten(), jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol, 'maxls': maxls}
        )
        if result.success or result.fun < _lj_energy(pos):
            return result.x.reshape(-1, 3)
    except Exception:
        pass
    return pos

def _polish(pos, deadline=None):
    """Multi-stage high-precision polish with progressively tighter tolerances."""
    best_pos = pos.copy()
    best_energy = _lj_energy(pos)
    
    stages = [
        {'maxiter': 2000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-16},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-18},
    ]
    
    x0 = best_pos.flatten()
    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 8.0:
            break
        try:
            def obj_and_grad(x):
                pos_flat = x.reshape(-1, 3)
                e, g = _lj_energy_grad(pos_flat)
                return e, g.flatten()
            
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

def _optimize_radii(radii_init, deadline=None):
    """Optimize 54 radial parameters on fixed icosahedral directions.
    
    This is a lower-dimensional optimization (54 vs 165 DOF) that can
    escape shallow local minima in Cartesian space.
    """
    def radial_energy(radii):
        # Center atom at origin, 54 atoms on icosahedral directions
        pos = np.zeros((55, 3))
        pos[1:] = radii[:, np.newaxis] * _DIRS
        return _lj_energy(pos)
    
    def radial_grad(radii):
        pos = np.zeros((55, 3))
        pos[1:] = radii[:, np.newaxis] * _DIRS
        e, g_cart = _lj_energy_grad(pos)
        # Project Cartesian gradient onto radial directions
        # dE/dr_i = dot(g_cart[i+1], dir_i)
        g_radial = np.sum(g_cart[1:] * _DIRS, axis=1)
        return e, g_radial
    
    best_radii = radii_init.copy()
    best_energy = radial_energy(best_radii)
    
    try:
        result = minimize(
            radial_energy, radii_init, jac=radial_grad, method='L-BFGS-B',
            options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-16, 'maxls': 200}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_radii = result.x
    except Exception:
        pass
    
    # Second pass with tighter tolerance
    if deadline is None or time.time() < deadline - 10.0:
        try:
            result = minimize(
                radial_energy, best_radii, jac=radial_grad, method='L-BFGS-B',
                options={'maxiter': 10000, 'gtol': 1e-14, 'ftol': 1e-18, 'maxls': 500}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_radii = result.x
        except Exception:
            pass
    
    # Convert back to Cartesian
    pos = np.zeros((55, 3))
    pos[1:] = best_radii[:, np.newaxis] * _DIRS
    return pos, best_energy

def _mackay_base(r1=1.135, r2=1.162, r3=1.283):
    """Construct base Mackay icosahedron with given radii."""
    pos = np.zeros((55, 3))
    pos[0] = [0, 0, 0]
    pos[1:13] = _V5 * r1
    pos[13:43] = _V2 * r2
    pos[43:55] = _V5 * r3
    return pos

def construct():
    """Construct 55-atom LJ cluster near global minimum."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = np.random.default_rng(42)
    
    # Start with literature-optimal Mackay geometry
    base = _mackay_base(1.135, 1.162, 1.283)
    
    best_energy = float('inf')
    best_pos = base.copy()
    
    # Step 1: Initial L-BFGS relaxation of base geometry
    relaxed = _relax_lbfgs(base, maxiter=3000, gtol=1e-10)
    if _is_valid(relaxed):
        e = _lj_energy(relaxed)
        if e < best_energy:
            best_energy = e
            best_pos = relaxed.copy()
    
    # Step 2: Radial manifold optimization
    if time.time() < deadline - 30.0:
        # Extract initial radii from relaxed position
        radii_init = np.linalg.norm(best_pos[1:], axis=1)
        radial_pos, radial_energy = _optimize_radii(radii_init, deadline)
        if _is_valid(radial_pos):
            # Polish the radial result in Cartesian space
            polished, pol_energy = _polish(radial_pos, deadline)
            if _is_valid(polished) and pol_energy < best_energy:
                best_energy = pol_energy
                best_pos = polished.copy()
    
    # Step 3: Multi-start with perturbations if time remains
    if time.time() < deadline - 20.0:
        noise_scales = [0.02, 0.05, 0.1]
        for scale in noise_scales:
            if time.time() > deadline - 15.0:
                break
            for trial in range(2):
                if time.time() > deadline - 12.0:
                    break
                noise = rng.normal(0, scale, size=base.shape)
                pos = base + noise
                relaxed = _relax_lbfgs(pos, maxiter=2000, gtol=1e-10)
                if _is_valid(relaxed):
                    e = _lj_energy(relaxed)
                    if e < best_energy:
                        best_energy = e
                        best_pos = relaxed.copy()
    
    # Step 4: Final high-precision polish
    if time.time() < deadline - 10.0:
        refined, refined_energy = _polish(best_pos, deadline)
        if _is_valid(refined) and refined_energy < best_energy:
            best_energy = refined_energy
            best_pos = refined
    
    return best_pos
