"""Program for Lennard-Jones 55-atom cluster global minimum search.

Uses the known Mackay icosahedron geometry with hierarchical relaxation:
1. Coarse-grained rigid-body orientation optimization (6D)
2. Fine-grained L-BFGS polishing of individual atomic coordinates
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 55
DEADLINE_SECONDS = 115  # Safety margin before 120s limit
_START_TIME = time.time()

def _time_remaining():
    return DEADLINE_SECONDS - (time.time() - _START_TIME)

def _get_icosahedron_vectors():
    """
    Returns 12 unit vectors along the 5-fold axes (icosahedron vertices)
    and 30 unit vectors along the 2-fold axes (icosahedron edge midpoints).
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi],
        [0, -1, phi],
        [0, 1, -phi],
        [0, -1, -phi],
        [1, phi, 0],
        [-1, phi, 0],
        [1, -phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [-phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, -1],
    ], dtype=float)
    
    # Normalize to unit length
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # 30 edge midpoints
    dists = np.linalg.norm(verts[:, np.newaxis, :] - verts[np.newaxis, :, :], axis=2)
    min_dist = np.min(dists[dists > 1e-8])
    
    edge_mids = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.isclose(dists[i,j], min_dist, atol=1e-8):
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid)
                edge_mids.append(mid)
                
    edge_mids = np.array(edge_mids)
    return verts, edge_mids

def _mackay_icosahedron_55(r1=1.135, r2=1.162, r3=1.283):
    """
    Construct the 55-atom two-shell Mackay icosahedron.
    1 center, 12 at 5-fold (r1), 30 at 2-fold (r2), 12 at 5-fold (r3).
    """
    v5, v2 = _get_icosahedron_vectors()
    
    points = [np.zeros(3)]
    
    # Add 12 atoms at 5-fold axes, radius r1
    for v in v5:
        points.append(v * r1)
        
    # Add 30 atoms at 2-fold axes, radius r2
    for v in v2:
        points.append(v * r2)
        
    # Add 12 atoms at 5-fold axes, radius r3
    for v in v5:
        points.append(v * r3)
        
    points = np.array(points, dtype=float)
    return points

def _lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    mask = r2 > 1e-10
    r2_safe = np.where(mask, r2, 1.0)
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    energy = 4 * np.sum(np.where(mask, r12 - r6, 0))
    return energy

def _lj_energy_and_grad(pos):
    """Compute total Lennard-Jones energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    mask = r2 > 1e-10
    r2_safe = np.where(mask, r2, 1.0)
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    energy = 4 * np.sum(np.where(mask, r12 - r6, 0))
    
    # Gradient: dE/dx_i = sum_j 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j)/r
    # Simplified: grad = 4 * sum_j (-12*r2^-7 + 6*r2^-4) * diff
    r2_inv4 = r2_safe ** (-4)
    r2_inv7 = r2_safe ** (-7)
    coeff = 4 * (-12 * r2_inv7 + 6 * r2_inv4)
    coeff = np.where(mask, coeff, 0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad

def _is_valid(pos):
    """Check if all inter-atomic distances are >= 0.5."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.5

def _relax(pos, maxiter=10000, ftol=1e-12, gtol=1e-10):
    """Relax positions using L-BFGS-B with analytical gradient."""
    def energy_func(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos_flat)
        return e, g.flatten()
        
    result = minimize(
        energy_func,
        pos.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    return result.x.reshape(-1, 3)

def _rotation_matrix_from_params(params):
    """Convert 6 parameters (3 translation + 3 Euler angles) to rotation+translation."""
    tx, ty, tz, rx, ry, rz = params
    # Rotation matrices
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    t = np.array([tx, ty, tz])
    return R, t

def _apply_rigid_transform(pos, params):
    """Apply rigid body transformation to positions."""
    R, t = _rotation_matrix_from_params(params)
    return pos @ R.T + t

def _coarse_energy_and_grad(pos, params):
    """Compute energy and gradient w.r.t. rigid body parameters."""
    transformed = _apply_rigid_transform(pos, params)
    e, g_cart = _lj_energy_and_grad(transformed)
    
    # Chain rule: dE/dparams = dE/dpos * dpos/dparams
    R, t = _rotation_matrix_from_params(params)
    tx, ty, tz, rx, ry, rz = params
    
    # Translation gradient
    grad_t = np.sum(g_cart, axis=0)
    
    # Rotation gradient (using skew-symmetric representation)
    # For small rotations, dpos/dangle involves cross products
    # grad_r = sum_i g_i x (R @ p_i)
    rotated_pos = pos @ R.T
    grad_r = np.zeros(3)
    for i in range(len(pos)):
        grad_r += np.cross(g_cart[i], rotated_pos[i])
    
    # Full gradient in parameter space [tx,ty,tz,rx,ry,rz]
    grad = np.concatenate([grad_t, grad_r])
    return e, grad

def construct():
    """
    Construct the 55-atom LJ cluster using hierarchical multi-scale relaxation.
    
    Step 1: Coarse-grained rigid-body optimization (6D) to find best orientation
    Step 2: Fine-grained L-BFGS polishing of individual atomic coordinates
    """
    _START_TIME_LOCAL = time.time()
    deadline = 115  # seconds, safety margin
    
    def time_remaining():
        return deadline - (time.time() - _START_TIME_LOCAL)
    
    rng = np.random.default_rng(42)
    
    # Start with standard Mackay icosahedron geometry
    base = _mackay_icosahedron_55(1.135, 1.162, 1.283)
    
    best_energy = float('inf')
    best_pos = base.copy()
    
    # Step 1: Coarse-grained rigid body optimization
    # Optimize over 6 parameters: 3 translation + 3 rotation angles
    if time_remaining() > 30:
        def coarse_obj(x):
            e, g = _coarse_energy_and_grad(base, x)
            return e, g
        
        # Try multiple initial orientations
        best_coarse = None
        best_coarse_energy = float('inf')
        
        # Initial identity transform
        init_params = np.zeros(6)
        
        for trial in range(3):
            if time_remaining() < 10:
                break
            
            if trial == 0:
                init = np.zeros(6)
            else:
                # Small random perturbations to explore different basins
                init = rng.normal(0, 0.1, size=6)
            
            result = minimize(
                coarse_obj,
                init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success:
                e_coarse = result.fun
                if e_coarse < best_coarse_energy:
                    best_coarse_energy = e_coarse
                    best_coarse = result.x
        
        # Apply best rigid transform to get improved starting positions
        if best_coarse is not None:
            coarse_pos = _apply_rigid_transform(base, best_coarse)
            # Center the cluster
            coarse_pos -= np.mean(coarse_pos, axis=0)
            
            # Step 2: Fine-grained relaxation from coarse result
            if time_remaining() > 20:
                relaxed = _relax(coarse_pos, maxiter=10000, ftol=1e-12, gtol=1e-10)
                if _is_valid(relaxed):
                    e = _lj_energy(relaxed)
                    if e < best_energy:
                        best_energy = e
                        best_pos = relaxed.copy()
    
    # Step 3: Direct relaxation from original base (fallback/backup)
    if time_remaining() > 15:
        relaxed = _relax(base, maxiter=15000, ftol=1e-14, gtol=1e-12)
        if _is_valid(relaxed):
            e = _lj_energy(relaxed)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
    
    # Step 4: High-precision polish if close to global minimum
    if time_remaining() > 10 and best_energy < -279.0:
        # Final polish with very tight tolerances
        relaxed = _relax(best_pos, maxiter=5000, ftol=1e-15, gtol=1e-14)
        if _is_valid(relaxed):
            e = _lj_energy(relaxed)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
    
    # Step 5: Try small perturbations if not at global minimum
    if time_remaining() > 15 and best_energy > -279.248:
        for trial in range(5):
            if time_remaining() < 5:
                break
            noise = rng.normal(0, 0.005, size=base.shape)
            pos = base + noise
            relaxed = _relax(pos, maxiter=8000, ftol=1e-12, gtol=1e-10)
            if _is_valid(relaxed):
                e = _lj_energy(relaxed)
                if e < best_energy:
                    best_energy = e
                    best_pos = relaxed.copy()
    
    return best_pos
