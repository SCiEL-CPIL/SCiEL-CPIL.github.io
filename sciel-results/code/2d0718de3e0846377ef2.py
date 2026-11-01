"""55-atom LJ cluster: Mackay icosahedron construction with optimized individual shell radii (54D manifold),
full Cartesian L-BFGS relaxation, and a final high-precision polish.
Combines 3-shell pre-optimization with 54D radial manifold optimization for a superior starting point
for the final Cartesian relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE_SECONDS = 115.0  # Safety margin for 120s limit
_BH_SEED = 42

def _lj_energy(pos):
    """Total LJ energy: 4*sum_{i<j}(r^-12 - r^-6)."""
    rij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(rij**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2**(-3)
    r12 = r6**2
    # Sum over all ordered pairs = 2 * sum over unordered pairs
    # Energy = 4 * sum_{i<j} (r^-12 - r^-6) = 2 * sum_{i!=j} (r^-12 - r^-6)
    return 2.0 * np.sum(r12 - r6)

def _lj_energy_grad(pos):
    """Gradient of LJ energy."""
    rij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(rij**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    # E = 2*sum_{i!=j}(r2^-6 - r2^-3)
    # dE/dx_i = 4*sum_{j!=i}(3*r2^-4 - 6*r2^-7)*(x_i - x_j)
    coeff = 4.0 * (3.0 * r2**(-4) - 6.0 * r2**(-7))
    np.fill_diagonal(coeff, 0.0)
    return np.sum(coeff[:, :, np.newaxis] * rij, axis=1)

def _get_mackay_directions():
    """Return (verts_unit (12,3), edge_mids_unit (30,3))."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms

    rij = verts_unit[:, np.newaxis, :] - verts_unit[np.newaxis, :, :]
    dists = np.sqrt(np.sum(rij**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    edge_mask = dists < min_dist * 1.001

    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            if edge_mask[i, j]:
                mid = (verts_unit[i] + verts_unit[j]) / 2.0
                edge_mids.append(mid / np.linalg.norm(mid))
    edge_mids = np.array(edge_mids)
    return verts_unit, edge_mids

_VERTS, _EDGES = _get_mackay_directions()

def _mackay_pos_3shell(r_in, r_edge, r_vert):
    """Build 55-atom Mackay icosahedron with three shell radii."""
    pts = [np.zeros(3)]
    for v in _VERTS:
        pts.append(r_in * v)
    for v in _EDGES:
        pts.append(r_edge * v)
    for v in _VERTS:
        pts.append(r_vert * v)
    return np.array(pts, dtype=float)

def _optimize_radii_3shell():
    """Optimize the three shell radii via Nelder-Mead."""
    def obj(p):
        ri, re, rv = p
        if ri < 0.3 or re < 0.5 or rv < 0.5:
            return 1e10
        if re < ri * 0.85 or rv < ri * 0.85:
            return 1e10
        return _lj_energy(_mackay_pos_3shell(ri, re, rv))

    r0 = np.array([1.054, 1.755, 2.000])
    res = minimize(obj, r0, method='Nelder-Mead',
                   options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14,
                            'adaptive': True})
    return res.x, res.fun

def _build_manifold_dirs():
    """
    Construct the 54 unit direction vectors for the non-center atoms.
    Order: 12 shell1 (verts), 30 shell2 (edges), 12 shell3 (verts).
    """
    dirs = np.vstack([_VERTS, _EDGES, _VERTS])
    return dirs

_DIRS = _build_manifold_dirs()

def _radial_to_pos(radii):
    """Convert 54 radial distances to 55 positions (center at 0)."""
    pos_noncenter = _DIRS * radii[:, np.newaxis]
    center = np.zeros((1, 3))
    return np.vstack([center, pos_noncenter])

def _radial_energy_grad(radii):
    """
    Energy and gradient w.r.t. radial parameters on the fixed-direction manifold.
    """
    pos = _radial_to_pos(radii)
    e = _lj_energy(pos)
    grad_pos = _lj_energy_grad(pos)
    # grad_radial[i] = dot(grad_pos[i+1], dir[i])
    # Center is fixed, so we only consider the 54 non-center atoms
    grad_rad = np.sum(grad_pos[1:] * _DIRS, axis=1)
    return e, grad_rad

def _optimize_radii_54d(r_init):
    """Optimize all 54 individual radial distances on the icosahedral manifold."""
    def obj_and_grad(x):
        return _radial_energy_grad(x)
    
    # Initial guess from 3-shell optimization
    x0 = np.zeros(54)
    x0[:12] = r_init[0]   # shell1
    x0[12:42] = r_init[1] # shell2
    x0[42:54] = r_init[2] # shell3
    
    try:
        res = minimize(obj_and_grad, x0, jac=True, method='L-BFGS-B',
                       bounds=[(0.5, 5.0)] * 54,
                       options={'maxiter': 2000, 'gtol': 1e-12, 'ftol': 1e-14,
                                'maxls': 50, 'iprint': -1})
        return res.x, res.fun
    except Exception:
        return x0, _lj_energy(_radial_to_pos(x0))

def _relax_cartesian(pos0, maxiter=5000, gtol=1e-12, ftol=1e-16):
    """L-BFGS-B Cartesian relaxation."""
    def ef(x):
        return _lj_energy(x.reshape(-1, 3))
    def gf(x):
        return _lj_energy_grad(x.reshape(-1, 3)).flatten()
    
    opts = {'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol,
            'maxls': 50, 'iprint': -1}
    res = minimize(ef, pos0.flatten(), jac=gf, method='L-BFGS-B', options=opts)
    return res.x.reshape(-1, 3), res.fun

def _basin_hopping(pos0, t0, deadline, rng):
    """Basin-hopping with rigid-body perturbations (translation + rotation)."""
    best_pos = pos0.copy()
    best_e = _lj_energy(pos0)
    current_pos = pos0.copy()
    current_e = best_e
    
    T = 0.5
    max_trans = 0.1
    max_rot = 0.05 # radians
    
    while time.time() - t0 < deadline - 5.0:
        # Random rigid-body displacement: translation + rotation
        disp = rng.standard_normal(3) * max_trans
        
        # Random rotation
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        angle = rng.standard_normal() * max_rot
        # Rodrigues rotation vector
        k = np.array([0, -axis[2], axis[1], axis[2], 0, -axis[0], -axis[1], axis[0], 0]).reshape(3,3)
        R = np.eye(3) + np.sin(angle)*k + (1-np.cos(angle))*(k@k)
        
        perturbed = current_pos @ R.T + disp
        
        # Relax to local minimum
        try:
            new_pos, new_e = _relax_cartesian(perturbed, maxiter=1000, gtol=1e-10, ftol=1e-14)
        except Exception:
            break
        
        # Check minimum distance constraint
        rij = new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :, :]
        r2 = np.sum(rij**2, axis=2)
        np.fill_diagonal(r2, np.inf)
        if np.min(r2) < 0.25:
            continue
        
        # Metropolis acceptance
        delta_e = new_e - current_e
        if delta_e < 0 or (T > 0 and rng.random() < np.exp(-delta_e / T)):
            current_pos = new_pos
            current_e = new_e
        
        if new_e < best_e:
            best_e = new_e
            best_pos = new_pos.copy()
        
        T *= 0.95
        
        if time.time() - t0 > deadline - 10:
            break
    
    return best_pos, best_e

def construct():
    t0 = time.time()
    rng = np.random.default_rng(_BH_SEED)

    # Step 1: Quick 3-shell radius optimization
    try:
        radii3, _ = _optimize_radii_3shell()
    except Exception:
        radii3 = np.array([1.054, 1.755, 2.000])
    
    pos_3shell = _mackay_pos_3shell(radii3[0], radii3[1], radii3[2])
    
    # Step 2: 54D Radial Manifold Optimization
    elapsed = time.time() - t0
    if elapsed < DEADLINE_SECONDS - 60:
        try:
            radii54, e_radial = _optimize_radii_54d(radii3)
            pos_manifold = _radial_to_pos(radii54)
        except Exception:
            pos_manifold = pos_3shell
    else:
        pos_manifold = pos_3shell
    
    # Step 3: Main Cartesian Relaxation from Manifold
    elapsed = time.time() - t0
    if elapsed < DEADLINE_SECONDS - 20:
        try:
            pos1, e1 = _relax_cartesian(pos_manifold, maxiter=3000, gtol=1e-12, ftol=1e-16)
        except Exception:
            pos1, e1 = pos_manifold, _lj_energy(pos_manifold)
    else:
        pos1, e1 = pos_manifold, _lj_energy(pos_manifold)
        
    best_pos = pos1
    best_e = e1

    # Step 4: Basin Hopping if time permits
    elapsed = time.time() - t0
    remaining = DEADLINE_SECONDS - elapsed - 10
    if remaining > 15:
        try:
            bh_pos, bh_e = _basin_hopping(pos1, t0, DEADLINE_SECONDS - 10, rng)
            if bh_e < best_e:
                best_e = bh_e
                best_pos = bh_pos
        except Exception:
            pass

    # Step 5: Final High-Precision Polish
    elapsed = time.time() - t0
    remaining = DEADLINE_SECONDS - elapsed - 5
    if remaining > 5:
        try:
            pos2, e2 = _relax_cartesian(best_pos, maxiter=2000, gtol=1e-14, ftol=1e-18)
            if e2 < best_e:
                best_e = e2
                best_pos = pos2
        except Exception:
            pass

    return best_pos
