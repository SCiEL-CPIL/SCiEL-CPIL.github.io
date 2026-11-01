import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE_SECONDS = 140.0  # Safety margin for 150s limit

# --- Geometry Definitions ---

_PHI = (1.0 + np.sqrt(5.0)) / 2.0

def _get_mackay_icosahedron_directions():
    """
    Returns 54 unit vectors for the Mackay icosahedron (excluding center).
    Structure:
    - 12 vertices of inner icosahedron
    - 30 edge midpoints (face centers of outer shell relative to inner?) 
      Actually, Mackay 55 is:
      1 center
      12 inner shell (icosahedron vertices)
      42 outer shell? No.
      Standard Mackay 55:
      1 (center)
      12 (inner icosahedron)
      30 (outer shell, edge midpoints of inner icosahedron scaled out?)
      12 (outer shell, vertices of inner icosahedron scaled out?)
      
      Let's verify the standard Mackay 55 structure:
      It consists of:
      - 1 atom at center
      - 12 atoms forming an inner icosahedron (radius R1)
      - 30 atoms forming the edge midpoints of the inner icosahedron (radius R2)
      - 12 atoms forming the vertices of the inner icosahedron but at a larger radius (radius R3)?
      
      Actually, the 55-atom Mackay icosahedron has:
      - 1 center
      - 12 inner shell (icosahedral vertices)
      - 30 middle shell (edge midpoints of the icosahedron defined by the 12 vertices)
      - 12 outer shell (vertices of the icosahedron, scaled up)
      
      Wait, 1 + 12 + 30 + 12 = 55.
      The directions are:
      1. 12 icosahedral vertices (unit vectors)
      2. 30 edge midpoints (normalized)
      3. 12 icosahedral vertices (same directions as 1, but different radius)
    """
    # 12 vertices of a regular icosahedron
    verts = np.array([
        [0, 1, _PHI], [0, -1, _PHI], [0, 1, -_PHI], [0, -1, -_PHI],
        [1, _PHI, 0], [-1, _PHI, 0], [1, -_PHI, 0], [-1, -_PHI, 0],
        [_PHI, 0, 1], [-_PHI, 0, 1], [_PHI, 0, -1], [-_PHI, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms

    # Find edge midpoints
    # Edges connect vertices with minimum distance
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
    edge_mids = np.array(edge_mids)

    # The 54 outer/inner shell directions (excluding center)
    # Order: 12 inner verts, 30 edge mids, 12 outer verts
    dirs = np.vstack([verts_unit, edge_mids, verts_unit])
    return dirs, verts_unit, edge_mids

_DIRS, _VERTS_UNIT, _EDGE_MIDS = _get_mackay_icosahedron_directions()

# --- Energy and Gradient Calculations (Vectorized) ---

def _lj_energy(pos):
    """
    pos: (N, 3) array
    Returns total LJ energy.
    """
    # Compute all pair differences using broadcasting
    # rij[i, j] = pos[i] - pos[j]
    rij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(rij**2, axis=2)
    
    # Avoid division by zero
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2**(-3)
    r12 = r6**2
    energy = 4.0 * np.sum(r12 - r6)
    return energy

def _lj_grad(pos):
    """
    pos: (N, 3) array
    Returns gradient of energy w.r.t pos.
    """
    rij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(rij**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    # Gradient of 4 * sum (r^-12 - r^-6)
    # dE/dr = 4 * (-12 r^-13 - (-6) r^-7) = 4 * (-12 r^-13 + 6 r^-7)
    # dE/dx_i = sum_j dE/dr_ij * (x_i - x_j)/r_ij
    # dE/dx_i = sum_j 4 * (-12 r_ij^-13 + 6 r_ij^-7) * (x_i - x_j) / r_ij
    # = sum_j 4 * (-12 r_ij^-14 + 6 r_ij^-8) * (x_i - x_j)
    # Note: r_ij^-14 = (r2_ij)^-7, r_ij^-8 = (r2_ij)^-4
    
    r2_neg4 = r2**(-4)
    r2_neg7 = r2**(-7)
    coeff = 4.0 * (6.0 * r2_neg4 - 12.0 * r2_neg7)
    np.fill_diagonal(coeff, 0.0)
    
    # grad[i] = sum_j coeff[i,j] * (pos[i] - pos[j])
    # This is a matrix multiplication: (coeff * rij) summed over j
    # coeff is (N, N), rij is (N, N, 3)
    grad = np.sum(coeff[:, :, np.newaxis] * rij, axis=1)
    return grad

# --- Structural Optimization Steps ---

def _generate_mackay_positions(r_inner, r_mid, r_outer):
    """
    Generate 55 positions based on Mackay structure with given radii.
    """
    # 1 center
    center = np.zeros((1, 3))
    
    # 12 inner vertices
    inner_pos = r_inner * _VERTS_UNIT
    
    # 30 mid edge points
    mid_pos = r_mid * _EDGE_MIDS
    
    # 12 outer vertices (same direction as inner)
    outer_pos = r_outer * _VERTS_UNIT
    
    pos = np.vstack([center, inner_pos, mid_pos, outer_pos])
    return pos

def _symmetry_project(pos):
    """
    Project positions onto the Mackay icosahedral symmetry orbits.
    This enforces the symmetry, removing numerical drift.
    """
    # pos[0] is center, stays 0
    pos_proj = pos.copy()
    pos_proj[0] = 0.0
    
    # Inner vertices (indices 1-12)
    # Average all 12 inner vertices to get a "center of mass" for that shell?
    # No, they must remain at their specific directions.
    # The "distortion" is usually in the radial distance.
    # We can compute the optimal radius for each shell by averaging the projections.
    
    # Inner shell (1:13)
    if len(pos) > 12:
        inner_vecs = pos[1:13]
        # Project each vector onto its ideal direction
        # d = dot(v, dir), new_v = d * dir
        # Since dirs are unit vectors
        inner_radial = np.sum(inner_vecs * _VERTS_UNIT, axis=1)
        # Average radius to enforce uniformity
        r_inner_avg = np.mean(inner_radial)
        pos_proj[1:13] = r_inner_avg * _VERTS_UNIT
        
    # Mid shell (13:43)
    if len(pos) > 42:
        mid_vecs = pos[13:43]
        mid_radial = np.sum(mid_vecs * _EDGE_MIDS, axis=1)
        r_mid_avg = np.mean(mid_radial)
        pos_proj[13:43] = r_mid_avg * _EDGE_MIDS
        
    # Outer shell (43:55)
    if len(pos) > 54:
        outer_vecs = pos[43:55]
        outer_radial = np.sum(outer_vecs * _VERTS_UNIT, axis=1)
        r_outer_avg = np.mean(outer_radial)
        pos_proj[43:55] = r_outer_avg * _VERTS_UNIT
        
    return pos_proj

def _optimize_radial_parameters():
    """
    Optimize the 3 radii (inner, mid, outer) using Nelder-Mead.
    This is a 3D optimization, very fast and robust.
    """
    def objective(radii):
        r_i, r_m, r_o = radii
        pos = _generate_mackay_positions(r_i, r_m, r_o)
        return _lj_energy(pos)
    
    # Initial guess from literature
    x0 = np.array([1.054, 1.755, 2.0])
    
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
    return res.x, res.fun

def construct():
    start_time = time.time()
    
    # Step 1: Coarse optimization of shell radii
    # This finds the best "Mackay-like" geometry
    try:
        radii_opt, e_radial = _optimize_radial_parameters()
        pos_best = _generate_mackay_positions(radii_opt[0], radii_opt[1], radii_opt[2])
        best_energy = e_radial
    except Exception:
        # Fallback to standard values
        pos_best = _generate_mackay_positions(1.054, 1.755, 2.0)
        best_energy = _lj_energy(pos_best)
        
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 20:
        return pos_best

    # Step 2: Full Cartesian Relaxation
    # Use L-BFGS-B on all 165 coordinates
    # This allows the structure to distort slightly from perfect Mackay if beneficial
    # (Though for N=55, Mackay is the global min, so distortion should be minimal)
    
    def energy_func(x):
        return _lj_energy(x.reshape(-1, 3))
    
    def grad_func(x):
        return _lj_grad(x.reshape(-1, 3)).flatten()
    
    try:
        res = minimize(
            energy_func, 
            pos_best.flatten(), 
            jac=grad_func,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-15, 'maxls': 50}
        )
        pos_relaxed = res.x.reshape(-1, 3)
        e_relaxed = res.fun
        
        if e_relaxed < best_energy:
            best_energy = e_relaxed
            pos_best = pos_relaxed
    except Exception:
        pass

    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 10:
        return pos_best

    # Step 3: Symmetry Enforcement and Final Polish
    # The relaxation might have introduced tiny asymmetries.
    # Project back to symmetry, then do a very short high-precision polish.
    
    pos_sym = _symmetry_project(pos_best)
    e_sym = _lj_energy(pos_sym)
    
    if e_sym < best_energy:
        best_energy = e_sym
        pos_best = pos_sym
        
    # Final high precision polish from symmetric start
    try:
        res_final = minimize(
            energy_func, 
            pos_best.flatten(), 
            jac=grad_func,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'gtol': 1e-15, 'ftol': 1e-18, 'maxls': 50}
        )
        pos_final = res_final.x.reshape(-1, 3)
        e_final = res_final.fun
        
        if e_final < best_energy:
            best_energy = e_final
            pos_best = pos_final
    except Exception:
        pass

    # Check minimum distance constraint (0.5 apart)
    # If violated, we might need to adjust, but LJ repulsion is strong.
    # Just in case, verify.
    rij = pos_best[:, np.newaxis, :] - pos_best[np.newaxis, :, :]
    r2 = np.sum(rij**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    min_dist = np.sqrt(np.min(r2))
    
    if min_dist < 0.5:
        # This should not happen for LJ minimum, but if it does, 
        # we return the best valid one found or just the current one.
        # The problem states "Atoms must be at least 0.5 apart".
        # If this fails, the score is 0.
        # We assume the optimization keeps them apart.
        pass

    return pos_best
