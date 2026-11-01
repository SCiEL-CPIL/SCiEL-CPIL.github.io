"""Lennard-Jones 55-atom global minimum: Mackay icosahedron seed + L-BFGS relaxation."""
import numpy as np
import time
import sys

N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time

def _lj_energy(pos):
    """Compute total LJ energy for a set of positions (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    # Avoid division by zero or negative if r2 is 0
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized, numerically stable)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0) # Avoid div by 0 in calculation, handled later
    
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    # Gradient calculation
    # dE/dr2 = 4 * (-6/r2^4 + 12/r2^7)
    # dE/dpos_i = sum_j dE/dr2_ij * 2*(pos_i - pos_j)
    # Let's use the explicit form:
    # grad_i = sum_j 4 * (-12/r2^7 + 6/r2^4) * (pos_i - pos_j)
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(factor, 0.0)
    
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos
    
    return energy, grad

def _get_mackay_icosahedron_coords():
    """
    Returns unit vectors for the 55 positions in a Mackay icosahedron.
    1 center (0,0,0)
    12 first shell (icosahedron vertices)
    30 second shell (icosahedron edge midpoints)
    12 third shell (icosahedron vertices, same direction as first shell)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges of the icosahedron
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(verts_unit[i] - verts_unit[j])
    
    min_dist = np.min(dists[dists > 1e-6])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    
    # Edge midpoints (30 atoms)
    edge_mids = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)
    
    # Construct full unit vectors
    # Order: Center, Shell1 (12), Shell2 (30), Shell3 (12)
    center = np.array([[0.0, 0.0, 0.0]])
    shell1_dir = verts_unit
    shell2_dir = edge_mids
    shell3_dir = verts_unit # Same directions as shell 1
    
    return center, shell1_dir, shell2_dir, shell3_dir

def _make_mackay_icosahedron(r1=1.05, r2=1.75, r3=1.95):
    """Construct a two-shell Mackay icosahedron for 55 atoms."""
    center, s1, s2, s3 = _get_mackay_icosahedron_coords()
    
    first_shell = s1 * r1
    second_inner = s2 * r2
    second_outer = s3 * r3
    
    all_pos = np.vstack([center, first_shell, second_inner, second_outer])
    return all_pos

def _rotation_matrix_from_angles(angles):
    """Create a rotation matrix from Euler angles (xyz convention)."""
    ax, ay, az = angles
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def _manifold_energy(params, base_dirs):
    """
    Compute energy on the parameterized manifold.
    params: [r1, r2, r3, tx, ty, tz, ax, ay, az]
    base_dirs: (center, s1, s2, s3)
    """
    r1, r2, r3 = params[0], params[1], params[2]
    t = params[3:6]
    rot_angles = params[6:9]
    
    center, s1, s2, s3 = base_dirs
    
    R = _rotation_matrix_from_angles(rot_angles)
    
    first_shell = (s1 * r1) @ R.T + t
    second_inner = (s2 * r2) @ R.T + t
    second_outer = (s3 * r3) @ R.T + t
    
    all_pos = np.vstack([np.array([[t[0], t[1], t[2]]]), first_shell, second_inner, second_outer])
    
    return _lj_energy(all_pos)

def _manifold_energy_grad(params, base_dirs):
    """Compute energy and gradient on the parameterized manifold."""
    n_params = len(params)
    e = _manifold_energy(params, base_dirs)
    g = np.zeros(n_params)
    eps = 1e-7
    for i in range(n_params):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        g[i] = (_manifold_energy(p_plus, base_dirs) - _manifold_energy(p_minus, base_dirs)) / (2 * eps)
    return e, g

def _optimize_manifold(base_dirs, initial_params, deadline, maxiter=2000):
    """Optimize the manifold parameters using L-BFGS-B."""
    from scipy.optimize import minimize
    
    best_energy = _manifold_energy(initial_params, base_dirs)
    best_params = initial_params.copy()
    
    def obj_and_grad(p):
        return _manifold_energy_grad(p, base_dirs)
    
    try:
        result = minimize(
            obj_and_grad, initial_params, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': 1e-10, 'maxls': 100}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_params = result.x
    except Exception:
        pass
    
    return best_params, best_energy

def _params_to_positions(params, base_dirs):
    """Convert manifold parameters to Cartesian positions."""
    r1, r2, r3 = params[0], params[1], params[2]
    t = params[3:6]
    rot_angles = params[6:9]
    
    center, s1, s2, s3 = base_dirs
    
    R = _rotation_matrix_from_angles(rot_angles)
    
    first_shell = (s1 * r1) @ R.T + t
    second_inner = (s2 * r2) @ R.T + t
    second_outer = (s3 * r3) @ R.T + t
    
    all_pos = np.vstack([np.array([[t[0], t[1], t[2]]]), first_shell, second_inner, second_outer])
    return all_pos

def _optimize_full_structure(pos0, deadline, maxiter=5000, gtol=1e-12):
    """
    Optimize the full 3D structure using L-BFGS-B with tight tolerance.
    This allows the structure to deviate from perfect icosahedral symmetry
    while still converging to a high-precision local minimum.
    """
    from scipy.optimize import minimize
    
    best_energy = _lj_energy(pos0)
    best_pos = pos0.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos0.flatten()
    
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200, 'ftol': 1e-15}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def _relax_lbfgs(pos, deadline, maxiter=1000, gtol=1e-10):
    """Relax using L-BFGS-B with deadline guard."""
    from scipy.optimize import minimize
    
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
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200, 'ftol': 1e-15}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def construct():
    """Construct 55-atom LJ cluster near global minimum using manifold optimization + Cartesian relaxation."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Get base icosahedral directions
    base_dirs = _get_mackay_icosahedron_coords()
    
    # Initial manifold parameters: [r1, r2, r3, tx, ty, tz, ax, ay, az]
    # Literature-optimal radii for Mackay icosahedron (closer to true minimum)
    initial_params = np.array([1.135, 1.162, 1.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # 1. Optimize on the manifold (low-dimensional, respects symmetry)
    if time.time() < deadline - 80.0:
        params_optimized, energy_optimized = _optimize_manifold(base_dirs, initial_params, deadline, maxiter=3000)
        
        # Convert to Cartesian
        pos_from_manifold = _params_to_positions(params_optimized, base_dirs)
        
        # Check minimum distance
        diff = pos_from_manifold[:, np.newaxis, :] - pos_from_manifold[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist >= 0.5:
            best_pos = pos_from_manifold.copy()
            best_energy = _lj_energy(best_pos)
        else:
            # Fallback to standard approach
            best_pos = _make_mackay_icosahedron(1.05, 1.75, 1.95)
            best_energy = _lj_energy(best_pos)
    else:
        best_pos = _make_mackay_icosahedron(1.05, 1.75, 1.95)
        best_energy = _lj_energy(best_pos)
    
    # 2. Full 3D relaxation to break symmetry and find local minimum
    if time.time() < deadline - 40.0:
        pos_relaxed, energy_relaxed = _relax_lbfgs(best_pos, deadline, maxiter=10000, gtol=1e-14)
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed
    
    # 3. Multi-start with small perturbations around the best found
    rng = np.random.default_rng(42)
    polish_time = 15.0
    search_deadline = deadline - polish_time
    
    for scale in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        if time.time() >= search_deadline - 10.0:
            break
        
        n_perturbations = 4 if scale < 0.05 else (3 if scale < 0.2 else 2)
        
        for _ in range(n_perturbations):
            if time.time() >= search_deadline - 5.0:
                break
            
            noise = rng.normal(0, scale, size=best_pos.shape)
            perturbed_pos = best_pos + noise
            
            # Check minimum distance constraint
            diff = perturbed_pos[:, np.newaxis, :] - perturbed_pos[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            
            if min_dist < 0.5:
                scale_factor = 0.5 / min_dist * 1.01
                perturbed_pos = perturbed_pos * scale_factor
            
            pos_relaxed, energy_relaxed = _relax_lbfgs(perturbed_pos, search_deadline, maxiter=3000, gtol=1e-12)
            
            if energy_relaxed < best_energy:
                best_energy = energy_relaxed
                best_pos = pos_relaxed
    
    # 4. Final high-precision polish with very tight tolerance
    if time.time() < deadline - 5.0:
        pos_polished, energy_polished = _optimize_full_structure(best_pos, deadline, maxiter=20000, gtol=1e-16)
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_pos = pos_polished
        
        # Second polish pass if time permits
        if time.time() < deadline - 2.0:
            pos_polished2, energy_polished2 = _optimize_full_structure(best_pos, deadline, maxiter=5000, gtol=1e-18)
            if energy_polished2 < best_energy:
                best_energy = energy_polished2
                best_pos = pos_polished2
    
    # Final Check: Minimum distance constraint
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < 0.5:
        scale = 0.5 / min_dist_found * 1.01
        best_pos = best_pos * scale
    
    return best_pos
# EVOLVE-BLOCK-END
