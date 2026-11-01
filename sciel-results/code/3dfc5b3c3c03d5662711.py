"""Improved Lennard-Jones cluster global minimum, 55 atoms.
Combines Mackay icosahedron radius optimization with Basin-Hopping global search.
"""
import numpy as np
import time
from scipy.optimize import minimize, basinhopping
from scipy.spatial.distance import pdist

N = 55
_DEADLINE = 145.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _remaining():
    return _DEADLINE - _elapsed()

def _lj_energy(pos):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    d = pdist(pos)
    d = np.maximum(d, 1e-10)
    r6 = d ** -6
    return 4.0 * np.sum(r6 ** 2 - r6)

def _lj_energy_grad(pos):
    """
    Compute total LJ energy and gradient (vectorized).
    pos: (N, 3) array
    Returns: energy (float), grad (N, 3) array
    """
    # Pairwise differences: pos[i] - pos[j]
    # Shape: (N, N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    
    # Squared distances
    r2 = np.sum(diff ** 2, axis=2)
    
    # Avoid division by zero for i=j
    np.fill_diagonal(r2, 1.0)
    
    # We need r^-6 and r^-12 terms.
    # r2^-3 = r^-6
    r2_inv3 = r2 ** -3
    r2_inv6 = r2_inv3 ** 2
    
    # Energy: 4 * sum (r^-12 - r^-6)
    # Note: pdist sums unique pairs. Here we sum all i!=j pairs, so we divide by 2.
    # E = 0.5 * sum_{i!=j} 4 * (r2_inv6 - r2_inv3)
    e_pair = 4.0 * (r2_inv6 - r2_inv3)
    np.fill_diagonal(e_pair, 0.0)
    energy = 0.5 * np.sum(e_pair)
    
    # Gradient derivation:
    # E = sum_{i<j} 4(r_ij^-12 - r_ij^-6)
    # dE/dx_i = sum_{j!=i} 4(-12 r_ij^-14 + 6 r_ij^-8) (x_i - x_j)
    # Let r2 = |x_i - x_j|^2.
    # r^-14 = r2^-7
    # r^-8 = r2^-4
    # Factor = 4 * (-12 * r2^-7 + 6 * r2^-4)
    # grad_i = sum_{j!=i} Factor_ij * (x_i - x_j)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # Factor matrix (N, N)
        factor = 4.0 * (-12.0 * (r2 ** -7) + 6.0 * (r2 ** -4))
    
    np.fill_diagonal(factor, 0.0)
    
    # grad_i = sum_j factor_ij * (pos_i - pos_j)
    #        = pos_i * sum_j factor_ij - sum_j factor_ij * pos_j
    
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    # sum_j factor_ij * pos_j
    # factor is (N, N), pos is (N, 3)
    # result is (N, 3)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    
    grad = pos * sum_factor - sum_factor_pos
    
    return energy, grad

def _get_mackay_directions():
    """Get the 55 unit direction vectors for a Mackay icosahedron and shell assignments."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find edges of icosahedron (pairs at minimum distance)
    D = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            D[i, j] = np.linalg.norm(verts[i] - verts[j])
    min_dist = D.min()
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(D[i, j] - min_dist) < 1e-10:
                edges.append((i, j))
    
    # 30 edge midpoints (normalized)
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    
    # Assemble directions and shell IDs
    directions = np.zeros((55, 3))
    shell_ids = np.zeros(55, dtype=int)
    
    # Atom 0: center
    directions[0] = [0.0, 0.0, 0.0]
    shell_ids[0] = 0
    
    # Atoms 1-12: Shell 1 (icosahedron vertices)
    for i in range(12):
        directions[1 + i] = verts[i]
        shell_ids[1 + i] = 1
    
    # Atoms 13-42: Shell 2 (edge midpoints)
    for i in range(30):
        directions[13 + i] = midpoints[i]
        shell_ids[13 + i] = 2
    
    # Atoms 43-54: Shell 3 (icosahedron vertices, outer)
    for i in range(12):
        directions[43 + i] = verts[i]
        shell_ids[43 + i] = 3
    
    return directions, shell_ids

def _make_mackay_from_radii(radii):
    """Generate Mackay icosahedron positions given shell radii [r1, r2, r3]."""
    directions, shell_ids = _get_mackay_directions()
    pos = np.zeros((55, 3))
    for i in range(1, 55):
        shell = shell_ids[i]
        if shell == 1:
            pos[i] = directions[i] * radii[0]
        elif shell == 2:
            pos[i] = directions[i] * radii[1]
        elif shell == 3:
            pos[i] = directions[i] * radii[2]
    return pos

def _local_minimize(pos, maxiter=500, gtol=1e-10):
    """L-BFGS-B local relaxation."""
    if _remaining() < 2.0:
        return pos
    
    def fun(x):
        x = x.reshape(-1, 3)
        e, g = _lj_energy_grad(x)
        return e, g.ravel()
    
    x0 = pos.reshape(-1)
    try:
        result = minimize(fun, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': gtol})
        return result.x.reshape(-1, 3)
    except Exception:
        return pos

def _polish(positions, deadline=None):
    """High-precision polish pass with progressively tighter tolerances."""
    if deadline is None:
        deadline = _DEADLINE
    
    best_energy = _lj_energy(positions)
    best_pos = positions.copy()
    
    def fun(x):
        x = x.reshape(-1, 3)
        e, g = _lj_energy_grad(x)
        return e, g.ravel()
    
    x0 = positions.reshape(-1)
    
    # Multi-stage polish: progressively tighter
    stages = [
        {'maxiter': 2000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-14},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-18},
    ]
    
    for stage_opts in stages:
        if time.time() > deadline - 5.0:
            break
        try:
            result = minimize(fun, x0, jac=True, method='L-BFGS-B', options=stage_opts)
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass
    
    return best_pos

def _check_min_dist(pos, min_d=0.5):
    """Check all pair distances >= min_d."""
    d = pdist(pos)
    return d.min() >= min_d - 1e-9

def construct():
    global _start_time
    _start_time = time.time()
    
    best_pos = None
    best_energy = np.inf
    
    # --- Phase 1: Quick Mackay Radius Optimization ---
    # The Mackay icosahedron is very close to the global minimum.
    # Optimizing the radii gives a high-quality starting point for global search.
    
    radius_candidates = [
        np.array([1.0, 1.8, 2.6]),
        np.array([1.1, 1.9, 2.7]),
        np.array([0.9, 1.7, 2.5]),
        np.array([1.05, 1.85, 2.65]),
    ]
    
    def _optimize_radii_quick(radii_init):
        def objective(radii):
            if _remaining() < 5.0:
                return 1e10
            radii = np.maximum(radii, 0.5)
            pos = _make_mackay_from_radii(radii)
            pos_relaxed = _local_minimize(pos, maxiter=100, gtol=1e-8)
            return _lj_energy(pos_relaxed)
        
        try:
            res = minimize(objective, radii_init, method='Nelder-Mead',
                           options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
            best_radii = res.x
            best_pos = _make_mackay_from_radii(best_radii)
            best_pos = _local_minimize(best_pos, maxiter=200, gtol=1e-10)
            best_e = _lj_energy(best_pos)
            return best_pos, best_e
        except Exception:
            return None, np.inf

    for radii_init in radius_candidates:
        if _remaining() < 10.0:
            break
        pos, e = _optimize_radii_quick(radii_init)
        if pos is not None and _check_min_dist(pos) and e < best_energy:
            best_energy = e
            best_pos = pos.copy()
    
    if best_pos is None:
        # Fallback to default Mackay
        default_radii = np.array([1.12, 2.14, 3.16])
        best_pos = _make_mackay_from_radii(default_radii)
        best_pos = _local_minimize(best_pos, maxiter=300)
        best_energy = _lj_energy(best_pos)

    # --- Phase 2: Basin Hopping Global Search ---
    # Use basin hopping to explore the landscape around the best Mackay candidate.
    # This helps escape local minima if the radius optimization got stuck.
    
    def _basin_hopping_search(start_pos, maxiter=10, stepsize=0.5):
        if _remaining() < 10.0:
            return start_pos, _lj_energy(start_pos)
        
        def energy_func(x):
            pos = x.reshape(-1, 3)
            e, _ = _lj_energy_grad(pos)
            return e
        
        def accept_func(e_new, e_old, T):
            # Standard Metropolis acceptance
            if e_new < e_old:
                return True
            if T == 0:
                return False
            return np.random.rand() < np.exp(-(e_new - e_old) / T)
        
        def step(x):
            # Random Gaussian step
            return x + np.random.normal(0, stepsize, x.shape)
        
        try:
            res = basinhopping(
                energy_func, 
                start_pos.reshape(-1), 
                minimizer_kwargs={'jac': lambda x: _lj_energy_grad(x.reshape(-1,3))[1].ravel(), 'method': 'L-BFGS-B', 'options': {'maxiter': 100, 'gtol': 1e-8}},
                niter=maxiter,
                stepsize=stepsize,
                accept=accept_func,
                step=step,
                T=1.0,
                disp=False
            )
            final_pos = res.x.reshape(-1, 3)
            final_e = res.fun
            return final_pos, final_e
        except Exception:
            return start_pos, _lj_energy(start_pos)

    if _remaining() > 20.0:
        # Run basin hopping from the current best
        bh_pos, bh_e = _basin_hopping_search(best_pos, maxiter=15, stepsize=0.3)
        if _check_min_dist(bh_pos) and bh_e < best_energy:
            best_energy = bh_e
            best_pos = bh_pos.copy()
        
        # Try a few more diverse starts if time permits
        if _remaining() > 15.0:
            rng = np.random.default_rng(123)
            for _ in range(2):
                if _remaining() < 10.0:
                    break
                # Perturb the best Mackay
                perturbed = best_pos + rng.normal(0, 0.2, best_pos.shape)
                if _check_min_dist(perturbed):
                    bh_pos2, bh_e2 = _basin_hopping_search(perturbed, maxiter=5, stepsize=0.2)
                    if _check_min_dist(bh_pos2) and bh_e2 < best_energy:
                        best_energy = bh_e2
                        best_pos = bh_pos2.copy()

    # --- Phase 3: Final Polish ---
    if _remaining() > 5.0:
        polished = _polish(best_pos, deadline=_DEADLINE)
        e = _lj_energy(polished)
        if _check_min_dist(polished) and e < best_energy:
            best_energy = e
            best_pos = polished.copy()
    
    return best_pos
