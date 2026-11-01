"""Lennard-Jones 55-atom cluster: Mackay icosahedron with per-atom radial manifold + full relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55
LJ_MIN = 2.0 ** (1.0 / 6.0)  # 1.122462... pair-potential minimum distance

# EVOLVE-BLOCK-START

def lj_energy(pos):
    """Total LJ energy 4*sum_{i<j}(r^-12 - r^-6)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    return 2.0 * (r12 - r6).sum()

def lj_energy_grad(pos):
    """LJ energy and gradient (each pair counted once)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 2.0 * (r12 - r6).sum()
    coeff = 24.0 * (2.0 * r2 ** (-7) - r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad

def _get_icosahedral_directions():
    """
    Returns:
        verts: (12, 3) array of unit vectors to icosahedron vertices.
        edge_mids: (30, 3) array of unit vectors to icosahedron edge midpoints.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms

    # Find edges: pairs of vertices with minimum distance
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            dists[i, j] = d
            dists[j, i] = d
    
    triu_indices = np.triu_indices(12, k=1)
    triu_dists = dists[triu_indices]
    sorted_indices = np.argsort(triu_dists)
    edge_pairs = list(zip(triu_indices[0][sorted_indices[:30]], triu_indices[1][sorted_indices[:30]]))
    
    edge_mids = []
    for i, j in edge_pairs:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        edge_mids.append(mid)
    
    edge_mids = np.array(edge_mids)
    return verts, edge_mids


def _construct_mackay_from_params(params, verts, edge_mids):
    """
    Construct the 55-atom Mackay icosahedron from parameters.
    params: array of shape (3,) or (54,)
    If shape (3,): [r1, r2, r3] — simple radial manifold
    If shape (54,): per-atom radial offsets from base Mackay geometry
    """
    if len(params) == 3:
        r1, r2, r3 = params
        center = np.zeros((1, 3))
        shell1 = verts * r1
        shell2 = edge_mids * r2
        shell3 = verts * r3
        all_pos = np.vstack([center, shell1, shell2, shell3])
        return all_pos
    else:
        # Per-atom radial parameters
        base_r1 = 1.12
        base_r2 = 2.24
        base_r3 = 3.36
        
        inner_scales = 1.0 + params[0:12]
        mid_scales = 1.0 + params[12:42]
        outer_scales = 1.0 + params[42:54]
        
        center = np.zeros((1, 3))
        
        shell1_radii = base_r1 * inner_scales
        shell1 = (verts * shell1_radii[:, np.newaxis])
        
        shell2_radii = base_r2 * mid_scales
        shell2 = (edge_mids * shell2_radii[:, np.newaxis])
        
        shell3_radii = base_r3 * outer_scales
        shell3 = (verts * shell3_radii[:, np.newaxis])
        
        all_pos = np.vstack([center, shell1, shell2, shell3])
        return all_pos


def _radial_energy_and_grad(params, verts, edge_mids):
    """
    Compute energy and gradient with respect to parameters.
    params: shape (3,) for simple radial or (54,) for per-atom radial.
    """
    pos = _construct_mackay_from_params(params, verts, edge_mids)
    energy, grad_cart = _lj_energy_grad(pos)
    
    if len(params) == 3:
        grad_r1 = np.sum(grad_cart[1:13] * verts)
        grad_r2 = np.sum(grad_cart[13:43] * edge_mids)
        grad_r3 = np.sum(grad_cart[43:55] * verts)
        grad_params = np.array([grad_r1, grad_r2, grad_r3])
    else:
        base_r1 = 1.12
        base_r2 = 2.24
        base_r3 = 3.36
        
        grad_params = np.zeros(54)
        
        for i in range(12):
            grad_params[i] = np.dot(grad_cart[1+i], base_r1 * verts[i])
        
        for i in range(30):
            grad_params[12+i] = np.dot(grad_cart[13+i], base_r2 * edge_mids[i])
        
        for i in range(12):
            grad_params[42+i] = np.dot(grad_cart[43+i], base_r3 * verts[i])
    
    return energy, grad_params


def _valid(pos):
    """Check minimum-pair-distance >= 0.5 and finiteness."""
    if not np.all(np.isfinite(pos)):
        return False
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1e18)
    return np.min(r2) >= 0.5 ** 2

def try_relax(positions, deadline):
    """L-BFGS relaxation with a hard deadline. Returns (pos, energy)."""
    best_energy, best_pos = lj_energy(positions), positions.copy()
    try:
        res = minimize(
            lj_energy_grad, positions, method='L-BFGS-B',
            options={'maxiter': 4000, 'ftol': 1e-12, 'gtol': 1e-9,
                     'maxfun': 8000, 'maxls': 20},
        )
        if np.all(np.isfinite(res.fun)) and res.fun < best_energy:
            best_energy, best_pos = res.fun, res.x.copy()
    except Exception:
        pass
    return best_pos, best_energy

def construct():
    """Return Cartesian coordinates (N,3) of the 55-atom LJ cluster."""
    start = time.time()
    deadline = start + 135.0   # hard wall-clock guard

    def remaining():
        return deadline - time.time()

    best_pos, best_energy = None, np.inf
    rng = np.random.default_rng(42)

    def consider(pos):
        nonlocal best_pos, best_energy
        if pos is None or not _valid(pos):
            return
        pos, e = try_relax(pos, deadline)
        if e < best_energy:
            best_energy, best_pos = e, pos.copy()

    # Get icosahedral directions
    verts, edge_mids = _get_icosahedral_directions()

    # Phase 1: Optimize simple 3-parameter radial manifold
    r0 = np.array([1.12, 2.24, 3.36])
    try:
        res_radial = minimize(
            _radial_energy_and_grad,
            r0,
            args=(verts, edge_mids),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-12, 'maxls': 50}
        )
        best_radii = res_radial.x
    except Exception:
        best_radii = r0
    
    # Construct the best symmetric configuration
    symmetric_pos = _construct_mackay_from_params(best_radii, verts, edge_mids)
    consider(symmetric_pos)

    # Phase 2: Expand to per-atom radial manifold (54 params)
    if remaining() > 20.0:
        params0 = np.zeros(54)
        try:
            res_expanded = minimize(
                _radial_energy_and_grad,
                params0,
                args=(verts, edge_mids),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-12, 'maxls': 100}
            )
            expanded_pos = _construct_mackay_from_params(res_expanded.x, verts, edge_mids)
            consider(expanded_pos)
        except Exception:
            pass

    # Phase 3: Full 3D relaxation from the best radial solution
    if remaining() > 15.0 and best_pos is not None:
        consider(best_pos)

    # Phase 4: Random perturbations to escape local minima
    for amp in (0.05, 0.15, 0.30):
        if remaining() < 18.0:
            break
        try:
            base = best_pos if best_pos is not None else symmetric_pos
            pert = base + rng.normal(0.0, amp, base.shape)
            consider(pert)
        except Exception:
            pass

    if best_pos is not None:
        return best_pos
    
    # Fallback: return the symmetric Mackay
    return symmetric_pos

# EVOLVE-BLOCK-END
