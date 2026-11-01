"""Lennard-Jones cluster global minimum, 55 atoms.
Strategy: Build the Mackay icosahedron (two-shell) as the initial guess,
then relax with L-BFGS to find the local minimum near the known global minimum.
"""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 120.0  # wall-clock limit with safety margin
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _remaining():
    return _DEADLINE - _elapsed()


def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for a set of positions.
    
    pos: (N, 3) array of positions.
    Returns: (energy, grad) where grad is (N, 3).
    """
    n = pos.shape[0]
    # Pairwise displacements: (n, n, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # r_i - r_j
    r2 = np.sum(diff ** 2, axis=2)  # (n, n)
    # Avoid division by zero on diagonal
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3  # r^6
    r12 = r6 ** 2  # r^12
    
    # Energy: 4 * sum_{i<j} (r^-12 - r^-6)
    # Using full matrix and dividing by 2
    inv_r6 = 1.0 / r6
    inv_r12 = inv_r6 ** 2
    pair_energy = 4.0 * (inv_r12 - inv_r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)
    
    # Gradient: dE/dr_i = sum_{j != i} 4 * (-12/r^13 + 6/r^7) * (r_i - r_j)/r
    # = sum_{j != i} 4 * (6/r^7 - 12/r^13) * (r_i - r_j) / r^2 * r^2 / r... 
    # Actually: dE/dr_i = sum_j 4 * (-12 r^-14 + 6 r^-8) * (r_i - r_j)
    # Let's be more careful.
    # E = sum_{i<j} 4*(r_ij^-12 - r_ij^-6)
    # dE/dr_i = sum_{j != i} 4 * (-12 r_ij^-14 + 6 r_ij^-8) * (r_i - r_j)
    # where r_ij = |r_i - r_j|
    
    # r_ij = sqrt(r2_ij)
    # r^-14 = r^-12 / r^2 = inv_r12 / r2
    # r^-8 = r^-6 / r^2 = inv_r6 / r2
    
    coeff = 4.0 * (-12.0 * inv_r12 / r2 + 6.0 * inv_r6 / r2)  # (n, n)
    np.fill_diagonal(coeff, 0.0)
    # grad_i = sum_j coeff_{ij} * (r_i - r_j)
    # diff is (n, n, 3) where diff[i,j] = r_i - r_j
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad


def _build_mackay_icosahedron():
    """Build the two-shell Mackay icosahedron for 55 atoms.
    
    Shell 1: 12 atoms at vertices of a regular icosahedron (radius ~1.0)
    Shell 2: 42 atoms at vertices of a truncated icosahedron-like structure
    
    Actually, the Mackay icosahedron has:
    - Inner shell: 12 atoms (icosahedron vertices)
    - Outer shell: 42 atoms (truncated icosahedron vertices)
    
    Let me use known coordinates. The 12 vertices of a regular icosahedron
    can be given by cyclic permutations of (0, ±1, ±phi) where phi = golden ratio.
    
    For the outer shell of 42 atoms, I'll use the vertices of a truncated icosahedron
    scaled appropriately, or better, use the known Mackay structure coordinates.
    
    Let me use a simpler approach: generate the icosahedron vertices, then for
    the outer shell, place atoms at the centers of faces and edges appropriately.
    
    Actually, let me just use well-known approximate coordinates and let relaxation
    fix them up.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio ~1.618
    
    # 12 vertices of regular icosahedron (cyclic perms of (0, ±1, ±phi))
    inner = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    
    # Normalize to unit radius
    inner_norms = np.linalg.norm(inner, axis=1)
    inner = inner / inner_norms[:, np.newaxis]
    
    # For the outer shell (42 atoms), use the vertices of a truncated icosahedron.
    # A truncated icosahedron has 60 vertices. We need 42, so this isn't quite right.
    # 
    # The Mackay icosahedron outer shell has 42 atoms. Let me think about this differently.
    # 
    # Actually, for n=55, the Mackay icosahedron structure is:
    # - 1 atom at center? No, Mackay clusters are hollow.
    # - The two-shell Mackay icosahedron for n=55 has 12 inner + 42 outer.
    #
    # Let me use a different approach: generate points on a sphere with icosahedral symmetry.
    # 
    # Alternative: Use the known fact that the 42 outer atoms form a structure related
    # to the icosahedron. Let me just place 42 points approximately on a larger sphere
    # with icosahedral symmetry and let relaxation fix it.
    #
    # Simpler approach: Use the 12 icosahedron vertices scaled up for outer shell,
    # plus add more points. But we need exactly 42 outer points.
    #
    # Let me use a practical approach: create 42 points by taking the icosahedron
    # vertices, edge midpoints, and face centers.
    # 
    # Icosahedron has: 12 vertices, 30 edges, 20 faces.
    # 12 + 30 = 42! So the outer shell = 12 vertices + 30 edge midpoints (scaled).
    
    # Get the 30 edges of the icosahedron
    # An edge connects two vertices at the minimum distance
    min_dist = np.inf
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(inner[i] - inner[j])
            if d < min_dist - 1e-10:
                min_dist = d
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(inner[i] - inner[j])
            if abs(d - min_dist) < 1e-10:
                edges.append((i, j))
    
    # Outer shell: 12 vertices at larger radius + 30 edge midpoints at larger radius
    # The ratio of outer to inner radius in Mackay icosahedron is approximately 2.0
    # Let me use a ratio that gives reasonable initial spacing
    
    r_inner = 1.0
    r_outer = 2.0
    
    outer_vertices = inner * r_outer
    outer_edge_mids = []
    for i, j in edges:
        mid = (inner[i] + inner[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        mid_normalized = mid / mid_norm
        outer_edge_mids.append(mid_normalized * r_outer)
    
    outer = np.vstack([outer_vertices] + [np.array(m) for m in outer_edge_mids])
    
    # Combine inner and outer shells
    all_points = np.vstack([inner * r_inner, outer])
    
    return all_points


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Build initial Mackay icosahedron guess
    pos = _build_mackay_icosahedron()
    
    # Ensure minimum distance constraint (0.5 apart)
    # Check and adjust if needed
    for _ in range(100):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        if min_dist >= 0.5:
            break
        # Scale up slightly
        pos *= 1.01
    
    # Relax using L-BFGS
    def func_and_grad(x):
        p = x.reshape(N, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()
    
    x0 = pos.ravel()
    e0, g0 = func_and_grad(x0)
    
    best_pos = pos.copy()
    best_energy = e0
    
    # Try L-BFGS-B with constraints
    try:
        result = minimize(
            func_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 5000,
                'maxfun': 5000,
                'ftol': 1e-15,
                'gtol': 1e-15,
                'maxls': 50,
                'maxcor': 20,
                'iprint': 0,
            }
        )
        if result.success and result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(N, 3)
    except Exception:
        pass
    
    # Check time and do additional polish if time permits
    if _remaining() > 10:
        try:
            result2 = minimize(
                func_and_grad,
                best_pos.ravel(),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 10000,
                    'maxfun': 10000,
                    'ftol': 1e-18,
                    'gtol': 1e-18,
                    'maxls': 100,
                    'maxcor': 30,
                    'iprint': 0,
                }
            )
            if result2.fun < best_energy:
                best_energy = result2.fun
                best_pos = result2.x.reshape(N, 3)
        except Exception:
            pass
    
    # Final check: ensure minimum distance
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 0.5:
        # Scale to fix
        min_d = np.min(dists)
        scale = 0.5 / min_d * 1.001
        best_pos *= scale
    
    return best_pos
# EVOLVE-BLOCK-END
