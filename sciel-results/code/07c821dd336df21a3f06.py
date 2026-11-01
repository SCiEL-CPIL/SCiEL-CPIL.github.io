"""Improved Lennard-Jones cluster global minimum, 55 atoms."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist

# Deadline guard: 150s wall-clock limit, use 140s for safety
_DEADLINE = 140.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _lj_energy_vec(pos):
    """Compute total Lennard-Jones energy using vectorized operations."""
    pos = np.asarray(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    r2 = np.sum(diff * diff, axis=2)  # (N, N)
    mask = ~np.eye(len(pos), dtype=bool)
    r2 = np.where(mask, r2, 1e10)
    
    # Avoid division by zero or negative r2 (should not happen)
    r2 = np.maximum(r2, 1e-10)
    
    r6 = r2 ** -3
    r12 = r6 ** 2
    
    e = 4.0 * np.sum(r12 - r6) / 2.0
    return e


def _lj_grad_vec(pos):
    """Compute gradient of LJ energy using vectorized operations."""
    pos = np.asarray(pos)
    N = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    r2 = np.sum(diff * diff, axis=2)  # (N, N)
    
    mask = ~np.eye(N, dtype=bool)
    r2_safe = np.where(mask, r2, 1.0)
    r2_safe = np.maximum(r2_safe, 1e-10)
    
    r6 = r2_safe ** -3
    r12 = r6 ** 2
    
    # Gradient of 4*(r^-12 - r^-6) w.r.t position i
    # d/dr (4*(r^-12 - r^-6)) = 4*(-12*r^-13 + 6*r^-7)
    # d/dx_i = (d/dr) * (r / x_i) ... actually d(r^2)/dx_i = 2(x_i - x_j)
    # Let's use the standard formula:
    # F_ij = 4 * (12*r^-14 - 6*r^-8) * (r_i - r_j)
    # Or in terms of r2:
    # coeff = 4 * (-6 * r2^-4 + 12 * r2^-7) ? No.
    # E = 4 * sum (r^-12 - r^-6)
    # dE/dx_i = sum_j 4 * (-12 r^-13 + 6 r^-7) * (x_i - x_j) / r
    # = sum_j 4 * (-12 r^-14 + 6 r^-8) * (x_i - x_j)
    # r^-14 = (r^2)^-7
    # r^-8 = (r^2)^-4
    
    coeff = 4.0 * (-12.0 * (r2_safe ** -7) + 6.0 * (r2_safe ** -4))
    coeff = np.where(mask, coeff, 0.0)
    
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad


def _lj_energy_and_grad(pos):
    """Combined energy and gradient for L-BFGS."""
    e = _lj_energy_vec(pos)
    g = _lj_grad_vec(pos)
    return e, g.ravel()


def _check_min_dist(pos, min_d=0.5):
    """Check all pair distances >= min_d."""
    d = pdist(pos)
    return d.min() >= min_d - 1e-9


def _relax_lbfgs(pos, maxiter=2000, tol=1e-12):
    """L-BFGS-B local relaxation with vectorized objective."""
    if _remaining() < 5.0:
        return pos
    
    def fun(x):
        x = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(x)
        return e, g
    
    x0 = pos.reshape(-1)
    try:
        result = minimize(fun, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': tol, 'maxls': 100})
        if result.success or result.fun < _lj_energy_vec(pos):
            return result.x.reshape(-1, 3)
    except Exception:
        pass
    return pos


def _make_mackay_icosahedron_55(r1=1.12, r2=2.24):
    """
    Construct a Mackay icosahedron with 55 atoms.
    1 center, 12 inner shell, 42 outer shell.
    """
    phi = (1 + np.sqrt(5)) / 2.0
    
    # Icosahedron vertices (12)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Shell 1: 12 icosahedron vertices at radius r1
    shell1 = verts * r1
    
    # Find edges of the icosahedron
    D = cdist(verts, verts)
    np.fill_diagonal(D, np.inf)
    min_dist = D.min()
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(D[i, j] - min_dist) < 1e-10:
                edges.append((i, j))
    
    # Edge midpoint directions (30 edges)
    edge_mids = []
    for i, j in edges:
        mid_dir = (verts[i] + verts[j]) / np.linalg.norm(verts[i] + verts[j])
        edge_mids.append(mid_dir)
    edge_mids = np.array(edge_mids)  # shape (30, 3)
    
    # Vertex directions for outer shell (12 points)
    vertex_outer = verts * r2
    
    # Combine: 12 vertex-outer + 30 edge-midpoint-outer = 42
    shell2 = np.vstack([vertex_outer, edge_mids * r2])
    
    # Center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    positions = np.vstack([center, shell1, shell2])
    return positions


def construct():
    global _start_time
    _start_time = time.time()
    
    best_pos = None
    best_energy = np.inf
    
    # Initial guess: Mackay Icosahedron with typical LJ radii
    # Optimal r1 is approx 1.12, r2 is approx 2.24
    # Let's try a few variations of radii to ensure we capture the basin
    
    radii_sets = [
        (1.12, 2.24),
        (1.10, 2.20),
        (1.15, 2.30),
        (1.0, 2.0),
        (1.2, 2.4)
    ]
    
    for r1, r2 in radii_sets:
        if _remaining() < 10.0:
            break
            
        try:
            pos = _make_mackay_icosahedron_55(r1, r2)
            # Relax to local minimum
            relaxed = _relax_lbfgs(pos, maxiter=1000, tol=1e-12)
            e = _lj_energy_vec(relaxed)
            
            if _check_min_dist(relaxed) and e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except Exception:
            continue
    
    # If we haven't found a good solution, try a multi-start from the best Mackay
    if best_pos is not None and _remaining() > 20.0:
        # Try perturbing the best solution to escape potential local minima
        # although Mackay should be the global min, numerical issues might cause issues
        rng = np.random.default_rng(42)
        for _ in range(5):
            if _remaining() < 10.0:
                break
            noise = rng.normal(0, 0.01, best_pos.shape)
            perturbed = best_pos + noise
            relaxed = _relax_lbfgs(perturbed, maxiter=500, tol=1e-12)
            e = _lj_energy_vec(relaxed)
            if _check_min_dist(relaxed) and e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
                
    # Final high-precision polish
    if best_pos is not None and _remaining() > 5.0:
        polished = _relax_lbfgs(best_pos, maxiter=2000, tol=1e-14)
        e = _lj_energy_vec(polished)
        if _check_min_dist(polished) and e < best_energy:
            best_energy = e
            best_pos = polished.copy()

    if best_pos is None:
        # Fallback: original cubic lattice
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12

    return best_pos
# EVOLVE-BLOCK-END
