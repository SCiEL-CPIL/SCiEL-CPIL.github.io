"""Improved Lennard-Jones cluster global minimum, 55 atoms."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

# Deadline guard: 150s wall-clock limit, use 145s for safety
_DEADLINE = 145.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _lj_energy_vec(pos):
    """Compute total Lennard-Jones energy using vectorized operations."""
    # pos shape: (N, 3)
    # Compute all pairwise distances
    # diff[i, j] = pos[i] - pos[j]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    # r^2
    r2 = np.sum(diff * diff, axis=2)  # (N, N)
    # Mask diagonal (self-interaction)
    mask = ~np.eye(len(pos), dtype=bool)
    r2 = np.where(mask, r2, 1e10)  # Large value to make contribution 0
    
    r6 = r2 ** -3
    r12 = r6 ** 2
    
    # Sum over all pairs, but we counted each pair twice (i,j and j,i)
    # The standard LJ energy is 4 * sum_{i<j} (r^-12 - r^-6)
    # So we sum the full matrix and divide by 2, then multiply by 4
    # Or just sum the upper triangle.
    # Let's sum the full matrix and divide by 2.
    e = 4.0 * np.sum(r12 - r6) / 2.0
    return e


def _lj_grad_vec(pos):
    """Compute gradient of LJ energy using vectorized operations."""
    # pos shape: (N, 3)
    N = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    r2 = np.sum(diff * diff, axis=2)  # (N, N)
    
    mask = ~np.eye(N, dtype=bool)
    # Avoid division by zero
    r2_safe = np.where(mask, r2, 1.0)
    
    r6 = r2_safe ** -3
    r12 = r6 ** 2
    
    # Gradient of 4*(r^-12 - r^-6) with respect to pos_i
    # f(r) = 4*(r^-12 - r^-6)
    # df/dr = 4*(-12*r^-13 + 6*r^-7)
    # grad_i = df/dr * (pos_i - pos_j)/r = df/dr * diff_ij / r
    # = 4*(-12*r^-13 + 6*r^-7) * diff_ij / r
    # = 4*(-12*r^-14 + 6*r^-8) * diff_ij
    # Note: r^-14 = r^-12 / r^2, r^-8 = r^-6 / r^2
    coeff = 4.0 * (-12.0 * r12 + 6.0 * r6) / r2_safe  # (N, N)
    coeff = np.where(mask, coeff, 0.0)
    
    # grad_i = sum_j coeff_ij * diff_ij
    # coeff shape (N, N), diff shape (N, N, 3)
    # We want sum over j (axis 1)
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


def _relax_lbfgs(pos, maxiter=1000, tol=1e-10):
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
                          options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': tol, 'maxls': 50})
        if result.success or result.fun < _lj_energy_vec(pos):
            return result.x.reshape(-1, 3)
    except Exception:
        pass
    return pos


def _make_mackay_icosahedron_55():
    """Construct a high-quality Mackay icosahedron initial guess."""
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
    
    # Radii for LJ optimization
    # Inner shell (12 atoms) at r1
    # Outer shell (42 atoms) at r2
    # Center atom at 0
    
    # Good initial guesses based on literature/relaxation
    r1 = 1.122  # ~ 2^(1/6)
    r2 = 2.244  # ~ 2 * 2^(1/6)
    
    # Shell 1: 12 icosahedron vertices at radius r1
    shell1 = verts * r1
    
    # Shell 2: 42 points
    # In Mackay icosahedron, the second shell consists of:
    # - 12 points corresponding to vertex directions (but further out)
    # - 30 points corresponding to edge midpoint directions
    # Total 42.
    
    # Find edges of the icosahedron
    # Edges connect vertices with minimum distance
    from scipy.spatial.distance import cdist
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
    
    # Vertex directions for outer shell (12 points, same directions as inner but at r2)
    vertex_outer = verts * r2
    
    # Combine: 12 vertex-outer + 30 edge-midpoint-outer = 42
    shell2 = np.vstack([vertex_outer, edge_mids * r2])
    
    # Center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    positions = np.vstack([center, shell1, shell2])
    return positions


def _make_random_cluster(n=55, seed=None):
    """Generate a random cluster with minimum distance 0.5."""
    rng = np.random.default_rng(seed)
    # Place atoms in a sphere of radius ~6
    pos = []
    for _ in range(n):
        for attempt in range(1000):
            p = rng.uniform(-6, 6, size=3)
            if len(pos) == 0:
                pos.append(p)
                break
            existing = np.array(pos)
            dists = np.linalg.norm(existing - p, axis=1)
            if dists.min() > 0.5:
                pos.append(p)
                break
    if len(pos) < n:
        # Fallback: cubic grid
        side = int(np.ceil(n ** (1/3)))
        pts = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    pts.append([i, j, k])
                    if len(pts) >= n:
                        break
        pos = pts[:n]
    return np.array(pos, dtype=float) * 1.12


def _make_fcc_cluster(n=55, seed=None):
    """Generate an FCC-like cluster."""
    rng = np.random.default_rng(seed)
    a = 1.12
    pos = []
    # Generate points in a larger box and take the closest n
    for i in range(-2, 5):
        for j in range(-2, 5):
            for k in range(-2, 5):
                # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
                for offset in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i, j, k]) * a + np.array(offset) * a
                    pos.append(p)
    
    pos = np.array(pos)
    # Sort by distance from origin
    dists = np.linalg.norm(pos, axis=1)
    indices = np.argsort(dists)
    pos = pos[indices[:n]]
    
    # Add small noise to break symmetry
    pos += rng.normal(0, 0.01, pos.shape)
    return pos


def construct():
    global _start_time
    _start_time = time.time()
    
    best_pos = None
    best_energy = np.inf
    
    # Strategy 1: Mackay Icosahedron (High Symmetry)
    if _remaining() > 10.0:
        try:
            mackay = _make_mackay_icosahedron_55()
            relaxed = _relax_lbfgs(mackay, maxiter=500)
            e = _lj_energy_vec(relaxed)
            if _check_min_dist(relaxed) and e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except Exception:
            pass
    
    # Strategy 2: Perturbed Mackay (Break Symmetry)
    if _remaining() > 10.0:
        try:
            rng = np.random.default_rng(123)
            mackay = _make_mackay_icosahedron_55()
            # Add small noise
            perturbed = mackay + rng.normal(0, 0.05, mackay.shape)
            relaxed = _relax_lbfgs(perturbed, maxiter=500)
            e = _lj_energy_vec(relaxed)
            if _check_min_dist(relaxed) and e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except Exception:
            pass
            
    # Strategy 3: FCC-like Cluster
    if _remaining() > 10.0:
        try:
            fcc = _make_fcc_cluster(55, seed=42)
            relaxed = _relax_lbfgs(fcc, maxiter=500)
            e = _lj_energy_vec(relaxed)
            if _check_min_dist(relaxed) and e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except Exception:
            pass
            
    # Strategy 4: Random Cluster
    if _remaining() > 10.0:
        try:
            random_pos = _make_random_cluster(55, seed=999)
            relaxed = _relax_lbfgs(random_pos, maxiter=500)
            e = _lj_energy_vec(relaxed)
            if _check_min_dist(relaxed) and e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except Exception:
            pass

    # Final Polish on the best result
    if best_pos is not None and _remaining() > 5.0:
        try:
            # Tighter tolerance for final polish
            polished = _relax_lbfgs(best_pos, maxiter=1000, tol=1e-12)
            e = _lj_energy_vec(polished)
            if _check_min_dist(polished) and e < best_energy:
                best_energy = e
                best_pos = polished.copy()
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback: original cubic lattice
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
    
    return best_pos
# EVOLVE-BLOCK-END
