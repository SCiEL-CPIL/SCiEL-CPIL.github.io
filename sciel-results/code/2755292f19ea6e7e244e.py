"""Lennard-Jones 55-atom global minimum search using scipy L-BFGS-B with multiple starts."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

def _lj_energy_flat(x):
    """Compute total LJ energy from flat array of shape (3N,)."""
    pos = x.reshape(N, 3)
    n = N
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r6 = r2 ** 3
    r12 = r6 * r6
    # Energy: 4*(r^-12 - r^-6) for each pair, but we double count so divide by 2
    e = 4.0 * (1.0 / r12 - 1.0 / r6)
    return 0.5 * np.sum(e)

def _lj_energy_grad_flat(x):
    """Compute total LJ energy and gradient from flat array of shape (3N,)."""
    pos = x.reshape(N, 3)
    n = N
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n, n, 3)
    r2 = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(r2, 1.0)
    r = np.sqrt(r2)
    r6 = r2 ** 3
    r12 = r6 * r6
    e = 4.0 * (1.0 / r12 - 1.0 / r6)
    energy = 0.5 * np.sum(e)
    # Gradient: for pair (i,j), grad_i = 4*(-12*r^-13 + 6*r^-7) * (pos[i]-pos[j])/r
    # = 4*(-12/r^14 + 6/r^8) * diff[i,j]
    # But we need to be careful with the factor
    # dE/dr = 4*(-12*r^-13 + 6*r^-7)
    # grad_i = dE/dr * (pos[i]-pos[j])/r = 4*(-12/r^14 + 6/r^8) * diff[i,j]
    # Let's compute: factor[i,j] = 4*(-12/r2^3.5 + 6/r2^1.5) / r ... 
    # Actually: 4*(-12/r^14 + 6/r^8) = 4*(-12/(r2^7) + 6/(r2^4))
    # So grad_i = 4*(-12/(r2^7) + 6/(r2^4)) * diff[i,j]
    # But wait, that's per pair. Let me re-derive.
    # E_pair = 4*(r^-12 - r^-6), r = |rij|
    # dE_pair/d(pos_i) = 4*(-12*r^-13 * d(r)/d(pos_i) + 6*r^-7 * d(r)/d(pos_i))
    # d(r)/d(pos_i) = (pos_i - pos_j)/r = rij/r
    # So dE_pair/d(pos_i) = 4*(-12*r^-13 + 6*r^-7) * rij/r
    # = 4*(-12/r^14 + 6/r^8) * rij
    # = 4*(-12/(r2^7) + 6/(r2^4)) * rij
    # where rij = pos_i - pos_j
    # For the full gradient at pos_i, sum over all j != i
    # grad_i = sum_{j != i} 4*(-12/(r2_ij^7) + 6/(r2_ij^4)) * (pos_i - pos_j)
    # We can compute this vectorially:
    # factor[i,j] = 4*(-12/(r2[i,j]^7) + 6/(r2[i,j]^4))
    # grad[i] = sum_j factor[i,j] * diff[i,j]
    factor = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(factor, 0.0)
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    return energy, grad.reshape(3 * N)

def _make_mackay_icosahedron():
    """Construct a two-shell Mackay icosahedron for 55 atoms."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # Standard icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find edges
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(verts[i] - verts[j])
    min_dist = np.min(dists[dists > 1e-6])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    
    # 30 edge midpoints (unit vectors)
    edge_mids = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)
    
    # 20 face centers (unit vectors)
    # For the standard icosahedron, we can find faces by looking at triples of vertices
    # that form triangles. A simpler approach: the face centers of an icosahedron
    # correspond to the vertices of a dodecahedron.
    # Dodecahedron vertices (20 vertices)
    # The dodecahedron vertices are related to the icosahedron face centers.
    # Let me use a known construction.
    # Actually, for the Mackay icosahedron, the 42 second-shell atoms are:
    # - 30 at edge midpoints (scaled)
    # - 12 at face centers... but there are only 20 faces.
    # Let me reconsider. The 55-atom Mackay icosahedron has:
    # 1 center, 12 first shell, 42 second shell.
    # The 42 second-shell atoms: in the Mackay construction, they are placed at
    # positions corresponding to the icosahedral symmetry.
    # A common construction: 30 atoms at the midpoints of the 30 edges, and
    # 12 atoms at the 12 vertices of a dodecahedron inscribed in the icosahedron.
    # But that gives 30+12=42. Let me use that.
    
    # Dodecahedron vertices (20 vertices) - but we only need 12 of them?
    # Actually, the Mackay icosahedron second shell has 42 atoms.
    # Let me just use a reasonable initial guess and let optimization fix it.
    
    # Use: 1 center, 12 first shell at r1, 30 edge-midpoint directions at r2, 12 vertex directions at r3
    r1 = 1.0
    r2 = 1.6
    r3 = 1.8
    
    first_shell = verts * r1
    second_shell_edges = edge_mids * r2
    second_shell_verts = verts * r3
    
    center = np.array([[0.0, 0.0, 0.0]])
    all_pos = np.vstack([center, first_shell, second_shell_edges, second_shell_verts])
    return all_pos

def _relax_scipy(pos, max_iter=500, tol=1e-10):
    """Relax positions using scipy L-BFGS-B."""
    x0 = pos.flatten()
    result = minimize(_lj_energy_flat, x0, jac=_lj_energy_grad_flat, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 50})
    return result.x.reshape(N, 3), result.fun

def _check_min_distance(pos, min_d=0.5):
    """Check if all pairs are at least min_d apart."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    return np.min(r2) >= min_d * min_d

def _fix_min_distance(pos, min_d=0.5):
    """Push apart overlapping atoms."""
    pos = pos.copy()
    for _ in range(200):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(r2, 1.0)
        r = np.sqrt(r2)
        min_idx = np.unravel_index(np.argmin(r), r.shape)
        if r[min_idx] >= min_d:
            break
        i, j = min_idx
        direction = pos[i] - pos[j]
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            direction = np.array([1.0, 0.0, 0.0])
            norm = 1.0
        direction = direction / norm
        push = (min_d - r[min_idx]) / 2.0 + 0.005
        pos[i] += push * direction
        pos[j] -= push * direction
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    best_pos = None
    best_energy = np.inf
    
    # Start with Mackay icosahedron
    try:
        mackay = _make_mackay_icosahedron()
        e0 = _lj_energy_flat(mackay.flatten())
        if np.isfinite(e0):
            best_energy = e0
            best_pos = mackay.copy()
    except Exception:
        pass
    
    # If Mackay failed, use cubic lattice
    if best_pos is None:
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
        best_energy = _lj_energy_flat(best_pos.flatten())
    
    # Relax the initial structure
    if time.time() < deadline:
        try:
            relaxed, e_rel = _relax_scipy(best_pos, max_iter=1000, tol=1e-12)
            if np.isfinite(e_rel) and e_rel < best_energy:
                best_energy = e_rel
                best_pos = relaxed
        except Exception:
            pass
    
    # Multiple restarts with perturbations
    rng = np.random.RandomState(42)
    num_restarts = 0
    while time.time() < deadline - 20.0:
        num_restarts += 1
        # Create perturbed version
        if num_restarts == 1:
            pos = best_pos.copy()
        else:
            noise_scale = 0.05 + 0.02 * (num_restarts % 10)
            pos = best_pos + noise_scale * rng.randn(N, 3)
        
        # Ensure minimum distance
        pos = _fix_min_distance(pos)
        
        # Relax
        try:
            relaxed, e_rel = _relax_scipy(pos, max_iter=500, tol=1e-12)
            if np.isfinite(e_rel) and e_rel < best_energy:
                best_energy = e_rel
                best_pos = relaxed
        except Exception:
            pass
    
    # Final check: ensure minimum distance
    best_pos = _fix_min_distance(best_pos)
    
    return best_pos
