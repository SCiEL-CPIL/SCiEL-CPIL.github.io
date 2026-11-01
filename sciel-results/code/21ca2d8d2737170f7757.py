"""Lennard-Jones 55-atom cluster: Mackay icosahedron with radial manifold optimization + L-BFGS relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START

def _lj_energy_and_grad(pos):
    """Compute Lennard-Jones energy and gradient for a set of positions."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r2_safe = np.maximum(r2, 1e-12)
    
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    
    # Energy: 4 * sum_{i<j} (r^-12 - r^-6)
    # Matrix has each pair twice, so: E = 2 * sum(r12 - r6)
    E = 2.0 * np.sum(r12 - r6)
    
    # Gradient
    # dE/dpos_i = 4 * sum_{j!=i} (-12*r^-14 + 6*r^-8) * (pos_i - pos_j)
    # In terms of r2: r^-14 = r2^(-7), r^-8 = r2^(-4)
    # coeff = 24 * (2*r2^(-7) - r2^(-4))
    coeff = 24.0 * (2.0 * r2_safe ** (-7) - r2_safe ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return E, grad

def _get_mackay_vectors():
    """
    Returns the 12 unit vectors to icosahedron vertices and 
    the 30 unit vectors to edge midpoints.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # Icosahedron vertices (unit length)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Find edges: pairs of vertices that are nearest neighbors
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            dists[i, j] = d
            dists[j, i] = d
    
    min_dist = np.min(dists[dists > 0])
    tol = min_dist * 1.01
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if dists[i, j] < tol:
                edges.append((i, j))
    
    # Edge midpoints (30 atoms)
    edge_mids = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid /= np.linalg.norm(mid)
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    
    return verts, edge_mids

def _build_mackay_from_radials(radials):
    """
    Build the 55-atom Mackay icosahedron from individual radial distances.
    radials: array of shape (54,) for the 54 non-center atoms.
    Order: 12 shell1 verts, 12 shell2 verts, 30 shell2 edge mids.
    """
    verts, edge_mids = _get_mackay_vectors()
    
    # radials[0:12] -> shell1 vertices
    # radials[12:24] -> shell2 vertices  
    # radials[24:54] -> shell2 edge midpoints
    
    center = np.array([[0.0, 0.0, 0.0]])
    shell1 = verts * radials[0:12, np.newaxis]
    shell2_verts = verts * radials[12:24, np.newaxis]
    shell2_mids = edge_mids * radials[24:54, np.newaxis]
    
    all_points = np.vstack([center, shell1, shell2_verts, shell2_mids])
    return all_points

def _radial_energy_and_grad(radials):
    """
    Compute energy and gradient with respect to radial distances.
    radials: shape (54,)
    """
    pos = _build_mackay_from_radials(radials)
    e, grad_pos = _lj_energy_and_grad(pos)
    
    # Chain rule: dE/dr_i = sum_k (dE/dpos_k) * (dpos_k/dr_i)
    # For atom k (k>0), pos_k = r_k * dir_k, so dpos_k/dr_k = dir_k
    # grad_radial[i] = dot(grad_pos[i+1], dir_i)
    
    verts, edge_mids = _get_mackay_vectors()
    dirs = np.vstack([verts, verts, edge_mids])  # 54 directions
    
    # grad_radial[i] = dot(grad_pos[i+1], dirs[i])
    grad_radial = np.sum(grad_pos[1:] * dirs, axis=1)
    
    return e, grad_radial

def _enforce_min_distance(pos, min_dist=0.5):
    """
    Enforce minimum distance constraint by pushing overlapping atoms apart.
    Returns a valid position array.
    """
    pos = pos.copy()
    n = pos.shape[0]
    # Iterate a few times to resolve overlaps
    for _ in range(50):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < min_dist:
                    if dist < 1e-10:
                        # Random nudge
                        d = np.random.randn(3) * 0.1
                        dist = np.linalg.norm(d)
                    # Push apart
                    overlap = min_dist - dist
                    if overlap > 0:
                        direction = d / dist
                        pos[i] += 0.5 * overlap * direction
                        pos[j] -= 0.5 * overlap * direction
                        moved = True
        if not moved:
            break
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_energy = np.inf
    best_pos = None
    
    # 1. Optimize the radial manifold (54 individual radii)
    # Initial guess: symmetric Mackay values
    initial_radials = np.array([
        1.12] * 12 + [2.24] * 12 + [2.24] * 30
    )
    
    try:
        if time.time() < deadline:
            res_radial = minimize(_radial_energy_and_grad, initial_radials, method='L-BFGS-B',
                                 bounds=[(0.8, 2.5)] * 54,
                                 options={'maxiter': 2000, 'gtol': 1e-6})
            radials_opt = res_radial.x
            pos_seed = _build_mackay_from_radials(radials_opt)
            best_energy = res_radial.fun
            best_pos = pos_seed
    except Exception:
        # Fallback to simple two-radius optimization
        verts, edge_mids = _get_mackay_vectors()
        pos_seed = np.vstack([
            np.array([[0.0, 0.0, 0.0]]),
            verts * 1.12,
            verts * 2.24,
            edge_mids * 2.24
        ])
        e, _ = _lj_energy_and_grad(pos_seed)
        best_energy = e
        best_pos = pos_seed
    
    # 2. Full relaxation with L-BFGS from the radial-optimized structure
    try:
        if time.time() < deadline:
            result = minimize(_lj_energy_and_grad, best_pos, method='L-BFGS-B',
                             options={'maxiter': 5000, 'gtol': 1e-8})
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x
    except Exception:
        pass
    
    # 3. Multi-start with perturbations to explore nearby minima
    np.random.seed(42)
    
    # Try several perturbation scales
    for scale in [0.01, 0.05, 0.1, 0.15, 0.2]:
        if time.time() > deadline:
            break
            
        for trial in range(4):  # 4 trials per scale
            if time.time() > deadline:
                break
                
            perturbed = best_pos + np.random.randn(55, 3) * scale
            # Enforce min distance 0.5
            perturbed = _enforce_min_distance(perturbed, 0.5)
            
            try:
                res = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B',
                             options={'maxiter': 3000, 'gtol': 1e-7})
                if res.fun < best_energy:
                    best_energy = res.fun
                    best_pos = res.x
            except Exception:
                pass
    
    # 4. If we have time, try a few larger perturbations to escape potential local minima
    if time.time() < deadline - 15.0:
        for scale in [0.3, 0.5]:
            if time.time() > deadline:
                break
                
            for trial in range(2):
                if time.time() > deadline:
                    break
                    
                perturbed = best_pos + np.random.randn(55, 3) * scale
                perturbed = _enforce_min_distance(perturbed, 0.5)
                
                try:
                    res = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B',
                                 options={'maxiter': 2000, 'gtol': 1e-6})
                    if res.fun < best_energy:
                        best_energy = res.fun
                        best_pos = res.x
                except Exception:
                    pass
    
    # 5. High-precision polish pass
    if time.time() < deadline - 5.0:
        try:
            result = minimize(_lj_energy_and_grad, best_pos, method='L-BFGS-B',
                             options={'maxiter': 10000, 'gtol': 1e-10, 'ftol': 1e-12})
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x
        except Exception:
            pass
    
    # Final sanity check: ensure min distance constraint
    best_pos = _enforce_min_distance(best_pos, 0.5)
    
    return best_pos

# EVOLVE-BLOCK-END
