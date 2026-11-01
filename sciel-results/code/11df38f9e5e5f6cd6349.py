"""Lennard-Jones 55-atom cluster: Mackay icosahedron with 3-parameter radial optimization + L-BFGS relaxation."""
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

def _get_mackay_directions():
    """
    Returns unit vectors for the Mackay icosahedron shells.
    Returns:
        verts1: 12 unit vectors to icosahedron vertices (Shell 1)
        verts2: 12 unit vectors to icosahedron vertices (Shell 2 vertices)
        mids2:  30 unit vectors to edge midpoints (Shell 2 edges)
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
    
    # Shell 1 uses the same vertex directions
    # Shell 2 uses the same vertex directions for the 12 vertices
    # Shell 2 uses the edge midpoints for the 30 edges
    return verts, verts, edge_mids

def _build_mackay_3r(r1, r2, r3):
    """
    Build the 55-atom Mackay icosahedron with independent radii.
    r1: radius of 1st shell (12 atoms)
    r2: radius of 2nd shell vertices (12 atoms)
    r3: radius of 2nd shell edge midpoints (30 atoms)
    """
    verts1, verts2, mids2 = _get_mackay_directions()
    
    center = np.array([[0.0, 0.0, 0.0]])
    shell1 = verts1 * r1
    shell2_verts = verts2 * r2
    shell2_mids = mids2 * r3
    
    all_points = np.vstack([center, shell1, shell2_verts, shell2_mids])
    return all_points

def _energy_for_radii_3r(params):
    """Objective function for optimizing radii r1, r2, r3."""
    r1, r2, r3 = params
    pos = _build_mackay_3r(r1, r2, r3)
    e, _ = _lj_energy_and_grad(pos)
    return e

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    # 1. Optimize the 3 radii of the symmetric Mackay icosahedron
    # Initial guess based on literature/previous runs
    # r1 ~ 1.12, r2 ~ 2.24, r3 ~ 2.24
    initial_guess = [1.12, 2.24, 2.24]
    
    best_pos = None
    best_energy = np.inf
    
    try:
        # Minimize radii using L-BFGS-B
        res_radii = minimize(_energy_for_radii_3r, initial_guess, method='L-BFGS-B', 
                             bounds=[(1.0, 1.5), (2.0, 2.5), (2.0, 2.5)], 
                             options={'maxiter': 2000, 'gtol': 1e-8})
        r1_opt, r2_opt, r3_opt = res_radii.x
    except Exception:
        r1_opt, r2_opt, r3_opt = initial_guess
        
    # Build the optimized symmetric structure
    pos_seed = _build_mackay_3r(r1_opt, r2_opt, r3_opt)
    
    # 2. Full relaxation with L-BFGS
    try:
        result = minimize(_lj_energy_and_grad, pos_seed, method='L-BFGS-B',
                         options={'maxiter': 10000, 'gtol': 1e-10, 'ftol': 1e-10})
        
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.copy()
    except Exception:
        pass
    
    # 3. Multi-start perturbations to ensure we are in the global basin
    # If the main relaxation is good, this is a safety net.
    # We use small perturbations to break any residual symmetry locks if needed.
    if best_energy > -270.0: # If we haven't reached near the known global minimum
        np.random.seed(42)
        for scale in [0.05, 0.1, 0.2]:
            if time.time() > deadline:
                break
            perturbed = pos_seed + np.random.randn(55, 3) * scale
            # Ensure minimum distance constraint is not grossly violated
            # (L-BFGS will fix it, but we avoid starting in a singularity)
            try:
                res = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B',
                               options={'maxiter': 5000, 'gtol': 1e-8, 'ftol': 1e-8})
                if res.fun < best_energy:
                    best_energy = res.fun
                    best_pos = res.x.copy()
            except Exception:
                pass
            
            if time.time() > deadline:
                break

    # 4. Fallback "Liquid Quench" if still not good enough
    # This addresses the directive to try a symmetry-breaking approach if the structured one fails
    if best_energy > -260.0 and time.time() < deadline:
        # Generate a random liquid configuration within a small sphere
        # Radius ~ 4.0 should contain 55 atoms at liquid density
        np.random.seed(123)
        liquid_pos = np.random.randn(55, 3) * 2.0
        # Simple rejection to avoid overlaps > 0.5
        valid = True
        for i in range(55):
            for j in range(i+1, 55):
                if np.linalg.norm(liquid_pos[i] - liquid_pos[j]) < 0.5:
                    valid = False
                    break
            if not valid: break
        
        if valid:
            try:
                res = minimize(_lj_energy_and_grad, liquid_pos, method='L-BFGS-B',
                               options={'maxiter': 10000, 'gtol': 1e-8, 'ftol': 1e-8})
                if res.fun < best_energy:
                    best_energy = res.fun
                    best_pos = res.x.copy()
            except Exception:
                pass

    if best_pos is None:
        # Absolute fallback: return the radially optimized seed
        return pos_seed
        
    return best_pos

# EVOLVE-BLOCK-END
