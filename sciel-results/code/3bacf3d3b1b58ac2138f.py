"""Lennard-Jones cluster global minimum, 55 atoms.
Strategy: build a Mackay icosahedron (2 shells: 1+12+42) as an initial guess,
then perform multi-restart local relaxation with L-BFGS.
"""
import numpy as np
import time

N = 55

def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for positions of shape (N,3)."""
    n = pos.shape[0]
    # Pairwise displacements
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n,n,3)
    r2 = np.sum(diff * diff, axis=2)  # (n,n)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r2 = np.maximum(r2, 1e-12)
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    # Energy: 4 * sum over i<j of (r^-12 - r^-6)
    # Using full matrix, each pair counted twice
    e_pair = 4.0 * (inv_r12 - inv_r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = 0.5 * np.sum(e_pair)
    
    # Gradient: dE/dr_i = sum_j 4 * (-12*r^-14 + 6*r^-8) * (r_i - r_j)
    # = sum_j 4 * (6*r^-8 - 12*r^-14) * (r_i - r_j)
    # Note: r^-14 = r^-12 / r^2, r^-8 = r^-6 / r^2
    coeff = 4.0 * (6.0 * inv_r6 - 12.0 * inv_r12) * inv_r2  # (n,n)
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad


def _relax(pos, maxiter=1000, gtol=1e-8):
    """L-BFGS relaxation using scipy."""
    from scipy.optimize import minimize
    
    def fun(x):
        p = x.reshape(-1, 3)
        e, _ = _lj_energy_and_grad(p)
        return e
    
    def jac(x):
        p = x.reshape(-1, 3)
        _, g = _lj_energy_and_grad(p)
        return g.ravel()
    
    res = minimize(fun, pos.ravel(), jac=jac, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'gtol': gtol})
    return res.x.reshape(-1, 3), res.fun


def _mckay_icosahedron():
    """Construct a Mackay icosahedron with 55 atoms (1+12+42)."""
    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # Icosahedron vertices (12) - normalized to unit length
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Inner shell: 12 atoms at radius r1
    # Outer shell: 42 atoms
    # Mackay icosahedron: inner shell at radius ~1.0, outer at ~1.7 or so
    # For LJ, the optimal spacing is 2^(1/6) ≈ 1.1225
    
    # Standard Mackay icosahedron construction:
    # Shell 1 (inner): 12 atoms at distance d1 from center
    # Shell 2 (outer): 42 atoms at distance d2 from center
    # d1/d2 ratio determined by geometry
    
    # The icosahedron has 12 vertices, 30 edges, 20 faces
    # Outer shell atoms are at:
    # - 12 positions along vertex directions (same as inner but further out)
    # - 30 positions along edge midpoints (directions)
    # Total outer: 12 + 30 = 42
    
    # Directions for outer shell:
    # 1. Along the 12 icosahedron vertex directions
    # 2. Along the 30 edge-midpoint directions
    
    # Edge midpoints: each edge connects two icosahedron vertices
    # We need to find all 30 edges
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = ico_verts[i] - ico_verts[j]
            dist = np.linalg.norm(d)
            # Edge length of regular icosahedron with circumradius 1
            # Edge length = 4/sqrt(10+2*sqrt(5)) ≈ 1.0515
            if dist < 1.1:  # threshold for being an edge
                edges.append((i, j))
    
    # Get edge midpoint directions
    edge_dirs = []
    for i, j in edges:
        mid = (ico_verts[i] + ico_verts[j]) / 2.0
        mid /= np.linalg.norm(mid)
        edge_dirs.append(mid)
    
    # Check we have 30 edges
    # If not, adjust threshold
    if len(edge_dirs) != 30:
        # Try different threshold
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = ico_verts[i] - ico_verts[j]
                dist = np.linalg.norm(d)
                if dist < 1.2:
                    edges.append((i, j))
        edge_dirs = []
        for i, j in edges:
            mid = (ico_verts[i] + ico_verts[j]) / 2.0
            mid /= np.linalg.norm(mid)
            edge_dirs.append(mid)
    
    # Now construct the 55-atom cluster
    # Inner shell: 12 atoms at radius r_in
    # Outer shell: 12 + 30 = 42 atoms at radius r_out
    
    # For a Mackay icosahedron, the ratio r_out/r_in is fixed by geometry
    # The inner shell atoms are at the icosahedron vertices scaled to r_in
    # The outer shell vertex atoms are at the same directions but at r_out
    # The outer shell edge atoms are at the edge-midpoint directions at r_out
    
    # Optimal radii for LJ: need to find r_in and r_out that minimize energy
    # As a starting guess, use r_in = 1.0, r_out = 1.7 (rough)
    # Then let relaxation find the optimum
    
    r_in = 1.0
    r_out = 1.7
    
    atoms = []
    # Center atom
    atoms.append(np.array([0.0, 0.0, 0.0]))
    
    # Inner shell: 12 atoms
    for v in ico_verts:
        atoms.append(v * r_in)
    
    # Outer shell: 12 atoms along vertex directions
    for v in ico_verts:
        atoms.append(v * r_out)
    
    # Outer shell: 30 atoms along edge directions
    for v in edge_dirs:
        atoms.append(v * r_out)
    
    pos = np.array(atoms[:55], dtype=float)
    
    # Scale so that nearest-neighbor distances are ~2^(1/6)
    # Compute current nearest-neighbor distance
    diffs = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diffs * diffs, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn_dists = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn_dists)
    
    # Scale to make avg_nn = 2^(1/6)
    scale = 2.0 ** (1.0/6.0) / avg_nn
    pos = pos * scale
    
    return pos


def construct():
    t_start = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Start with Mackay icosahedron
    pos0 = _mckay_icosahedron()
    
    best_pos = pos0.copy()
    best_energy = _lj_energy_and_grad(pos0)[0]
    
    # Try multiple perturbations and relax
    rng = np.random.default_rng(42)
    
    # First, relax the initial guess
    if time.time() - t_start > deadline:
        return best_pos
    
    try:
        relaxed, e = _relax(pos0, maxiter=500, gtol=1e-10)
        if e < best_energy:
            best_energy = e
            best_pos = relaxed.copy()
    except Exception:
        pass
    
    # Multi-restart with small perturbations
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() - t_start > deadline - 10.0:
            break
        
        # Perturb the best position
        noise_scale = 0.1 + 0.05 * i
        perturbed = best_pos + rng.normal(0, noise_scale, size=best_pos.shape)
        
        # Ensure minimum separation of 0.5
        # Simple check: if any pair is too close, push apart
        for _ in range(50):
            diffs = perturbed[:, np.newaxis, :] - perturbed[np.newaxis, :, :]
            r2 = np.sum(diffs * diffs, axis=2)
            np.fill_diagonal(r2, 1.0)
            close_pairs = (r2 < 0.25) & (np.arange(55)[:, None] < np.arange(55)[None, :])
            if not np.any(close_pairs):
                break
            # Push apart
            for idx in range(55):
                for jdx in range(idx+1, 55):
                    if r2[idx, jdx] < 0.25:
                        d = perturbed[idx] - perturbed[jdx]
                        dist = np.linalg.norm(d)
                        if dist < 1e-8:
                            d = rng.normal(0, 0.1, size=3)
                        d /= np.linalg.norm(d)
                        perturbed[idx] += 0.5 * d
                        perturbed[jdx] -= 0.5 * d
        
        try:
            relaxed, e = _relax(perturbed, maxiter=300, gtol=1e-9)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() - t_start < deadline - 5.0:
        try:
            polished, e = _relax(best_pos, maxiter=2000, gtol=1e-12)
            if e < best_energy:
                best_energy = e
                best_pos = polished.copy()
        except Exception:
            pass
    
    return best_pos
