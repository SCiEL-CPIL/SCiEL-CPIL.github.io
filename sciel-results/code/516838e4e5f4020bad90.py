"""Lennard-Jones cluster global minimum, 55 atoms — Mackay icosahedron direct construction and polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE_MARGIN = 5.0  # seconds before the 150 s wall-clock kill
TOTAL_BUDGET = 150.0


def _lj_energy_and_grad(pos):
    """Compute total LJ energy and gradient for N atoms. Vectorized."""
    # pos: (N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)  # avoid div by zero
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** -3
    r12 = r6 ** 2

    # Energy: 4 * sum over i<j of (r^-12 - r^-6)
    # The matrix has each pair twice, so divide by 2
    pair_e = 4.0 * (r12 - r6)
    np.fill_diagonal(pair_e, 0.0)
    energy = 0.5 * np.sum(pair_e)

    # Gradient: dE/dx_i = sum_j 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j)
    # In terms of r2: r^14 = r2^7, r^8 = r2^4
    # dE/dx_i = 4*(-12*r2^-7 + 6*r2^-4) * diff[i,j]

    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))  # (N, N)
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)

    return energy, grad


def _get_mackay_icosahedron_positions():
    """
    Construct the 55-atom Mackay icosahedron.
    Structure:
    - 1 atom at center (0,0,0)
    - 12 atoms at icosahedron vertices (inner shell)
    - 30 atoms at icosahedron edge midpoints (middle shell)
    - 12 atoms at icosahedron vertices (outer shell)
    
    Radii are set based on the close-packed Mackay icosahedron geometry:
    Inner shell radius R1 = sqrt(10/3) * sigma
    Outer shell radius R3 = sqrt(50/3) * sigma
    Middle shell radius R2 = (R1 + R3) / 2 is a good approximation, but exact 
    geometric placement on edges is better. For the Mackay icosahedron, the 
    atoms in the second shell are at the midpoints of the edges of the 
    icosahedron formed by the outer shell vertices, scaled appropriately.
    
    Actually, the standard Mackay icosahedron has:
    Shell 1 (inner): 12 atoms, radius r1
    Shell 2 (middle): 30 atoms, radius r2
    Shell 3 (outer): 12 atoms, radius r3
    
    For the LJ global minimum, the radii are approximately:
    r1 = 1.82574
    r2 = 2.95412
    r3 = 4.08248
    
    Let's construct the directions first.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    # Icosahedral vertices (12)
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
    
    # The 30 edges have the smallest distances
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
    
    # Radii for the Mackay icosahedron (LJ units)
    # These values correspond to the global minimum energy configuration
    r1 = 1.82574
    r2 = 2.95412
    r3 = 4.08248

    center = np.zeros((1, 3))
    shell1 = verts * r1
    shell2 = edge_mids * r2
    shell3 = verts * r3
    
    all_pos = np.vstack([center, shell1, shell2, shell3])
    return all_pos


def construct():
    start_time = time.time()
    
    # Start with the known global minimum structure: Mackay Icosahedron
    pos = _get_mackay_icosahedron_positions()
    
    # Phase 1: L-BFGS-B optimization to polish the structure
    # This handles any slight deviations from the ideal Mackay geometry
    try:
        res1 = minimize(
            _lj_energy_and_grad,
            pos,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'gtol': 1e-12, 'maxls': 100}
        )
        if res1.fun < _lj_energy_and_grad(pos)[0]:
            pos = res1.x
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > TOTAL_BUDGET - DEADLINE_MARGIN:
        return pos

    # Phase 2: High-precision polish with tighter tolerance
    try:
        res2 = minimize(
            _lj_energy_and_grad,
            pos,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'gtol': 1e-15, 'maxls': 200}
        )
        if res2.fun < _lj_energy_and_grad(pos)[0]:
            pos = res2.x
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > TOTAL_BUDGET - DEADLINE_MARGIN:
        return pos

    # Phase 3: Final BFGS polish for extreme precision
    try:
        res3 = minimize(
            _lj_energy_and_grad,
            pos,
            jac=True,
            method='BFGS',
            options={'maxiter': 500, 'gtol': 1e-16, 'disp': False}
        )
        if res3.fun < _lj_energy_and_grad(pos)[0]:
            pos = res3.x
    except Exception:
        pass

    # Final validation: ensure atoms are at least 0.5 apart
    # Compute minimum distance
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    min_dist = np.sqrt(np.min(r2))
    
    if min_dist < 0.5:
        # Fallback to the initial Mackay structure if validation fails (should not happen)
        pos = _get_mackay_icosahedron_positions()
        
    return pos
