"""
Lennard-Jones 55-atom cluster optimisation.

Strategy:
  1. Generate a good initial guess close to the Mackay icosahedron structure.
  2. Use L-BFGS-B (scipy.optimize.minimize) to relax to a local minimum.
  3. Multi-start: try several perturbations and keep the best.
  4. Respect the 150 s wall-clock deadline with a safety margin.

The 55-atom Mackay icosahedron has:
  - 1 central atom
  - 12 atoms on the first shell (icosahedral vertices) at distance r1
  - 42 atoms on the second shell (subdivided icosahedral vertices) at distance r2

The optimal radii for LJ are approximately:
  r1 ≈ 1.12 (equilibrium distance)
  r2 ≈ 2.05

We generate the icosahedral coordinates exactly and place the 42 second-shell
atoms at the 42 positions obtained by the Mackay subdivision.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 55
DEADLINE = 145.0  # safety margin before 150 s kill

# ---------------------------------------------------------------------------
# Lennard-Jones energy and gradient
# ---------------------------------------------------------------------------

def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for N atoms.  pos shape (N,3)."""
    N = pos.shape[0]
    # Pairwise differences: (N, N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # diff[i,j] = pos[i]-pos[j]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)  # avoid divide by zero
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** -3
    r12 = r6 ** 2
    # Energy per pair (count each pair once): 4*(r^-12 - r^-6)
    # Sum over i<j  =>  0.5 * (sum over all i≠j)
    energy = 4.0 * 0.5 * np.sum(r12 - r6 - np.diag(r12 - r6))
    # Actually simpler: just sum over all pairs i<j
    # Let me redo: energy = sum_{i<j} 4*(r_ij^-12 - r_ij^-6)
    # Using the matrix: energy = 0.5 * sum_{i≠j} 4*(r12 - r6)
    energy = 2.0 * np.sum(np.triu(r12 - r6, k=1))
    # Gradient: dE/dpos[i] = sum_{j≠i} dE/dpos[i]
    # d/dx_i [4*(r^-12 - r^-6)] = 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j) / r
    # = 4*(6*r^-8 - 12*r^-14) * (x_i - x_j) / r
    # = 24*(r^-8 - 2*r^-14) * (x_i - x_j) / r
    # Note: d(r)/d(x_i) = (x_i - x_j)/r
    # So dE/dx_i = sum_j 4*(12*r^-13 - 6*r^-7) * (x_j - x_i)/r
    #            = sum_j (48*r^-13 - 24*r^-7) * (x_j - x_i) / r
    # Let's use: grad_i = sum_{j≠i} 4 * (12/r^13 - 6/r^7) * (pos_i - pos_j) / r
    # Actually: d/dr [4(r^-12 - r^-6)] = 4(-12 r^-13 + 6 r^-7) = -48 r^-13 + 24 r^-7
    # dE/dpos_i = sum_j dE/dr_ij * d r_ij / dpos_i = sum_j (-48 r^-13 + 24 r^-7) * (pos_i - pos_j)/r
    #           = sum_j (24 r^-7 - 48 r^-13) * (pos_i - pos_j) / r
    #           = sum_j (24 r^-8 - 48 r^-14) * (pos_i - pos_j)
    r = np.sqrt(r2)
    np.fill_diagonal(r, 1.0)
    coeff = (24.0 * r2 ** -4 - 48.0 * r2 ** -7)  # 24 r^-8 - 48 r^-14
    np.fill_diagonal(coeff, 0.0)
    # grad[i] = sum_j coeff[i,j] * (pos[i] - pos[j])
    # = sum_j coeff[i,j]*pos[i] - sum_j coeff[i,j]*pos[j]
    # = (sum_j coeff[i,j]) * pos[i] - sum_j coeff[i,j] * pos[j]
    row_sum = np.sum(coeff, axis=1)
    grad = row_sum[:, np.newaxis] * pos - coeff @ pos
    return energy, grad


# ---------------------------------------------------------------------------
# Icosahedral geometry
# ---------------------------------------------------------------------------

def _icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin,
    with circumscribed radius 1."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ], dtype=float)
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts /= norms
    return verts  # (12, 3)


def _mackay_subdivision_42():
    """Generate 42 points on the icosahedral surface corresponding to
    the second shell of the 55-atom Mackay icosahedron.

    The 42 points are generated to preserve I_h symmetry:
    - 30 points at the edge midpoints of the icosahedron
    - 12 points at the vertices of the icosahedron (same directions as shell 1)
    """
    ico = _icosahedron_vertices()  # (12, 3), unit radius
    n = 12

    # Find edges: pairs of vertices that are close (adjacent on icosahedron)
    dists = np.linalg.norm(ico[:, np.newaxis, :] - ico[np.newaxis, :, :], axis=2)
    edge_threshold = 1.2  # generous
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] < edge_threshold:
                edges.append((i, j))
    
    # Edge midpoints (in unit icosahedron coordinates)
    edge_midpoints = []
    for i, j in edges:
        mid = (ico[i] + ico[j]) / 2.0
        edge_midpoints.append(mid)
    edge_midpoints = np.array(edge_midpoints)  # (30, 3)

    # The other 12 directions are the vertices themselves
    # This ensures I_h symmetry for the second shell
    vertex_dirs = ico.copy() # (12, 3)
    
    outer_dirs = np.vstack([edge_midpoints, vertex_dirs])  # (42, 3)
    # Normalize directions
    outer_norms = np.linalg.norm(outer_dirs, axis=1, keepdims=True)
    outer_dirs /= outer_norms
    return ico, outer_dirs  # ico: (12,3) unit, outer_dirs: (42,3) unit


# ---------------------------------------------------------------------------
# Construct initial positions
# ---------------------------------------------------------------------------

def _make_initial_guess(r1=1.1225, r2=2.051):
    """Return (N,3) initial positions close to Mackay icosahedron."""
    ico, outer_dirs = _mackay_subdivision_42()
    center = np.zeros((1, 3))
    shell1 = ico * r1
    shell2 = outer_dirs * r2
    pos = np.vstack([center, shell1, shell2])  # (55, 3)
    return pos


# ---------------------------------------------------------------------------
# Relaxation
# ---------------------------------------------------------------------------

def _relax(pos, time_limit=10.0):
    """Relax positions using L-BFGS-B.  Returns (positions, energy)."""
    start_time = time.time()

    def obj_and_grad(p):
        pos = p.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos)
        return e, g.ravel()

    res = minimize(
        obj_and_grad,
        pos.ravel(),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 20000},
    )
    elapsed = time.time() - start_time
    if res.success or hasattr(res, 'x'):
        new_pos = res.x.reshape(-1, 3)
        e, _ = _lj_energy_and_grad(new_pos)
        return new_pos, e, elapsed
    return pos, _lj_energy_and_grad(pos)[0], elapsed


# ---------------------------------------------------------------------------
# Multi-start search
# ---------------------------------------------------------------------------

def construct():
    t0 = time.time()

    # --- Start 1: clean Mackay guess ---
    pos0 = _make_initial_guess()
    pos_best, e_best, _ = _relax(pos0, time_limit=15.0)

    elapsed = time.time() - t0
    if elapsed > DEADLINE - 30.0:
        return pos_best

    # --- Additional starts with random perturbations ---
    rng = np.random.RandomState(123)
    n_starts = 5
    for s in range(n_starts):
        if time.time() - t0 > DEADLINE - 20.0:
            break
        # Random perturbation
        noise = rng.normal(0, 0.15, size=(N, 3))
        pos_pert = pos_best + noise
        pos_new, e_new, _ = _relax(pos_pert, time_limit=15.0)
        if e_new < e_best:
            pos_best, e_best = pos_new, e_new

    # --- Try a few more perturbations if time allows ---
    if time.time() - t0 < DEADLINE - 15.0:
        for s in range(3):
            if time.time() - t0 > DEADLINE - 10.0:
                break
            noise = rng.normal(0, 0.3, size=(N, 3))
            pos_pert = pos0 + noise
            pos_new, e_new, _ = _relax(pos_pert, time_limit=12.0)
            if e_new < e_best:
                pos_best, e_best = pos_new, e_new

    # Final high-precision polish
    if time.time() - t0 < DEADLINE - 5.0:
        pos_final, e_final, _ = _relax(pos_best, time_limit=5.0)
        if e_final < e_best:
            pos_best, e_best = pos_final, e_final

    return pos_best
