"""Lennard-Jones cluster global minimum, 55 atoms.
Strategy: Mackay icosahedron template with radial parameter optimization,
then full Cartesian L-BFGS polish.
"""
N = 55
import numpy as np
import time
import math
from scipy.optimize import minimize

# Deadline guard
_START = time.time()
DEADLINE = 110.0  # safety margin before 120s kill

def _elapsed():
    return time.time() - _START

def _remaining():
    return max(0.0, DEADLINE - _elapsed())

def _lj_energy(pos):
    """Compute total Lennard-Jones energy 4*sum(r^-12 - r^-6) over all pairs."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    r2 = r2[mask]
    r6 = 1.0 / r2 ** 3
    r12 = r6 ** 2
    return 4.0 * np.sum(r12 - r6)

def _lj_grad(pos):
    """Gradient of LJ energy."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid div by zero
    r6 = 1.0 / r2 ** 3
    r12 = r6 ** 2
    # dE/dr2 = 4 * (-12*r^-14 * 2r + 6*r^-8 * 2r) ... let's do it properly
    # E_pair = 4*(r^-12 - r^-6)
    # dE_pair/dr = 4*(-12*r^-13 + 6*r^-7) = 4*(-12/r^13 + 6/r^7)
    # dE_pair/dr2 = dE_pair/dr * dr/dr2 = dE_pair/dr * (1/(2r)) = 4*(-6/r^14 + 3/r^8) = 4*(3*r^-8 - 6*r^-14)
    # = 4*(3*r6 - 6*r12) where r6 = r^-6, r12 = r^-12... wait
    # r^-8 = r6/r2, r^-14 = r12/r2
    # dE_pair/dr2 = 4*(3*r6/r2 - 6*r12/r2) = 4*(3*r6 - 6*r12)/r2
    # grad_i = sum_{j != i} dE_pair/dr2 * (pos_i - pos_j) ... actually:
    # dE/dr2_ij * (pos_i - pos_j) for the gradient w.r.t. pos_i
    # Let's use: grad_i = sum_j 4*(3*r6_ij - 6*r12_ij)/r2_ij * (pos_i - pos_j)
    coeff = 4.0 * (3.0 * r6 - 6.0 * r12) / r2  # shape (n, n)
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad

def _mckay_icosahedron(r1, r2, r3):
    """
    Generate 55-atom Mackay icosahedron positions.
    Shell 0: 1 atom at center (radius 0)
    Shell 1: 12 atoms at icosahedron vertices (radius r1)
    Shell 2: 30 atoms at edge midpoints (radius r2)
    Shell 3: 12 atoms at face centers (radius r3)
    Total: 1 + 12 + 30 + 12 = 55
    """
    # Icosahedron vertices: (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    verts_norm = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Edge midpoints: each edge connects two vertices.
    # For an icosahedron, each vertex is connected to 5 others.
    # 30 edges total. We need to find all pairs that are edges.
    # Two vertices are connected if their dot product is -1/phi ≈ -0.618 (for unit icosahedron)
    # Actually for the standard icosahedron with vertices as above (unit length after normalization),
    # adjacent vertices have dot product = -1/phi.
    dot_thresh = -1.0 / phi + 1e-9
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.dot(verts_norm[i], verts_norm[j])
            if abs(d - (-1.0 / phi)) < 1e-6:
                edges.append((i, j))
    
    # Face centers: 20 faces. Each face is a triangle of 3 vertices.
    # We need 12 of them for the Mackay icosahedron (one per vertex direction? No...)
    # Actually in Mackay icosahedron, the 3rd shell has 12 atoms at face centers.
    # Wait: 1 + 12 + 30 + 12 = 55. The 12 face-center atoms correspond to... 
    # Actually for a two-shell Mackay icosahedron: inner shell = 13 (1+12), outer shell = 42 (30+12).
    # The 30 are edge midpoints, 12 are face centers of the outer icosahedron.
    # But an icosahedron has 20 faces, not 12. Let me reconsider.
    
    # Actually, the Mackay icosahedron for N=55:
    # - 1 center
    # - 12 at icosahedron vertices (1st shell)
    # - 30 at edge midpoints (2nd shell)  
    # - 12 at face centers... but icosahedron has 20 faces.
    
    # Let me reconsider: perhaps it's:
    # - 1 center
    # - 12 vertices
    # - 30 edge midpoints  
    # - 12 face centers (out of 20 faces, only 12 are used?)
    
    # Actually, I think the correct decomposition for the Mackay icosahedron N=55 is:
    # Inner icosahedron: 13 atoms (1 center + 12 vertices)
    # Outer icosahedron: 42 atoms = 12 vertices + 30 edge midpoints
    # But that's 13 + 42 = 55. The outer shell has 12 vertices + 30 edge midpoints = 42.
    
    # So we have TWO icosahedra: inner (13 atoms) and outer (42 atoms).
    # Inner: center + 12 vertices at radius r1
    # Outer: 12 vertices + 30 edge midpoints at radii r2 (vertices) and r3 (edge midpoints)
    
    # Let me redo:
    # Shell A (inner): 1 center + 12 icosahedron vertices at radius r1
    # Shell B (outer): 12 icosahedron vertices at radius r2 + 30 edge midpoints at radius r3
    # Total: 1 + 12 + 12 + 30 = 55
    
    # The outer icosahedron vertices are at the same directions as inner vertices.
    # Edge midpoints are at the average of adjacent vertex directions.
    
    # Edge midpoint directions:
    edge_mids = []
    for (i, j) in edges:
        mid = (verts_norm[i] + verts_norm[j]) / np.linalg.norm(verts_norm[i] + verts_norm[j])
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    
    # Build positions
    pos = []
    # Center
    pos.append(np.array([0.0, 0.0, 0.0]))
    # Inner vertices
    for v in verts_norm:
        pos.append(r1 * v)
    # Outer vertices (same directions as inner)
    for v in verts_norm:
        pos.append(r2 * v)
    # Outer edge midpoints
    for m in edge_mids:
        pos.append(r3 * m)
    
    return np.array(pos)

def _radial_energy(params):
    """Energy as function of radial parameters [r1, r2, r3]."""
    r1, r2, r3 = params
    if r1 < 0.1 or r2 < 0.1 or r3 < 0.1:
        return 1e10
    pos = _mckay_icosahedron(r1, r2, r3)
    return _lj_energy(pos)

def _lbfgs_polish(pos, time_limit=30.0):
    """Polish positions using L-BFGS."""
    t0 = time.time()
    try:
        res = minimize(
            _lj_energy, pos, jac=_lj_grad, method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        if time.time() - t0 > time_limit:
            return pos
        return res.x
    except Exception:
        return pos

def construct():
    global _START
    _START = time.time()
    
    # Known approximate Mackay icosahedron parameters
    # From literature, the optimal radii are approximately:
    # r1 ≈ 1.0-1.2 (inner shell), r2 ≈ 2.0-2.5 (outer vertices), r3 ≈ 2.0-2.5 (outer edge mids)
    
    best_energy = 1e10
    best_pos = None
    
    # Try multiple initial guesses for radial parameters
    initial_guesses = [
        [1.1, 2.2, 2.2],
        [1.0, 2.0, 2.0],
        [1.2, 2.4, 2.4],
        [1.1, 2.3, 2.3],
        [1.05, 2.1, 2.15],
        [1.15, 2.25, 2.3],
        [0.9, 2.0, 2.1],
        [1.3, 2.5, 2.5],
    ]
    
    for guess in initial_guesses:
        if _remaining() < 15.0:
            break
        
        # Nelder-Mead optimization in radial space
        try:
            res = minimize(
                _radial_energy, guess, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            r_opt = res.x
            pos = _mckay_icosahedron(*r_opt)
            e = _lj_energy(pos)
            
            if e < best_energy:
                best_energy = e
                best_pos = pos
        except Exception:
            continue
    
    # If we found a good radial solution, polish in Cartesian space
    if best_pos is not None and _remaining() > 10.0:
        polished = _lbfgs_polish(best_pos, time_limit=min(_remaining() - 5.0, 40.0))
        e_polished = _lj_energy(polished)
        if e_polished < best_energy:
            best_energy = e_polished
            best_pos = polished
    
    # If still no good result, fall back to a reasonable default
    if best_pos is None:
        pos = _mckay_icosahedron(1.1, 2.2, 2.2)
        best_pos = pos
        best_energy = _lj_energy(pos)
    
    # Final tight polish if time permits
    if _remaining() > 5.0:
        polished = _lbfgs_polish(best_pos, time_limit=min(_remaining() - 2.0, 20.0))
        e_final = _lj_energy(polished)
        if e_final < best_energy:
            best_pos = polished
            best_energy = e_final
    
    # Ensure minimum distance constraint (0.5 apart)
    # If any pair is too close, nudge them apart
    n = len(best_pos)
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1.0)
    if np.min(dists) < 0.5:
        # Simple fix: scale up slightly
        scale = 0.5 / np.min(dists) * 1.01
        best_pos = best_pos * scale
    
    return best_pos
