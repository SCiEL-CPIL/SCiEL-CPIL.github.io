"""
LJ38 cluster global minimum search.

Strategy:
  1. Build the known FCC truncated-octahedron geometry (the putative global minimum).
  2. Optimize the lattice distortion parameter to better match surface effects.
  3. Apply a high-precision L-BFGS-B polish to remove any residual numerical error.
  4. Return the polished coordinates.

The FCC truncated octahedron for N=38 is the established global minimum of the
Lennard-Jones cluster with energy ≈ -173.928427 (Cambridge Cluster Database).
By constructing it explicitly, optimizing distortion, and polishing, we avoid the
icosahedral funnel that traps naive local searches from a cubic-lattice seed.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 38


def _lj_energy_and_grad(pos):
    """
    Total LJ energy and gradient for positions pos (N,3).
    Uses a numerically stable formulation with r2 directly.
    """
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # Avoid division by zero on diagonal
    
    r6 = r2 ** 3
    r12 = r6 ** 2

    # Energy: sum over all pairs, but since we compute both (i,j) and (j,i),
    # we multiply by 0.5 to get the correct total.
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    # Gradient derivation:
    # E = 2 * sum_{i<j} (r_ij^-12 - r_ij^-6)
    # dE/dx_i = 4 * sum_{j != i} (x_i - x_j) * (-6 * r_ij^-14 + 3 * r_ij^-8)
    # Let's verify signs.
    # U = 4 * (r^-12 - r^-6)
    # dU/dr = 4 * (-12 r^-13 + 6 r^-7)
    # dU/dx_i = (dU/dr) * (dr/dx_i) = 4 * (-12 r^-13 + 6 r^-7) * (x_i - x_j)/r
    #         = 4 * (-12 r^-14 + 6 r^-8) * (x_i - x_j)
    #         = 4 * (x_i - x_j) * (-12 / r^14 + 6 / r^8)
    #
    # Using r2:
    # r^-14 = (r2)^-7
    # r^-8  = (r2)^-4
    # coeff = 4 * (-12 * (r2)^-7 + 6 * (r2)^-4)
    
    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    
    # grad_i = sum_j coeff_ij * (x_i - x_j)
    #        = x_i * sum_j coeff_ij - sum_j coeff_ij * x_j
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad


def _make_fcc_truncated_octahedron(distortion=1.0):
    """
    Build the FCC truncated-octahedron geometry for LJ38 with optional distortion.
    
    The distortion parameter allows scaling the z-axis independently to account
    for surface effects that might compress or expand the cluster vertically.
    """
    # Generate FCC points
    max_coord = 5
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    points.append((float(i), float(j), float(k)))
    
    points = np.array(points)
    
    # Center the cluster
    points -= np.mean(points, axis=0)
    
    # Sort by distance from origin
    dists = np.sum(points ** 2, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:N]]
    
    # Re-center after selection
    points -= np.mean(points, axis=0)
    
    # Apply distortion to z-axis
    points[:, 2] *= distortion
    
    # Scale so that nearest-neighbor distance = 2^(1/6)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    min_r = np.sqrt(min_r2)
    
    if min_r > 1e-10:
        scale = 2.0 ** (1.0 / 6.0) / min_r
        points = points * scale
    
    return points


def _relax(pos, max_iter=2000, gtol=1e-10, ftol=1e-14):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    res = minimize(
        obj_grad,
        pos.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol},
    )
    p_out = res.x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out


def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure all pairs are at least min_sep apart."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = (min_sep / min_d) * 1.01
        pos = pos * scale
    return pos


def construct():
    """
    Construct the 38-atom LJ cluster at (or very near) the global minimum.
    Returns array of shape (38, 3).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_coords = None
    best_energy = np.inf
    
    # Step 1: Build FCC truncated octahedron with default distortion
    coords = _make_fcc_truncated_octahedron(distortion=1.0)
    coords = _ensure_min_sep(coords)
    
    # Step 2: Optimize distortion parameter
    # We'll try a few distortion values to find the one that gives the lowest energy after relaxation
    distortions = [1.0, 0.95, 1.05, 0.9, 1.1]
    
    for dist in distortions:
        if time.time() > deadline - 30.0:
            break
            
        try:
            coords_dist = _make_fcc_truncated_octahedron(distortion=dist)
            coords_dist = _ensure_min_sep(coords_dist)
            
            # Quick relaxation to see if this distortion helps
            p, e = _relax(coords_dist, max_iter=1000, gtol=1e-8, ftol=1e-12)
            
            if e < best_energy:
                best_energy = e
                best_coords = p
        except Exception:
            continue
    
    # If no distortion worked well, fall back to default
    if best_coords is None:
        try:
            best_coords, best_energy = _relax(coords, max_iter=2000, gtol=1e-10, ftol=1e-14)
        except Exception:
            best_coords = coords
            best_energy = _lj_energy_and_grad(best_coords)[0]
    
    # Step 3: High-precision polish with L-BFGS-B
    if time.time() < deadline - 20.0 and best_coords is not None:
        try:
            polished, e_polished = _relax(best_coords, max_iter=5000, gtol=1e-12, ftol=1e-16)
            if e_polished < best_energy:
                best_coords = polished
                best_energy = e_polished
        except Exception:
            pass
    
    # Step 4: Additional polish if time permits
    if time.time() < deadline - 10.0 and best_coords is not None:
        try:
            polished, e_polished = _relax(best_coords, max_iter=5000, gtol=1e-14, ftol=1e-18)
            if e_polished < best_energy:
                best_coords = polished
                best_energy = e_polished
        except Exception:
            pass
    
    # Final validation
    best_coords = _ensure_min_sep(best_coords)
    
    return best_coords
