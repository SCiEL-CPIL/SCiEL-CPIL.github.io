"""
LJ38 cluster global minimum search.

Strategy:
  1. Generate the FCC truncated-octahedron geometry (the known global minimum
     for LJ38, energy ≈ -173.928427).
  2. Generate several other plausible starting geometries (icosahedral shell,
     random perturbed FCC, etc.).
  3. For each starting geometry, perform L-BFGS local relaxation using
     scipy.optimize.minimize with analytic gradient.
  4. Keep the best (lowest-energy) configuration found.
  5. Use a deadline guard to ensure we return well before the 120 s limit.
"""
import time
import numpy as np
from itertools import combinations

N = 38


def _lj_energy_and_grad(pos):
    """Compute total LJ energy and gradient for a set of positions.

    pos: (N, 3) array of Cartesian coordinates.
    Returns (energy, grad) where grad has same shape as pos.
    """
    n = len(pos)
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    r2 = np.sum(diff ** 2, axis=2)  # (N, N)
    # Avoid division by zero on diagonal
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r = np.sqrt(r2)
    r6 = r2 ** 3
    r12 = r6 ** 2

    # Energy: sum over i<j of 4*(r^-12 - r^-6)
    # Using full matrix and dividing by 2
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    # Gradient: dE/dr_i = sum_{j != i} 4 * (-12*r^-13*(r_i - r_j)/r + 6*r^-7*(r_i - r_j)/r)
    #           = sum_{j != i} 4 * (-12/r^14 + 6/r^8) * (r_i - r_j)
    # Let c = 4 * (-12/r^14 + 6/r^8) = 4*(6/r^8 - 12/r^14) = 24/r^8 - 48/r^14
    # grad_i = sum_j c_ij * (r_i - r_j)
    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)  # (N, N)
    np.fill_diagonal(coeff, 0.0)
    # grad_i = sum_j coeff[i,j] * (pos[i] - pos[j])
    #        = sum_j coeff[i,j]*pos[i] - sum_j coeff[i,j]*pos[j]
    #        = pos[i] * sum_j coeff[i,j] - (coeff @ pos)[i]
    row_sum = np.sum(coeff, axis=1, keepdims=True)  # (N, 1)
    grad = row_sum * pos - coeff @ pos

    return energy, grad


def _make_fcc_truncated_octahedron():
    """
    Construct the FCC truncated octahedron for N=38.

    Uses the median nearest-neighbour distance for robust scaling to the
    LJ minimum r0 = 2^(1/6).
    """
    # Generate FCC lattice points in a cubic box
    pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                pts.append(np.array([i, j, k], dtype=float))
                pts.append(np.array([i + 0.5, j + 0.5, k], dtype=float))
                pts.append(np.array([i + 0.5, j, k + 0.5], dtype=float))
                pts.append(np.array([i, j + 0.5, k + 0.5], dtype=float))
    pts = np.array(pts)
    pts = np.unique(np.round(pts, 12), axis=0)

    # Center
    center = pts.mean(axis=0)
    pts = pts - center

    # Select 38 closest to centroid
    dists = np.sum(pts ** 2, axis=1)
    idx = np.argsort(dists)[:N]
    pos = pts[idx].copy()

    # Compute nearest-neighbour distances to robustly scale
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, np.inf)
    r = np.sqrt(r2)
    # For each atom, its nearest neighbour distance
    nn_dists = np.min(r, axis=1)
    # Use median for robustness
    median_nn = np.median(nn_dists)

    # Scale so that median NN distance = r0
    r0 = 2.0 ** (1.0 / 6.0)
    if median_nn > 1e-12:
        scale = r0 / median_nn
        pos = pos * scale

    # Re-center
    pos -= pos.mean(axis=0)
    return pos


def _make_icosahedral_structure():
    """
    Generate an icosahedral-like structure for 38 atoms.
    Icosahedral shell: 12 vertices + 1 center + additional layers.
    For 38: 12 (outer) + 12 (middle) + 12 (inner) + 1 (center) = 37... 
    Let's do 12+12+12+2 or similar.
    
    Actually, a common icosahedral arrangement for ~38:
    - 1 center
    - 12 at radius r1 (icosahedron vertices)
    - 12 at radius r2 (edge midpoints or face centers)
    - 12 at radius r3
    - 1 extra? 
    
    Let's just make a reasonable icosahedral-like starting point.
    """
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(ico_verts, axis=1, keepdims=True)
    ico_verts /= norms
    
    # Build layers
    r1 = 1.0
    r2 = 1.8
    r3 = 2.6
    r4 = 3.4
    
    pts = []
    # Center
    pts.append([0, 0, 0])
    # Layer 1: icosahedron vertices at r1
    pts.extend(ico_verts * r1)
    # Layer 2: icosahedron vertices at r2 (slightly rotated)
    # Use the same directions but different radius
    pts.extend(ico_verts * r2)
    # Layer 3: icosahedron vertices at r3
    pts.extend(ico_verts * r3)
    # That's 1 + 12 + 12 + 12 = 37. Add one more near the center.
    pts.append([0.3, 0.3, 0.3])
    
    pts = np.array(pts[:N], dtype=float)
    # Scale
    from scipy.spatial.distance import cdist
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    nn_dists = np.min(D, axis=1)
    avg_nn = np.mean(nn_dists)
    target_nn = 2.0 ** (1.0 / 6.0)
    scale = target_nn / avg_nn if avg_nn > 0 else 1.0
    pts *= scale
    
    return pts


def _make_random_perturbed_fcc(rng):
    """Generate FCC truncated octahedron with random perturbations."""
    base = _make_fcc_truncated_octahedron()
    perturbation = 0.1 * rng.standard_normal(base.shape)
    return base + perturbation


def _relax_lbfgs(pos, max_iter=500, deadline=None, gtol=1e-10, ftol=1e-16):
    """
    Relax positions using L-BFGS-B with analytic gradient.
    Performs a two-stage relaxation for higher precision.
    Returns (positions, energy).
    """
    from scipy.optimize import minimize
    
    def fun_and_grad(x):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError
        pos = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    try:
        # Stage 1: Standard relaxation
        result1 = minimize(
            fun_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': gtol, 'ftol': ftol, 'maxls': 50}
        )
        
        pos_opt = result1.x.reshape(-1, 3)
        e_opt, _ = _lj_energy_and_grad(pos_opt)
        
        # Stage 2: High-precision polish
        # Check time if deadline is provided
        if deadline is not None and time.time() > deadline - 5.0:
            return pos_opt, e_opt
            
        result2 = minimize(
            fun_and_grad,
            result1.x,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-12, 'ftol': 1e-16, 'maxls': 50}
        )
        
        pos_polished = result2.x.reshape(-1, 3)
        e_polished, _ = _lj_energy_and_grad(pos_polished)
        
        if e_polished < e_opt:
            return pos_polished, e_polished
        else:
            return pos_opt, e_opt
            
    except TimeoutError:
        return pos, _lj_energy_and_grad(pos)[0]
    except Exception:
        return pos, _lj_energy_and_grad(pos)[0]


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110 s to be safe within 120 s limit
    
    rng = np.random.default_rng(42)
    
    best_pos = None
    best_energy = np.inf
    
    # 1. FCC truncated octahedron (best known)
    try:
        fcc = _make_fcc_truncated_octahedron()
        pos_opt, e_opt = _relax_lbfgs(fcc, max_iter=20000, deadline=deadline, gtol=1e-10, ftol=1e-16)
        
        # Verify minimum separation
        from scipy.spatial.distance import cdist
        D = cdist(pos_opt, pos_opt)
        np.fill_diagonal(D, np.inf)
        if np.min(D) >= 0.5:
            if e_opt < best_energy:
                best_energy = e_opt
                best_pos = pos_opt.copy()
    except Exception:
        pass
    
    # High-precision polish pass on the best result so far
    if best_pos is not None and time.time() < deadline - 10.0:
        try:
            pos_polished, e_polished = _relax_lbfgs(best_pos, max_iter=10000, deadline=deadline, gtol=1e-12, ftol=1e-16)
            from scipy.spatial.distance import cdist
            D = cdist(pos_polished, pos_polished)
            np.fill_diagonal(D, np.inf)
            if np.min(D) >= 0.5 and e_polished < best_energy:
                best_energy = e_polished
                best_pos = pos_polished.copy()
        except Exception:
            pass
    
    if time.time() > deadline - 20.0:
        if best_pos is not None:
            # Ensure minimum separation
            from scipy.spatial.distance import cdist
            D = cdist(best_pos, best_pos)
            np.fill_diagonal(D, np.inf)
            if np.min(D) < 0.5:
                scale = 0.5 / np.min(D) * 1.01
                best_pos = best_pos * scale
            return best_pos
    
    # 2. Icosahedral structure
    try:
        ico = _make_icosahedral_structure()
        pos_opt, e_opt = _relax_lbfgs(ico, max_iter=15000, deadline=deadline, gtol=1e-10, ftol=1e-16)
        
        # Verify minimum separation
        from scipy.spatial.distance import cdist
        D = cdist(pos_opt, pos_opt)
        np.fill_diagonal(D, np.inf)
        if np.min(D) >= 0.5 and e_opt < best_energy:
            best_energy = e_opt
            best_pos = pos_opt.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 20.0:
        if best_pos is not None:
            from scipy.spatial.distance import cdist
            D = cdist(best_pos, best_pos)
            np.fill_diagonal(D, np.inf)
            if np.min(D) < 0.5:
                scale = 0.5 / np.min(D) * 1.01
                best_pos = best_pos * scale
            return best_pos
    
    # 3. Multi-scale perturbations around the best FCC result
    base = _make_fcc_truncated_octahedron()
    for sigma in [0.01, 0.03, 0.05, 0.08, 0.12, 0.18]:
        if time.time() > deadline - 15.0:
            break
        for _ in range(3):
            if time.time() > deadline - 10.0:
                break
            try:
                pos_perturbed = base + rng.standard_normal(base.shape) * sigma
                pos_opt, e_opt = _relax_lbfgs(pos_perturbed, max_iter=10000, deadline=deadline, gtol=1e-10, ftol=1e-16)
                
                # Verify minimum separation
                from scipy.spatial.distance import cdist
                D = cdist(pos_opt, pos_opt)
                np.fill_diagonal(D, np.inf)
                if np.min(D) >= 0.5 and e_opt < best_energy:
                    best_energy = e_opt
                    best_pos = pos_opt.copy()
            except Exception:
                continue
    
    # 4. Simple cubic (original seed) as fallback
    try:
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        cubic = np.array(points[:N], dtype=float) * 1.12
        pos_opt, e_opt = _relax_lbfgs(cubic, max_iter=5000, deadline=deadline, gtol=1e-10, ftol=1e-16)
        
        # Verify minimum separation
        from scipy.spatial.distance import cdist
        D = cdist(pos_opt, pos_opt)
        np.fill_diagonal(D, np.inf)
        if np.min(D) >= 0.5 and e_opt < best_energy:
            best_energy = e_opt
            best_pos = pos_opt.copy()
    except Exception:
        pass
    
    # If no valid result, return the FCC structure scaled
    if best_pos is None:
        try:
            best_pos = _make_fcc_truncated_octahedron()
        except Exception:
            # Last resort: simple cubic
            side = int(np.ceil(N ** (1 / 3)))
            points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
            points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
            best_pos = np.array(points[:N], dtype=float) * 1.12
    
    # Final high-precision polish if time remains
    if time.time() < deadline - 5.0:
        try:
            pos_polished, e_polished = _relax_lbfgs(best_pos, max_iter=10000, deadline=deadline, gtol=1e-12, ftol=1e-16)
            from scipy.spatial.distance import cdist
            D = cdist(pos_polished, pos_polished)
            np.fill_diagonal(D, np.inf)
            if np.min(D) >= 0.5 and e_polished < best_energy:
                best_energy = e_polished
                best_pos = pos_polished.copy()
        except Exception:
            pass
    
    # Final check: ensure minimum separation
    from scipy.spatial.distance import cdist
    D = cdist(best_pos, best_pos)
    np.fill_diagonal(D, np.inf)
    if np.min(D) < 0.5:
        scale = 0.5 / np.min(D) * 1.01
        best_pos = best_pos * scale
    
    return best_pos
