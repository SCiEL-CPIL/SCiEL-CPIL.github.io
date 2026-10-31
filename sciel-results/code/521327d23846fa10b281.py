"""
LJ38 cluster global minimum search.

Strategy:
  1. Generate the FCC truncated-octahedron geometry using the known shell
     structure (1 center + 12 + 12 + 13 atoms in FCC directions).
  2. Generate perturbed FCC variants as additional starting points.
  3. For each starting geometry, perform L-BFGS local relaxation with
     analytic gradient.
  4. Apply a high-precision polish pass on the best result.
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
    Construct the FCC truncated octahedron for N=38 using the known shell structure.
    
    The LJ38 global minimum is a face-centered-cubic truncated octahedron with
    shells: 1 (center) + 12 (first shell) + 12 (second shell) + 13 (third shell) = 38.
    
    The FCC lattice directions are generated from all permutations of (±1, ±1, 0)
    type vectors. We build the shells at radii that give nearest-neighbor
    distance ≈ 2^(1/6) and let the optimizer fine-tune.
    """
    from scipy.spatial.distance import cdist
    
    # FCC directions: all distinct permutations of (1, 1, 0) with all sign combos
    # This gives 12 directions for the first shell
    fcc_dirs = []
    for perm in [(1, 1, 0), (1, 0, 1), (0, 1, 1)]:
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    d = (sx * perm[0], sy * perm[1], sz * perm[2])
                    # Only include if sum of coords is even (FCC condition)
                    if (d[0] + d[1] + d[2]) % 2 == 0:
                        fcc_dirs.append(d)
    
    # Deduplicate
    fcc_dirs = list(set(fcc_dirs))
    fcc_dirs = np.array(fcc_dirs, dtype=float)
    # Normalize
    norms = np.linalg.norm(fcc_dirs, axis=1, keepdims=True)
    fcc_dirs /= norms
    
    # Second shell: permutations of (2, 0, 0) type - the 6 face centers
    face_dirs = []
    for axis in range(3):
        for sign in (1, -1):
            d = [0.0, 0.0, 0.0]
            d[axis] = float(sign)
            face_dirs.append(d)
    face_dirs = np.array(face_dirs, dtype=float)
    face_dirs /= np.linalg.norm(face_dirs, axis=1, keepdims=True)
    
    # Third shell: combinations of FCC and face directions
    # For 13 atoms in the third shell, use FCC dirs + one extra
    # Actually, let's use a different approach: generate FCC lattice points
    # and select by shell structure
    
    # Better approach: use the explicit FCC lattice construction
    # FCC points are at (i, j, k) where i+j+k is even, scaled by a/2
    # where a is the cubic lattice constant.
    # Nearest neighbor distance in FCC = a/sqrt(2)
    # For LJ equilibrium: a/sqrt(2) = 2^(1/6), so a = sqrt(2) * 2^(1/6)
    
    a = np.sqrt(2.0) * 2.0 ** (1.0 / 6.0)
    half_a = a / 2.0
    
    # Generate FCC lattice points within a radius
    max_idx = 3
    fcc_pts = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                if (i + j + k) % 2 == 0:
                    pt = np.array([i * half_a, j * half_a, k * half_a])
                    fcc_pts.append(pt)
    
    fcc_pts = np.array(fcc_pts, dtype=float)
    dists = np.linalg.norm(fcc_pts, axis=1)
    order = np.argsort(dists)
    fcc_pts = fcc_pts[order]
    
    # Take 38 closest to origin
    pts = fcc_pts[:N].copy()
    
    # Center
    pts -= np.mean(pts, axis=0)
    
    # Use median-NN scaling for robustness
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    nn_dists = np.min(D, axis=1)
    median_nn = np.median(nn_dists)
    target_nn = 2.0 ** (1.0 / 6.0)
    scale = target_nn / median_nn if median_nn > 0 else 1.0
    pts *= scale
    
    return pts


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


def _relax_lbfgs(pos, max_iter=500, deadline=None, gtol=1e-8, ftol=1e-12):
    """
    Relax positions using L-BFGS-B with analytic gradient.
    Returns (positions, energy).
    """
    from scipy.optimize import minimize
    
    def fun_and_grad(x):
        pos = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    result = minimize(
        fun_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'gtol': gtol, 'ftol': ftol}
    )
    
    pos_opt = result.x.reshape(-1, 3)
    e_opt, _ = _lj_energy_and_grad(pos_opt)
    
    return pos_opt, e_opt


def _polish_high_precision(pos, deadline=None):
    """
    High-precision polish: run L-BFGS with very tight tolerances on the
    best configuration found.
    """
    remaining = deadline - time.time() if deadline else 60.0
    if remaining < 5.0:
        return pos, _lj_energy_and_grad(pos)[0]
    
    max_iter = 1000
    # Use even tighter tolerances
    pos_opt, e_opt = _relax_lbfgs(
        pos, max_iter=max_iter, deadline=deadline,
        gtol=1e-14, ftol=1e-18
    )
    
    return pos_opt, e_opt


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115 s to be safe within 120 s limit
    
    rng = np.random.default_rng(42)
    
    best_pos = None
    best_energy = np.inf
    
    # Phase 1: Relax the primary FCC start with moderate precision
    try:
        fcc = _make_fcc_truncated_octahedron()
        # Ensure minimum separation
        from scipy.spatial.distance import cdist
        D = cdist(fcc, fcc)
        np.fill_diagonal(D, np.inf)
        if np.min(D) < 0.5:
            scale = 0.5 / np.min(D) * 1.01
            fcc = fcc * scale
        
        pos_opt, e_opt = _relax_lbfgs(fcc, max_iter=500, deadline=deadline,
                                       gtol=1e-10, ftol=1e-14)
        
        D = cdist(pos_opt, pos_opt)
        np.fill_diagonal(D, np.inf)
        if np.min(D) >= 0.5 and e_opt < best_energy:
            best_energy = e_opt
            best_pos = pos_opt.copy()
    except Exception:
        pass
    
    # Phase 2: Try perturbed FCC variants if time permits
    elapsed = time.time() - start_time
    if elapsed < 60.0:
        for i in range(3):
            if time.time() > deadline - 40:
                break
            try:
                perturbed = _make_random_perturbed_fcc(rng)
                D = cdist(perturbed, perturbed)
                np.fill_diagonal(D, np.inf)
                if np.min(D) < 0.5:
                    scale = 0.5 / np.min(D) * 1.01
                    perturbed = perturbed * scale
                
                pos_opt, e_opt = _relax_lbfgs(perturbed, max_iter=300, deadline=deadline,
                                               gtol=1e-10, ftol=1e-14)
                
                D = cdist(pos_opt, pos_opt)
                np.fill_diagonal(D, np.inf)
                if np.min(D) >= 0.5 and e_opt < best_energy:
                    best_energy = e_opt
                    best_pos = pos_opt.copy()
            except Exception:
                continue
    
    # Phase 3: Try icosahedral start if time permits
    if time.time() < deadline - 30:
        try:
            ico = _make_icosahedral_structure()
            D = cdist(ico, ico)
            np.fill_diagonal(D, np.inf)
            if np.min(D) < 0.5:
                scale = 0.5 / np.min(D) * 1.01
                ico = ico * scale
            
            pos_opt, e_opt = _relax_lbfgs(ico, max_iter=300, deadline=deadline,
                                           gtol=1e-10, ftol=1e-14)
            
            D = cdist(pos_opt, pos_opt)
            np.fill_diagonal(D, np.inf)
            if np.min(D) >= 0.5 and e_opt < best_energy:
                best_energy = e_opt
                best_pos = pos_opt.copy()
        except Exception:
            pass
    
    # Fallback if nothing worked
    if best_pos is None:
        try:
            best_pos = _make_fcc_truncated_octahedron()
        except Exception:
            side = int(np.ceil(N ** (1 / 3)))
            points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
            points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
            best_pos = np.array(points[:N], dtype=float) * 1.12
    
    # Phase 4: High-precision polish on the best result
    if time.time() < deadline - 10:
        try:
            best_pos, best_energy = _polish_high_precision(best_pos, deadline=deadline)
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
