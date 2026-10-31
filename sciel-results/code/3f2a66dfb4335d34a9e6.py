"""
LJ38 cluster global minimum search.

Strategy:
1. Construct the known FCC truncated-octahedron geometry analytically.
2. Construct an icosahedral starting geometry as an alternative basin.
3. Use L-BFGS relaxation with analytic gradients from multiple starting points.
4. Final high-precision polish to converge tightly to the local minimum.

Entry point: construct() -> (N, 3) array of Cartesian coordinates.
"""
import time
import numpy as np

N = 38


def _lj_energy(pos):
    """Total LJ energy for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 ** 2
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    return 0.5 * np.sum(pair_energy)


def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 ** 2

    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad


def _make_fcc_truncated_octahedron():
    """
    Build the FCC truncated-octahedron geometry for LJ38.
    
    The 38-atom FCC cluster is a truncated octahedron taken from the FCC lattice.
    We generate FCC points, sort by distance from origin, and take the 38 closest.
    Then scale to equilibrium nearest-neighbor distance and center.
    """
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))

    pts = np.array(pts, dtype=np.float64)
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]

    # Take the 38 closest to the origin
    pts = pts[:N].copy()

    # Center
    pts -= np.mean(pts, axis=0)

    # Scale so that the mean nearest-neighbor distance is 2^(1/6)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        pts *= scale

    return pts


def _make_icosahedral_structure():
    """
    Build an icosahedral starting geometry for LJ38.
    
    The icosahedral cluster has:
    - 1 atom at center
    - 12 atoms at icosahedral vertices (inner shell)
    - 20 atoms at face centers of the icosahedron (outer shell)
    - 5 atoms to reach 38 (additional outer layer)
    
    We construct the standard icosahedral geometry and scale to LJ equilibrium.
    """
    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    # Icosahedron vertices (12 points)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=np.float64)

    # Normalize to unit sphere
    verts /= np.linalg.norm(verts[0])

    # Center atom
    center = np.array([[0.0, 0.0, 0.0]])

    # Face centers of icosahedron (20 faces)
    # Each face is a triangle of 3 vertices. We compute the centroid of each face.
    # Icosahedron faces (indices into verts):
    faces = [
        [0, 1, 8], [0, 1, 10], [0, 2, 10], [0, 2, 11], [0, 3, 11],
        [0, 3, 9], [0, 8, 9], [0, 9, 10], [0, 10, 11], [0, 11, 8],
        [1, 4, 8], [1, 4, 12], [1, 5, 12], [1, 5, 10], [1, 12, 9],
        [2, 6, 10], [2, 6, 13], [2, 7, 13], [2, 7, 11], [2, 13, 9],
        [3, 8, 12], [3, 8, 14], [3, 9, 14], [3, 11, 15], [3, 11, 14],
        [4, 12, 13], [4, 12, 15], [4, 13, 14], [4, 15, 14], [5, 10, 13],
        [5, 10, 15], [5, 13, 14], [5, 14, 15], [6, 13, 15], [6, 13, 14],
        [6, 14, 15], [7, 11, 14], [7, 11, 15], [7, 14, 15],
    ]
    # That's 38 faces listed but icosahedron has only 20 faces. Let me use the correct 20 faces.
    # Correct icosahedron face list:
    faces = [
        [0, 1, 8], [0, 1, 10], [0, 2, 10], [0, 2, 11], [0, 3, 11],
        [0, 3, 9], [0, 8, 9], [0, 9, 10], [0, 10, 11], [0, 11, 8],
        [1, 4, 8], [1, 4, 12], [1, 5, 12], [1, 5, 10], [1, 12, 9],
        [2, 6, 10], [2, 6, 13], [2, 7, 13], [2, 7, 11], [2, 13, 9],
        [3, 8, 12], [3, 8, 14], [3, 9, 14], [3, 11, 15], [3, 11, 14],
        [4, 12, 13], [4, 12, 15], [4, 13, 14], [4, 15, 14], [5, 10, 13],
        [5, 10, 15], [5, 13, 14], [5, 14, 15], [6, 13, 15], [6, 13, 14],
        [6, 14, 15], [7, 11, 14], [7, 11, 15], [7, 14, 15],
    ]
    # Actually let me just use a simpler approach: generate points on a sphere in icosahedral pattern
    # and use the known shell structure.
    
    # Simpler approach: use the known icosahedral shell radii
    # Inner shell: 12 atoms at radius r1
    # Outer shell: 25 atoms at radius r2 (for 38 total: 1+12+25)
    # For LJ equilibrium, r1 ≈ 2^(1/6) * 1.0, r2 ≈ 2^(1/6) * sqrt(2) approximately
    
    r_eq = 2.0 ** (1.0 / 6.0)
    
    # Inner shell: icosahedron vertices scaled
    inner = verts * r_eq
    
    # Outer shell: 25 atoms
    # 20 face centers + 5 additional
    # Face centers of the 12-vertex icosahedron
    # Use the 20 triangular faces
    face_list = [
        [0, 1, 8], [0, 1, 10], [0, 2, 10], [0, 2, 11], [0, 3, 11],
        [0, 3, 9], [0, 8, 9], [0, 9, 10], [0, 10, 11], [0, 11, 8],
        [1, 4, 8], [1, 4, 12], [1, 5, 12], [1, 5, 10], [1, 12, 9],
        [2, 6, 10], [2, 6, 13], [2, 7, 13], [2, 7, 11], [2, 13, 9],
    ]
    # That's only 20 but indices go up to 15 which is wrong for 12 vertices.
    # Let me use a different approach: just place 25 points on a sphere at the outer radius
    # in an approximately icosahedral pattern.
    
    # For simplicity, use a Fibonacci sphere for the outer 25 points
    n_outer = 25
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    outer = np.zeros((n_outer, 3))
    for i in range(n_outer):
        z = 1.0 - 2.0 * i / (n_outer - 1)
        radius_xy = np.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * i
        outer[i] = [radius_xy * np.cos(theta), radius_xy * np.sin(theta), z]
    
    # Scale outer shell
    outer *= r_eq * 1.8  # approximate outer shell radius
    
    # Combine: center + inner + outer
    all_pts = np.vstack([center, inner, outer])
    
    # Scale to ensure nearest-neighbor distance is 2^(1/6)
    diff = all_pts[:, np.newaxis, :] - all_pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = r_eq / avg_nn
        all_pts *= scale

    return all_pts


def _relax(pos, max_iter=400, gtol=1e-10, ftol=1e-14):
    """L-BFGS-B local relaxation with analytic gradient."""
    from scipy.optimize import minimize

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


def _polish(pos, max_iter=1000):
    """High-precision polish pass with very tight tolerances."""
    return _relax(pos, max_iter=max_iter, gtol=1e-12, ftol=1e-16)


def _ensure_min_sep(pos, min_sep=0.5):
    """Scale up the cluster if any pair is closer than min_sep."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = min_sep / min_d * 1.01
        pos = pos * scale
    return pos


def _try_external_relax(pos, max_iter=1000):
    """Try to use science_chemistry.relax if available."""
    try:
        import science_chemistry
        if hasattr(science_chemistry, 'relax'):
            relaxed = science_chemistry.relax(pos.copy())
            if relaxed is not None and len(relaxed) == N:
                return np.array(relaxed, dtype=np.float64)
    except Exception:
        pass
    return None


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin within 150 s hard limit

    # ---- 1. Build the direct FCC truncated-octahedron geometry ----
    try:
        fcc = _make_fcc_truncated_octahedron()
    except Exception:
        fcc = None

    if fcc is None:
        # Fallback: cubic lattice fragment
        side = int(np.ceil(N ** (1.0 / 3.0)))
        pts = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    pts.append([i, j, k])
        pts = np.array(pts[:N], dtype=np.float64)
        pts -= np.mean(pts, axis=0)
        fcc = pts * 1.12

    best_pos = fcc.copy()
    best_energy = _lj_energy(fcc)

    # ---- 2. Build icosahedral starting geometry ----
    try:
        ico = _make_icosahedral_structure()
    except Exception:
        ico = None

    # ---- 3. Try external relax on FCC (may be faster/more robust) ----
    if time.time() < deadline - 40.0:
        try:
            ext_relaxed = _try_external_relax(fcc)
            if ext_relaxed is not None:
                ext_energy = _lj_energy(ext_relaxed)
                if ext_energy < best_energy:
                    best_energy = ext_energy
                    best_pos = ext_relaxed.copy()
        except Exception:
            pass

    # ---- 4. L-BFGS relaxation from FCC ----
    if time.time() < deadline - 30.0:
        try:
            p, e = _relax(fcc, max_iter=400)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    # ---- 5. L-BFGS relaxation from icosahedral structure ----
    if ico is not None and time.time() < deadline - 30.0:
        try:
            p, e = _relax(ico, max_iter=400)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    # ---- 6. Perturbed FCC variants to escape shallow local minima ----
    rng = np.random.default_rng(12345)
    for trial in range(8):
        if time.time() > deadline - 25.0:
            break
        try:
            noise_scale = 0.02 + 0.01 * trial
            perturbed = fcc + noise_scale * rng.standard_normal(fcc.shape)
            perturbed = _ensure_min_sep(perturbed)
            p, e = _relax(perturbed, max_iter=250)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            continue

    # ---- 7. Perturbed icosahedral variants ----
    if ico is not None:
        for trial in range(4):
            if time.time() > deadline - 20.0:
                break
            try:
                noise_scale = 0.02 + 0.01 * trial
                perturbed = ico + noise_scale * rng.standard_normal(ico.shape)
                perturbed = _ensure_min_sep(perturbed)
                p, e = _relax(perturbed, max_iter=250)
                if e < best_energy:
                    best_energy = e
                    best_pos = p.copy()
            except Exception:
                continue

    # ---- 8. High-precision polish ----
    best_pos = _ensure_min_sep(best_pos)
    if time.time() < deadline - 10.0:
        try:
            p, e = _polish(best_pos, max_iter=800)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    # ---- 9. Final micro-polish ----
    best_pos = _ensure_min_sep(best_pos)
    if time.time() < deadline - 3.0:
        try:
            p, e = _polish(best_pos, max_iter=500)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    best_pos = _ensure_min_sep(best_pos)
    return best_pos
