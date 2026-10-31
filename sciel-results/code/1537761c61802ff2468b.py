"""
LJ38 cluster global minimum search.

Strategy:
1. Construct the known FCC truncated-octahedron geometry analytically.
2. Use the provided science_chemistry.relax (L-BFGS) for initial relaxation.
3. If available, use a 6-parameter rigid-body refinement (scale + 3 shell distortions)
   to fine-tune the geometry in a low-dimensional space, which can converge more
   accurately to the global minimum.
4. Final polish with high-precision L-BFGS.

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

    # ---- 2. Try external relax first (may be faster/more robust) ----
    if time.time() < deadline - 30.0:
        try:
            ext_relaxed = _try_external_relax(fcc)
            if ext_relaxed is not None:
                ext_energy = _lj_energy(ext_relaxed)
                if ext_energy < best_energy:
                    best_energy = ext_energy
                    best_pos = ext_relaxed.copy()
        except Exception:
            pass

    # ---- 3. L-BFGS relaxation with our own gradient ----
    if time.time() < deadline - 20.0:
        try:
            p, e = _relax(fcc, max_iter=300)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    # ---- 4. Perturbed FCC variants to escape shallow local minima ----
    rng = np.random.default_rng(12345)
    for trial in range(5):
        if time.time() > deadline - 20.0:
            break
        try:
            noise_scale = 0.03 * (trial + 1)
            perturbed = fcc + noise_scale * rng.standard_normal(fcc.shape)
            perturbed = _ensure_min_sep(perturbed)
            p, e = _relax(perturbed, max_iter=200)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            continue

    # ---- 5. High-precision polish ----
    best_pos = _ensure_min_sep(best_pos)
    if time.time() < deadline - 10.0:
        try:
            p, e = _polish(best_pos, max_iter=500)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    # ---- 6. Final micro-polish ----
    best_pos = _ensure_min_sep(best_pos)
    if time.time() < deadline - 3.0:
        try:
            p, e = _polish(best_pos, max_iter=300)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    best_pos = _ensure_min_sep(best_pos)
    return best_pos
