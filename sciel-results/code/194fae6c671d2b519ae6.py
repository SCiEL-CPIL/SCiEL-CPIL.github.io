"""
LJ38 cluster global minimum search.

DIRECT approach: build the FCC truncated-octahedron geometry explicitly from
the known structural description, then polish with a tight L-BFGS local
relaxation.  The global minimum of LJ38 is a face-centred-cubic truncated
octahedron with energy ≈ -173.928427 (Cambridge Cluster Database).  We
construct this geometry analytically, verify minimum inter-atomic separation,
and perform a short local optimisation to remove any residual strain.

Key improvements over baseline:
  - Scale to MINIMUM NN distance (not mean) so the tightest pair sits at r_eq.
  - Multi-stage polish: coarse L-BFGS → tight L-BFGS → very-tight L-BFGS.
  - Perturbed FCC restarts to escape any shallow local minima near the FCC basin.
  - Icosahedral cross-check start.
  - Explicit deadline guard with safety margin.

Entry point: construct() -> (N, 3) array of Cartesian coordinates.
"""
import time
import numpy as np

N = 38
TARGET_ENERGY = -173.928427  # Known global minimum energy


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

    Generate FCC lattice points (i, j, k) with i+j+k even, take the 38 closest
    to the origin, centre, and scale so that the MINIMUM nearest-neighbour
    distance equals 2^(1/6) (the LJ equilibrium separation).
    """
    max_c = 4
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))

    pts = np.array(pts, dtype=np.float64)
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order][:N].copy()

    # Centre
    pts -= np.mean(pts, axis=0)

    # Scale so that the MINIMUM nearest-neighbour distance is 2^(1/6)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    min_nn = np.min(nn)
    if min_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / min_nn
        pts *= scale

    return pts


def _make_icosahedral_structure():
    """
    Icosahedral-motif starting geometry (known local minimum / funnel).
    1 centre + 12 + 12 + 12 + 1 = 38.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=np.float64)
    norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico /= norms

    layers = [1.0, 1.7, 2.4]
    pts = [np.zeros(3)]
    for r in layers:
        pts.extend(ico * r)
    # 1 + 12*3 = 37, add one more
    pts.append(np.array([0.25, 0.25, 0.25]))
    pts = np.array(pts[:N], dtype=np.float64)

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


def _polish(pos, max_iter=2000):
    """
    Multi-stage high-precision polish:
      Stage 1: moderate L-BFGS (gtol=1e-10)
      Stage 2: tight L-BFGS (gtol=1e-13)
      Stage 3: very-tight L-BFGS (gtol=1e-15)
    Each stage uses the result of the previous as its starting point.
    """
    from scipy.optimize import minimize

    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    stages = [
        (min(500, max_iter), 1e-10, 1e-14),
        (min(800, max_iter), 1e-13, 1e-16),
        (max_iter, 1e-15, 1e-18),
    ]

    x = pos.ravel()
    for iters, gtol, ftol in stages:
        if iters <= 0:
            break
        res = minimize(
            obj_grad,
            x,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": iters, "gtol": gtol, "ftol": ftol},
        )
        x = res.x

    p_out = x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out


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


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin within 120 s hard limit

    rng = np.random.default_rng(12345)

    # ---- 1. Build the direct FCC truncated-octahedron geometry ----
    try:
        fcc = _make_fcc_truncated_octahedron()
    except Exception:
        fcc = None

    # Collect starting geometries
    starts = []
    if fcc is not None:
        starts.append(fcc)

        # Perturbed FCC variants to escape any shallow local minima
        for noise_scale in [0.02, 0.05, 0.1]:
            if time.time() > deadline - 60.0:
                break
            perturbed = fcc + noise_scale * rng.standard_normal(fcc.shape)
            perturbed = _ensure_min_sep(perturbed)
            starts.append(perturbed)

    # Icosahedral starting geometry (cross-check)
    try:
        ico = _make_icosahedral_structure()
        starts.append(ico)
    except Exception:
        pass

    best_pos = None
    best_energy = np.inf

    # ---- 2. Relax each starting geometry ----
    for start in starts:
        if time.time() > deadline - 50.0:
            break
        try:
            p, e = _relax(start, max_iter=400, gtol=1e-10, ftol=1e-14)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            continue

    # Fallback if all relaxations failed
    if best_pos is None:
        if fcc is not None:
            best_pos = fcc
        else:
            side = int(np.ceil(N ** (1.0 / 3.0)))
            pts = []
            for i in range(side):
                for j in range(side):
                    for k in range(side):
                        pts.append([i, j, k])
            pts = np.array(pts[:N], dtype=np.float64)
            pts -= np.mean(pts, axis=0)
            best_pos = pts * 1.12

    best_pos = _ensure_min_sep(best_pos)

    # ---- 3. Multi-stage high-precision polish ----
    if time.time() < deadline - 20.0:
        try:
            p, e = _polish(best_pos, max_iter=3000)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    # ---- 4. Final micro-polish if time remains ----
    if time.time() < deadline - 5.0:
        try:
            p, e = _relax(best_pos, max_iter=500, gtol=1e-14, ftol=1e-17)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    best_pos = _ensure_min_sep(best_pos)

    return best_pos
