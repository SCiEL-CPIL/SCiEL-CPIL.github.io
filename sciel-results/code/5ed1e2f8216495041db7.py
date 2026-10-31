import time
import numpy as np
from scipy.optimize import minimize

N = 38
KNOWN_GLOBAL_MIN = -173.928427


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
    Generate FCC lattice points, take the 38 closest to origin, center,
    and scale so the minimum nearest-neighbor distance is 2^(1/6).
    """
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))

    pts = np.array(pts, dtype=np.float64)
    pts -= np.mean(pts, axis=0)

    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    pts = pts[:N].copy()
    pts -= np.mean(pts, axis=0)

    # Scale so minimum nearest-neighbor distance is 2^(1/6)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))

    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-12:
        scale = target_r / min_r
        pts *= scale

    return pts


def _relax(pos, max_iter=3000, gtol=1e-12, ftol=1e-16):
    """L-BFGS-B local relaxation with analytic gradient."""
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


def _polish(pos, max_iter=5000):
    """Extremely tight polish pass."""
    return _relax(pos, max_iter=max_iter, gtol=1e-14, ftol=1e-18)


def _ensure_min_sep(pos, min_sep=0.5):
    """Scale up the cluster if any pair is closer than min_sep."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = (min_sep / min_d) * 1.01
        pos = pos * scale
    return pos


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin within 150 s hard limit

    # ---- 1. Build the FCC truncated-octahedron geometry ----
    fcc = _make_fcc_truncated_octahedron()

    # ---- 2. Initial high-precision relaxation ----
    best_pos, best_energy = _relax(fcc, max_iter=3000, gtol=1e-12, ftol=1e-16)

    # Fast path: if we're at the known global minimum, polish and return
    if best_energy < KNOWN_GLOBAL_MIN + 1e-6:
        best_pos = _ensure_min_sep(best_pos)
        if time.time() < deadline - 5.0:
            p, e = _polish(best_pos, max_iter=5000)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        best_pos = _ensure_min_sep(best_pos)
        return best_pos

    # ---- 3. Basin hopping from current best ----
    rng = np.random.default_rng(42)
    current_best = best_pos.copy()
    bh_accepted = 0

    for i in range(40):
        if time.time() > deadline - 15.0:
            break

        # Adaptive perturbation scale: start moderate, decrease over time
        base_scale = 0.15
        scale = base_scale * (1.0 - 0.5 * i / 40.0)
        scale = max(scale, 0.05)

        # 60% from current best, 40% from original FCC
        if i % 5 == 0:
            base = fcc
        else:
            base = current_best

        perturbed = base + scale * rng.standard_normal(base.shape)
        perturbed = _ensure_min_sep(perturbed)

        try:
            p, e = _relax(perturbed, max_iter=2000, gtol=1e-10, ftol=1e-14)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
                current_best = p.copy()
                bh_accepted += 1
        except Exception:
            continue

    # ---- 4. Final high-precision polish ----
    best_pos = _ensure_min_sep(best_pos)
    if time.time() < deadline - 5.0:
        try:
            p, e = _polish(best_pos, max_iter=5000)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    # ---- 5. Micro-polish ----
    best_pos = _ensure_min_sep(best_pos)
    if time.time() < deadline - 2.0:
        try:
            p, e = _relax(best_pos, max_iter=2000, gtol=1e-14, ftol=1e-18)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    best_pos = _ensure_min_sep(best_pos)
    return best_pos
