"""
LJ38 cluster global minimum search.

Design family (kept): build the FCC truncated-octahedron geometry analytically,
then polish with tight L-BFGS local relaxation.  Strengthened with:
  * basin hopping (annealed perturbations + local relax) to escape local minima,
  * a 1-D global scale optimisation to remove residual strain,
  * an explicit wall-clock deadline guard.

Entry point: construct() -> (N, 3) array of Cartesian coordinates.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 38
TARGET_ENERGY = -173.928427  # Known global minimum energy (Cambridge)
DEADLINE = 150.0
SAFETY = 20.0  # return at least this many seconds before the hard kill


def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
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


def _lj_energy(pos):
    e, _ = _lj_energy_and_grad(pos)
    return e


def _make_fcc_truncated_octahedron():
    """Build the 38-atom FCC truncated-octahedron from the FCC lattice."""
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
    pts = pts[order][:N].copy()
    pts -= np.mean(pts, axis=0)
    # scale so min nearest-neighbour distance = 2^(1/6)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_nn = np.sqrt(np.min(r2))
    if min_nn > 1e-12:
        pts *= (2.0 ** (1.0 / 6.0)) / min_nn
    return pts


def _relax(pos, max_iter=400, gtol=1e-10, ftol=1e-14):
    """L-BFGS-B local relaxation with analytic gradient."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()
    try:
        res = minimize(obj_grad, pos.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol})
        p_out = res.x.reshape(-1, 3)
    except Exception:
        p_out = pos
    e_out = _lj_energy(p_out)
    return p_out, e_out


def _optimize_scale(pos, n=25):
    """Refine the uniform scale factor (1-D) to minimise energy."""
    def e_of_log_s(t):
        return _lj_energy(pos * np.exp(t))
    # bracket around 0
    best_t, best_e = 0.0, e_of_log_s(0.0)
    for dt in np.linspace(-0.05, 0.05, n):
        e = e_of_log_s(dt)
        if e < best_e:
            best_t, best_e = dt, e
    # refine around best
    lo, hi = best_t - 0.005, best_t + 0.005
    for _ in range(40):
        mids = np.linspace(lo, hi, 31)
        vals = [e_of_log_s(t) for t in mids]
        i = int(np.argmin(vals))
        best_t, best_e = mids[i], vals[i]
        if i == 0:
            hi = mids[1]
        elif i == len(mids) - 1:
            lo = mids[-2]
        else:
            lo, hi = mids[i - 1], mids[i + 1]
    return pos * np.exp(best_t), best_e


def _ensure_min_sep(pos, min_sep=0.5):
    """Scale up the cluster if any pair is closer than min_sep."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        pos = pos * (min_sep / min_d * 1.01)
    return pos


def _basin_hopping(pos, rng, t0, deadline, max_iter=250, gtol=1e-10):
    """Annealed basin hopping: perturb + relax, keep best, accept by Metropolis."""
    best_pos = pos.copy()
    best_e = _lj_energy(pos)
    cur_pos = pos.copy()
    cur_e = best_e
    T = 0.05
    T_min = 0.0005
    for it in range(2000):
        if time.time() > deadline - SAFETY:
            break
        # perturb current
        noise = 0.05 * rng.standard_normal(cur_pos.shape)
        cand = cur_pos + noise
        cand = _ensure_min_sep(cand)
        cand, e_cand = _relax(cand, max_iter=max_iter, gtol=gtol)
        cand, e_cand = _optimize_scale(cand)
        cand = _ensure_min_sep(cand)
        # Metropolis accept for the walk
        delta = e_cand - cur_e
        if delta < 0 or rng.random() < np.exp(-delta / T):
            cur_pos, cur_e = cand, e_cand
        if e_cand < best_e:
            best_e, best_pos = e_cand, cand.copy()
        T = max(T_min, T * 0.97)
        if best_e < TARGET_ENERGY + 0.001:
            break
    return best_pos, best_e


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - SAFETY
    rng = np.random.default_rng(2024)

    fcc = _make_fcc_truncated_octahedron()
    fcc = _ensure_min_sep(fcc)
    fcc, e_fcc = _relax(fcc, max_iter=500, gtol=1e-10)
    fcc, e_fcc = _optimize_scale(fcc)
    fcc = _ensure_min_sep(fcc)

    best_pos, best_e = fcc.copy(), e_fcc

    # basin hopping from the FCC geometry
    if time.time() < deadline - SAFETY:
        bp, be = _basin_hopping(fcc, rng, t0, deadline, max_iter=250)
        if be < best_e:
            best_e, best_pos = be, bp

    # a couple of independent basin-hopping restarts from perturbed FCC
    for restart in range(3):
        if time.time() > deadline - SAFETY - 25.0:
            break
        try:
            start = fcc + 0.08 * rng.standard_normal(fcc.shape)
            start = _ensure_min_sep(start)
            bp, be = _basin_hopping(start, rng, t0, deadline, max_iter=200)
            if be < best_e:
                best_e, best_pos = be, bp
        except Exception:
            continue

    # final high-precision polish
    if time.time() < deadline - 12.0:
        try:
            p, e = _relax(best_pos, max_iter=3000, gtol=1e-12, ftol=1e-16)
            p, e = _optimize_scale(p)
            if e < best_e:
                best_e, best_pos = e, p
        except Exception:
            pass

    best_pos = _ensure_min_sep(best_pos)
    return best_pos
