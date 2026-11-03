"""Packing 300 equal circles in the unit disk: lattice seed + smooth soft-min polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 300
_DEADLINE = 135.0  # safety margin under the 150 s wall-clock kill
_T0 = time.time()


def _elapsed():
    return time.time() - _T0


def _pairwise_and_boundary(pts):
    """pts: (N,2). Return pairwise half-distances (M,) and boundary clearances (N,)."""
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    iu = np.triu_indices(N, k=1)
    d_half = d[iu] * 0.5
    bound = 1.0 - np.sqrt((pts ** 2).sum(-1))
    return d_half, bound


def _score(pts):
    d_half, bound = _pairwise_and_boundary(pts)
    return float(min(d_half.min(), bound.min()))


def _softmin_value_and_grad(pts, tau):
    """f = -softmin_tau(S), S = {d_ij/2} U {1-|c_i|}; returns (value, grad) to MINIMIZE.

    softmin_tau(S) = -tau*logsumexp(-S/tau);  f = tau*logsumexp(-S/tau).
    df/dS_k = -softmax(-S/tau)_k.
    """
    d_half, bound = _pairwise_and_boundary(pts)
    S = np.concatenate([d_half, bound])
    x = -S / tau
    m = x.max()
    ex = np.exp(x - m)
    w = ex / ex.sum()
    value = tau * (m + np.log(ex.sum()))

    grad = np.zeros_like(pts)
    # boundary: dS_i/d c_i = -c_i/|c_i|  ->  df/d c_i = w_bi * c_i/|c_i|
    wb = w[-N:]
    r = np.sqrt((pts ** 2).sum(-1))
    r = np.maximum(r, 1e-12)
    grad += wb[:, None] * pts / r[:, None]

    # pairwise: dS_ij/d c_i = (c_i-c_j)/(2 d_ij)  ->  df/d c_i = -w_p*(c_i-c_j)/(2 d_ij)
    wp = w[:len(d_half)]
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    d = np.maximum(d, 1e-12)
    iu = np.triu_indices(N, k=1)
    vec = diff[iu]
    coef = wp / (2.0 * d[iu])
    np.add.at(grad, iu[0], -vec * coef[:, None])
    np.add.at(grad, iu[1], vec * coef[:, None])
    return value, grad


def _anneal(pts, tau_start, tau_end, iters_per_stage, stages, max_iter_total):
    best = pts.copy()
    best_score = _score(best)
    cur = pts.copy()
    budget = max_iter_total
    for s in range(stages):
        if _elapsed() > _DEADLINE - 8.0:
            break
        tau = tau_start * (tau_end / tau_start) ** (s / max(1, stages - 1))
        res = minimize(
            _softmin_value_and_grad,
            cur,
            args=(tau,),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": iters_per_stage, "maxls": 20, "gtol": 1e-6},
        )
        budget -= iters_per_stage
        cand = res.x.reshape(N, 2)
        if np.all(np.isfinite(cand)):
            sc = _score(cand)
            if sc > best_score:
                best_score = sc
                best = cand.copy()
            cur = cand
        if budget <= 0:
            break
    return best, best_score


def _lattice_seed():
    for spacing in [0.11, 0.105, 0.10, 0.095, 0.09, 0.085, 0.08]:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        keep = radius <= 1.0 - spacing / 2.0 - 1e-9
        points = points[keep]
        if len(points) >= N:
            idx = np.argsort(radius, kind="stable")[:N]
            return points[idx]
    return _ring_seed()


def _ring_seed():
    pts = [np.array([0.0, 0.0])]
    used = 1
    ring = 1
    while used < N:
        count = min(6 * ring, N - used)
        ang = 2.0 * np.pi * np.arange(count) / count
        rad = ring * 0.11
        pts.extend([np.array([rad * np.cos(a), rad * np.sin(a)]) for a in ang])
        used += count
        ring += 1
    return np.array(pts[:N])


def construct():
    pts = _lattice_seed()
    base_score = _score(pts)
    best, best_score = _anneal(pts, tau_start=0.05, tau_end=0.004,
                               iters_per_stage=120, stages=6, max_iter_total=720)
    if best_score < base_score:
        return pts
    return best


if __name__ == "__main__":
    c = construct()
    print("score", _score(c), "shape", c.shape)
