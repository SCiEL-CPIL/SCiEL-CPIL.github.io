"""Rewritten program for Erdos minimum overlap; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
HALF = N // 2
NORM = 1.0 / N  # so that IFFT of H*conj(H) equals circular autocorrelation

import time
import numpy as np
from scipy.optimize import fmin_l_bfgs_b


def _time_remaining(deadline):
    return deadline - time.time()


def _compute_overlaps(h):
    """Return O_k = sum_i h_i (1 - h_{i+k}) for all k, via FFT (circular)."""
    H = np.fft.fft(h)
    C = np.fft.ifft(H * np.conj(H)).real
    return HALF - C


def _max_overlap(h):
    return float(np.max(_compute_overlaps(h)))


def _score(h):
    return _max_overlap(h)


def _project_box(h):
    return np.clip(h, 0.0, 1.0)


def _bang_bang(h):
    """Round to a binary vector with exactly HALF ones (greedy by value)."""
    idx = np.argsort(-h)
    out = np.zeros(N)
    out[idx[:HALF]] = 1.0
    return out


def _bang_bang_search(h, deadline, max_rounds=60, max_evals=120000):
    """Flip-pair local search on the bang-bang vector to maximize min_k C_k."""
    x = _bang_bang(h)
    ones = np.flatnonzero(x > 0.5)
    zeros = np.flatnonzero(x <= 0.5)
    if len(ones) < 2 or len(zeros) < 2:
        return x
    ov = _compute_overlaps(x)
    best_min = float(np.min(ov))
    evals = 0
    for _ in range(max_rounds):
        if _time_remaining(deadline) < 1.0 or evals > max_evals:
            break
        improved = False
        for i in ones:
            if _time_remaining(deadline) < 1.0 or evals > max_evals:
                break
            for j in zeros:
                if i == j:
                    continue
                evals += 1
                x[i] = 0.0
                x[j] = 1.0
                ov = _compute_overlaps(x)
                m = float(np.min(ov))
                if m > best_min + 1e-12:
                    best_min = m
                    improved = True
                else:
                    x[i] = 1.0
                    x[j] = 0.0
        if not improved:
            break
        ones = np.flatnonzero(x > 0.5)
        zeros = np.flatnonzero(x <= 0.5)
    return x


def _fista_surrogate(h0, deadline, t, max_iter=400):
    """FISTA on f(h)=sum_k max(0,O_k-t), then project+round to bang-bang."""
    x = _project_box(h0.copy())
    y = x.copy()
    tval = 1.0
    L = 4.0 * N
    best = None
    for it in range(max_iter):
        if _time_remaining(deadline) < 1.0:
            break
        O = _compute_overlaps(y)
        mask = O > t
        d = np.zeros(N)
        d[mask] = 1.0
        grad = d - np.array([np.roll(d, k) for k in range(N)]).sum(axis=0)
        # backtracking line search
        step = 1.0 / L
        fy = np.sum(np.maximum(0.0, O - t))
        while True:
            x_new = _project_box(y - step * grad)
            O_new = _compute_overlaps(x_new)
            fx_new = np.sum(np.maximum(0.0, O_new - t))
            if fx_new <= fy + 0.5 * step * np.dot(grad, x_new - y):
                break
            step *= 0.5
            if step < 1e-9:
                break
        y = x_new + ((tval - 1.0) / (tval + 1.5)) * (x_new - x)
        tval = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tval * tval))
        x = x_new
        cur = _score(x)
        if best is None or cur < best[1]:
            best = (x.copy(), cur)
    return best[0] if best is not None else _project_box(h0)


def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin under 150s kill

    best_h = None
    best_score = np.inf

    def consider(h):
        nonlocal best_h, best_score
        h = _project_box(h)
        s = np.sum(h)
        if s > 0:
            h = h / s * HALF
            h = _project_box(h)
        sc = _score(h)
        if sc < best_score:
            best_score = sc
            best_h = h.copy()
        return h, sc

    # --- Initial structured candidates ---
    candidates = [np.full(N, 0.5)]
    x = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    for freq in [1, 2, 3, 5, 7, 11]:
        for amp in [0.3, 0.45, 0.6]:
            hc = 0.5 + amp * np.cos(freq * x + 0.3 * freq)
            candidates.append(hc)
    rng = np.random.default_rng(12345)
    for _ in range(6):
        candidates.append(rng.random(N))

    for c in candidates:
        if _time_remaining(deadline) < 5.0:
            break
        h, sc = consider(c)
        bb = _bang_bang_search(h, deadline, max_rounds=40, max_evals=60000)
        consider(bb)
        if _time_remaining(deadline) < 5.0:
            break
        # FISTA polish from current best bang-bang
        t_cur = best_score - 1e-3
        hf = _fista_surrogate(bb, deadline, t=max(0.0, t_cur), max_iter=250)
        consider(hf)
        bb2 = _bang_bang_search(hf, deadline, max_rounds=40, max_evals=60000)
        consider(bb2)

    # --- Final polish: bang-bang search on the best found, with more budget ---
    if best_h is not None and _time_remaining(deadline) > 3.0:
        bb = _bang_bang_search(best_h, deadline, max_rounds=80, max_evals=200000)
        consider(bb)

    # --- Final validation: ensure sum == N/2 exactly and bounds hold ---
    h_final = _project_box(best_h)
    s = np.sum(h_final)
    if s > 0:
        h_final = h_final / s * HALF
        h_final = _project_box(h_final)
    # Force exact sum by nudging (keep within [0,1])
    diff = HALF - np.sum(h_final)
    if abs(diff) > 1e-12:
        # distribute diff across entries with room
        if diff > 0:
            order = np.argsort(1.0 - h_final)  # most room first
        else:
            order = np.argsort(h_final)  # most value first
        per = diff / N
        for idx in order:
            if abs(diff) < 1e-12:
                break
            room = (1.0 - h_final[idx]) if diff > 0 else h_final[idx]
            step = min(abs(diff), room) * (1.0 if diff > 0 else -1.0)
            h_final[idx] += step
            diff -= step
    h_final = _project_box(h_final)
    return h_final
