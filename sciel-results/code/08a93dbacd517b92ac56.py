import numpy as np
import time
import random
from scipy.optimize import minimize

N = 600
DEADLINE = 147
M = 24  # number of cosine-bump modes (basis size)


def construct():
    start_time = time.time()

    def time_left():
        return DEADLINE - (time.time() - start_time)

    i = np.arange(N, dtype=float)
    size = 2 * N

    # ---- cosine-bump basis: nonnegative, compact support, centered ----
    # h(t) = sum_m a_m * bump(t - c_m),  a_m >= 0, t in [-0.5, 0.5]
    centers = (np.arange(M) + 0.5) / M          # M equally spaced centers
    widths = 0.5 / M                             # adjacent bumps just touch
    basis = np.zeros((M, N))
    for m in range(M):
        d = np.abs(i / N - centers[m])
        d = np.minimum(d, 1.0 - d)               # wrap distance on the circle
        basis[m] = np.cos(np.pi * d / widths) ** 2
    basis = basis / basis.sum(axis=1, keepdims=True)  # normalize each bump

    def h_from_a(a):
        return basis @ a

    def compute_autoconv_max(h):
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return float(np.max(conv))

    # ---- smooth-max surrogate objective + gradient (w.r.t. a) ----
    def smooth_obj_grad(a, beta):
        h = h_from_a(a)
        s = h.sum()
        h = h / s
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        m = conv.max()
        e = np.exp((conv - m) * beta)
        w = e / e.sum()
        w_full = np.zeros(size)
        w_full[:size] = w
        W = np.fft.fft(w_full, n=size)
        grad_h = 2.0 * np.fft.ifft(H * W).real[:N]
        grad_a = basis.T @ (grad_h / s)
        val = m + np.log(e.sum()) / beta
        return val, grad_a

    # ---- fast coordinate polish (analytic FFT updates) ----
    def polish(h, max_iter=600):
        best = compute_autoconv_max(h)
        for _ in range(max_iter):
            if time_left() < 4:
                break
            h_pad = np.zeros(size)
            h_pad[:N] = h
            H = np.fft.fft(h_pad, n=size)
            conv = np.fft.ifft(H * H).real
            idx = int(np.argmax(conv))
            step = h.mean() * 0.2
            improved = False
            for k in range(30):
                if time_left() < 4:
                    break
                new_h = h.copy()
                new_h[idx] = max(0.0, h[idx] - step)
                s = new_h.sum()
                if s <= 0:
                    break
                new_h = new_h / s
                # analytic FFT update (only index idx changes)
                d = (new_h[idx] - h[idx]) / s
                H_new = H + d * np.exp(-2j * np.pi * idx * np.arange(size) / size)
                conv_new = np.fft.ifft(H_new * H_new).real
                v = conv_new.max()
                if v < best - 1e-12:
                    h = new_h
                    H = H_new
                    best = v
                    improved = True
                    step *= 0.85
                    break
                step *= 0.7
            if not improved:
                break
        return h, best

    # ---- diverse nonnegative initializations (a >= 0) ----
    def a_from_profile(profile):
        # project a nonneg profile onto the bump basis (least squares, a>=0)
        try:
            a, *_ = np.linalg.lstsq(basis, profile, rcond=None)
        except Exception:
            a = np.ones(M)
        return np.maximum(a, 0.0)

    inits = []
    inits.append(np.ones(M))
    for k in [1, 2, 3, 4, 6]:
        a = np.zeros(M)
        for j in range(k):
            a[int(round(j * M / k))] = 1.0
        inits.append(a)
    for p in [0.5, 1.0, 2.0, 4.0]:
        a = np.ones(M) * (0.3 + 0.7 * np.abs(np.cos(np.pi * np.arange(M) / M)) ** p)
        inits.append(a)
    for _ in range(8):
        a = np.abs(np.random.randn(M)) + 0.05
        inits.append(a)

    best_a = None
    best_val = np.inf
    for a in inits:
        if time_left() < 8:
            break
        h = h_from_a(a)
        s = h.sum()
        if s <= 0:
            continue
        h = h / s
        v = compute_autoconv_max(h)
        if v < best_val:
            best_val = v
            best_a = a.copy()
    if best_a is None:
        best_a = np.ones(M)
        best_val = compute_autoconv_max(h_from_a(best_a) / h_from_a(best_a).sum())

    # ---- parametric L-BFGS-B with iterative beta schedule ----
    def run_lbfgs(a0, beta, maxiter):
        try:
            res = minimize(
                lambda a: smooth_obj_grad(a, beta)[0],
                a0.copy(),
                jac=lambda a: smooth_obj_grad(a, beta)[1],
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-7, 'maxfun': maxiter * 3},
            )
            return res
        except Exception:
            return None

    for beta in [25.0, 60.0, 120.0, 250.0, 400.0]:
        if time_left() < 25:
            break
        res = run_lbfgs(best_a, beta, 120)
        if res is not None and res.fun < best_val:
            h = h_from_a(np.maximum(res.x, 0.0))
            s = h.sum()
            if s > 0:
                h = h / s
                v = compute_autoconv_max(h)
                if v < best_val:
                    best_val = v
                    best_a = np.maximum(res.x, 0.0).copy()

    # ---- multi-restart on the parametric family ----
    if time_left() > 30:
        n_restarts = int(min(6, time_left() // 6))
        for r in range(n_restarts):
            if time_left() < 20:
                break
            a0 = best_a.copy()
            a0 += 0.1 * np.abs(np.random.randn(M))
            a0 = np.maximum(a0, 0.0)
            for beta in [60.0, 200.0]:
                if time_left() < 15:
                    break
                res = run_lbfgs(a0, beta, 80)
                if res is not None:
                    h = h_from_a(np.maximum(res.x, 0.0))
                    s = h.sum()
                    if s > 0:
                        h = h / s
                        v = compute_autoconv_max(h)
                        if v < best_val:
                            best_val = v
                            best_a = np.maximum(res.x, 0.0).copy()
                            a0 = best_a.copy()

    # ---- fast coordinate polish on the full h ----
    h = h_from_a(best_a)
    h = h / h.sum()
    if time_left() > 12:
        h, v = polish(h)
        if v < best_val:
            best_val = v

    h = np.maximum(h, 0.0)
    h = h / h.sum()
    return h.tolist()
