"""First autocorrelation inequality: minimize normalized max autoconvolution.

Strategy: Gaussian-superposition parametric model (global structure) + soft-max
surrogate gradient descent (smooth max) + L-BFGS local polish.
"""
import time
import numpy as np

N = 600
DEADLINE = time.time() + 115.0  # safety margin under 150s wall-clock limit


def _grid():
    x = np.linspace(-0.25, 0.25, N + 1)
    return x[:-1] + 0.25 / N  # interval centers


_X = _grid()


def _autoconv(h):
    # circular autoconvolution of the step heights (periodic model)
    return np.fft.ifft(np.fft.fft(h) ** 2).real


def _score(h):
    ac = _autoconv(h)
    m = np.max(ac)
    return m / (np.sum(h) ** 2 / N)


def _soft_max(ac, t):
    m = np.max(ac)
    return m + (1.0 / t) * np.log(np.sum(np.exp(t * (ac - m))))


def _soft_max_grad(ac, t):
    m = np.max(ac)
    e = np.exp(t * (ac - m))
    w = e / np.sum(e)
    return w  # softmax weights


def _gmm_params_to_h(params):
    # params: [mu1, mu2, mu3, s1, s2, s3, a1, a2, a3]
    mus = params[0:3]
    sigs = np.abs(params[3:6]) + 1e-4
    amps = np.abs(params[6:9])
    h = np.zeros(N)
    for mu, s, a in zip(mus, sigs, amps):
        h += a * np.exp(-0.5 * ((_X - mu) / s) ** 2)
    return h


def _build_and_score(params):
    h = _gmm_params_to_h(params)
    ac = _autoconv(h)
    m = np.max(ac)
    ratio = m / (np.sum(h) ** 2 / N)
    return h, ratio


def _grad_soft(params, t):
    h = _gmm_params_to_h(params)
    ac = _autoconv(h)
    sm = _soft_max(ac, t)
    w = _soft_max_grad(ac, t)
    S = np.sum(h)
    base = N / (S * S)
    # d(sm)/d(ac_k) = w_k ; d(ac)/d(h) = ifft(fft(h)*2*conj(fft(h)))
    F = np.fft.fft(h)
    d_ac_d_h = np.fft.ifft(F * 2.0 * np.conj(F)).real
    d_sm_d_h = base * w * d_ac_d_h
    # d(h)/d(params)
    h_grad = np.zeros_like(params)
    mus = params[0:3]
    sigs = np.abs(params[3:6]) + 1e-4
    amps = np.abs(params[6:9])
    for i in range(3):
        g = _X - mus[i]
        h_grad[i] = np.sum(d_sm_d_h * amps[i] * np.exp(-0.5 * (g / sigs[i]) ** 2) * (-g / sigs[i] ** 2))
        # d/ds of exp(-0.5 (g/s)^2) = exp(...) * (g^2 / s^3)
        h_grad[3 + i] = np.sum(d_sm_d_h * amps[i] * np.exp(-0.5 * (g / sigs[i]) ** 2) * (g ** 2 / sigs[i] ** 3))
        h_grad[6 + i] = np.sum(d_sm_d_h * np.exp(-0.5 * (g / sigs[i]) ** 2))
    return sm, h_grad


def _soft_max_loss(params, t):
    h = _gmm_params_to_h(params)
    ac = _autoconv(h)
    return _soft_max(ac, t)


def _soft_max_loss_grad(params, t):
    h = _gmm_params_to_h(params)
    ac = _autoconv(h)
    m = np.max(ac)
    e = np.exp(t * (ac - m))
    w = e / np.sum(e)
    S = np.sum(h)
    base = N / (S * S)
    F = np.fft.fft(h)
    d_ac_d_h = np.fft.ifft(F * 2.0 * np.conj(F)).real
    d_sm_d_h = base * w * d_ac_d_h
    g = np.zeros_like(params)
    mus = params[0:3]
    sigs = np.abs(params[3:6]) + 1e-4
    amps = np.abs(params[6:9])
    for i in range(3):
        gg = _X - mus[i]
        g[i] = np.sum(d_sm_d_h * amps[i] * np.exp(-0.5 * (gg / sigs[i]) ** 2) * (-gg / sigs[i] ** 2))
        g[3 + i] = np.sum(d_sm_d_h * amps[i] * np.exp(-0.5 * (gg / sigs[i]) ** 2) * (gg ** 2 / sigs[i] ** 3))
        g[6 + i] = np.sum(d_sm_d_h * np.exp(-0.5 * (gg / sigs[i]) ** 2))
    return _soft_max(ac, t), g


def construct():
    from scipy.optimize import minimize

    best_h = np.ones(N)
    best_score = _score(best_h)

    # ---- Phase 1: soft-max gradient descent on Gaussian superposition ----
    rng = np.random.default_rng(0)
    for restart in range(8):
        if time.time() > DEADLINE - 10:
            break
        params = np.array([
            -0.1 + 0.2 * rng.random(), 0.0 + 0.1 * rng.random(), 0.1 - 0.2 * rng.random(),
            0.05 + 0.03 * rng.random(), 0.06 + 0.03 * rng.random(), 0.07 + 0.03 * rng.random(),
            1.0 + rng.random(), 1.0 + rng.random(), 1.0 + rng.random(),
        ])
        # anneal temperature high -> low
        for t in [1000.0, 300.0, 100.0, 40.0, 15.0, 6.0, 3.0]:
            if time.time() > DEADLINE - 10:
                break
            lr = 0.02
            for step in range(300):
                if time.time() > DEADLINE - 10:
                    break
                val, grad = _soft_max_loss_grad(params, t)
                g = np.clip(grad, -100.0, 100.0)
                params = params - lr * g
                # keep amplitudes positive-ish by re-anchoring
                params[6:9] = np.abs(params[6:9]) + 0.05
                params[3:6] = np.abs(params[3:6]) + 0.01
            # record
            h, s = _build_and_score(params)
            if s < best_score:
                best_score = s
                best_h = h.copy()

    # ---- Phase 2: local polish via L-BFGS on soft-max surrogate ----
    if time.time() < DEADLINE - 15:
        h0 = best_h
        # initialize params from a 3-component fit: split the best_h into 3 Gaussians
        # simple init: use peak locations
        peak = int(np.argmax(h0))
        p1 = _X[peak]
        # find secondary peaks
        h0s = h0.copy()
        mus = [p1]
        sigs = [0.06]
        amps = [h0[peak]]
        for k in range(2):
            # dampen already-assigned region
            for mu, s, a in zip(mus, sigs, amps):
                h0s = h0s - a * np.exp(-0.5 * ((_X - mu) / s) ** 2)
            pk = int(np.argmax(h0s))
            mus.append(_X[pk])
            sigs.append(0.06)
            amps.append(max(h0[pk], 0.1))
        params0 = np.array(mus + sigs + amps)

        def obj(p):
            return _soft_max_loss(p, 8.0)

        def obj_grad(p):
            v, g = _soft_max_loss_grad(p, 8.0)
            return g

        try:
            res = minimize(obj, params0, jac=obj_grad, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8,
                                    'maxfun': 400})
            h, s = _build_and_score(res.x)
            if s < best_score:
                best_score = s
                best_h = h.copy()
        except Exception:
            pass

    # ---- Phase 3: direct coordinate polish on heights (soft-max surrogate) ----
    if time.time() < DEADLINE - 5:
        h = best_h.copy()
        cur = _score(h)
        # small local perturbations around the peak region to reduce the max
        # use gradient of soft-max w.r.t. heights directly
        for t in [20.0, 8.0, 3.0]:
            for it in range(200):
                if time.time() > DEADLINE - 3:
                    break
                ac = _autoconv(h)
                m = np.max(ac)
                e = np.exp(t * (ac - m))
                w = e / np.sum(e)
                S = np.sum(h)
                base = N / (S * S)
                F = np.fft.fft(h)
                d_ac_d_h = np.fft.ifft(F * 2.0 * np.conj(F)).real
                grad = base * w * d_ac_d_h
                # gradient step with projection to nonnegativity
                lr = 1e-3
                h_new = h - lr * grad
                h_new = np.maximum(h_new, 0.0)
                s_new = _score(h_new)
                if s_new < cur:
                    h = h_new
                    cur = s_new
                    if cur < best_score:
                        best_score = cur
                        best_h = h.copy()
                else:
                    lr *= 0.5
                    if lr < 1e-9:
                        break

    # ensure nonnegative
    best_h = np.maximum(best_h, 0.0)
    return [float(v) for v in best_h]
