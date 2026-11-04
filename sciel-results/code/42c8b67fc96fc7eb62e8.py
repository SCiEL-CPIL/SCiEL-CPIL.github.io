"""Differentiable soft-max Ising representation for Erdos minimum overlap."""
import numpy as np
from scipy.optimize import minimize

N = 100

def _heights(theta):
    return 1.0 / (1.0 + np.exp(-np.clip(theta, -30.0, 30.0)))

def _overlap(h, k):
    return np.dot(h, np.roll(h, k))

def _max_overlap(h):
    ks = np.arange(1, N)
    return np.max([_overlap(h, k) for k in ks])

def _sumsq(h):
    return np.sum(h * h)

def construct():
    import time
    t0 = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    # Objective: smooth log-sum-exp over all non-zero shifts of overlap
    # h = sigmoid(theta), constraint sum(h)=N/2 handled by adding a
    # penalty term (soft) to keep heights near mean 0.5.
    def objective(theta):
        h = _heights(theta)
        # sum constraint penalty
        s = np.sum(h) - N / 2.0
        pen = 1e4 * s * s
        # overlaps at all shifts 1..N-1
        ks = np.arange(1, N)
        ov = np.array([_overlap(h, k) for k in ks])
        # log-sum-exp approximation of max
        m = np.max(ov)
        lse = m + np.log(np.sum(np.exp(ov - m)))
        return pen + lse

    def objective_grad(theta):
        h = _heights(theta)
        sh = h * (1.0 - h)
        ks = np.arange(1, N)
        ov = np.array([_overlap(h, k) for k in ks])
        m = np.max(ov)
        e = np.exp(ov - m)
        w = e / np.sum(e)
        # gradient of lse w.r.t h: sum_k w_k * d ov_k/dh
        # d ov_k/dh_i = h[(i+k) mod N]
        grad_h = np.zeros(N)
        for k in range(1, N):
            grad_h += w[k - 1] * np.roll(h, -k)
        # sum penalty grad
        s = np.sum(h) - N / 2.0
        grad_h += 2e4 * s
        return grad_h * sh

    best_h = np.full(N, 0.5)
    best_val = _max_overlap(best_h)

    rng = np.random.default_rng(0)
    n_restarts = 0
    while time.time() < DEADLINE - 3.0:
        # random-ish init: small noise around 0, or a structured init
        if n_restarts == 0:
            theta0 = np.zeros(N)
        elif n_restarts % 2 == 1:
            theta0 = rng.normal(0, 1.0, N)
        else:
            # Barker-like structured init: alternating with some pattern
            theta0 = rng.choice([-1.5, 1.5], N)
            if n_restarts > 2:
                theta0 = rng.normal(0, 0.5, N) + 0.5 * np.sin(2 * np.pi * rng.integers(1, 5) * np.arange(N) / N)
        res = minimize(objective, theta0, jac=objective_grad,
                       method='L-BFGS-B',
                       options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        h = _heights(res.x)
        # Project to exact sum N/2 via a small affine correction, then clip
        s = np.sum(h)
        h = h + (N / 2.0 - s) / N
        h = np.clip(h, 0.0, 1.0)
        # Re-project sum exactly with a second pass
        s = np.sum(h)
        if s > 0:
            h = h + (N / 2.0 - s) / N
            h = np.clip(h, 0.0, 1.0)
        val = _max_overlap(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        n_restarts += 1
        if time.time() > DEADLINE - 3.0:
            break

    # Final exact-sum correction: adjust by distributing residual
    s = np.sum(best_h)
    resid = N / 2.0 - s
    # distribute evenly, then clip and fix
    best_h = best_h + resid / N
    best_h = np.clip(best_h, 0.0, 1.0)
    # exact fix: find a position to absorb any tiny residual
    s = np.sum(best_h)
    resid = N / 2.0 - s
    if abs(resid) > 1e-12:
        # adjust the element with the most room
        if resid > 0:
            idx = np.argmax(1.0 - best_h)
        else:
            idx = np.argmax(best_h)
        room = 1.0 - best_h[idx] if resid > 0 else best_h[idx]
        add = min(resid, room) if resid > 0 else max(resid, -room)
        best_h[idx] += add
    # ensure sum exactly N/2 to floating precision
    s = np.sum(best_h)
    best_h[-1] += (N / 2.0 - s)
    best_h = np.clip(best_h, 0.0, 1.0)
    # final exact fix on last element
    s = np.sum(best_h)
    best_h[-1] += (N / 2.0 - s)

    return best_h
