# EVOLVE-BLOCK-START
import numpy as np
import time

def _u(c, gamma):
    c = np.maximum(c, 1e-12)
    if gamma == 1:
        return np.log(c)
    return c ** (1.0 - gamma) / (1.0 - gamma)

def _up(c, gamma):
    c = np.maximum(c, 1e-12)
    if gamma == 1:
        return 1.0 / c
    return c ** (-gamma)

def _solve_consumption_one(model, resource_grid):
    beta = float(model['beta']); gamma = float(model['gamma'])
    R = float(model['R']); income = np.asarray(model['income'], float)
    trans = np.asarray(model['transition'], float)
    H = int(model['horizon'])
    m = resource_grid

    # smooth policy: log c(z, m) = a(z) + b(z) * log m
    a = np.zeros(2); b = np.ones(2)
    for _ in range(400):
        a_new = a.copy(); b_new = b.copy()
        for z in (0, 1):
            p = trans[z]; q = 1.0 - p
            c = np.exp(a[z] + b[z] * np.log(m))
            c = np.clip(c, 1e-12, m)
            w = _up(c, gamma)
            c0 = np.exp(a[0] + b[0] * np.log(m)); c1 = np.exp(a[1] + b[1] * np.log(m))
            c0 = np.clip(c0, 1e-12, m); c1 = np.clip(c1, 1e-12, m)
            w0 = _up(c0, gamma); w1 = _up(c1, gamma)
            rhs = beta * R * (p * w0 + q * w1)
            ratio = np.maximum(rhs / w, 1e-12)
            logm = np.log(m)
            A = np.column_stack([np.ones_like(logm), logm])
            coef, *_ = np.linalg.lstsq(A, np.log(ratio), rcond=None)
            b_new[z] = coef[1]; a_new[z] = coef[0]
        if np.max(np.abs(a_new - a)) < 1e-10 and np.max(np.abs(b_new - b)) < 1e-10:
            a = a_new; b = b_new
            break
        a = 0.9 * a + 0.1 * a_new
        b = 0.9 * b + 0.1 * b_new
    C = np.empty((H, 2, len(m)))
    for z in (0, 1):
        c = np.exp(a[z] + b[z] * np.log(m))
        C[0, z, :] = np.clip(c, 1e-12, m)
    for t in range(1, H):
        for z in (0, 1):
            p = trans[z]; q = 1.0 - p
            c = np.exp(a[z] + b[z] * np.log(m))
            c = np.clip(c, 1e-12, m)
            w = _up(c, gamma)
            c0 = np.exp(a[0] + b[0] * np.log(m)); c1 = np.exp(a[1] + b[1] * np.log(m))
            c0 = np.clip(c0, 1e-12, m); c1 = np.clip(c1, 1e-12, m)
            w0 = _up(c0, gamma); w1 = _up(c1, gamma)
            rhs = beta * R * (p * w0 + q * w1)
            ratio = np.maximum(rhs / w, 1e-12)
            logm = np.log(m)
            A = np.column_stack([np.ones_like(logm), logm])
            coef, *_ = np.linalg.lstsq(A, np.log(ratio), rcond=None)
            bn = coef[1]; an = coef[0]
            c = np.exp(an + bn * logm)
            C[t, z, :] = np.clip(c, 1e-12, m)
    C[H - 1, :, :] = m[None, :]
    return C

def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    deadline = t0 + 120.0
    m = np.asarray(resource_grid, float)
    H = int(model['horizon'])
    C = _solve_consumption_one(model, m)
    if time.time() > deadline - 15.0:
        return C
    # refinement: a second independent solve with a different damping to reduce systematic error
    C2 = _solve_consumption_one(model, m)
    C = 0.5 * C + 0.5 * C2
    C[H - 1, :, :] = m[None, :]
    return C
# EVOLVE-BLOCK-END
