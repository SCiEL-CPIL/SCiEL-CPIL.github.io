import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 146.0  # safety margin under 150s

tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)          # (165, 3)
A = tri_idx[:, 0]
B = tri_idx[:, 1]
C = tri_idx[:, 2]

H = np.sqrt(3) / 2

def bary_to_cart(bary):
    u = bary[:, 0]; v = bary[:, 1]; w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def clamp_bary(bary, eps=1e-9):
    u = np.clip(bary[:, 0].copy(), eps, 1 - eps)
    v = np.clip(bary[:, 1].copy(), eps, 1 - eps)
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
    return np.column_stack([u, v])

def calc_areas_bary(bary):
    u = bary[:, 0]; v = bary[:, 1]; w = 1.0 - u - v
    ui = u[A]; vi = v[A]; wi = w[A]
    uj = u[B]; vj = v[B]; wj = w[B]
    uk = u[C]; vk = v[C]; wk = w[C]
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    return np.abs(det)

def hard_min_value(bary):
    return float(np.min(calc_areas_bary(bary)))

# ---------- Epigraph formulation (smooth, differentiable) ----------
def epigraph_obj(bary_flat, t, beta=40.0):
    bary = bary_flat.reshape(N, 2)
    areas = calc_areas_bary(bary)
    d = areas - t
    if np.any(d < -1e-12):
        return 1e10 + np.sum(d[d < 0] ** 2)
    scaled = -beta * d
    m = np.max(scaled)
    if m > 700:
        return -np.min(d)
    lse = m + np.log(np.sum(np.exp(scaled - m)))
    return lse / beta

def epigraph_grad(bary_flat, t, beta=40.0):
    bary = bary_flat.reshape(N, 2)
    u = bary[:, 0]; v = bary[:, 1]; w = 1.0 - u - v
    ui = u[A]; vi = v[A]; wi = w[A]
    uj = u[B]; vj = v[B]; wj = w[B]
    uk = u[C]; vk = v[C]; wk = w[C]
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    sgn = np.sign(det)
    # d(det)/du_i = (v_j - v_k), etc.
    ddu_i = sgn * (vj - vk); ddu_j = sgn * (vk - vi); ddu_k = sgn * (vi - vj)
    ddv_i = sgn * (wk - wj); ddv_j = sgn * (wi - wk); ddv_k = sgn * (wj - wi)
    d = calc_areas_bary(bary) - t
    scaled = -beta * d
    m = np.max(scaled)
    if m > 700:
        wgt = np.zeros_like(d); wgt[np.argmin(d)] = 1.0
    else:
        e = np.exp(scaled - m)
        wgt = e / np.sum(e) * beta
    grad = np.zeros_like(bary_flat)
    gi = wgt * ddu_i; gj = wgt * ddu_j; gk = wgt * ddu_k
    grad[A, 0] += gi; grad[B, 0] += gj; grad[C, 0] += gk
    gi = wgt * ddv_i; gj = wgt * ddv_j; gk = wgt * ddv_k
    grad[A, 1] += gi; grad[B, 1] += gj; grad[C, 1] += gk
    return grad

# ---------- Coordinated ascent on active (smallest) triangles ----------
def coordinated_ascent(bary, t_target, max_iter=400, tol=1e-13, time_budget=4.0):
    bary = bary.copy()
    start = time.time()
    for _ in range(max_iter):
        areas = calc_areas_bary(bary)
        amin = np.min(areas)
        if amin >= t_target:
            break
        active = np.where(areas <= amin + 1e-10)[0]
        if len(active) == 0:
            break
        P = tri_idx[active]                      # (m,3)
        u = bary[:, 0]; v = bary[:, 1]; w = 1.0 - u - v
        ui = u[P[:,0]]; vi = v[P[:,0]]; wi = w[P[:,0]]
        uj = u[P[:,1]]; vj = v[P[:,1]]; wj = w[P[:,1]]
        uk = u[P[:,2]]; vk = v[P[:,2]]; wk = w[P[:,2]]
        det = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
        sgn = np.sign(det)
        G = np.zeros((N, 2))
        def add(idx, du, dv):
            np.add.at(G[:,0], idx, du)
            np.add.at(G[:,1], idx, dv)
        add(P[:,0], sgn*(vj-vk), sgn*(wk-wj))
        add(P[:,1], sgn*(vk-vi), sgn*(wi-wk))
        add(P[:,2], sgn*(vi-vj), sgn*(wj-wi))
        # normalize gradient magnitude
        gnorm = np.linalg.norm(G)
        if gnorm < 1e-15:
            break
        G = G / gnorm
        # line search
        base = amin
        best_step = 0.0; best_val = base
        step = 0.05
        for _ls in range(24):
            cand = clamp_bary(bary + step * G)
            val = np.min(calc_areas_bary(cand))
            if val > best_val + 1e-15:
                best_val = val; best_step = step
            else:
                step *= 0.5
                if step < 1e-6:
                    break
        if best_step <= 0.0:
            break
        bary = clamp_bary(bary + best_step * G)
        if best_val - base < tol:
            bary = clamp_bary(bary + best_step * G)
            break
        if time.time() - start > time_budget:
            break
    return bary

# ---------- Initial configurations ----------
def make_symmetric_init():
    # Known good symmetric 11-point arrangement (centres + 3 corners + 6 edge-midpoints)
    pts = np.array([
        [0.5, 0.5],
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.0, 0.5], [0.5, 1.0],
        [1.0, 0.5], [0.5, 0.5], [0.0, 0.0], [1.0, 0.0]
    ])
    # de-duplicate / build a clean symmetric set:
    pts = np.array([
        [0.5, 0.5],
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.0, 0.5], [0.5, 1.0], [1.0, 0.5],
        [1/3, 1/3], [2/3, 1/3], [1/3, 2/3]
    ])
    return clamp_bary(pts)

def make_hex_init():
    best = None; bestv = -1
    for scale in [0.18, 0.22, 0.26, 0.3, 0.34]:
        pts = []
        for k in range(int(1.0/scale)+2):
            wv = k*scale
            if wv > 1.0: break
            for j in range(int((1-wv)/scale)+2):
                uv = j*scale
                vv = (1-wv)-uv
                if uv >= -1e-9 and vv >= -1e-9 and uv+vv <= 1+1e-9:
                    pts.append([uv, vv])
        if len(pts) >= N:
            rng = np.random.default_rng(7)
            for _ in range(6):
                idx = rng.choice(len(pts), N, replace=False)
                b = np.array(pts)[idx]
                val = hard_min_value(b)
                if val > bestv:
                    bestv = val; best = b.copy()
    if best is None:
        rng = np.random.default_rng(7)
        u = rng.random(N); v = rng.random(N)*(1-u)
        best = np.column_stack([u, v])
    return best

def make_random_init(seed):
    rng = np.random.default_rng(seed)
    u = rng.random(N); v = rng.random(N)*(1-u)
    return np.column_stack([u, v])

def make_perturbed(base, sigma, seed):
    rng = np.random.default_rng(seed)
    return clamp_bary(base + rng.normal(0, sigma, base.shape))

def refine_local(bary, time_budget=4.0):
    start = time.time()
    b = bary.copy()
    t = hard_min_value(b)

    # epigraph L-BFGS-B with increasing beta
    for beta in [20.0, 60.0, 150.0]:
        if time.time() - start > time_budget*0.5:
            break
        def f(x):
            return epigraph_obj(x, t, beta)
        def g(x):
            return epigraph_grad(x, t, beta)
        try:
            r = minimize(f, b.flatten(), jac=g, method='L-BFGS-B',
                         options={'maxiter': 250, 'ftol': 1e-14, 'gtol': 1e-10})
            if r.fun < 1e9:
                nb = clamp_bary(r.x.reshape(N,2))
                if hard_min_value(nb) > t:
                    b = nb; t = hard_min_value(b)
        except Exception:
            pass

    # coordinated ascent on active triangles
    if time.time() - start < time_budget*0.85:
        b = coordinated_ascent(b, t, time_budget=max(0.0, time_budget - (time.time()-start)))
        t = hard_min_value(b)

    # final epigraph polish
    if time.time() - start < time_budget:
        def f(x):
            return epigraph_obj(x, t, 200.0)
        def g(x):
            return epigraph_grad(x, t, 200.0)
        try:
            r = minimize(f, b.flatten(), jac=g, method='L-BFGS-B',
                         options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-11})
            if r.fun < 1e9:
                nb = clamp_bary(r.x.reshape(N,2))
                if hard_min_value(nb) > t:
                    b = nb; t = hard_min_value(b)
        except Exception:
            pass
    return b

def construct():
    start = time.time()
    deadline = start + TIME_LIMIT

    best = None; bestv = -np.inf
    def update(b):
        nonlocal best, bestv
        b = clamp_bary(b)
        val = hard_min_value(b)
        if val > bestv:
            bestv = val; best = b.copy()
            return True
        return False

    # 1. Strong symmetric start
    s = refine_local(make_symmetric_init(), time_budget=6.0)
    update(s)

    # 2. Hexagonal start
    if time.time() < deadline - 20:
        h = refine_local(make_hex_init(), time_budget=5.0)
        update(h)

    # 3. Multi-start (random + perturbed) with local refinement
    rng = np.random.default_rng(123)
    rc = 0
    while time.time() < deadline - 12:
        rc += 1
        if best is not None and rc % 2 == 0:
            init = make_perturbed(best, 0.04, 2000+rc)
        else:
            init = make_random_init(3000+rc)
        update(refine_local(init, time_budget=3.0))

    # 4. Final high-precision coordinated ascent + epigraph polish
    if best is not None and time.time() < deadline - 6:
        b = coordinated_ascent(best, hard_min_value(best),
                               time_budget=max(0.0, deadline - time.time() - 3))
        update(b)
        if time.time() < deadline - 2:
            t = hard_min_value(b)
            def f(x): return epigraph_obj(x, t, 400.0)
            def g(x): return epigraph_grad(x, t, 400.0)
            try:
                r = minimize(f, b.flatten(), jac=g, method='L-BFGS-B',
                             options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-12})
                if r.fun < 1e9:
                    update(clamp_bary(r.x.reshape(N,2)))
            except Exception:
                pass

    if best is None:
        best = make_symmetric_init()
    return bary_to_cart(best)
