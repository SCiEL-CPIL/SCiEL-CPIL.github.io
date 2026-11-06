"""Broadband antireflection coating design via characteristic-matrix method."""
import time
import math
import random
import numpy as np

# ---- fixed problem constants (from the public contract) ----
N0 = 1.0          # ambient (air)
NS = 1.52         # substrate (glass)
NL = 1.38         # low index
NH = 2.35         # high index
W0, W1 = 400.0, 800.0
MAXL = 20
MAXT = 1000.0
MAXTH = 400.0

# exact evaluation grid (401 equally spaced wavelengths)
WV = np.linspace(W0, W1, 401)
# coarse grid for fast search
WC = np.linspace(W0, W1, 21)


def _reflectance_stack(mat, th, wv):
    """Exact normal-incidence reflectance (fraction) for a stack (air -> substrate)."""
    k = 2.0 * np.pi / wv
    n = np.where(mat == 'H', NH, NL)
    n = np.broadcast_to(n, (len(mat), len(wv)))
    t = np.broadcast_to(th, (len(mat), len(wv)))
    phi = k * n * t
    c = np.cos(phi)
    s = np.sin(phi)
    # characteristic matrices
    M11 = c
    M12 = 1j * s / n
    M21 = 1j * n * s
    M22 = c
    for i in range(len(mat) - 1, -1, -1):
        M11 = M11 * c[i] + M12 * M21[i]
        M12 = M11 * M12 + M22 * M22[i]
        M21 = M21 * c[i] + M22 * M21[i]
        M22 = M21 * M12 + M22 * M22[i]
    Y = (M22 + NS * M12) / (M21 + NS * M11)
    R = (Y - N0) * np.conj(Y - N0) / ((Y + N0) * np.conj(Y + N0))
    return np.real(R)


def _mean_r(mat, th, wv):
    return float(np.mean(_reflectance_stack(mat, th, wv))) * 100.0


def _valid(mat, th):
    if len(mat) == 0 or len(mat) != len(th):
        return False
    for t in th:
        if t < -1e-9 or t > MAXTH + 1e-9:
            return False
    if sum(th) > MAXT + 1e-6:
        return False
    return True


def _fix(th):
    out = []
    for t in th:
        if t < 0:
            t = 0.0
        elif t > MAXTH:
            t = MAXTH
        out.append(t)
    return out


def _de_pop(pop, F, CR):
    n = len(pop)
    new = []
    for i in range(n):
        a, b, c = random.sample(range(n), 3)
        while a == i or b == i or c == i:
            a, b, c = random.sample(range(n), 3)
        cand = [pop[a][j] + F * (pop[b][j] - pop[c][j]) for j in range(len(pop[i]))]
        new.append(cand)
    return new


def _de(pop, fit, F, CR, iters, time_budget, wv):
    n = len(pop)
    for _ in range(iters):
        if time.time() > time_budget:
            break
        for cand in _de_pop(pop, F, CR):
            cand = _fix(cand)
            cf = _mean_r(mat, cand, wv)
            for i in range(n):
                if cf < fit[i]:
                    pop[i] = cand
                    fit[i] = cf
    return pop, fit


def _nm(mat, th, wv, time_budget):
    best = list(th)
    bestf = _mean_r(mat, best, wv)
    n = len(best)
    scale = 30.0
    simplex = [bestf]
    points = [best]
    for i in range(n):
        p = list(best)
        p[i] = p[i] + scale
        p = _fix(p)
        points.append(p)
        simplex.append(_mean_r(mat, p, wv))
    for _ in range(600):
        if time.time() > time_budget:
            break
        order = np.argsort(simplex)
        points = [points[i] for i in order]
        simplex = [simplex[i] for i in order]
        worst = points[-1]
        wval = simplex[-1]
        centroid = [sum(points[i][j] for i in range(n)) / n for j in range(n)]
        alpha = 1.0
        r = [centroid[j] + alpha * (centroid[j] - worst[j]) for j in range(n)]
        r = _fix(r)
        rf = _mean_r(mat, r, wv)
        if rf < simplex[0]:
            g = [centroid[j] + 2.0 * (centroid[j] - worst[j]) for j in range(n)]
            g = _fix(g)
            gf = _mean_r(mat, g, wv)
            if gf < rf:
                points[-1] = g
                simplex[-1] = gf
            else:
                points[-1] = r
                simplex[-1] = rf
        elif rf < simplex[-2]:
            points[-1] = r
            simplex[-1] = rf
        else:
            gamma = 0.5
            s = [centroid[j] + gamma * (worst[j] - centroid[j]) for j in range(n)]
            s = _fix(s)
            sf = _mean_r(mat, s, wv)
            if sf < wval:
                points[-1] = s
                simplex[-1] = sf
            else:
                for i in range(1, n):
                    points[i] = [points[i][j] + 0.5 * (points[0][j] - points[i][j]) for j in range(n)]
                    points[i] = _fix(points[i])
                    simplex[i] = _mean_r(mat, points[i], wv)
    return points[0], simplex[0]


def design(problem):
    global mat
    t_start = time.time()
    deadline = t_start + 55.0          # safety margin under the 60 s CPU budget
    rnd = random.Random(12345)
    rng = np.random.default_rng(12345)

    # ---- 1) coarse structure search over (nL, nH, Lfirst) ----
    best_struct = None
    best_struct_r = None
    for nL in range(1, 7):
        for nH in range(1, 7):
            if nL + nH > MAXL:
                continue
            for Lfirst in (0, 1):
                mat = ['L'] * nL + ['H'] * nH
                if Lfirst:
                    mat = mat[::-1]
                th0 = [60.0] * len(mat)
                pop = [list(th0)]
                fit = [_mean_r(mat, th0, WC)]
                for _ in range(5):
                    cand = [rnd.uniform(0.0, MAXTH) for _ in mat]
                    cand = _fix(cand)
                    pop.append(cand)
                    fit.append(_mean_r(mat, cand, WC))
                pop, fit = _de(pop, fit, 0.8, 0.9, 20, deadline, WC)
                i = int(np.argmin(fit))
                if best_struct_r is None or fit[i] < best_struct_r:
                    best_struct_r = fit[i]
                    best_struct = (list(mat), list(pop[i]))

    mat = best_struct[0]
    th = best_struct[1]

    # ---- 2) exact differential evolution on the chosen structure ----
    n = len(mat)
    pop = [list(th)]
    fit = [_mean_r(mat, th, WV)]
    for _ in range(7):
        cand = [rnd.uniform(0.0, MAXTH) for _ in mat]
        cand = _fix(cand)
        pop.append(cand)
        fit.append(_mean_r(mat, cand, WV))
    pop, fit = _de(pop, fit, 0.7, 0.8, 30, deadline, WV)
    i = int(np.argmin(fit))
    th = pop[i]

    # ---- 3) Nelder-Mead polish ----
    th, _ = _nm(mat, th, WV, deadline)

    # ---- final validity + clamp ----
    if not _valid(mat, th):
        th = _fix(th)
        if not _valid(mat, th):
            mat = ['L']
            th = [W0 / (4.0 * NL)]
    return mat, [float(t) for t in th]
