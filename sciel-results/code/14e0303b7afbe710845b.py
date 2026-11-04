import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0
SQRT3 = np.sqrt(3.0)

_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
N_TRIPLES = len(_IDX_TRIPLES)


def _bary_to_cart(bary):
    x = bary[:, 0] + 0.5 * bary[:, 1]
    y = 0.5 * SQRT3 * bary[:, 1]
    return np.column_stack([x, y])


def _all_dets_from_bary(bary):
    idx = _IDX_TRIPLES
    ui = bary[idx[:, 0], 0]
    vi = bary[idx[:, 0], 1]
    uj = bary[idx[:, 1], 0]
    vj = bary[idx[:, 1], 1]
    uk = bary[idx[:, 2], 0]
    vk = bary[idx[:, 2], 1]
    du_ji = uj - ui
    dv_ji = vj - vi
    du_ki = uk - ui
    dv_ki = vk - vi
    return 0.5 * SQRT3 * (du_ji * dv_ki - dv_ji * du_ki)


def _min_area_ratio(bary):
    dets = _all_dets_from_bary(bary)
    return 0.5 * np.min(np.abs(dets)) / TRI_AREA


def _project(bary):
    bary = np.array(bary, dtype=float)
    bary = np.maximum(bary, 0.0)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] /= s[mask, None]
    return bary


def _slsqp_fun(x):
    bary = x[:2 * N].reshape(N, 2)
    t = x[2 * N]
    dets = _all_dets_from_bary(bary)
    return -t, dets - t


def _slsqp_grad(x):
    bary = x[:2 * N].reshape(N, 2)
    t = x[2 * N]
    dets = _all_dets_from_bary(bary)
    signs = np.sign(dets)
    idx = _IDX_TRIPLES
    ui = bary[idx[:, 0], 0]
    vi = bary[idx[:, 0], 1]
    uj = bary[idx[:, 1], 0]
    vj = bary[idx[:, 1], 1]
    uk = bary[idx[:, 2], 0]
    vk = bary[idx[:, 2], 1]
    c = 0.5 * SQRT3
    grad_det = np.zeros((N_TRIPLES, N, 2))
    grad_det[:, idx[:, 0], 0] = c * (vj - vk)
    grad_det[:, idx[:, 0], 1] = c * (uj - uk)
    grad_det[:, idx[:, 1], 0] = c * (vk - vi)
    grad_det[:, idx[:, 1], 1] = -c * (uk - ui)
    grad_det[:, idx[:, 2], 0] = -c * (vj - vi)
    grad_det[:, idx[:, 2], 1] = c * (uj - ui)
    grad_bary = -np.sum(signs[:, None, None] * grad_det, axis=0)
    g = np.zeros_like(x)
    g[:2 * N] = grad_bary.flatten()
    g[2 * N] = 1.0
    return g


def _polish_slsqp(bary):
    bary = _project(bary)
    t0 = _min_area_ratio(bary)
    x0 = np.concatenate([bary.flatten(), [t0]])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 1.0)]
    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * i + 1]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2 * i] - x[2 * i + 1]})
    try:
        res = minimize(_slsqp_fun, x0, jac=_slsqp_grad, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': 300, 'ftol': 1e-14})
        if res.success or res.fun < -t0:
            bary_new = _project(res.x[:2 * N].reshape(N, 2))
            score = _min_area_ratio(bary_new)
            if score > t0:
                return bary_new
    except Exception:
        pass
    return bary


def _nm_smooth(x, beta=200.0):
    bary = _project(x.reshape(N, 2))
    dets = np.abs(_all_dets_from_bary(bary))
    max_d = np.max(dets)
    if max_d < 1e-15:
        return 0.0
    nb = -beta * dets
    return (np.max(nb) + np.log(np.sum(np.exp(nb - np.max(nb))))) / beta


def _polish_nm(bary, maxiter=300):
    try:
        res = minimize(_nm_smooth, bary.flatten(), method='Nelder-Mead',
                       args=(200.0,),
                       options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        bary_new = _project(res.x.reshape(N, 2))
        if _min_area_ratio(bary_new) > _min_area_ratio(bary):
            return bary_new
    except Exception:
        pass
    return bary


def _full_polish(bary):
    b1 = _polish_slsqp(bary)
    b2 = _polish_nm(b1)
    b3 = _polish_slsqp(b2)
    b4 = _polish_nm(b3)
    best = max([bary, b1, b2, b3, b4], key=_min_area_ratio)
    return best


def _de_population(n_pop, rng, best_bary, best_score):
    pop = []
    # Include best
    pop.append(best_bary.copy())
    # Structured seeds
    seeds = [
        np.array([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5],
                  [1/3, 1/3], [0.25, 0], [0.75, 0], [0.25, 0.25], [0.5, 0.75]], dtype=float),
        np.array([[0, 0], [1, 0], [0, 1], [1/3, 1/3], [1/3, 2/3], [2/3, 1/3],
                  [0.5, 0.5], [0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.5, 0.3]], dtype=float),
    ]
    for s in seeds:
        pop.append(_project(s))
    # Jittered copies of best
    gap = max(0.001, 0.02 - best_score)
    for _ in range(n_pop - 1 - len(seeds)):
        jitter = rng.normal(0, gap * 1.5, size=(N, 2))
        pop.append(_project(best_bary + jitter))
    # Random
    for _ in range(max(0, n_pop - len(pop))):
        b = np.random.RandomState(rng.randint(1, 99999)).uniform(0.05, 0.95, (N, 2))
        b = np.maximum(b, 0)
        s = b[:, 0] + b[:, 1]
        mask = s > 1.0
        if np.any(mask):
            b[mask] /= s[mask, None]
        pop.append(b)
    return pop


def _de_step(pop, F, CR, rng):
    n_pop = len(pop)
    scores = np.array([_min_area_ratio(p) for p in pop])
    new_pop = []
    for i in range(n_pop):
        # Select 3 distinct random indices different from i
        candidates = [j for j in range(n_pop) if j != i]
        a, b, c = rng.choice(candidates, size=3, replace=False)
        v = pop[a] + F * (pop[b] - pop[c])
        # Crossover
        trial = pop[i].copy()
        j_rand = rng.randint(N)
        for j in range(N):
            if rng.random() < CR or j == j_rand:
                trial[j] = v[j]
        trial = _project(trial)
        # Select
        if _min_area_ratio(trial) >= scores[i]:
            new_pop.append(trial)
        else:
            new_pop.append(pop[i])
    return new_pop


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def time_left():
        return deadline - time.time()

    rng = np.random.RandomState(2024)

    # Initial best
    bary_init = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5],
        [0.0, 0.5], [1/3, 1/3], [0.25, 0.0], [0.75, 0.0],
        [0.25, 0.25], [0.5, 0.75]
    ], dtype=float)
    best_bary = _project(bary_init)
    best_score = _min_area_ratio(best_bary)

    # Initial polish
    if time_left() > 50:
        best_bary = _full_polish(best_bary)
        best_score = _min_area_ratio(best_bary)

    # DE parameters
    n_pop = 30
    F = 0.5
    CR = 0.7
    de_gens_per_cycle = 15

    # Main loop: DE cycles with polishing
    cycle = 0
    while time_left() > 30.0:
        cycle += 1

        # Build population around current best
        pop = _de_population(n_pop, rng, best_bary, best_score)
        pop_scores = np.array([_min_area_ratio(p) for p in pop])
        best_i = np.argmax(pop_scores)
        if pop_scores[best_i] > best_score:
            best_score = pop_scores[best_i]
            best_bary = pop[best_i].copy()

        # Run DE generations
        for gen in range(de_gens_per_cycle):
            if time_left() < 35:
                break
            pop = _de_step(pop, F, CR, rng)
            # Track best
            for i, p in enumerate(pop):
                s = _min_area_ratio(p)
                if s > best_score:
                    best_score = s
                    best_bary = p.copy()

            # Adapt F slightly
            F = max(0.3, F - 0.01 * gen)

        # Polish the best found
        if time_left() > 25:
            polished = _full_polish(best_bary)
            s = _min_area_ratio(polished)
            if s > best_score:
                best_score = s
                best_bary = polished.copy()

    # Final polish with remaining time
    if time_left() > 10:
        best_bary = _full_polish(best_bary)
        best_score = _min_area_ratio(best_bary)

    if time_left() > 5:
        # One more round of targeted local search
        for _ in range(5):
            if time_left() < 3:
                break
            gap = max(0.0005, 0.01 - best_score)
            jitter = rng.normal(0, gap, size=(N, 2))
            candidate = _project(best_bary + jitter)
            polished = _full_polish(candidate)
            s = _min_area_ratio(polished)
            if s > best_score:
                best_score = s
                best_bary = polished.copy()

    points = _bary_to_cart(best_bary)
    return points
