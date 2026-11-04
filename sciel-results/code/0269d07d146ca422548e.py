"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3.0)
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3 / 2.0]])
TRI_AREA = S3 / 4.0

_IDX_TRIPLES = np.array(
    [(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)]
)
_I = _IDX_TRIPLES[:, 0]
_J = _IDX_TRIPLES[:, 1]
_K = _IDX_TRIPLES[:, 2]


def _bary_to_cart(bc):
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = c * (S3 / 2.0)
    return np.column_stack([x, y])


def _cart_to_bary(points):
    x = points[:, 0]
    y = points[:, 1]
    c = y / (S3 / 2.0)
    b = x - 0.5 * c
    return np.column_stack([b, c])


def _project_to_bary_simplex(bc):
    b = bc[:, 0].astype(float).copy()
    c = bc[:, 1].astype(float).copy()
    b = np.clip(b, 0.0, None)
    c = np.clip(c, 0.0, None)
    excess = b + c - 1.0
    mask = excess > 0
    if np.any(mask):
        s = 1.0 / (1.0 + excess[mask])
        b[mask] *= s
        c[mask] *= s
    return np.column_stack([b, c])


def _clip_to_triangle(points):
    x = points[:, 0].astype(float).copy()
    y = points[:, 1].astype(float).copy()
    y = np.clip(y, 0.0, S3 / 2.0)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(x, x_min, x_max)
    return np.column_stack([x, y])


def _dets_from_bary(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    ui, vi, wi = u[_I], v[_I], w[_I]
    uj, vj, wj = u[_J], v[_J], w[_J]
    uk, vk, wk = u[_K], v[_K], w[_K]
    det = ui * (vj * wk - wj * vk) - vi * (uj * wk - wj * uk) + wi * (uj * vk - vj * uk)
    return det


def _min_area_ratio_from_bary(bary):
    dets = _dets_from_bary(bary)
    return float(np.min(np.abs(dets)))


def _objective_slsqp(x):
    bary = _project_to_bary_simplex(x.reshape(N, 2))
    dets = _dets_from_bary(bary)
    return -float(np.min(np.abs(dets)))


def _objective_smooth(x, beta):
    bary = _project_to_bary_simplex(x.reshape(N, 2))
    dets = _dets_from_bary(bary)
    A = np.abs(dets)
    m = np.min(A)
    shifted = A - m
    z = -beta * shifted
    zmax = np.max(z)
    s = np.sum(np.exp(z - zmax))
    logsum = zmax + np.log(s)
    smooth_min = m - (1.0 / beta) * logsum
    return -smooth_min


def _polish(bary_init, deadline):
    """Multi-stage polishing: SLSQP smooth -> SLSQP exact -> Nelder-Mead."""
    best_bary = bary_init.copy()
    best_score = _min_area_ratio_from_bary(best_bary)
    x0 = best_bary.flatten()

    # SLSQP with smooth objective
    if time.time() < deadline - 5.0:
        try:
            res = minimize(
                lambda x: _objective_smooth(x, 100.0), x0, method='SLSQP',
                bounds=[(0.0, 1.0)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            b = _project_to_bary_simplex(res.x.reshape(N, 2))
            s = _min_area_ratio_from_bary(b)
            if s > best_score:
                best_bary = b
                best_score = s
                x0 = b.flatten()
        except Exception:
            pass

    # SLSQP with exact objective
    if time.time() < deadline - 3.0:
        try:
            res = minimize(
                _objective_slsqp, x0, method='SLSQP',
                bounds=[(0.0, 1.0)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            b = _project_to_bary_simplex(res.x.reshape(N, 2))
            s = _min_area_ratio_from_bary(b)
            if s > best_score:
                best_bary = b
                best_score = s
        except Exception:
            pass

    # Nelder-Mead on Cartesian
    if time.time() < deadline - 1.0:
        def obj_cart(x):
            pts = _clip_to_triangle(x.reshape(N, 2))
            return -_min_area_ratio_from_bary(_cart_to_bary(pts))
        try:
            res = minimize(
                obj_cart, _bary_to_cart(best_bary).flatten(), method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            b = _cart_to_bary(_clip_to_triangle(res.x.reshape(N, 2)))
            s = _min_area_ratio_from_bary(b)
            if s > best_score:
                best_bary = b
                best_score = s
        except Exception:
            pass

    return best_bary, best_score


def _swap_refine(bary, deadline):
    """Iterative swap refinement: find min-area triangles, try swapping involved points."""
    best_bary = bary.copy()
    best_score = _min_area_ratio_from_bary(best_bary)

    for _ in range(50):
        if time.time() > deadline:
            break
        dets = _dets_from_bary(best_bary)
        A = np.abs(dets)
        min_area = np.min(A)
        min_indices = np.where(A <= min_area * 1.000001)[0]

        involved = set()
        for idx in min_indices:
            involved.add(_I[idx])
            involved.add(_J[idx])
            involved.add(_K[idx])

        improved = False
        for p in list(involved):
            if time.time() > deadline:
                break
            for q in range(N):
                if q == p:
                    continue
                new_bary = best_bary.copy()
                new_bary[p], new_bary[q] = new_bary[q], new_bary[p]
                new_bary = _project_to_bary_simplex(new_bary)
                new_score = _min_area_ratio_from_bary(new_bary)
                if new_score > best_score * (1 + 1e-10):
                    best_bary = new_bary
                    best_score = new_score
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return best_bary, best_score


def _differential_evolution_objective(x):
    bary = _project_to_bary_simplex(x.reshape(N, 2))
    return -_min_area_ratio_from_bary(bary)


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s

    rng = np.random.default_rng(42)

    # Phase 1: Differential Evolution for global search
    best_bary = None
    best_score = -float('inf')

    if time.time() < deadline - 20.0:
        try:
            # DE with a reasonable population size
            pop_size = 30
            pop = np.array([_project_to_bary_simplex(rng.uniform(0, 1, (N, 2))) for _ in range(pop_size)])
            pop_scores = np.array([_min_area_ratio_from_bary(p) for p in pop])
            best_idx = np.argmax(pop_scores)
            best_bary = pop[best_idx].copy()
            best_score = pop_scores[best_idx]

            # DE iterations
            F = 0.8
            CR = 0.9
            for gen in range(200):
                if time.time() > deadline - 15.0:
                    break
                for i in range(pop_size):
                    if time.time() > deadline - 15.0:
                        break
                    a, b, c = [j for j in range(pop_size) if j != i][:3]
                    v = pop[a] + F * (pop[b] - pop[c])
                    trial = pop[i].copy()
                    for d in range(N * 2):
                        if rng.random() < CR or d == rng.integers(N * 2):
                            trial[d] = v[d]
                    trial_bary = _project_to_bary_simplex(trial.reshape(N, 2))
                    trial_score = _min_area_ratio_from_bary(trial_bary)
                    if trial_score >= pop_scores[i]:
                        pop[i] = trial_bary
                        pop_scores[i] = trial_score
                        if trial_score > best_score:
                            best_score = trial_score
                            best_bary = trial_bary.copy()
        except Exception:
            pass

    # Phase 2: Polish the best from DE
    if best_bary is not None and time.time() < deadline - 10.0:
        best_bary, best_score = _polish(best_bary, deadline - 5.0)

    # Phase 3: Multi-start local refinement from diverse initializations
    n_starts = 6
    for s in range(n_starts):
        if time.time() > deadline - 8.0:
            break
        # Create diverse initial points using stratified sampling
        bary_init = np.zeros((N, 2))
        for i in range(N):
            # Place points in a structured pattern with random perturbation
            row = i // 3
            col = i % 3
            bary_init[i, 0] = (col + 0.5) / 3.0 + rng.uniform(-0.1, 0.1)
            bary_init[i, 1] = (row + 0.5) / 3.0 * (1.0 - (col + 0.5) / 3.0) + rng.uniform(-0.05, 0.05)
        bary_init = _project_to_bary_simplex(bary_init)

        polished_bary, polished_score = _polish(bary_init, deadline - 3.0)
        if polished_score > best_score:
            best_score = polished_score
            best_bary = polished_bary

    # Phase 4: Final swap refinement
    if best_bary is not None and time.time() < deadline - 2.0:
        best_bary, best_score = _swap_refine(best_bary, deadline - 0.5)

    if best_bary is None:
        # Fallback: uniform random points
        best_bary = _project_to_bary_simplex(rng.uniform(0, 1, (N, 2)))

    return _clip_to_triangle(_bary_to_cart(best_bary))
