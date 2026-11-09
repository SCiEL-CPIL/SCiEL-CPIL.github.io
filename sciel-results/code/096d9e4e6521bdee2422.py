# EVOLVE-BLOCK-START
import time
import numpy as np

# ---- Fixed problem data ----
_N = 10
_DEMANDS = [
    (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
    (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
    (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
]
_D = len(_DEMANDS)

# Build exact edge masks for each routing direction.
# CW from s to t uses edges s, s+1, ..., t-1 (mod N). CCW uses all other edges.
_CW = np.zeros((_N, _D), dtype=np.float64)
_CCW = np.zeros((_N, _D), dtype=np.float64)
for j, (s, t, w) in enumerate(_DEMANDS):
    if s <= t:
        cw_edges = list(range(s, t))
    else:
        cw_edges = list(range(s, _N)) + list(range(0, t))
    for e in cw_edges:
        _CW[e, j] = 1.0
    for e in range(_N):
        if _CW[e, j] == 0.0:
            _CCW[e, j] = 1.0

# Load contribution matrix L[i, j] = weight of demand j if it uses edge i.
# CW choice adds w, CCW choice adds w as well (each demand adds its weight to its route).
_L = np.zeros((_N, _D), dtype=np.float64)
for j, (s, t, w) in enumerate(_DEMANDS):
    for e in range(_N):
        if _CW[e, j] == 1.0:
            _L[e, j] = w
        elif _CCW[e, j] == 1.0:
            _L[e, j] = w


def _load(b):
    """Edge loads for a binary routing vector b (1 = CCW, 0 = CW)."""
    # load_i = sum_j L[i,j] * (CW or CCW mask). For each demand, exactly one mask is 1.
    # load = L_cw * (1-b) + L_ccw * b  where L_cw[e,j]=w if CW else 0, L_ccw similarly.
    return _CW * _L * (1.0 - b) + _CCW * _L * b


def _max_load(b):
    return float(np.max(_load(b)))


def ring_routing() -> np.ndarray:
    """Return a (15,) routing array; entry >= 0.5 routes CCW, else CW.
    Minimizes the maximum edge load."""
    start = time.time()
    deadline = start + 590.0  # safety margin well under 630s

    # ---- Phase 1: smooth continuous optimization (log-sum-exp surrogate) ----
    # Variables: x = [t (scalar), p (D,)] where p_j = probability of CCW.
    # load_i = sum_j L[i,j] * (CWmask*(1-p) + CCWmask*p)
    # surrogate = sum_i exp((load_i - t) / tau)  (minimizes t ~ max load)
    from scipy.optimize import minimize

    def surrogate_and_grad(v, tau):
        t = v[0]
        p = v[1:]
        p = np.clip(p, 0.0, 1.0)
        # load vector
        load = _CW * _L * (1.0 - p) + _CCW * _L * p
        z = (load - t) / tau
        # numerical stability
        zmax = np.max(z)
        ez = np.exp(z - zmax)
        val = np.sum(ez) * np.exp(zmax)
        # gradients
        dval_dz = ez * np.exp(zmax)  # = exp(z)
        dval_dt = -np.sum(dval_dz) / tau
        # d load_i / dp_j = L[i,j]*(CCWmask - CWmask)
        dload_dp = (_CCW - _CW) * _L
        dval_dp = dval_dz @ dload_dp / tau
        grad = np.empty_like(v)
        grad[0] = dval_dt
        grad[1:] = dval_dp
        return val, grad

    # Try a few temperature values (annealing-like) and keep best rounded solution.
    best_b = np.zeros(_D)
    best_ml = _max_load(best_b)

    # initial continuous solution: start near CW (p=0) with t at current max
    t0 = float(np.max(_CW * _L * np.ones(_D)))  # all-CW max load
    v0 = np.concatenate(([t0], np.zeros(_D)))

    taus = [3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08]
    v_cur = v0.copy()
    for tau in taus:
        if time.time() > deadline - 60.0:
            break
        try:
            res = minimize(
                lambda v: surrogate_and_grad(v, tau)[0],
                v_cur,
                jac=lambda v: surrogate_and_grad(v, tau)[1],
                method='L-BFGS-B',
                bounds=[(0.0, 100.0)] + [(0.0, 1.0)] * _D,
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40},
            )
            v_cur = res.x
        except Exception:
            pass
        # round
        b = (v_cur[1:] >= 0.5).astype(np.float64)
        ml = _max_load(b)
        if ml < best_ml:
            best_ml = ml
            best_b = b.copy()

    # ---- Phase 2: local search (1-flip and 2-flip) with deadline guard ----
    b = best_b.copy()
    cur = _max_load(b)
    while time.time() < deadline:
        improved = False
        # 1-flip
        for j in range(_D):
            b[j] = 1.0 - b[j]
            ml = _max_load(b)
            if ml < cur - 1e-12:
                cur = ml
                improved = True
            else:
                b[j] = 1.0 - b[j]
        # 2-flip
        for j in range(_D):
            for k in range(j + 1, _D):
                if time.time() > deadline:
                    break
                b[j] = 1.0 - b[j]
                b[k] = 1.0 - b[k]
                ml = _max_load(b)
                if ml < cur - 1e-12:
                    cur = ml
                    improved = True
                else:
                    b[j] = 1.0 - b[j]
                    b[k] = 1.0 - b[k]
            if time.time() > deadline:
                break
        if not improved:
            break

    # Return as >=0.5 for CCW (1), else CW (0)
    return b

# EVOLVE-BLOCK-END
