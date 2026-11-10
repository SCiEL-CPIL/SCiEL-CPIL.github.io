# EVOLVE-BLOCK-START
import time
import numpy as np

# ---- fixed problem constants -------------------------------------------------
K = 10
NPT = 21
M = np.arange(-K, K + 1, dtype=np.float64)   # m values, length 21

# ---- precomputed quadrature / basis (module load, tiny) ----------------------
def _precompute():
    # 70-point Gauss-Legendre on [-1,1]
    x, w = np.polynomial.legendre.leggauss(70)
    # 130 equally spaced phi
    phi = np.arange(130, dtype=np.float64) * (2.0 * np.pi / 130.0)
    # Condon-Shortley associated Legendre P_l^m(cos theta)
    P = np.zeros((70, 11))
    for m in range(0, 11):
        P[:, m] = np.polynomial.legendre.legval(x,
                    [0.0]*m + [1.0]) * ((-1.0)**m) * np.sqrt(np.prod(1.0 - x*x))**m
    # normalization N_m
    N = np.zeros(11)
    for m in range(0, 11):
        N[m] = np.sqrt(21.0/(4.0*np.pi) * np.math.factorial(10-m)/np.math.factorial(10+m))
    # basis columns: for m=-10..10
    B = np.zeros((70, 130, 21))
    for i, m in enumerate(M):
        if m == 0:
            B[:, :, i] = N[0] * P[:, 0][:, None]
        elif m > 0:
            B[:, :, i] = np.sqrt(2.0) * N[m] * P[:, m][:, None] * np.cos(m*phi)[None, :]
        else:
            B[:, :, i] = np.sqrt(2.0) * N[-m] * P[:, -m][:, None] * np.sin(-m*phi)[None, :]
    # measure weight: dOmega = dphi * d(cos theta); total mass 4*pi
    W = np.outer(w, np.full(130, 2.0*np.pi/130.0))
    W = W[:, :, None]  # (70,130,1)
    return B, W

_B, _W = _precompute()

def _R(c):
    f = np.tensordot(c, _B, axes=(0, 2))   # (70,130)
    f2 = f*f
    n2 = np.sum(f2 * _W)
    n4 = np.sum(f2*f2 * _W)
    if n2 <= 0:
        return np.inf
    return np.sqrt(n4/n2)

# ---- optimization ------------------------------------------------------------
def rudin_problem10() -> np.ndarray:
    t0 = time.time()
    LIMIT = 560.0   # safety margin under 630s
    best_c = np.ones(NPT, dtype=np.float64)
    best_R = _R(best_c)

    def time_left():
        return LIMIT - (time.time() - t0)

    # 1) global search with differential evolution (bounded, robust)
    try:
        from scipy.optimize import differential_evolution, minimize
        bounds = [(-10.0, 10.0)] * NPT
        de_res = differential_evolution(
            _R, bounds, maxiter=60, popsize=15, tol=1e-6,
            seed=1234, polish=False, init='sobol',
            callback=lambda xk, fk: (_ for _ in ()).throw(TimeoutError) if time_left() < 120 else None
        )
        if de_res is not None and np.isfinite(de_res.fun) and de_res.fun < best_R:
            best_R = de_res.fun
            best_c = de_res.x.astype(np.float64)
    except (TimeoutError, Exception):
        pass

    # 2) local refinement (L-BFGS-B) from several starts
    starts = [best_c.copy()]
    rng = np.random.default_rng(0)
    for _ in range(8):
        starts.append(rng.uniform(-10, 10, size=NPT))
    try:
        from scipy.optimize import minimize
        for s in starts:
            if time_left() < 30:
                break
            res = minimize(_R, s, method='L-BFGS-B',
                           bounds=[(-100.0, 100.0)]*NPT,
                           options=dict(maxiter=400, ftol=1e-12, gtol=1e-10))
            if res is not None and np.isfinite(res.fun) and res.fun < best_R:
                best_R = res.fun
                best_c = res.x.astype(np.float64)
    except Exception:
        pass

    # 3) final guard: ensure validity
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(NPT, dtype=np.float64)
    mx = np.max(np.abs(best_c))
    if mx < 1e-3 or mx > 1e5:
        if mx > 0:
            best_c = best_c * (1.0/mx) * 1.0   # rescale into range
        else:
            best_c = np.ones(NPT, dtype=np.float64)

    return best_c


# EVOLVE-BLOCK-END
