"""Bermudan put exercise policy: backward induction on a price grid with
antithetic-MC continuation values, deadline-guarded, with a final polish pass."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_param(contract, key, default):
    if isinstance(contract, dict):
        v = contract.get(key, default)
        return default if v is None else v
    return getattr(contract, key, default)

def _build_price_grid(r, sigma, T, n_exercise, n_grid):
    dt = T / n_exercise
    mu = r - 0.5 * sigma * sigma
    # state at each exercise time t_k = k*dt (k=1..n_exercise)
    log_mean = mu * (np.arange(1, n_exercise + 1) * dt)
    log_sd = sigma * np.sqrt(np.arange(1, n_exercise + 1) * dt)
    lo = np.exp(log_mean - 4.0 * log_sd)
    hi = np.exp(log_mean + 4.0 * log_sd)
    lo = np.minimum.accumulate(lo)
    hi = np.maximum.accumulate(hi)
    lo = np.maximum(lo, 1e-4)
    grids = []
    for i in range(n_exercise):
        grids.append(np.linspace(lo[i], hi[i], n_grid))
    return grids, dt

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    DEADLINE = 130.0  # 20s safety margin before 150s kill

    K = float(_get_param(contract, 'K', 100.0))
    r = float(_get_param(contract, 'r', 0.05))
    sigma = float(_get_param(contract, 'sigma', 0.2))
    T = float(_get_param(contract, 'T', 1.0))
    n_exercise = int(_get_param(contract, 'n_exercise', 12))
    n_grid = int(_get_param(contract, 'n_grid', 256))

    grids, dt = _build_price_grid(r, sigma, T, n_exercise, n_grid)
    disc = np.exp(-r * dt)
    n = n_exercise

    # V[k] = value vector at exercise time k (k = n-1 .. 0), shape (n, n_grid)
    V = np.empty((n, n_grid))
    # terminal: only intrinsic payoff (no continuation)
    V[n - 1] = np.maximum(K - grids[n - 1], 0.0)
    # exercise threshold S*(k): below this we exercise
    thr = np.empty(n)
    thr[n - 1] = K

    def _mc_continue(grid, V_next, n_paths, rng):
        # estimate discounted E[ V_next(S_{t+1}) | S_t = grid ] via antithetic MC
        S = grid  # (m,)
        m = S.shape[0]
        n_half = n_paths // 2
        acc = np.zeros(m)
        # generate normal increments
        z = rng.standard_normal((n_half, m))
        # antithetic pair: z and -z
        dS1 = S[None, :] * (sigma * np.sqrt(dt) * z)
        dS2 = -dS1
        S1 = S[None, :] * np.exp((r - 0.5 * sigma * sigma) * dt + dS1)
        S2 = S[None, :] * np.exp((r - 0.5 * sigma * sigma) * dt + dS2)
        # clip to next grid range for interpolation
        lo = grids[-1].min() if False else V_next.shape[0]  # placeholder
        lo_next = grids[n - 1].min()  # last grid is terminal; but we use per-step
        # use the grid that corresponds to time t+1 (which is grids[k+1])
        return S1, S2

    # We'll implement continuation estimation inline per step to use the right next grid.
    def _interp(V_next, grid_next, S):
        # S: (n_half, m) or (m,); returns same shape
        S = np.asarray(S)
        return np.interp(S.ravel(), grid_next, V_next).reshape(S.shape)

    def _continue_value(k, n_paths, rng):
        # k is index (0-based) of current step; next step is k+1
        grid = grids[k]
        grid_next = grids[k + 1]
        V_next = V[k + 1]
        m = grid.shape[0]
        n_half = n_paths // 2
        z = rng.standard_normal((n_half, m))
        drift = (r - 0.5 * sigma * sigma) * dt
        vol = sigma * np.sqrt(dt)
        S1 = grid[None, :] * np.exp(drift + vol * z)
        S2 = grid[None, :] * np.exp(drift - vol * z)
        c1 = _interp(V_next, grid_next, S1)
        c2 = _interp(V_next, grid_next, S2)
        return 0.5 * (c1 + c2).mean(axis=0)

    # initial path count, shrink if time running out
    n_paths = 20000
    rng = np.random.default_rng(12345)

    for k in range(n - 2, -1, -1):
        if time.time() - start_time > DEADLINE - 15:
            # not enough time for a careful step; fall back to intrinsic-only
            V[k] = np.maximum(K - grids[k], 0.0)
            thr[k] = K
            continue
        cont = _continue_value(k, n_paths, rng)
        intrinsic = np.maximum(K - grids[k], 0.0)
        V[k] = np.maximum(intrinsic, disc * cont)
        # threshold: largest S where V > intrinsic (i.e. continuation worthwhile)
        diff = V[k] - intrinsic
        # grid is increasing in S; exercise (intrinsic >= disc*cont) at low S
        # find first index where diff <= 0 -> that S is threshold (exercise below)
        idx = np.where(diff <= 0.0)[0]
        if idx.size == 0:
            thr[k] = grids[k].max()
        else:
            thr[k] = grids[k][idx[0]]

    # FINAL POLISH PASS: refine thresholds using a tighter local solve with more paths
    if time.time() - start_time < DEADLINE - 10:
        n_paths_polish = 40000
        rng2 = np.random.default_rng(98765)
        # recompute continuation with more paths for earlier steps (most sensitive)
        # only polish steps where continuation was active (thr > small)
        for k in range(n - 2, -1, -1):
            if time.time() - start_time > DEADLINE - 8:
                break
            if thr[k] <= 1e-6:
                continue
            cont = _continue_value(k, n_paths_polish, rng2)
            intrinsic = np.maximum(K - grids[k], 0.0)
            V[k] = np.maximum(intrinsic, disc * cont)
            diff = V[k] - intrinsic
            idx = np.where(diff <= 0.0)[0]
            if idx.size == 0:
                thr[k] = grids[k].max()
            else:
                thr[k] = grids[k][idx[0]]

    # Build policy object
    # For exercise at time t, find the exercise index k such that t is closest to t_k
    # t_k = (k+1)*dt for k in 0..n-1
    class ExercisePolicy:
        def __init__(self, grids, thr, dt, n_exercise, K):
            self.grids = grids
            self.thr = thr
            self.dt = dt
            self.n_exercise = n_exercise
            self.K = K
            self.t_ex = (np.arange(1, n_exercise + 1)) * dt
            self._last = None

        def exercise(self, current_state, t):
            # current_state: array of spot prices; t: scalar time
            current_state = np.asarray(current_state, dtype=float)
            # find exercise index: nearest t_ex to t
            k = int(np.argmin(np.abs(self.t_ex - t)))
            th = self.thr[k]
            # exercise if spot <= threshold
            return (current_state <= th).astype(bool)

    return ExercisePolicy(grids, thr, dt, n_exercise, K)
# EVOLVE-BLOCK-END
