import numpy as np
import time
import math


def _make_policy(thresholds, n_steps, dt, T):
    class ExercisePolicy:
        def __init__(self, thresholds, n_steps, dt, T):
            self.thresholds = np.asarray(thresholds, dtype=np.float64)
            self.n_steps = n_steps
            self.dt = dt
            self.T = T

        def exercise(self, current_state, time):
            cs = np.asarray(current_state, dtype=np.float64)
            if time is None:
                t = 0.0
            else:
                try:
                    t = float(time)
                except Exception:
                    t = 0.0
            if self.dt > 0:
                j = int(round(t / self.dt))
            else:
                j = 1
            j = max(1, min(j, self.n_steps))
            threshold = self.thresholds[j - 1]
            return cs <= threshold

    return ExercisePolicy(thresholds, n_steps, dt, T)


def _simple_thresholds(K, r, sigma, T, n_steps):
    """Fallback: time-decaying thresholds, always valid."""
    dt = T / n_steps
    th = np.empty(n_steps)
    for j in range(n_steps):
        rem = (n_steps - j) * dt
        th[j] = K * (1.0 - math.exp(-2.5 * rem / max(dt, 1e-9)))
    th[-1] = K
    return th


def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    DEADLINE = 145.0
    SAFETY = 5.0

    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 12

    def _get(obj, *names, default=None):
        for nm in names:
            if isinstance(obj, dict):
                if nm in obj:
                    return obj[nm]
            elif hasattr(obj, nm):
                return getattr(obj, nm)
        return default

    try:
        K = float(_get(contract, 'K', 'strike', 'Strike', default=K))
        r = float(_get(contract, 'r', 'rate', 'Rate', 'risk_free', default=r))
        sigma = float(_get(contract, 'sigma', 'vol', 'volatility', default=sigma))
        T = float(_get(contract, 'T', 'maturity', 'Maturity', default=T))
        n_steps = int(_get(contract, 'n_dates', 'num_dates', 'exercise_dates', default=n_steps))
    except Exception:
        pass

    if n_steps < 1:
        n_steps = 1
    dt = T / n_steps if n_steps > 0 else T
    mu_log = (r - 0.5 * sigma * sigma) * dt
    vol_log = sigma * math.sqrt(dt)
    disc = math.exp(-r * dt)

    # Grid in log-space for better resolution near zero and at K
    # Focus where the put has value: S < K mostly, but some above K for continuation
    S_min = max(0.01 * K, 1e-6)
    S_max = 2.0 * K
    n_grid = 500
    logS_grid = np.linspace(math.log(S_min), math.log(S_max), n_grid)
    S_grid = np.exp(logS_grid)

    # Gauss-Hermite quadrature for the normal density
    n_quad = 64
    zq, wq = np.polynomial.hermite.hermgauss(n_quad)
    zq = zq * math.sqrt(2.0)
    wq = wq / math.sqrt(math.pi)

    def cont_value(V_next):
        """Compute e^{-r dt} * E[V_next(S * exp(drift + vol_step*Z)) | S]
        for each S in S_grid, using Gauss-Hermite quadrature in log-space."""
        # For each S_i, log(S_{i+1}) = log(S_i) + drift + vol_step * Z
        log_S_next = logS_grid[:, None] + mu_log + vol_log * zq[None, :]
        # Clip to grid log range for interpolation
        log_S_next = np.clip(log_S_next, logS_grid[0], logS_grid[-1])
        # Interpolate V_next in log-space
        V_interp = np.interp(log_S_next, logS_grid, V_next)  # (n_grid, n_quad)
        cont = disc * np.sum(wq[None, :] * V_interp, axis=1)
        return cont

    def find_threshold(immediate, cont):
        """Find the threshold S where immediate(S) >= cont(S).
        Returns the largest S where exercise is optimal."""
        diff = immediate - cont
        # Only consider S < K (for S >= K, immediate = 0, never exercise)
        valid = S_grid < K
        if not np.any(valid):
            return 0.0
        diff_v = diff[valid]
        S_v = S_grid[valid]
        pos_idx = np.where(diff_v >= 0)[0]
        if len(pos_idx) == 0:
            return 0.0
        last = pos_idx[-1]
        if last + 1 < len(S_v):
            d1 = diff_v[last]
            d2 = diff_v[last + 1]
            if d2 < 0 and d1 >= 0:
                # Linear interpolation for finer threshold
                frac = d1 / (d1 - d2 + 1e-300)
                thr = S_v[last] + frac * (S_v[last + 1] - S_v[last])
            else:
                thr = S_v[last]
        else:
            thr = S_v[last]
        # Cap at K
        if thr > K:
            thr = K
        if thr < 0:
            thr = 0.0
        return thr

    # Terminal value: V_n(S) = max(K - S, 0)
    V_next = np.maximum(K - S_grid, 0.0)
    thresholds = np.zeros(n_steps)

    # Backward induction
    for t_idx in range(n_steps - 1, -1, -1):
        # Check time budget
        if time.time() - t_start > DEADLINE - SAFETY:
            # Fill remaining with heuristic
            for j in range(t_idx, -1, -1):
                rem = (n_steps - j) * dt
                thresholds[j] = K * (1.0 - math.exp(-2.5 * rem / max(dt, 1e-9)))
            thresholds[-1] = K
            break

        immediate = np.maximum(K - S_grid, 0.0)
        cont = cont_value(V_next)
        V_cur = np.maximum(immediate, cont)

        thresholds[t_idx] = find_threshold(immediate, cont)
        V_next = V_cur

    # Enforce monotonic non-increasing across dates (earlier dates: higher threshold)
    for j in range(1, n_steps):
        if thresholds[j] > thresholds[j - 1]:
            thresholds[j] = thresholds[j - 1]

    # Keep thresholds within a sane range
    thresholds = np.clip(thresholds, 0.0, K)

    # If any threshold is still zero (shouldn't happen), use fallback
    if not np.all(np.isfinite(thresholds)) or np.any(thresholds < 0):
        thresholds = _simple_thresholds(K, r, sigma, T, n_steps)

    return _make_policy(thresholds, n_steps, dt, T)
