import numpy as np


def fit_exercise_policy(training_paths, contract):
    # ---- Extract model parameters from the contract (with safe defaults) ----
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12

    def _get(obj, *names, default=None):
        for n in names:
            if isinstance(obj, dict):
                if n in obj:
                    return obj[n]
            else:
                if hasattr(obj, n):
                    return getattr(obj, n)
        return default

    try:
        K = float(_get(contract, 'K', 'strike', 'Strike', default=K))
        r = float(_get(contract, 'r', 'rate', 'Rate', 'risk_free', default=r))
        sigma = float(_get(contract, 'sigma', 'vol', 'Vol', 'volatility', default=sigma))
        T = float(_get(contract, 'T', 'maturity', 'Maturity', 'horizon', default=T))
        nd = _get(contract, 'n_dates', 'n_exercise', 'n_bermudan', 'exercise_dates', default=n_dates)
        if nd is not None:
            n_dates = int(nd)
    except Exception:
        pass

    # ---- Backward induction on a fine CRR binomial tree ----
    # Increased N for higher precision in the exercise boundary
    N = 10000
    dt = T / N
    sqrt_dt = np.sqrt(dt)
    u = np.exp(sigma * sqrt_dt)
    d = 1.0 / u
    disc = np.exp(-r * dt)
    pu = (np.exp(r * dt) - d) / (u - d)
    pd = 1.0 - pu
    S0 = 100.0

    steps_per_date = max(1, N // n_dates)
    bermudan_levels = set(i * steps_per_date for i in range(1, n_dates + 1))

    # Pre-compute S values for Bermudan levels to save time in loop
    S_at_level = {}
    for level in bermudan_levels:
        j = np.arange(level + 1)
        S_at_level[level] = S0 * np.power(u, 2.0 * j - level)

    V_levels = [None] * (N + 1)

    # Final (maturity) level: put expires, value = intrinsic
    k = N
    S_final = S_at_level.get(k, S0 * np.power(u, 2.0 * np.arange(k + 1) - k))
    V_levels[k] = np.maximum(K - S_final, 0.0)

    thresholds = {}
    # If maturity coincides with the last exercise date, exercise iff ITM -> boundary = K
    if k in bermudan_levels:
        thresholds[k] = K

    for k in range(N - 1, -1, -1):
        V_next = V_levels[k + 1]
        cont = disc * (pu * V_next[0:k + 1] + pd * V_next[1:k + 2])
        if k in bermudan_levels:
            S = S_at_level[k]
            ex = np.maximum(K - S, 0.0)
            V_curr = np.maximum(ex, cont)
            # Locate the exercise/hold boundary by linear interpolation of (ex - cont)
            diff = ex - cont
            idx = np.where(diff >= 0.0)[0]
            if idx.size == 0:
                star = 0.0
            elif idx[-1] >= k:
                star = S[k]
            else:
                jstar = int(idx[-1])
                d0 = diff[jstar]
                d1 = diff[jstar + 1]
                denom = d0 - d1
                if abs(denom) < 1e-14:
                    star = S[jstar]
                else:
                    frac = d0 / denom
                    star = S[jstar] + frac * (S[jstar + 1] - S[jstar])
            thresholds[k] = star
        else:
            V_curr = cont
        V_levels[k] = V_curr

    date_thresholds = []
    for i in range(1, n_dates + 1):
        level = i * steps_per_date
        date_thresholds.append(float(thresholds.get(level, K if level == N else 0.0)))

    class _Policy:
        __slots__ = ('date_thresholds', 'n_dates', 'T')

        def __init__(self, date_thresholds, n_dates, T):
            self.date_thresholds = date_thresholds
            self.n_dates = n_dates
            self.T = T

        def exercise(self, current_state, time):
            cs = np.asarray(current_state, dtype=float)
            t = time
            if isinstance(t, (list, tuple, np.ndarray)):
                t = float(np.asarray(t, dtype=float).mean())
            else:
                t = float(t)
            # Primary: time is a fraction of the horizon in (0, T].
            # Fallback: if a value > 1 is passed, treat it as a 1..n_dates index.
            if t > 1.0 + 1e-9:
                i = int(round(t))
            else:
                i = int(round(t / self.T * self.n_dates))
            if i < 1:
                i = 1
            elif i > self.n_dates:
                i = self.n_dates
            star = self.date_thresholds[i - 1]
            return cs <= star

    return _Policy(date_thresholds, n_dates, T)
