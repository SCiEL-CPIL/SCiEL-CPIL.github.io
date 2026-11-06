import numpy as np
import time
from scipy.special import ndtr


# EVOLVE-BLOCK-START
def _european_put(s, k, r, sigma, tau):
    """Analytic European put price under GBM."""
    if tau <= 0:
        return max(k - s, 0.0)
    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return k * np.exp(-r * tau) * ndtr(-d2) - s * ndtr(-d1)


def _continuation(V_next, grid, mu_log, vol_log, dt, r):
    """Compute continuation value using analytic European put approximation.
    
    For Bermudan puts, the continuation value at step j (with tau = (j+1)*dt remaining
    to maturity) can be well-approximated by the European put price. This is much
    faster than numerical integration and provides a good baseline for the threshold.
    """
    n = grid.shape[0]
    disc = np.exp(-r * dt)
    # For each grid point, compute European put with remaining time
    # V_next is defined on grid, but we approximate it as European put
    # Actually, we should use the actual V_next values via interpolation, but
    # for speed, we can compute European put directly
    # However, to be more accurate, let's interpolate V_next onto a finer grid
    # and use that for the expectation
    
    # Alternative: use the fact that for a put, the optimal exercise boundary
    # is where payoff >= continuation. We can compute continuation as:
    # E[V_next(S_next)] where S_next = S * exp(N(mu_log, vol_log^2))
    # This is a convolution, but we can approximate it with European put
    # which is exact for the case where V_next = payoff (at final step)
    
    # For better accuracy, let's do a simplified numerical integration
    # but with fewer points and vectorized operations
    n_int = 201  # fewer integration points for speed
    x_grid = np.linspace(-4.0 * vol_log, 4.0 * vol_log, n_int)
    z = (x_grid - mu_log) / vol_log
    density = np.exp(-0.5 * z * z) / (vol_log * np.sqrt(2.0 * np.pi))
    wx = np.zeros_like(x_grid)
    if n_int > 1:
        wx[0] = 0.5 * (x_grid[1] - x_grid[0])
        wx[-1] = 0.5 * (x_grid[-1] - x_grid[-2])
        if n_int > 2:
            wx[1:-1] = 0.5 * (x_grid[2:] - x_grid[:-2])
    wd = wx * density
    
    cont = np.empty(n)
    expx = np.exp(x_grid)
    chunk = 512
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        S_next = grid[a:b, None] * expx[None, :]
        Vn = np.interp(S_next.ravel(), grid, V_next).reshape(S_next.shape)
        cont[a:b] = Vn @ wd
    return disc * cont


def _threshold_for(V, payoff, grid, V_next, mu_log, vol_log, dt, r, k, sigma, tau_remaining):
    """Largest grid point where exercise (payoff) is optimal, refined by bisection.

    Exercise is optimal at s iff payoff(s) >= disc * E[V_next(S_next|s)].
    payoff is decreasing in s, continuation is increasing in s -> single crossing.
    """
    n = grid.shape[0]
    # coarse mask
    cont = _continuation(V_next, grid, mu_log, vol_log, dt, r)
    ex_mask = payoff >= cont - 1e-12
    if not np.any(ex_mask):
        return grid[0]
    idx_ex = np.where(ex_mask)[0]
    lo = idx_ex[-1]
    # find a bracket: lo is exercise, find hi (next non-exercise)
    hi = lo
    # move hi to the right until non-exercise or end
    while hi + 1 < n and payoff[hi + 1] >= cont[hi + 1] - 1e-12:
        hi += 1
    if hi >= n - 1:
        return grid[hi]
    # Now lo..hi are exercise, hi+1 is continue. Threshold between grid[hi] and grid[hi+1].
    # Bisection on the smooth function f(s) = payoff(s) - cont(s).
    def f(s):
        c = _continuation(V_next, np.array([s]), mu_log, vol_log, dt, r)[0]
        return max(k - s, 0.0) - c
    a = grid[hi]
    b = grid[hi + 1]
    fa = f(a)
    fb = f(b)
    if fa < 0:
        return a
    if fb > 0:
        return b
    for _ in range(50):
        mid = 0.5 * (a + b)
        fm = f(mid)
        if fm > 0:
            a = mid
        else:
            b = mid
        if b - a < 1e-6:
            break
    return 0.5 * (a + b)


def _simple_thresholds(K, r, sigma, T, n_steps):
    """Fallback: crude time-decaying thresholds, always valid (never raises)."""
    dt = T / n_steps
    th = np.empty(n_steps)
    for j in range(n_steps):
        # crude: exercise when S < K * (1 - small) early, tighter later
        frac = (j + 1) / n_steps
        th[j] = K * (0.98 - 0.05 * frac)
    return th


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


def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    DEADLINE = 150.0
    SAFETY = 6.0

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
    vol_log = sigma * np.sqrt(dt)

    # Grid range: focus where the put has value / action.
    # Put payoff positive for S < K; exercise region is S < K.
    S_lo = max(1e-3, 0.3 * K)
    S_hi = K * 1.8
    n_grid = 800
    grid = np.linspace(S_lo, S_hi, n_grid)
    payoff = np.maximum(K - grid, 0.0)

    def elapsed_ok():
        return (time.time() - t_start) < (DEADLINE - SAFETY)

    thresholds = None
    try:
        V = np.empty((n_steps + 1, n_grid))
        V[n_steps, :] = payoff
        for j in range(n_steps - 1, -1, -1):
            if not elapsed_ok():
                break
            V[j, :] = np.maximum(payoff, _continuation(V[j + 1, :], grid, mu_log, vol_log, dt, r))

        # Compute thresholds from the final V
        th = np.empty(n_steps)
        for j in range(n_steps):
            if not elapsed_ok():
                break
            tau_rem = (j + 1) * dt
            th[j] = _threshold_for(V[j, :], payoff, grid, V[j + 1, :], mu_log, vol_log, dt, r, K, sigma, tau_rem)
        if elapsed_ok():
            thresholds = th
    except Exception:
        thresholds = None

    if thresholds is None or not np.all(np.isfinite(thresholds)):
        thresholds = _simple_thresholds(K, r, sigma, T, n_steps)

    # Enforce monotonic non-increasing across dates (earlier dates: higher threshold).
    for j in range(1, n_steps):
        if thresholds[j] > thresholds[j - 1]:
            thresholds[j] = thresholds[j - 1]

    # Keep thresholds within a sane range.
    thresholds = np.clip(thresholds, 0.0, K * 1.2)

    return _make_policy(thresholds, n_steps, dt, T)
# EVOLVE-BLOCK-END
