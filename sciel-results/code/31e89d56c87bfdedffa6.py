import numpy as np
import time
from scipy.special import roots_hermite


def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # ---- Parameters ----
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12  # monthly exercise dates, no time-0

    training_paths = np.asarray(training_paths, dtype=float)
    N, n_tp = training_paths.shape
    dt = T / n_exercise
    discount = np.exp(-r * dt)

    # ---- Build a dense log-space price grid ----
    # Captures put payoff asymmetry (log-space is natural for GBM)
    s_min = max(training_paths.min() * 0.85, 1e-2)
    s_max = max(training_paths.max() * 1.15, 1.5 * K)
    n_grid = 300
    log_edges = np.linspace(np.log(s_min), np.log(s_max), n_grid + 1)
    log_centers = 0.5 * (log_edges[:-1] + log_edges[1:])
    grid = np.exp(log_centers)  # (n_grid,)
    gmin, gmax = grid[0], grid[-1]

    # ---- Gauss-Hermite quadrature for E[ f(S_{t+dt}) | S_t = s ] ----
    # Risk-neutral GBM: log(S_{t+dt}/s) ~ N(mu, s2), mu = (r - 0.5*sigma^2)*dt
    mu = (r - 0.5 * sigma * sigma) * dt
    s2 = sigma * sigma * dt
    n_q = 32
    x, w = roots_hermite(n_q)
    # Transform standard normal z -> log-return u = mu + sqrt(s2)*z ; z = sqrt(2)*x
    u = mu + np.sqrt(2.0 * s2) * x
    qw = 0.5 * np.sqrt(np.pi) * w  # weights for E[f(u)] over log-return

    # ---- Backward induction on the grid ----
    # V[k] = value function on grid at exercise date k (V[n_exercise] = maturity)
    V = np.zeros((n_exercise + 1, n_grid))
    V[n_exercise] = np.maximum(K - grid, 0.0)

    # Continuation value on the grid at date k (continuing to k+1)
    def continuation_on_grid(V_next):
        # S_next = grid * exp(u)  -> shape (n_grid, n_q)
        S_next = np.outer(grid, np.exp(u))
        # Interpolate V_next onto S_next (log-space)
        idx = np.searchsorted(log_centers, np.log(S_next), side='right') - 1
        idx = np.clip(idx, 0, n_grid - 2)
        c0 = V_next[idx]
        c1 = V_next[idx + 1]
        lc0 = log_centers[idx]
        lc1 = log_centers[idx + 1]
        t = (np.log(S_next) - lc0) / (lc1 - lc0)
        t = np.clip(t, 0.0, 1.0)
        interp = c0 + t * (c1 - c0)
        # Weighted sum over quadrature
        return np.sum(interp * qw, axis=1)

    # ---- Evaluate the (smooth) continuation value at arbitrary spot points ----
    def continuation_at(S):
        S = np.asarray(S, dtype=float)
        S_next = np.outer(S, np.exp(u))  # (M, n_q)
        idx = np.searchsorted(log_centers, np.log(S_next), side='right') - 1
        idx = np.clip(idx, 0, n_grid - 2)
        c0 = V[k + 1][idx]
        c1 = V[k + 1][idx + 1]
        lc0 = log_centers[idx]
        lc1 = log_centers[idx + 1]
        t = (np.log(S_next) - lc0) / (lc1 - lc0)
        t = np.clip(t, 0.0, 1.0)
        interp = c0 + t * (c1 - c0)
        return np.sum(interp * qw, axis=1)

    # ---- Thresholds (fallback policy) ----
    thresholds = np.zeros(n_exercise)

    # ---- Optional: refine continuation with a monotone fit to training data ----
    # The quadrature value is the primary, dense, smooth estimate. We optionally
    # blend in a monotone regression of the realized continuation y = disc*V[k+1]
    # on S_k (training paths) to reduce MC noise, keeping it monotone so the
    # threshold stays well-defined.
    use_refine = N >= 2000

    for k in range(n_exercise - 1, 0, -1):
        if time.time() > deadline - 20:
            # Not enough time: fall back to thresholds (already computed for
            # earlier dates); fill remaining with a conservative heuristic.
            for j in range(k, 0, -1):
                thresholds[j - 1] = K * 0.8
            # Mark that we should use threshold policy
            use_grid_policy = False
            break

        # Smooth continuation value on the grid (primary)
        cont_grid = discount * continuation_on_grid(V[k + 1])

        # Optional refinement: monotone fit of realized continuation on training paths
        S_k = training_paths[:, k]
        y_real = discount * V[k + 1]  # NOTE: V[k+1] is on grid, not path-aligned.
        # Instead, compute realized continuation for the training paths directly:
        y_real = continuation_at(S_k)
        y_real = discount * y_real

        if use_refine:
            # Monotone (isotonic) regression of y_real on S_k for smoothing
            order = np.argsort(S_k)
            Ss = S_k[order]
            ys = y_real[order]
            # Light smoothing: bin means then monotone envelope
            nb = min(150, max(20, N // 40))
            q_edges = np.quantile(Ss, np.linspace(0, 1, nb + 1))
            q_edges = np.unique(q_edges)
            if len(q_edges) >= 3:
                bi = np.clip(np.searchsorted(q_edges, Ss, side='right') - 1, 0, len(q_edges) - 2)
                sums = np.bincount(bi, weights=ys, minlength=len(q_edges) - 1)
                cnts = np.bincount(bi, minlength=len(q_edges) - 1)
                cnts = np.maximum(cnts, 1)
                means = sums / cnts
                centers = 0.5 * (q_edges[:-1] + q_edges[1:])
                # Monotone (non-increasing in S for a put continuation) envelope
                mono = means.copy()
                for i in range(1, len(mono)):
                    if mono[i] > mono[i - 1]:
                        mono[i] = mono[i - 1]
                # Blend quadrature (dense) with monotone fit (noise-reduced)
                cont_fit = np.interp(grid, centers, mono,
                                     left=mono[0], right=mono[-1])
                cont_grid = 0.5 * cont_grid + 0.5 * np.maximum(cont_fit, 0)
            else:
                pass

        cont_grid = np.maximum(cont_grid, 0)
        V[k] = np.maximum(np.maximum(K - grid, 0), cont_grid)

        # Threshold on the grid: largest S where exercise >= continuation
        ex_grid = np.maximum(K - grid, 0)
        should = ex_grid >= V[k]  # V[k] here is max(ex, cont) == cont where cont>ex
        # We want largest S with ex_grid >= cont_grid
        should = ex_grid >= cont_grid
        if np.any(should):
            thresholds[k - 1] = grid[np.where(should)[0][-1]]
        else:
            thresholds[k - 1] = 0.0

    thresholds[n_exercise - 1] = K  # maturity: exercise if in the money

    # If we ran out of time before finishing, the grid V for uncomputed dates is
    # zero; in that case rely on the threshold policy.
    use_grid_policy = (time.time() <= deadline - 20) and (V[1].sum() > 0)

    class BermudanPutPolicy:
        def __init__(self, thresholds, n_exercise, K, grid, log_centers,
                     V, use_grid_policy, n_grid):
            self.thresholds = thresholds
            self.n_exercise = n_exercise
            self.K = K
            self.grid = grid
            self.log_centers = log_centers
            self.V = V
            self.use_grid_policy = use_grid_policy
            self.n_grid = n_grid

        def _interp(self, S, row):
            lc = self.log_centers
            idx = np.searchsorted(lc, np.log(S), side='right') - 1
            idx = np.clip(idx, 0, self.n_grid - 2)
            c0 = row[idx]
            c1 = row[idx + 1]
            t = (np.log(S) - lc[idx]) / (lc[idx + 1] - lc[idx])
            t = np.clip(t, 0.0, 1.0)
            return c0 + t * (c1 - c0)

        def exercise(self, current_state, time_step):
            current_state = np.asarray(current_state, dtype=float)
            try:
                if isinstance(time_step, (int, np.integer)):
                    idx = int(time_step)
                else:
                    idx = int(np.round(time_step * 12))

                if idx < 1 or idx > self.n_exercise:
                    return np.zeros(len(current_state), dtype=bool)

                # Maturity: exercise if in the money
                if idx == self.n_exercise:
                    return current_state < self.K

                if self.use_grid_policy:
                    # Smooth, dense decision: exercise if payoff >= value on grid
                    val = self._interp(current_state, self.V[idx])
                    payoff = np.maximum(self.K - current_state, 0.0)
                    return payoff >= val
                else:
                    thresh = self.thresholds[idx - 1]
                    return current_state <= thresh
            except Exception:
                return np.zeros(len(current_state), dtype=bool)

    return BermudanPutPolicy(thresholds, n_exercise, K, grid, log_centers,
                             V, use_grid_policy, n_grid)
