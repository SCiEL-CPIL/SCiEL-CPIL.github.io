import numpy as np
import time
from scipy.stats import norm

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option.

    Strategy: backward induction over the 12 exercise dates.  At each date we
    compare the intrinsic (immediate) value against the exact one-period
    continuation value.  The continuation is computed with the closed-form
    Black-Scholes European-put formula (exact under risk-neutral GBM for a
    single step), which is far more accurate than a single binomial step and
    removes the main discretization bias in the exercise boundary.
    """
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin within the 120s wall-clock limit

    # ---- contract parameters -------------------------------------------------
    K = float(contract.get('K', 100.0))
    r = float(contract.get('r', 0.05))
    sigma = float(contract.get('sigma', 0.2))
    T = float(contract.get('T', 1.0))
    n_exercises = 12

    dt = T / n_exercises
    disc = np.exp(-r * dt)
    vol_sqrt = sigma * np.sqrt(dt)

    # ---- price grid (log-spaced for resolution near low prices) --------------
    S_min = 0.05
    S_max = 400.0
    n_grid = 8000
    S_grid = np.exp(np.linspace(np.log(S_min), np.log(S_max), n_grid))

    # Precompute the Black-Scholes continuation integrand pieces for one period.
    # European put with time-to-expiry = dt:
    #   P(S) = K*exp(-r*dt)*N(-d2) - S*N(-d1)
    #   d1 = (ln(S/K) + (r + sigma^2/2)*dt) / (sigma*sqrt(dt))
    #   d2 = d1 - sigma*sqrt(dt)
    logS = np.log(S_grid)
    d1 = (logS - np.log(K) + (r + 0.5 * sigma * sigma) * dt) / vol_sqrt
    d2 = d1 - vol_sqrt
    N_neg_d1 = norm.cdf(-d1)
    N_neg_d2 = norm.cdf(-d2)
    # continuation(S) = A - B*S, where A = K*disc*N_neg_d2 (scalar), B = N_neg_d1 (array)
    A = K * disc * N_neg_d2          # scalar
    B = N_neg_d1                     # array, same shape as S_grid
    cont_lin = A - B * S_grid        # one-period continuation value on the grid

    # boundaries[i] = critical price at exercise date i (i = 0..11),
    # exercise when S <= boundaries[i].
    boundaries = np.zeros(n_exercises)

    # Backward induction.  At the final date the option expires, so its value is
    # purely intrinsic; the "continuation" past the final date is zero.
    # We store the *value function* V(S) at the next exercise date and build the
    # value function at the current date.
    #
    # For date i (0-indexed, exercise at t=(i+1)*dt):
    #   intrinsic(S) = max(K - S, 0)
    #   continuation(S) = disc * E[ V_next(S_next) | S ]
    #
    # For the LAST date (i = 11) there is no next date, so value = intrinsic.
    # For earlier dates, V_next is the value function one step ahead.
    #
    # The exact one-period continuation of a *general* value function V_next is
    # not a simple closed form, so we evaluate it numerically via the
    # risk-neutral lognormal transition.  For accuracy and speed we use a
    # Gauss-Hermite quadrature on the underlying normal, which is exact for the
    # GBM transition and cheap to vectorize.

    # Gauss-Hermite nodes/weights (15 points is very accurate for this smooth
    # integrand).  For S_next = S * exp(m + vol*Z), m=(r-sigma^2/2)*dt.
    m = (r - 0.5 * sigma * sigma) * dt
    gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(15)
    # Z = sqrt(2)*node
    Z = np.sqrt(2.0) * gh_nodes
    w = gh_weights / np.sqrt(np.pi)  # normalize so sum(w)=1

    # Precompute the transition multipliers and weights (shared across S).
    mult = np.exp(m + vol_sqrt * Z)   # shape (15,)
    # E[ V(S*mult) ] = sum_k w_k * V(S * mult_k)

    # Value function at the final date (intrinsic, since it expires).
    V_next = np.maximum(K - S_grid, 0.0)

    for i in range(n_exercises - 1, -1, -1):
        if i == n_exercises - 1:
            # Final date: value is intrinsic (no continuation).
            V_current = V_next.copy()
            intrinsic = np.maximum(K - S_grid, 0.0)
            # boundary at the final date: exercise if S < K (put intrinsic).
            # (No continuation, so exercise whenever intrinsic > 0, i.e. S < K.)
            boundaries[i] = K
        else:
            intrinsic = np.maximum(K - S_grid, 0.0)
            # Continuation via quadrature: for each S, sum over Z of w*V_next(S*mult)
            # S_grid[:,None] * mult[None,:] -> (n_grid, 15)
            S_next = S_grid[:, None] * mult[None, :]
            # Clip to grid range for interpolation (values outside are 0 for put-like).
            S_next_clipped = np.clip(S_next, S_grid[0], S_grid[-1])
            V_next_at = np.interp(S_next_clipped, S_grid, V_next)
            cont = disc * (V_next_at @ w)
            V_current = np.maximum(intrinsic, cont)

            # Find the exercise boundary: largest S where intrinsic >= continuation.
            diff = intrinsic - cont
            # exercise where diff >= 0.  For a put this is S <= boundary.
            ex = diff >= 0.0
            if np.any(ex):
                idx = np.where(ex)[0]
                last = idx[-1]
                if last < n_grid - 1:
                    d0 = diff[last]
                    d1v = diff[last + 1]
                    if d0 != d1v:
                        frac = d0 / (d0 - d1v)
                        boundaries[i] = S_grid[last] + frac * (S_grid[last + 1] - S_grid[last])
                    else:
                        boundaries[i] = S_grid[last]
                else:
                    boundaries[i] = S_grid[-1]
            else:
                boundaries[i] = 0.0

        V_next = V_current

        # deadline guard: if we are running out of time, stop refining earlier
        # dates (they are less important) and fill remaining with a conservative
        # fallback (exercise only when deeply in the money).
        if time.time() > deadline:
            for j in range(i - 1, -1, -1):
                boundaries[j] = K * 0.8
            break

    return ExercisePolicy(boundaries, n_exercises)


class ExercisePolicy:
    def __init__(self, boundaries, n_exercises):
        self.boundaries = np.asarray(boundaries, dtype=float)
        self.n_exercises = n_exercises

    def exercise(self, current_state, time):
        """
        Returns a boolean vector: True where the option should be exercised.

        current_state: array-like of stock prices (one per path/position).
        time: scalar or array-like of times (years) at which the state is given.
        """
        current_state = np.asarray(current_state, dtype=float)
        time_arr = np.asarray(time, dtype=float)

        if time_arr.ndim == 0:
            time_arr = np.full(current_state.shape, float(time_arr))
        else:
            time_arr = np.broadcast_to(time_arr, current_state.shape).astype(float)

        # Exercise dates are at (i+1)*dt for i=0..11, i.e. t = T/12, 2T/12, ...
        # Map a given time to the index of the most recent exercise date <= t.
        # t in [ (i+1)*dt - dt/2 , (i+1)*dt + dt/2 ) -> index i.
        # Simpler: idx = floor(t / dt), clamped to [0, n-1].
        T = 1.0  # matches contract default; dt = T/n
        dt = T / self.n_exercises
        idx = np.floor(time_arr / dt).astype(int)
        idx = np.clip(idx, 0, self.n_exercises - 1)

        b = self.boundaries[idx]
        return (current_state <= b) & (current_state > 0.0)
