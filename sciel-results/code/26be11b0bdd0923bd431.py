# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.stats import norm
from scipy.optimize import minimize

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using a parametric smooth boundary
    (sigmoid of polynomial in remaining time fraction) initialized from
    European put thresholds and refined via L-BFGS-B on the Monte Carlo
    objective, with local multi-start refinement.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit

    # Extract parameters from contract
    K = contract.get('K', contract.get('strike', 100.0))
    r = contract.get('r', contract.get('rate', 0.05))
    sigma = contract.get('sigma', contract.get('vol', 0.2))
    T = contract.get('T', contract.get('maturity', 1.0))
    n_dates = contract.get('n_dates', contract.get('n_exercise', 12))

    if n_dates is None:
        n_dates = 12

    paths = np.asarray(training_paths, dtype=np.float64)
    n_paths = paths.shape[0]

    # Extract spot prices at each exercise date
    if paths.shape[1] >= n_dates + 1:
        exercise_spots = paths[:, 1:n_dates+1]
    else:
        exercise_spots = paths[:, 1:]
        n_dates = exercise_spots.shape[1]

    dt = T / n_dates
    disc = math.exp(-r * dt)

    def european_put_price(S, K_val, r_val, sigma_val, tau):
        """Compute European put price."""
        if tau <= 0:
            return np.maximum(K_val - S, 0)
        S = np.maximum(S, 1e-12)
        sqrt_tau = np.sqrt(tau)
        d1 = (np.log(S / K_val) + (r_val + 0.5 * sigma_val**2) * tau) / (sigma_val * sqrt_tau)
        d2 = d1 - sigma_val * sqrt_tau
        N = norm.cdf
        return K_val * np.exp(-r_val * tau) * N(-d2) - S * N(-d1)

    def compute_european_thresholds():
        thresholds = np.zeros(n_dates)
        for t in range(n_dates):
            tau_t = (n_dates - t) * dt
            if tau_t <= 0:
                thresholds[t] = K
            else:
                S_grid = np.linspace(0.1, K * 1.5, 1000)
                cont_value = european_put_price(S_grid, K, r, sigma, tau_t)
                exercise_payoff = np.maximum(K - S_grid, 0)
                diff = exercise_payoff - cont_value

                positive_mask = diff > 0
                if not np.any(positive_mask):
                    thresholds[t] = 0.1
                elif np.all(positive_mask):
                    thresholds[t] = K * 1.5
                else:
                    idx = np.where(positive_mask[:-1] & ~positive_mask[1:])[0]
                    if len(idx) > 0:
                        i = idx[0]
                        S1, S2 = S_grid[i], S_grid[i+1]
                        d1_val, d2_val = diff[i], diff[i+1]
                        if d1_val != d2_val:
                            thresholds[t] = S1 - d1_val * (S2 - S1) / (d2_val - d1_val)
                        else:
                            thresholds[t] = S1
                    else:
                        thresholds[t] = K
        return thresholds

    initial_thresholds = compute_european_thresholds()

    # Parametric smooth boundary: sigmoid of polynomial in u (remaining time fraction)
    p = 4  # polynomial degree
    n_params = p + 1

    u_values = np.array([(n_dates - t) / n_dates for t in range(n_dates)])

    # Fit: log(threshold/K) - log(1 - threshold/K) = sum_{k=0}^{p} a_k * u^k
    thresholds_clipped = np.clip(initial_thresholds, 1e-6, K - 1e-6)
    y_values = np.log(thresholds_clipped / (K - thresholds_clipped))

    # Design matrix for polynomial fit
    X = np.zeros((n_dates, n_params))
    for k in range(n_params):
        X[:, k] = u_values ** k

    try:
        a_init, residuals, rank, sv = np.linalg.lstsq(X, y_values, rcond=None)
    except:
        a_init = np.zeros(n_params)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def thresholds_from_params(a):
        """Compute thresholds from polynomial parameters."""
        u_t = u_values
        poly_val = np.zeros(n_dates)
        for k in range(n_params):
            poly_val += a[k] * u_t ** k
        return K * sigmoid(poly_val)

    def compute_payoff_under_policy(paths_exercise, thresholds, K_val, r_val, n_d, T_val):
        """
        Simulate the Bermudan put payoff under a given threshold policy.
        """
        n_p = paths_exercise.shape[0]
        payoffs = np.zeros(n_p)
        exercised = np.zeros(n_p, dtype=bool)

        dt_val = T_val / n_d
        disc_val = math.exp(-r_val * dt_val)

        for t in range(n_d):
            mask = ~exercised
            if not np.any(mask):
                break
            s_t = paths_exercise[:, t]
            should_exercise = s_t < thresholds[t]
            new_exercise = mask & should_exercise

            if np.any(new_exercise):
                disc_factor = disc_val ** t
                payoffs[new_exercise] = (K_val - s_t[new_exercise]) * disc_factor
                exercised[new_exercise] = True

        return payoffs

    def negative_mean_payoff(a):
        thresholds = thresholds_from_params(a)
        payoffs = compute_payoff_under_policy(exercise_spots, thresholds, K, r, n_dates, T)
        return -np.mean(payoffs)

    # Primary optimization using L-BFGS-B
    try:
        result = minimize(
            negative_mean_payoff,
            a_init,
            method='L-BFGS-B',
            options={
                'maxiter': 500,
                'ftol': 1e-14,
                'gtol': 1e-12,
                'maxfun': 5000
            }
        )
        a_opt = result.x
    except Exception:
        a_opt = a_init

    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 15.0:
        thresholds = thresholds_from_params(a_opt)
    else:
        # Local refinement with multi-start
        best_a = a_opt.copy()
        best_val = -negative_mean_payoff(a_opt)

        remaining_time = DEADLINE - (time.time() - start_time) - 10.0
        if remaining_time > 5.0:
            # Local perturbations to escape potential local minima
            n_restarts = min(12, int(remaining_time / 2.0))
            rng = np.random.RandomState(42)
            for restart in range(n_restarts):
                a_try = a_opt + rng.normal(0, 0.05, size=n_params)
                try:
                    val = -negative_mean_payoff(a_try)
                    if val > best_val:
                        best_val = val
                        best_a = a_try
                except Exception:
                    pass

                if time.time() - start_time > DEADLINE - 10.0:
                    break

            # If improvement found, do a local optimization from the best point
            if best_val > -negative_mean_payoff(a_opt):
                try:
                    result2 = minimize(
                        negative_mean_payoff,
                        best_a,
                        method='L-BFGS-B',
                        options={
                            'maxiter': 200,
                            'ftol': 1e-14,
                            'gtol': 1e-12,
                            'maxfun': 2000
                        }
                    )
                    if -result2.fun > best_val:
                        best_a = result2.x
                        best_val = -result2.fun
                except Exception:
                    pass

        thresholds = thresholds_from_params(best_a)

    # Ensure thresholds are within valid range
    thresholds = np.clip(thresholds, 0.0, K * 2.0)

    class ExercisePolicy:
        def __init__(self, thresholds, n_steps):
            self.thresholds = thresholds
            self.n_steps = n_steps

        def exercise(self, current_state, time):
            """
            Determine whether to exercise at the given time.
            time is 1-indexed, so time=1 is the first exercise date.
            """
            t_idx = time - 1
            if t_idx < 0 or t_idx >= self.n_steps:
                return np.zeros(len(current_state), dtype=bool)

            threshold = self.thresholds[t_idx]
            return current_state < threshold

    policy = ExercisePolicy(thresholds, n_dates)
    return policy
# EVOLVE-BLOCK-END
