"""Bermudan put exercise-policy: parametric boundary via stochastic gradient ascent."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

def fit_exercise_policy(training_paths, contract):
    t_start = _time.time()
    DEADLINE = 145.0  # safety margin under 150s wall-clock

    # Extract parameters from contract
    K = contract['K']
    r = contract['r']
    sigma = contract['sigma']
    T = contract['T']
    n_dates = contract['n_dates']  # 12 monthly exercise dates
    n_paths = training_paths.shape[0]
    n_time = training_paths.shape[1]  # should be 13 (time 0 to T)

    # Exercise dates: t_1, ..., t_12 (no time-0 exercise)
    dt = T / n_dates
    exercise_times = np.arange(1, n_dates + 1) * dt
    # Discount factors for each exercise date
    disc = np.exp(-r * exercise_times)

    # Log-moneyness: x = ln(S/K)
    log_S = np.log(training_paths)
    x = log_S - np.log(K)  # shape (n_paths, n_time)

    # Time parameter: tau = t/T for each exercise date
    # For a path at exercise date j (index j in 1..12), the time is t_j
    # We use tau_j = t_j / T
    tau = exercise_times / T  # shape (n_dates,)

    # Parametric boundary: exercise if x <= a[j] + b[j] * ln(tau_j)
    # But ln(tau) is constant per date, so this simplifies to a threshold per date.
    # Actually, the directive says: ln(S/K) = a(t) + b(t)*ln(t/T)
    # This means the boundary in x-space is: x_boundary(t) = a(t) + b(t)*ln(t/T)
    # Exercise if x <= x_boundary(t) for a put (lower spot -> exercise)
    #
    # Since tau is fixed per date, ln(tau) is a constant per date.
    # So for each date j, the boundary is simply: a_j + b_j * ln(tau_j)
    # We can just define threshold_j = a_j + b_j * ln(tau_j) and optimize that directly.
    # But to follow the parametric form, let's keep a and b and compute the boundary.

    # Initialize: use a simple heuristic. For a Bermudan put, exercise when
    # intrinsic value is high relative to continuation.
    # Start with thresholds based on European put comparison.
    # Simple init: exercise when S < K * exp(-r*t) roughly (intrinsic > discount)
    # Actually, let's just start with a reasonable threshold.
    # x = ln(S/K), so S = K*exp(x). Exercise if S < K*exp(a_j + b_j*ln(tau_j))
    # Let's initialize a_j such that threshold is around ln(0.9) = -0.105 for early dates
    # and closer to 0 for later dates.

    # Initialize a and b
    a = np.zeros(n_dates)
    b = np.zeros(n_dates)

    # Better initialization: use the idea that for a put, we exercise when
    # the stock is low. A reasonable starting point:
    # threshold_j = -0.1 * (1 - j/n_dates)  (more negative early, less negative late)
    # Since boundary = a_j + b_j * ln(tau_j), and ln(tau_j) < 0 for all j < T,
    # let's set b_j = 0 initially and a_j = threshold_j.
    for j in range(n_dates):
        a[j] = -0.15 * (1.0 - (j + 1) / n_dates) + 0.02 * (j / n_dates)
        b[j] = 0.0

    def _boundary(thresholds):
        """Compute the exercise boundary x-values for each date."""
        # thresholds[j] = a[j] + b[j] * ln(tau[j])
        return thresholds

    def _evaluate_policy(thresholds, paths_x, max_time=None):
        """
        Evaluate the discounted payoff for a given set of thresholds.
        thresholds: array of shape (n_dates,) with x-boundaries for each exercise date.
        paths_x: log-moneyness array of shape (n_paths, n_time).
        Returns: mean discounted payoff.
        """
        total_value = np.zeros(n_paths)
        exercised = np.zeros(n_paths, dtype=bool)

        # Process dates from last to first (backward induction on paths)
        for j in range(n_dates - 1, -1, -1):
            if max_time is not None and (_time.time() - t_start) > max_time:
                break
            # Paths not yet exercised at date j
            mask = ~exercised
            if not np.any(mask):
                break
            # Check if exercise condition is met: x[j] <= thresholds[j]
            x_j = paths_x[mask, j + 1]  # index j+1 because time 0 is index 0
            ex_mask = x_j <= thresholds[j]
            if np.any(ex_mask):
                # Get indices of paths that exercise at date j
                idx = np.where(mask)[0]
                ex_idx = idx[ex_mask]
                # Payoff at exercise: K - S = K * (1 - exp(x))
                payoff = K * (1.0 - np.exp(paths_x[ex_idx, j + 1]))
                total_value[ex_idx] = disc[j] * payoff
                exercised[ex_idx] = True

        # Paths that never exercise get payoff at maturity (last date)
        # Actually, the last exercise date is at T, so if not exercised by then,
        # they get the terminal payoff which is the same as exercising at the last date.
        # Wait, the last exercise date IS at T. So if not exercised at date n_dates,
        # they must be exercised at T (or get zero if OTM).
        # Let me re-examine: the 12 exercise dates are at t_1,...,t_12 where t_12 = T.
        # So if a path hasn't exercised by date 12, it gets the terminal payoff at T.
        # But in my loop, j goes from n_dates-1 down to 0, corresponding to dates 1..12.
        # j = n_dates-1 is the last date (T). So paths not exercised after j=0 (first date)
        # but before j=n_dates-1... wait, I'm going backward.
        # Let me redo: j from n_dates-1 down to 0.
        # j = n_dates-1 corresponds to the last exercise date (index n_dates in paths, which is T).
        # j = 0 corresponds to the first exercise date (index 1 in paths).
        # So after the loop, any path not exercised has not met the condition at any date.
        # But at the last date (T), the payoff is just intrinsic value.
        # So paths not exercised should get the terminal payoff if ITM.

        # Actually, I think the issue is that the last date should always be "exercised" 
        # in the sense that you get the terminal payoff. Let me handle this:
        # After the loop, remaining paths get terminal payoff.
        remaining = ~exercised
        if np.any(remaining):
            # Terminal payoff at T
            x_T = paths_x[remaining, n_time - 1]
            payoff = K * np.maximum(1.0 - np.exp(x_T), 0.0)
            total_value[remaining] = disc[-1] * payoff

        return np.mean(total_value)

    def _gradient_step(thresholds, paths_x, lr, max_time=None):
        """
        Perform one stochastic gradient ascent step on thresholds.
        Uses finite differences.
        """
        n = len(thresholds)
        eps = 1e-4
        base_val = _evaluate_policy(thresholds, paths_x, max_time)
        grad = np.zeros(n)

        for j in range(n):
            if max_time is not None and (_time.time() - t_start) > max_time:
                break
            thresholds_plus = thresholds.copy()
            thresholds_plus[j] += eps
            val_plus = _evaluate_policy(thresholds_plus, paths_x, max_time)
            grad[j] = (val_plus - base_val) / eps

        # Gradient ascent
        new_thresholds = thresholds + lr * grad
        return new_thresholds, base_val

    # Optimize thresholds using stochastic gradient ascent
    # Use a subset of paths for speed
    n_sub = min(n_paths, 2048)
    sub_idx = np.random.choice(n_paths, n_sub, replace=False)
    sub_x = x[:, :n_time]  # Use all paths for now, or subset

    # Actually, let's use a subset for speed
    if n_paths > 2048:
        sub_x = x[sub_idx, :]
    else:
        sub_x = x

    thresholds = a + b * np.log(tau)
    best_val = _evaluate_policy(thresholds, sub_x)
    best_thresholds = thresholds.copy()

    # Adaptive learning rate
    lr = 0.05
    patience = 10
    no_improve = 0

    max_iter = 200
    for it in range(max_iter):
        if (_time.time() - t_start) > DEADLINE - 5.0:
            break

        thresholds, val = _gradient_step(thresholds, sub_x, lr, DEADLINE - 10.0)

        if val > best_val:
            best_val = val
            best_thresholds = thresholds.copy()
            no_improve = 0
        else:
            no_improve += 1
            # Reduce learning rate
            lr *= 0.9

        if no_improve >= patience:
            break

    # Final evaluation with full training set
    final_val = _evaluate_policy(best_thresholds, x)

    # Create policy object
    class ExercisePolicy:
        def __init__(self, thresholds, K, r, sigma, T, n_dates):
            self.thresholds = thresholds
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_dates = n_dates

        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: array of times (should match exercise dates)
            Returns: boolean vector indicating whether to exercise.
            """
            S = np.asarray(current_state)
            t = np.asarray(time)
            x = np.log(S) - np.log(self.K)

            # Determine which exercise date each time corresponds to
            dt = self.T / self.n_dates
            # Map time to date index
            date_idx = np.round(t / dt).astype(int) - 1  # t=dt -> 0, t=2dt -> 1, etc.
            date_idx = np.clip(date_idx, 0, self.n_dates - 1)

            # Get thresholds for each path's date
            thresh = self.thresholds[date_idx]

            # Exercise if x <= threshold (for a put)
            return x <= thresh

    return ExercisePolicy(best_thresholds, K, r, sigma, T, n_dates)
# EVOLVE-BLOCK-END
