import numpy as np
import time

class ExercisePolicy:
    def __init__(self, critical_prices):
        self.critical_prices = np.asarray(critical_prices, dtype=np.float64)

    def exercise(self, current_state, time):
        current_state = np.asarray(current_state, dtype=np.float64)
        k_float = time * 12.0
        k_int = int(round(k_float))
        if k_int < 1 or k_int > 12:
            return np.zeros(current_state.shape, dtype=bool)
        crit = self.critical_prices[k_int - 1]
        return current_state <= crit


def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = t_start + 140.0

    K = float(contract.get('K', 100.0))
    r = float(contract.get('r', 0.05))
    sigma = float(contract.get('sigma', 0.2))
    T = float(contract.get('T', 1.0))

    # ---------- 1. High-resolution binomial tree ----------
    N = 12000
    dt = T / N
    sqrt_dt = np.sqrt(dt)
    u = np.exp(sigma * sqrt_dt)
    d = 1.0 / u
    log_d = np.log(d)
    log_u_minus_d = np.log(u) - log_d  # = sigma*sqrt(dt)
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    S0 = 100.0

    # Terminal node prices (log-space)
    i_vals = np.arange(N + 1)
    S_N = S0 * np.exp(N * log_d + i_vals * log_u_minus_d)
    V = np.maximum(K - S_N, 0.0)

    crit = np.zeros(12, dtype=np.float64)
    S_curr = S_N.copy()
    steps_per_month = N // 12  # 1000

    for j in range(N, 0, -1):
        V_cont = disc * (p * V[:j] + (1.0 - p) * V[1:j + 1])
        S_prev = S_curr[:j] * d  # S_prev[i] = S_curr[i] / u = S_curr[i] * d
        V_ex = np.maximum(K - S_prev, 0.0)
        V_new = np.maximum(V_cont, V_ex)

        step_num = j - 1
        if step_num % steps_per_month == 0 and step_num > 0:
            k_idx = step_num // steps_per_month  # 0..11
            mask = V_ex >= V_cont
            if np.any(mask):
                crit[k_idx] = S_prev[np.where(mask)[0][-1]]
            else:
                crit[k_idx] = 0.0

        V = V_new
        S_curr = S_prev

        if time.time() > deadline - 110.0:
            break

    crit[11] = K
    crit = np.clip(crit, 0.0, K)
    # Enforce monotonicity
    for k in range(1, 12):
        if crit[k] < crit[k - 1]:
            crit[k] = crit[k - 1]
    crit[11] = K

    # ---------- 2. Prepare training data ----------
    paths = np.asarray(training_paths, dtype=np.float64)
    if paths.ndim == 1:
        paths = paths.reshape(-1, 12)
    M = paths.shape[0]
    t_steps = np.arange(1, 13) / 12.0
    disc_factors = np.exp(-r * t_steps)

    def compute_value(c_prices):
        W = paths <= c_prices[np.newaxis, :]
        exercised = np.zeros(M, dtype=bool)
        ex_step = np.full(M, -1, dtype=np.int32)
        for k in range(12):
            cond = W[:, k] & ~exercised
            if np.any(cond):
                exercised[cond] = True
                ex_step[cond] = k
        if not np.any(exercised):
            return 0.0
        ex_idx = np.where(exercised)[0]
        S_ex = paths[ex_idx, ex_step[ex_idx]]
        return np.mean(disc_factors[ex_step[ex_idx]] * (K - S_ex))

    # ---------- 3. Continuation-value regression for better initial boundaries ----------
    if time.time() < deadline - 100.0:
        # Walk backwards through exercise dates, fitting continuation value
        # At step 12: continuation = 0 (option expires)
        # At step k: continuation = disc * E[V_{k+1} | S_k] for non-exercised paths
        # We approximate by regressing the "next step value" on current spot

        # V_next[k] = value at step k+1 given optimal exercise from k+1..12
        # Initialize V_next[11] = payoff at step 12 (last exercise date)
        V_next = np.zeros((12, M))
        # At the last exercise date (index 11), if not exercised, option expires -> 0
        # But we exercise at 12 if ITM, so continuation after 12 is 0
        # V_next[11] represents the value at step 12 (which is the payoff if we get there)
        # Actually, let's think of it as:
        # cont_value[k, m] = continuation value at exercise date k for path m
        # = disc * E[ max(payoff at k+1, cont at k+1) | alive at k ]
        # We approximate E by regressing on S_k

        # Start: at step 12 (index 11), the "continuation" after exercise is 0
        # The value at step 12 for path m = max(K - S_12, 0) if we're alive
        # For regression at step k, we need the realized value at step k+1 for alive paths

        realized = np.zeros((12, M))
        alive = np.ones(M, dtype=bool)

        for k in range(11, -1, -1):
            # For alive paths at step k, compute what happens at step k+1 (if k<11)
            if k < 11:
                # Value at step k+1 for alive paths
                # = max(payoff at k+1, continuation at k+1)
                # We approximate continuation at k+1 by the regression from step k+1
                payoff_next = np.maximum(K - paths[:, k + 1], 0.0)
                # Use the fitted continuation from step k+1 (already computed since we go backwards)
                cont_next = realized[k + 1].copy()
                # Actually realized[k+1] should be the total value at step k+1
                # Let's define: realized[k] = total option value at exercise date k (if alive)
                # For k=11: realized[11] = max(K - S_11, 0) [since no continuation after]
                # For k<11: realized[k] = max(K - S_k, disc * E[realized[k+1] | S_k])

                # Fit regression: realized[k+1] ~ f(S_{k+1}) for alive paths
                # Then continuation at k = disc * f(S_k) ... no, that's not right
                # We need E[realized[k+1] | S_k], not E[realized[k+1] | S_{k+1}]
                # Simpler: just use the realized values directly with a simple model

                # For now, use a simple approximation:
                # continuation at k = disc * mean(realized[k+1] for alive paths)
                # This is crude but gives a reasonable starting point
                if np.any(alive):
                    mean_next = np.mean(realized[k + 1][alive])
                    cont_val = disc_factors[k] * mean_next  # discount from k to 0
                    # Actually discount_factors[k] = exp(-r * t_k), which discounts from t_k to 0
                    # We need discount from t_k to t_{k+1}: exp(-r * dt_step) where dt_step = 1/12
                    step_disc = np.exp(-r / 12.0)
                    cont_val = step_disc * mean_next
                else:
                    cont_val = 0.0

                payoff_k = np.maximum(K - paths[:, k], 0.0)
                realized[k] = np.where(payoff_k >= cont_val, payoff_k, cont_val)
                # Update alive: paths that exercised at k are no longer alive
                exercised_now = alive & (payoff_k >= cont_val)
                alive[exercised_now] = False
            else:
                # k = 11 (last exercise date)
                realized[11] = np.maximum(K - paths[:, 11], 0.0)
                exercised_now = alive & (realized[11] > 0)
                alive[exercised_now] = False

        # Use realized values to estimate critical prices
        # For each k, find the S_k where payoff_k = continuation_k
        # This is an approximation; the tree values are better
        # Skip this refinement if tree values look good

    # ---------- 4. Coordinate-wise optimization ----------
    # Use golden-section search for each coordinate
    golden_ratio = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618
    inv_golden = 1.0 - golden_ratio  # ~0.382

    def golden_section_minimize(f, a, b, tol=1e-5, max_iter=60):
        """Minimize f on [a,b] using golden-section search."""
        c = b - golden_ratio * (b - a)
        d_pt = a + golden_ratio * (b - a)
        fc = f(c)
        fd = f(d_pt)
        for _ in range(max_iter):
            if b - a < tol:
                break
            if fc < fd:
                b = d_pt
                d_pt = c
                fd = fc
                c = b - golden_ratio * (b - a)
                fc = f(c)
            else:
                a = c
                c = d_pt
                fc = fd
                d_pt = a + golden_ratio * (b - a)
                fd = f(d_pt)
        return 0.5 * (a + b)

    # Multiple passes of coordinate-wise optimization
    n_passes = 6
    for pass_num in range(n_passes):
        if time.time() > deadline - 20.0:
            break

        for k in range(11):  # 0..10
            if time.time() > deadline - 15.0:
                break

            lower = max(0.0, crit[k - 1] - 2.0) if k > 0 else 0.0
            upper = min(K, crit[k + 1] + 2.0) if k < 11 else K

            if lower >= upper:
                continue

            def obj_k(val):
                old = crit[k]
                crit[k] = val
                val_out = compute_value(crit)
                crit[k] = old
                return -val_out

            try:
                best = golden_section_minimize(obj_k, lower, upper, tol=1e-5, max_iter=50)
                if best > lower and best < upper:
                    crit[k] = best
            except Exception:
                pass

        # Enforce monotonicity
        for k in range(1, 12):
            if crit[k] < crit[k - 1]:
                crit[k] = crit[k - 1]
        crit[11] = K

    # ---------- 5. Fine polish with tighter bounds ----------
    if time.time() < deadline - 10.0:
        for k in range(11):
            if time.time() > deadline - 8.0:
                break

            lower = max(0.0, crit[k - 1]) if k > 0 else 0.0
            upper = min(K, crit[k + 1]) if k < 11 else K

            if lower >= upper:
                continue

            # Expand slightly for exploration
            margin = 0.5
            lower = max(0.0, lower - margin)
            upper = min(K, upper + margin)

            if lower >= upper:
                continue

            def obj_k2(val):
                old = crit[k]
                crit[k] = val
                val_out = compute_value(crit)
                crit[k] = old
                return -val_out

            try:
                best = golden_section_minimize(obj_k2, lower, upper, tol=1e-6, max_iter=60)
                if best > lower and best < upper:
                    crit[k] = best
            except Exception:
                pass

        # Final monotonicity
        for k in range(1, 12):
            if crit[k] < crit[k - 1]:
                crit[k] = crit[k - 1]
        crit[11] = K

    return ExercisePolicy(crit)
