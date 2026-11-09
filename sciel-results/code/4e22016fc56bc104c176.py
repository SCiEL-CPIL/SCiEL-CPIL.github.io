"""Bermudan put exercise policy via backward induction on log-space grid."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0

    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_ex = contract.get('n_exercise', 12)

    dt = T / n_ex
    # Exercise dates: t1, t2, ..., t12 (no time-0)
    # We index backward: at date i (1-indexed), remaining steps to maturity = n_ex - i

    # Build log-space grid
    # For a put, we need good resolution at low S (deep ITM) and around K
    # Log grid from very low to high
    log_min = np.log(max(0.5, S0 * 0.01))
    log_max = np.log(max(S0 * 20.0, K * 5.0))
    n_grid = 4096
    log_grid = np.linspace(log_min, log_max, n_grid)
    grid = np.exp(log_grid)

    # Terminal value (at maturity T): max(K - S, 0)
    # But the last exercise date is at T, so at T we just take max(K-S, 0)
    # Actually for Bermudan, the last exercise is at T.
    # Backward induction:
    #   V_n(S) = max(K-S, 0)  at S at time T
    #   V_{i}(S) = max(K-S, 0, exp(-r*dt) * E[V_{i+1}(S_{i+1}) | S_i = S])
    #
    # The continuation value at step i is: exp(-r*dt) * E[V_{i+1}(S_{i+1}) | S_i = S]
    # Under GBM: S_{i+1} = S_i * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    # So given S_i = S, S_{i+1} = S * exp(mu*dt + sigma*sqrt(dt)*Z) where mu = r - sigma^2/2

    mu = r - 0.5 * sigma * sigma
    vol_sqrt_dt = sigma * np.sqrt(dt)

    # Precompute the shift in log-space: log(S_{i+1}) = log(S_i) + mu*dt + vol_sqrt_dt * Z
    # For a given S_i on the grid, S_{i+1} is a shifted lognormal.
    # We need to compute E[V_{i+1}(S_{i+1}) | S_i = S] for each S on the grid.
    # V_{i+1} is piecewise linear on the log_grid.
    # E[V_{i+1}(S_{i+1})] where S_{i+1} = S * exp(mu*dt + vol_sqrt_dt * Z)
    # log(S_{i+1}) = log(S) + mu*dt + vol_sqrt_dt * Z
    # This is a lognormal distribution with mean log(S) + mu*dt and std vol_sqrt_dt.

    # For efficiency, precompute the continuation value operator.
    # Given a piecewise-linear function V on log_grid, compute:
    #   C(S) = exp(-r*dt) * E[V(S * exp(mu*dt + vol_sqrt_dt*Z))]
    # for S on the grid.

    # We'll use the fact that for a lognormal, we can compute the expectation
    # by interpolating V at the shifted grid points.

    # Actually, the cleanest approach: for each grid point S_i, the distribution of
    # log(S_{i+1}) is Normal(log(S_i) + mu*dt, vol_sqrt_dt).
    # We need E[V(S_{i+1})] where V is piecewise linear in log(S).
    #
    # Let's precompute: for each grid point j, log_grid[j], the continuation value
    # is exp(-r*dt) * integral of V(exp(x)) * N(x; log_grid[j] + mu*dt, vol_sqrt_dt^2) dx
    #
    # Since V is piecewise linear in log-space, V(exp(x)) is piecewise linear in x.
    # The integral of a piecewise linear function against a Gaussian can be computed
    # analytically per segment.

    # Let me define: V(x) for x in [log_grid[j], log_grid[j+1]] as:
    #   V(x) = V_j * (log_grid[j+1] - x)/(log_grid[j+1] - log_grid[j]) + V_{j+1} * (x - log_grid[j])/(log_grid[j+1] - log_grid[j])
    #
    # The integral over segment [a, b] of V(x) * phi(x; m, s^2) dx where phi is Gaussian pdf.
    #
    # Let's precompute the Gaussian CDF and PDF values.

    from scipy.stats import norm

    # Precompute for each pair of adjacent grid points, the contribution to the integral
    # for each source point.

    # Actually, let me use a simpler but still accurate approach:
    # Monte Carlo within the backward induction? No, that's slow.
    #
    # Better: use the fact that the expectation of a piecewise-linear function under
    # a lognormal can be computed by summing over segments.
    #
    # For segment [x_j, x_{j+1}] with V linear:
    #   V(x) = A + B*x where A = V_j - B*x_j, B = (V_{j+1} - V_j)/(x_{j+1} - x_j)
    #   integral of (A + B*x) * N(x; m, s^2) dx over [x_j, x_{j+1}]
    #   = A * (Phi((x_{j+1}-m)/s) - Phi((x_j-m)/s)) + B * [x*(Phi((x-m)/s)) - s*phi((x-m)/s)] evaluated at bounds
    #
    # This is doable. Let me implement it.

    def compute_continuation(V, log_grid, mu_dt, vol_sd, r_dt):
        """
        V: values on log_grid (piecewise linear)
        Returns: continuation values on log_grid, i.e., exp(-r*dt) * E[V(S_{next})]
        where log(S_{next}) ~ N(log(S) + mu_dt, vol_sd^2)
        """
        n = len(log_grid)
        cont = np.zeros(n)

        # Precompute segment slopes and intercepts
        # V(x) on [x_j, x_{j+1}]: V = V[j] + slope[j] * (x - x[j])
        dx = np.diff(log_grid)  # length n-1
        slope = np.diff(V) / dx  # length n-1
        # intercept for segment j: V[j] - slope[j]*log_grid[j]
        intercept = V[:-1] - slope * log_grid[:-1]  # length n-1

        # For each source point i, we need:
        #   E[V(S_next)] = sum over segments j of integral of V(x) * N(x; m_i, s^2) dx over [x_j, x_{j+1}]
        # where m_i = log_grid[i] + mu_dt, s = vol_sd

        # Precompute Phi and phi for efficiency
        # For each segment j and each source i, we need:
        #   Phi((x_{j+1} - m_i)/s) - Phi((x_j - m_i)/s)
        #   and the B term: integral of x * N(x; m_i, s^2) dx over [x_j, x_{j+1}]
        #   = m_i * (Phi((x_{j+1}-m_i)/s) - Phi((x_j-m_i)/s)) + s * (phi((x_j-m_i)/s) - phi((x_{j+1}-m_i)/s))

        # This is O(n^2) which for n=4096 is ~16M operations. Should be OK in numpy.

        # Vectorize over segments for each source, or over sources for each segment.
        # Let's do it segment by segment to keep memory manageable.

        s2 = vol_sd * vol_sd
        inv_s = 1.0 / vol_sd

        for j in range(n - 1):
            a = log_grid[j]
            b = log_grid[j + 1]
            A = intercept[j]
            B = slope[j]

            # m_i = log_grid + mu_dt (vector of length n)
            m = log_grid + mu_dt

            # z_a = (a - m) / s, z_b = (b - m) / s
            z_a = (a - m) * inv_s
            z_b = (b - m) * inv_s

            Phi_a = norm.cdf(z_a)
            Phi_b = norm.cdf(z_b)
            phi_a = norm.pdf(z_a)
            phi_b = norm.pdf(z_b)

            # Integral of (A + B*x) * N(x; m, s^2) dx over [a, b]
            # = A * (Phi_b - Phi_a) + B * [ m*(Phi_b - Phi_a) + s*(phi_a - phi_b) ]

            integral = A * (Phi_b - Phi_a) + B * (m * (Phi_b - Phi_a) + vol_sd * (phi_a - phi_b))

            cont += integral

        cont *= np.exp(-r * dt)
        return cont

    # Initialize value at maturity (last exercise date = T)
    # At T, V = max(K - S, 0)
    V = np.maximum(K - grid, 0.0)

    # Backward induction from step n_ex-1 down to 0
    # Step i (0-indexed from 0 to n_ex-1) corresponds to exercise date (i+1)
    # At the last step (i = n_ex-1), V is already the terminal value.
    # We go backward: for i = n_ex-2 down to 0:
    #   cont = compute_continuation(V, ...)
    #   V = max(immediate_payoff, cont)

    # But wait: the exercise dates are at t1, t2, ..., t12 where t12 = T.
    # At t12 (= T), the value is max(K-S, 0).
    # At t11, we can exercise or continue. If continue, we get E[V(t12)].
    # So backward induction:
    #   V at t12: max(K-S, 0)
    #   V at t11: max(max(K-S,0), exp(-r*dt)*E[V(t12)])
    #   ...
    #   V at t1: max(max(K-S,0), exp(-r*dt)*E[V(t2)])

    # We need the value at t1 for the exercise decision at t1.
    # The exercise policy at date i is: exercise if immediate >= continuation.

    # Let's store the exercise boundaries or the value functions.
    # For the policy object, we need to return True/False for each state at each time.
    # The policy will use the computed value functions to decide.

    # Store value functions at each exercise date
    value_funcs = []  # value_funcs[i] = V at exercise date (i+1)
    value_funcs.append(V.copy())  # V at t12 (maturity)

    for step in range(n_ex - 2, -1, -1):
        # Check deadline
        if time.time() > deadline - 5.0:
            # Not enough time, break and use what we have
            break

        cont = compute_continuation(V, log_grid, mu * dt, vol_sqrt_dt, r * dt)
        immediate = np.maximum(K - grid, 0.0)
        V = np.maximum(immediate, cont)
        value_funcs.append(V.copy())

    # Now value_funcs has n_ex entries (or fewer if we broke early)
    # value_funcs[0] is V at t1, value_funcs[-1] is V at t12

    # For the exercise policy:
    # At exercise date i (1-indexed), the state is S.
    # We need to decide: exercise or not.
    # Exercise if immediate >= continuation.
    # Continuation at date i = exp(-r*dt) * E[V(t_{i+1}) | S]
    # We can precompute the continuation values at each date.

    # Let me recompute: at date i, continuation = exp(-r*dt)*E[V_{i+1}(S_{i+1})]
    # where V_{i+1} is the value function at date i+1.

    # For the policy, I'll store the continuation value functions at each date.
    # Then at scoring time, for a given state S and time index, I interpolate
    # the continuation value and compare with immediate payoff.

    # Recompute continuation values at each date
    n_steps = len(value_funcs)
    cont_funcs = []
    for i in range(n_steps - 1):
        # value_funcs[i] is V at date (i+1) from the front... wait, let me re-index.
        # value_funcs[0] = V at t1 (first exercise date)
        # value_funcs[-1] = V at t12 (maturity)
        #
        # At date t1, continuation = exp(-r*dt)*E[V(t2)]
        # V(t2) is value_funcs[1]
        # So cont at t1 = compute_continuation(value_funcs[1], ...)
        #
        # At date t11, continuation = exp(-r*dt)*E[V(t12)]
        # V(t12) is value_funcs[-1] = value_funcs[n_steps-1]
        # So cont at t11 = compute_continuation(value_funcs[n_steps-1], ...)

        V_next = value_funcs[i + 1]
        cont_i = compute_continuation(V_next, log_grid, mu * dt, vol_sqrt_dt, r * dt)
        cont_funcs.append(cont_i)

    # cont_funcs[i] = continuation value at exercise date (i+1)
    # Number of cont_funcs = n_steps - 1
    # Exercise dates with a decision: t1 through t11 (11 dates)
    # At t12, no decision (it's maturity, always "exercise" in the sense that the option expires)

    # Actually, at t12, the option expires. The "exercise" at t12 is automatic.
    # The policy needs to return a boolean for each of the 12 exercise dates.
    # At t12, we should return True (exercise/redeem at maturity).
    # At t1 through t11, we return True if immediate >= continuation.

    # Let me create the ExercisePolicy class

    class ExercisePolicy:
        def __init__(self, log_grid, grid, cont_funcs, K, n_ex):
            self.log_grid = log_grid
            self.grid = grid
            self.cont_funcs = cont_funcs  # list of arrays, one per decision date (t1..t11)
            self.K = K
            self.n_ex = n_ex

        def _interp(self, x, x_grid, y_vals):
            """Piecewise linear interpolation of y_vals on x_grid at points x."""
            x_grid = np.asarray(x_grid)
            y_vals = np.asarray(y_vals)
            x = np.asarray(x, dtype=float)
            # Clip to grid range
            x_clipped = np.clip(x, x_grid[0], x_grid[-1])
            # Search sorted
            idx = np.searchsorted(x_grid, x_clipped, side='right') - 1
            idx = np.clip(idx, 0, len(x_grid) - 2)
            x0 = x_grid[idx]
            x1 = x_grid[idx + 1]
            y0 = y_vals[idx]
            y1 = y_vals[idx + 1]
            dx = x1 - x0
            # Avoid division by zero
            frac = np.where(dx > 1e-15, (x_clipped - x0) / dx, 0.0)
            return y0 + frac * (y1 - y0)

        def exercise(self, current_state, time_idx):
            """
            current_state: array of spot prices
            time_idx: which exercise date (0-indexed, 0 = t1, 11 = t12)
            Returns: boolean array, True if exercise
            """
            current_state = np.asarray(current_state, dtype=float)
            n_states = len(current_state)

            if time_idx >= self.n_ex - 1:
                # Last exercise date (maturity): always exercise (redeem)
                return np.ones(n_states, dtype=bool)

            # time_idx is 0 to n_ex-2, corresponding to t1 to t11
            cont_vals = self.cont_funcs[time_idx]
            log_state = np.log(np.maximum(current_state, 1e-10))
            cont_interp = self._interp(log_state, self.log_grid, cont_vals)
            immediate = np.maximum(self.K - current_state, 0.0)
            return immediate >= cont_interp

    return ExercisePolicy(log_grid, grid, cont_funcs, K, n_ex)

# EVOLVE-BLOCK-END
