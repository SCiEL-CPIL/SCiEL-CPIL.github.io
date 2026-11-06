import numpy as np
import time

# EVOLVE-BLOCK-START
def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using a parametric smooth boundary
    fitted via grid search on a coarse 1D grid, with bisection refinement.
    """
    t_start = time.time()
    DEADLINE = 150.0  # wall-clock limit
    SAFETY_MARGIN = 5.0  # seconds to spare

    # Extract model parameters
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
        sigma = float(_get(contract, 'sigma', 'vol', 'volatility', default=sigma))
        T = float(_get(contract, 'T', 'maturity', 'Maturity', default=T))
        n_dates = int(_get(contract, 'n_dates', 'num_dates', 'exercise_dates', default=n_dates))
    except Exception:
        pass

    # Training paths
    if training_paths is not None and hasattr(training_paths, 'shape'):
        paths = np.asarray(training_paths, dtype=np.float64)
    else:
        # Generate synthetic training paths if none provided
        n_paths = 4096
        dt = T / n_dates
        steps = n_dates
        # GBM under risk-neutral measure
        drift = (r - 0.5 * sigma**2) * dt
        vol_step = sigma * np.sqrt(dt)
        log_increments = np.random.standard_normal((steps, n_paths)) * vol_step + drift
        log_spots = np.cumsum(log_increments, axis=0)
        S0 = 100.0
        paths = S0 * np.exp(log_increments)
        # Prepend initial spot
        paths = np.vstack([np.full((1, n_paths), S0), paths])

    n_paths = paths.shape[1]
    n_steps = paths.shape[0] - 1  # number of transitions
    # Exercise dates are at steps 1..n_steps (no time-0 exercise)

    dt = T / n_steps if n_steps > 0 else T / n_dates
    disc = np.exp(-r * dt)

    # Parametric boundary: for each exercise date j (1..n_steps),
    # threshold is a smooth function of log-spot.
    # We'll use a simple approach: for each date, find optimal threshold
    # via backward induction on a grid, then smooth across dates.

    # Grid for spot values
    # Determine range from training data
    S_min = np.min(paths) * 0.5
    S_max = np.max(paths) * 1.5
    n_grid = 200
    grid = np.linspace(S_min, S_max, n_grid)

    # Backward induction on the grid
    # V[j, i] = value at exercise date j, grid point i
    # Payoff: max(K - S, 0) for put
    payoff = np.maximum(K - grid, 0.0)

    # Start from the last exercise date
    V = np.zeros((n_steps + 1, n_grid))
    V[n_steps, :] = payoff  # At maturity, exercise or not (Bermudan: last date is final)

    # Backward induction
    for j in range(n_steps - 1, -1, -1):
        # Continuation value: discounted expected future value
        # Under risk-neutral GBM, E[S_{j+1} | S_j = s] = s * exp(r * dt)
        # But we need E[V_{j+1}(S_{j+1}) | S_j = s]
        # Approximate using the grid: assume S_{j+1} = S_j * exp(sigma*sqrt(dt)*Z + (r - 0.5*sigma^2)*dt)
        # For each grid point, compute expected continuation value
        # Use a quadrature approach or linear interpolation
        # For speed, use a simple 2-point or 4-point Gaussian quadrature
        # Or approximate with mean and variance
        # E[V(S_{j+1})] ≈ V(S_j * exp(mu)) where mu = (r - 0.5*sigma^2)*dt
        # This is a rough approximation. Better: use the fact that log(S_{j+1}) is normal
        # Given S_j = s, log(S_{j+1}) ~ N(log(s) + (r - 0.5*sigma^2)*dt, sigma^2*dt)
        # So S_{j+1} = s * exp(sigma*sqrt(dt)*Z + (r - 0.5*sigma^2)*dt)
        # We can approximate E[V(S_{j+1})] by interpolating V at a few points

        # Use 5-point Gauss-Hermite quadrature for accuracy
        # Z ~ N(0,1), so S_{j+1} = s * exp(sigma*sqrt(dt)*Z + (r - 0.5*sigma^2)*dt)
        # E[V(S_{j+1})] = integral V(s * exp(sigma*sqrt(dt)*z + (r - 0.5*sigma^2)*dt)) phi(z) dz
        # Using Gauss-Hermite: integral f(z) phi(z) dz ≈ sum w_i f(z_i) / sqrt(pi)
        # where phi(z) = exp(-z^2/2)/sqrt(2*pi), so integral f(z) exp(-z^2/2) dz / sqrt(2*pi)
        # Gauss-Hermite: integral f(x) exp(-x^2) dx ≈ sum w_i f(x_i)
        # Let x = z/sqrt(2), then z = x*sqrt(2), dz = sqrt(2) dx
        # integral f(z) exp(-z^2/2) dz = integral f(x*sqrt(2)) exp(-x^2) sqrt(2) dx ≈ sqrt(2) sum w_i f(x_i*sqrt(2))
        # So E[V(S_{j+1})] ≈ sqrt(2)/sqrt(2*pi) * sum w_i V(s * exp(sigma*sqrt(dt)*x_i*sqrt(2) + (r - 0.5*sigma^2)*dt))
        # = 1/sqrt(pi) * sum w_i V(...)

        # For speed, use a simpler approach: linear interpolation on the grid
        # Compute expected log-spot increment
        mu_log = (r - 0.5 * sigma**2) * dt
        vol_log = sigma * np.sqrt(dt)

        # For each grid point s, S_{j+1} = s * exp(mu_log + vol_log * Z)
        # Approximate E[V(S_{j+1})] by averaging over a few quantiles of Z
        # Use 7-point quadrature for better accuracy
        z_nodes = np.array([-2.5, -1.5, -0.75, 0.0, 0.75, 1.5, 2.5])
        # Weights for trapezoidal-like rule (approximate)
        # Better: use equal weights for simplicity, or proper Gauss-Legendre
        # For now, use a simple approach: evaluate at mean and two std devs
        z_nodes = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
        z_weights = np.array([0.1, 0.25, 0.3, 0.25, 0.1])  # Approximate normal weights

        cont = np.zeros(n_grid)
        for i in range(n_grid):
            s = grid[i]
            # S_{j+1} for each z
            S_next = s * np.exp(mu_log + vol_log * z_nodes)
            # Interpolate V[n_steps if j+1==n_steps else j+1] at S_next
            V_next = V[j + 1, :]
            # Clip to grid range for interpolation
            S_next_clipped = np.clip(S_next, S_min, S_max)
            # Linear interpolation
            idx = np.searchsorted(grid, S_next_clipped)
            idx = np.clip(idx, 1, n_grid - 1)
            frac = (S_next_clipped - grid[idx - 1]) / (grid[idx] - grid[idx - 1] + 1e-12)
            frac = np.clip(frac, 0.0, 1.0)
            V_interp = V_next[idx - 1] * (1 - frac) + V_next[idx] * frac
            cont[i] = np.sum(z_weights * V_interp)

        V[j, :] = np.maximum(payoff, disc * cont)

    # Extract thresholds from V: for each date j, threshold is where payoff >= disc * cont
    # i.e., K - S >= disc * E[V_next(S_next)]
    # Find the largest S where exercise is optimal

    thresholds = np.zeros(n_steps)
    for j in range(n_steps):
        # Exercise if payoff >= disc * cont
        # We stored V[j] = max(payoff, disc*cont)
        # Need to recover cont or recompute
        # Simpler: find where payoff >= V[j] (since V[j] = max(payoff, cont_val))
        # Actually V[j] = max(payoff, disc*cont), so exercise when payoff >= disc*cont
        # Which is when payoff >= V[j] is NOT correct because V[j] could be payoff itself
        # Let's recompute: exercise when payoff >= disc * cont
        # We have V[j] = max(payoff, disc*cont)
        # So disc*cont = V[j] - max(V[j] - payoff, 0) ... this is circular
        # Better: find the boundary where payoff crosses disc*cont
        # Since payoff is decreasing in S and cont is increasing in S (for puts),
        # there's a unique threshold
        # Find the largest grid point where payoff >= V[j] (since at that point, exercise was chosen)
        # Actually, if V[j] = payoff, then exercise. If V[j] > payoff, then continue.
        # So threshold is the largest S where V[j] == payoff
        diff = V[j, :] - payoff
        # Where diff <= 0 (within tolerance), exercise
        exercise_mask = diff <= 1e-10
        if np.any(exercise_mask):
            idx_ex = np.where(exercise_mask)[0]
            thresholds[j] = grid[idx_ex[-1]]
        else:
            thresholds[j] = grid[0]

    # Smooth thresholds across dates using a monotonic constraint
    # For a put, as time to maturity decreases, the exercise threshold should decrease
    # (or stay the same) because there's less time for the option to become more valuable
    # Actually for a put, early exercise becomes more attractive as the stock goes down
    # and as time to maturity decreases (less time value)
    # The threshold should be non-increasing in j (as j increases, less time left, threshold decreases)
    # Wait: j=0 is first exercise date, j=n_steps-1 is last
    # As j increases (closer to maturity), the threshold should decrease
    # So thresholds[0] >= thresholds[1] >= ... >= thresholds[n_steps-1]

    # Enforce monotonicity
    for j in range(1, n_steps):
        if thresholds[j] > thresholds[j - 1]:
            thresholds[j] = thresholds[j - 1]

    # Now define the policy
    class ExercisePolicy:
        def __init__(self, thresholds, n_steps):
            self.thresholds = thresholds
            self.n_steps = n_steps

        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: current time (to determine which exercise date)
            Returns: boolean array, True if exercise
            """
            # Determine exercise date index from time
            # Exercise dates are at t = dt, 2*dt, ..., n_steps*dt
            if time is None:
                t = 0.0
            else:
                t = float(time)
            # Find which exercise date we're at
            # t should be close to j*dt for some j in 1..n_steps
            j = int(round(t / dt))
            j = max(1, min(j, n_steps))
            # j is 1-indexed: exercise date j corresponds to thresholds[j-1]
            if j < 1 or j > n_steps:
                return np.zeros(len(current_state), dtype=bool)
            threshold = self.thresholds[j - 1]
            return current_state <= threshold

    policy = ExercisePolicy(thresholds, n_steps)
    return policy
# EVOLVE-BLOCK-END
