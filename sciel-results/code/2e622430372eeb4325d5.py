# EVOLVE-BLOCK-START
import numpy as np
import time


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Maximizes H(x-y) / max(H(x), H(x+y), H(x+2y)).
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin for 600s limit

    n = 10
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')  # X[y, x] = x, Y[y, x] = y

    # Precompute projection values
    proj_0 = X  # x
    proj_1 = X + Y  # x + y
    proj_2 = X + 2 * Y  # x + 2y
    proj_m1 = X - Y  # x - y

    # Unique values and counts for each projection
    # For r=0: values 0..9
    # For r=1: values 0..18
    # For r=2: values 0..27
    # For r=-1: values -9..9

    def get_projection_distribution(P_flat, proj_vals, unique_vals):
        """Compute the marginal distribution of a projection given joint P."""
        # P_flat has shape (100,), proj_vals has shape (100,)
        # unique_vals is sorted array of unique projection values
        dist = np.zeros(len(unique_vals))
        for i, uv in enumerate(unique_vals):
            mask = proj_vals == uv
            dist[i] = np.sum(P_flat[mask])
        return dist

    def shannon_entropy_bits(dist):
        """Compute Shannon entropy in bits, ignoring zero probabilities."""
        dist = dist[dist > 1e-15]
        if len(dist) == 0:
            return 0.0
        return -np.sum(dist * np.log2(dist))

    # Unique values for each projection
    uv_0 = np.arange(10)
    uv_1 = np.arange(19)
    uv_2 = np.arange(28)
    uv_m1 = np.arange(-9, 10)

    # Flatten projections for vectorized ops
    p0_flat = proj_0.flatten()
    p1_flat = proj_1.flatten()
    p2_flat = proj_2.flatten()
    pm1_flat = proj_m1.flatten()

    def compute_objective(P):
        """Compute D/M ratio for a given distribution P (shape 10x10)."""
        P_flat = P.flatten()
        # Normalize to ensure sum = 1
        s = np.sum(P_flat)
        if s <= 1e-15:
            return 0.0
        P_flat = P_flat / s

        d_m1 = get_projection_distribution(P_flat, pm1_flat, uv_m1)
        d_0 = get_projection_distribution(P_flat, p0_flat, uv_0)
        d_1 = get_projection_distribution(P_flat, p1_flat, uv_1)
        d_2 = get_projection_distribution(P_flat, p2_flat, uv_2)

        H_m1 = shannon_entropy_bits(d_m1)
        H_0 = shannon_entropy_bits(d_0)
        H_1 = shannon_entropy_bits(d_1)
        H_2 = shannon_entropy_bits(d_2)

        M = max(H_0, H_1, H_2)
        if M <= 1e-9:
            return 0.0

        return H_m1 / M

    # Parametric ansatz: P[y, x] proportional to exp(alpha*(x-y)^2 + beta*(x+y) + gamma*x*y)
    # We optimize over (alpha, beta, gamma) to find good distributions.
    # The key insight: to maximize H(x-y), we want the distribution of x-y to be as flat as possible.
    # To minimize max(H(x), H(x+y), H(x+2y)), we want those projections to be concentrated.

    # Strategy: use a parametric family and do a grid search + local optimization.
    # P[y,x] ∝ exp(alpha * (x-y)^2 + beta * (x+y) + gamma * x * y)

    # Let's think about what parameters do:
    # - alpha > 0: weights cells far from diagonal more, spreading out x-y (good for H_m1)
    # - alpha < 0: concentrates on diagonal, x-y concentrated (bad for H_m1)
    # - beta: shifts mass in x+y direction
    # - gamma: creates correlation

    # Grid search over parameter space
    best_P = None
    best_val = 0.0

    # Start with a broad grid search
    alpha_range = np.linspace(-2.0, 2.0, 25)
    beta_range = np.linspace(-1.0, 1.0, 15)
    gamma_range = np.linspace(-1.0, 1.0, 15)

    for a in alpha_range:
        for b in beta_range:
            for g in gamma_range:
                if time.time() > deadline - 30:
                    break
                # Compute log weights
                log_w = a * (X - Y) ** 2 + b * (X + Y) + g * X * Y
                # Subtract max for numerical stability
                log_w -= np.max(log_w)
                w = np.exp(log_w)
                s = np.sum(w)
                if s <= 1e-15:
                    continue
                P = w / s
                val = compute_objective(P)
                if val > best_val:
                    best_val = val
                    best_P = P.copy()

    if best_P is None:
        # Fallback to uniform
        best_P = np.ones((n, n), dtype=np.float64) / (n * n)
        best_val = compute_objective(best_P)

    # Local optimization using gradient-free method (Nelder-Mead style)
    # Starting from the best grid point, refine
    # We'll use scipy's minimize if available, otherwise manual

    try:
        from scipy.optimize import minimize

        def neg_objective(params):
            a, b, g = params
            log_w = a * (X - Y) ** 2 + b * (X + Y) + g * X * Y
            log_w -= np.max(log_w)
            w = np.exp(log_w)
            s = np.sum(w)
            if s <= 1e-15:
                return 0.0
            P = w / s
            return -compute_objective(P)

        # Find the best grid point parameters
        best_params = None
        best_neg_val = 0.0
        for a in alpha_range:
            for b in beta_range:
                for g in gamma_range:
                    nv = neg_objective([a, b, g])
                    if nv < best_neg_val:
                        best_neg_val = nv
                        best_params = [a, b, g]

        if best_params is not None and time.time() < deadline - 60:
            result = minimize(
                neg_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result.success or result.fun < best_neg_val:
                a, b, g = result.x
                log_w = a * (X - Y) ** 2 + b * (X + Y) + g * X * Y
                log_w -= np.max(log_w)
                w = np.exp(log_w)
                s = np.sum(w)
                if s > 1e-15:
                    P = w / s
                    val = compute_objective(P)
                    if val > best_val:
                        best_val = val
                        best_P = P.copy()

    except Exception:
        pass

    # Try a different parametric family: P[y,x] ∝ exp(alpha * |x-y| + beta * (x+y)^2 + gamma * x * y)
    if time.time() < deadline - 30:
        alpha_range2 = np.linspace(-1.0, 1.0, 15)
        beta_range2 = np.linspace(-0.5, 0.5, 10)
        gamma_range2 = np.linspace(-1.0, 1.0, 10)

        for a in alpha_range2:
            for b in beta_range2:
                for g in gamma_range2:
                    if time.time() > deadline - 20:
                        break
                    log_w = a * np.abs(X - Y) + b * (X + Y) ** 2 + g * X * Y
                    log_w -= np.max(log_w)
                    w = np.exp(log_w)
                    s = np.sum(w)
                    if s <= 1e-15:
                        continue
                    P = w / s
                    val = compute_objective(P)
                    if val > best_val:
                        best_val = val
                        best_P = P.copy()

    # Final polish: direct optimization of P using projected gradient
    # Maximize H(x-y) - lambda * max(H(x), H(x+y), H(x+2y))
    # Use a soft-max approximation for the max
    if time.time() < deadline - 10:
        # Start from best_P and do a few iterations of gradient ascent on log P
        # with a smooth surrogate for the max

        def smooth_max(vals, temperature=0.1):
            """Smooth approximation of max using log-sum-exp."""
            m = np.max(vals)
            return m + temperature * np.log(np.sum(np.exp((vals - m) / temperature)))

        def smooth_objective(P):
            P_flat = P.flatten()
            s = np.sum(P_flat)
            if s <= 1e-15:
                return 0.0
            P_flat = P_flat / s

            d_m1 = get_projection_distribution(P_flat, pm1_flat, uv_m1)
            d_0 = get_projection_distribution(P_flat, p0_flat, uv_0)
            d_1 = get_projection_distribution(P_flat, p1_flat, uv_1)
            d_2 = get_projection_distribution(P_flat, p2_flat, uv_2)

            H_m1 = shannon_entropy_bits(d_m1)
            H_0 = shannon_entropy_bits(d_0)
            H_1 = shannon_entropy_bits(d_1)
            H_2 = shannon_entropy_bits(d_2)

            # Use soft max
            M_smooth = smooth_max([H_0, H_1, H_2], temperature=0.05)
            if M_smooth <= 1e-9:
                return 0.0
            return H_m1 / M_smooth

        # Simple projected gradient ascent on P
        P = best_P.copy()
        lr = 0.01
        for it in range(50):
            if time.time() > deadline - 5:
                break
            # Compute gradient numerically
            grad = np.zeros_like(P)
            eps = 1e-4
            f0 = smooth_objective(P)
            for i in range(10):
                for j in range(10):
                    P_plus = P.copy()
                    P_plus[i, j] += eps
                    P_plus = P_plus / np.sum(P_plus)
                    f_plus = smooth_objective(P_plus)
                    grad[i, j] = (f_plus - f0) / eps
                    P_plus = P.copy()

            # Gradient step
            P = P + lr * grad
            # Project to simplex
            P = np.clip(P, 0, None)
            s = np.sum(P)
            if s > 1e-15:
                P = P / s

            val = compute_objective(P)
            if val > best_val:
                best_val = val
                best_P = P.copy()

    # Ensure final validity
    best_P = np.clip(best_P, 0, None)
    s = np.sum(best_P)
    if s <= 1e-15:
        best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        best_P = best_P / s

    return best_P


# EVOLVE-BLOCK-END
