import numpy as np
import time

def _ridge_solve(V, y, alpha):
    """Solve ridge regression: min ||y - V@theta||^2 + alpha*||theta||^2
    Returns theta."""
    G = V.T @ V
    g = V.T @ y
    try:
        eigvals, eigvecs = np.linalg.eigh(G)
        eigvals = np.maximum(eigvals, 1e-15)
        eigvecs_T_g = eigvecs.T @ g
        theta = eigvecs @ (eigvecs_T_g / (eigvals + alpha))
    except np.linalg.LinAlgError:
        theta = np.zeros(V.shape[1])
    return theta

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 145.0

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n == 0:
        return np.zeros(6, dtype=np.float64)

    if n < 10:
        mean_y = np.mean(y) if n > 0 else 0.0
        return np.array([mean_y, 0, 0, 0, 0, 0], dtype=np.float64)

    # Scale x to [-1, 1] for better conditioning
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_range = 1.0
        x_min = x_min - 0.5
        x_max = x_min + 0.5

    A_scale = 2.0 / x_range
    B_scale = -(x_max + x_min) / x_range
    x_scaled = A_scale * x + B_scale

    # Build Vandermonde in scaled space
    V_all = np.vander(x_scaled, 6, increasing=True)

    # Use multiple random splits for model selection
    rng = np.random.RandomState(123)
    n_splits = 5
    best_cv_score = np.inf
    best_degree = 0
    best_alpha = 1.0

    degrees = [0, 1, 2, 3, 4, 5]
    alphas = np.logspace(-5, 4, 30)

    # Boundedness check points
    x_check = np.array([-1.5, -1.2, -0.8, 0.8, 1.2, 1.5])

    for deg in degrees:
        if time.time() > deadline - 20:
            break

        V_deg = V_all[:, :deg + 1]
        cv_mse_sums = np.zeros(len(alphas))
        cv_bound_sums = np.zeros(len(alphas))

        for split_i in range(n_splits):
            if time.time() > deadline - 20:
                break

            idx = rng.permutation(n)
            val_size = max(10, n // 5)
            val_idx = idx[:val_size]
            train_idx = idx[val_size:]

            V_train = V_deg[train_idx]
            y_train = y[train_idx]
            V_val = V_deg[val_idx]
            y_val = y[val_idx]

            G = V_train.T @ V_train
            g = V_train.T @ y_train

            try:
                eigvals, eigvecs = np.linalg.eigh(G)
                eigvals = np.maximum(eigvals, 1e-15)
                eigvecs_T_g = eigvecs.T @ g
                denom = eigvals[:, np.newaxis] + alphas[np.newaxis, :]
                theta_batch = eigvecs @ (eigvecs_T_g[:, np.newaxis] / denom)
            except np.linalg.LinAlgError:
                continue

            y_pred_batch = V_val @ theta_batch
            residuals = y_pred_batch - y_val[:, np.newaxis]
            mse_per_alpha = np.mean(residuals ** 2, axis=0)
            cv_mse_sums += mse_per_alpha

            V_check = np.vander(x_check, deg + 1, increasing=True)
            y_pred_check = V_check @ theta_batch
            bound_per_alpha = np.max(np.abs(y_pred_check), axis=0)
            cv_bound_sums += bound_per_alpha

        if time.time() > deadline - 20:
            break

        avg_cv_mse = cv_mse_sums / n_splits
        avg_bound = cv_bound_sums / n_splits

        y_scale = np.std(y) + 1e-10
        bound_penalty_weight = 0.1
        combined = avg_cv_mse + bound_penalty_weight * (avg_bound / y_scale) ** 2

        min_idx = np.argmin(combined)
        score = combined[min_idx]

        if score < best_cv_score:
            best_cv_score = score
            best_degree = deg
            best_alpha = alphas[min_idx]

    # Final solve with best parameters
    V_deg = V_all[:, :best_degree + 1]
    theta_scaled = _ridge_solve(V_deg, y, best_alpha)

    # Polish with nearby alphas
    if time.time() < deadline - 10:
        alpha_fine = best_alpha * np.array([0.5, 0.7, 1.0, 1.3, 2.0])
        best_polish_score = np.inf
        best_polish_theta = theta_scaled
        for af in alpha_fine:
            theta_try = _ridge_solve(V_deg, y, af)
            if n > 50:
                val_idx = np.arange(0, n, 5)
                pred = V_deg[val_idx] @ theta_try
                mse = np.mean((pred - y[val_idx]) ** 2)
            else:
                mse = 0
            if mse < best_polish_score:
                best_polish_score = mse
                best_polish_theta = theta_try
        theta_scaled = best_polish_theta

    # Convert from scaled monomial basis to original monomial basis
    result_poly = np.zeros(6, dtype=np.float64)

    if best_degree >= 0:
        current_power = np.array([1.0, 0, 0, 0, 0, 0])

        for i in range(best_degree + 1):
            if i < len(theta_scaled):
                result_poly += theta_scaled[i] * current_power

            if i < best_degree:
                new_power = np.zeros(6)
                for j in range(6):
                    if j < len(current_power):
                        new_power[j] += current_power[j] * B_scale
                        if j + 1 < 6:
                            new_power[j + 1] += current_power[j] * A_scale
                current_power = new_power

    if not np.all(np.isfinite(result_poly)):
        result_poly = np.array([np.mean(y), 0, 0, 0, 0, 0], dtype=np.float64)

    return result_poly
