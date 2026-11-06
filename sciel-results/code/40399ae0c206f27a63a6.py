"""Odd-power polynomial closure with refined regularization grid and cross-validation polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return np.zeros(6)

    if x.size < 10:
        try:
            coeff = np.zeros(6)
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            coeff[:2] = c_lin
            if not np.all(np.isfinite(coeff)):
                return np.zeros(6)
            return coeff
        except Exception:
            return np.zeros(6)

    # The subgrid tendency is approximately an odd function of X.
    # Fit using only odd powers: x, x^3, x^5
    A = np.column_stack([x, x**3, x**5])
    n = x.size

    best_coeff_odd = None
    best_val_err = np.inf

    if n > 50:
        rng = np.random.default_rng(42)
        indices = rng.permutation(n)
        val_size = max(10, n // 10)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        A_train = A[train_idx]
        y_train = y[train_idx]
        A_val = A[val_idx]
        y_val = y[val_idx]

        # Compute trace for lambda scaling
        G_train = A_train.T @ A_train
        trace_A = np.trace(G_train)
        b_train = A_train.T @ y_train

        if trace_A > 0:
            base_lam = trace_A / 3.0
            # Refined lambda grid with more points on the lower end
            lambdas = np.array([1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]) * base_lam
        else:
            lambdas = np.array([1e-6, 1e-4, 1e-2, 1.0, 100.0])

        # First pass: coarse grid
        for lam in lambdas:
            if time.time() > deadline:
                break
            try:
                coeff_candidate = np.linalg.solve(
                    G_train + lam * np.eye(3), 
                    b_train
                )
                pred_val = A_val @ coeff_candidate
                val_err = np.mean((pred_val - y_val)**2)
                if val_err < best_val_err:
                    best_val_err = val_err
                    best_coeff_odd = coeff_candidate
            except np.linalg.LinAlgError:
                continue

        # If we have time, do a refined search around the best lambda
        if best_coeff_odd is not None and time.time() < deadline - 30:
            # Find the best lambda from the first pass
            best_lam_idx = 0
            min_err = best_val_err
            for i, lam in enumerate(lambdas):
                try:
                    coeff_candidate = np.linalg.solve(
                        G_train + lam * np.eye(3), 
                        b_train
                    )
                    pred_val = A_val @ coeff_candidate
                    val_err = np.mean((pred_val - y_val)**2)
                    if val_err < min_err:
                        min_err = val_err
                        best_lam_idx = i
                except np.linalg.LinAlgError:
                    continue

            # Refined grid around best lambda
            best_lam = lambdas[best_lam_idx]
            if best_lam > 0:
                # Create a finer grid around the best lambda
                ratios = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0])
                refined_lambdas = best_lam * ratios
                # Also include some smaller values for safety
                refined_lambdas = np.concatenate([best_lam * np.array([0.01, 0.05, 0.1]), refined_lambdas])

                for lam in refined_lambdas:
                    if time.time() > deadline - 10:
                        break
                    try:
                        coeff_candidate = np.linalg.solve(
                            G_train + lam * np.eye(3), 
                            b_train
                        )
                        pred_val = A_val @ coeff_candidate
                        val_err = np.mean((pred_val - y_val)**2)
                        if val_err < best_val_err:
                            best_val_err = val_err
                            best_coeff_odd = coeff_candidate
                    except np.linalg.LinAlgError:
                        continue

        # Fallback if no lambda worked
        if best_coeff_odd is None:
            try:
                lam = 1e-6 * (trace_A / 3.0 + 1e-12)
                best_coeff_odd = np.linalg.solve(G_train + lam * np.eye(3), b_train)
            except np.linalg.LinAlgError:
                best_coeff_odd = np.linalg.lstsq(A_train, y_train, rcond=None)[0]
    else:
        # Not enough data for split, use a fixed small lambda
        try:
            G = A.T @ A
            b = A.T @ y
            trace_A = np.trace(G)
            if trace_A > 0:
                lam = 1e-6 * (trace_A / 3.0)
            else:
                lam = 1e-6
            best_coeff_odd = np.linalg.solve(G + lam * np.eye(3), b)
        except np.linalg.LinAlgError:
            best_coeff_odd = np.linalg.lstsq(A, y, rcond=None)[0]

    # Map odd coefficients to full degree-5 polynomial
    coeff = np.zeros(6)
    coeff[1] = best_coeff_odd[0]
    coeff[3] = best_coeff_odd[1]
    coeff[5] = best_coeff_odd[2]

    # Verify finite
    if not np.all(np.isfinite(coeff)):
        try:
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            coeff = np.zeros(6)
            coeff[:2] = c_lin
        except Exception:
            coeff = np.zeros(6)

    return coeff
# EVOLVE-BLOCK-END
