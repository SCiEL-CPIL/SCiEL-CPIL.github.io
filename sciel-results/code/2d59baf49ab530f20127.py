"""Combined program: Odd-power polynomial closure with adaptive regularization on raw scale."""
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

    # If very few points, fall back to linear
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

    # The subgrid tendency h*c*mean(Y) is approximately an odd function of X
    # due to the symmetry of the Lorenz-96 fast dynamics.
    # Fit using only odd powers: x, x^3, x^5
    # This reduces parameters from 6 to 3, improving generalization.
    
    # Build design matrix with odd powers on raw x
    # Check for potential overflow in x**5 if x is large, but L96 states are bounded (~10-20)
    # so x**5 is safe (~3.2e6).
    A = np.column_stack([x, x**3, x**5])
    
    n = x.size
    best_coeff_odd = None
    best_val_err = np.inf
    
    # If we have enough data, use a simple train/val split to select lambda
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
        
        # Grid of lambda values
        # The scale of A.T @ A is roughly N. 
        # We want lambda to be small relative to the diagonal elements.
        # Diagonal elements are sums of x^2, x^6, x^10.
        # For x ~ 10, x^10 ~ 1e10. N ~ 1000. Diagonal ~ 1e13.
        # So lambda should be in range [1e-4 * trace, 1e4 * trace] maybe?
        # Let's use a relative scale.
        trace_A = np.trace(A_train.T @ A_train)
        if trace_A > 0:
            base_lam = trace_A / 3.0
            lambdas = np.array([1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6]) * base_lam
        else:
            lambdas = np.array([1e-6, 1e-4, 1e-2, 1.0, 100.0])

        for lam in lambdas:
            if time.time() > deadline:
                break
            try:
                coeff_candidate = np.linalg.solve(
                    A_train.T @ A_train + lam * np.eye(3), 
                    A_train.T @ y_train
                )
                pred_val = A_val @ coeff_candidate
                val_err = np.mean((pred_val - y_val)**2)
                if val_err < best_val_err:
                    best_val_err = val_err
                    best_coeff_odd = coeff_candidate
            except np.linalg.LinAlgError:
                continue
        
        # If no lambda worked (unlikely), fall back to small lambda
        if best_coeff_odd is None:
            try:
                lam = 1e-6 * (np.trace(A.T @ A) / 3.0 + 1e-12)
                best_coeff_odd = np.linalg.solve(A.T @ A + lam * np.eye(3), A.T @ y)
            except np.linalg.LinAlgError:
                best_coeff_odd = np.linalg.lstsq(A, y, rcond=None)[0]
    else:
        # Not enough data for split, use a fixed small lambda
        try:
            trace_A = np.trace(A.T @ A)
            if trace_A > 0:
                lam = 1e-6 * (trace_A / 3.0)
            else:
                lam = 1e-6
            best_coeff_odd = np.linalg.solve(A.T @ A + lam * np.eye(3), A.T @ y)
        except np.linalg.LinAlgError:
            best_coeff_odd = np.linalg.lstsq(A, y, rcond=None)[0]

    # Map odd coefficients to full degree-5 polynomial
    # coeff[1] = c1, coeff[3] = c3, coeff[5] = c5
    coeff = np.zeros(6)
    coeff[1] = best_coeff_odd[0]
    coeff[3] = best_coeff_odd[1]
    coeff[5] = best_coeff_odd[2]

    # Verify finite
    if not np.all(np.isfinite(coeff)):
        # Fallback to linear
        try:
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            coeff = np.zeros(6)
            coeff[:2] = c_lin
        except Exception:
            coeff = np.zeros(6)

    return coeff
# EVOLVE-BLOCK-END
