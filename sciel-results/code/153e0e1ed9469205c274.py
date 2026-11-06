"""Rewritten program: odd-power polynomial closure with adaptive regularization."""
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
    
    # The subgrid tendency h*c*mean(Y) is approximately an odd function of X
    # due to the symmetry of the Lorenz-96 fast dynamics.
    # Fit using only odd powers: x, x^3, x^5
    # This reduces parameters from 6 to 3, improving generalization.
    
    # Build design matrix with odd powers
    # Use scaled features for better conditioning
    x_max = np.max(np.abs(x)) if x.size > 0 else 1.0
    if x_max < 1e-12:
        x_max = 1.0
    
    # Scale x to [-1, 1] range for better numerical conditioning
    x_scaled = x / x_max
    
    A = np.column_stack([x_scaled, x_scaled**3, x_scaled**5])
    
    # Use ridge regression with adaptive regularization
    # Try a small regularization
    try:
        # Compute optimal lambda via cross-validation-like approach
        # Use a grid of lambda values and pick the one that minimizes residual
        # on a held-out subset (if we have enough data)
        
        n = x.size
        if n > 100:
            # Use a simple train/val split for lambda selection
            indices = np.random.default_rng(42).permutation(n)
            val_size = max(10, n // 10)
            val_idx = indices[:val_size]
            train_idx = indices[val_size:]
            
            A_train = A[train_idx]
            y_train = y[train_idx]
            A_val = A[val_idx]
            y_val = y[val_idx]
            
            # Try several lambda values
            lambdas = np.array([1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
            best_lambda = 1e-6
            best_val_err = np.inf
            
            for lam in lambdas:
                try:
                    coeff_candidate = np.linalg.solve(
                        A_train.T @ A_train + lam * np.eye(3), 
                        A_train.T @ y_train
                    )
                    pred_val = A_val @ coeff_candidate
                    val_err = np.mean((pred_val - y_val)**2)
                    if val_err < best_val_err:
                        best_val_err = val_err
                        best_lambda = lam
                except np.linalg.LinAlgError:
                    continue
            
            coeff_odd = np.linalg.solve(
                A.T @ A + best_lambda * np.eye(3), 
                A.T @ y
            )
        else:
            # Not enough data for split, use a fixed small lambda
            best_lambda = 1e-6
            coeff_odd = np.linalg.solve(
                A.T @ A + best_lambda * np.eye(3), 
                A.T @ y
            )
    except np.linalg.LinAlgError:
        # Fallback to lstsq
        coeff_odd = np.linalg.lstsq(A, y, rcond=None)[0]
    
    # Map odd coefficients back to original x scale
    # If x_scaled = x / x_max, then:
    # c1*x_scaled + c3*x_scaled^3 + c5*x_scaled^5
    # = c1*(x/x_max) + c3*(x/x_max)^3 + c5*(x/x_max)^5
    # = (c1/x_max)*x + (c3/x_max^3)*x^3 + (c5/x_max^5)*x^5
    coeff = np.zeros(6)
    coeff[1] = coeff_odd[0] / x_max
    coeff[3] = coeff_odd[1] / (x_max**3)
    coeff[5] = coeff_odd[2] / (x_max**5)
    
    # Verify finite
    if not np.all(np.isfinite(coeff)):
        # Fallback to linear
        coeff = np.zeros(6)
        try:
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            coeff[:2] = c_lin
        except:
            coeff[1] = np.mean(y) / x_max if x_max > 1e-12 else 0.0
    
    return coeff
# EVOLVE-BLOCK-END
