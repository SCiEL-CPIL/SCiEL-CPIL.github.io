"""Improved program for the suite task; uses ridge regression for robust polynomial closure."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Returns 6 coefficients in ascending powers of x.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Ensure inputs are float64 numpy arrays
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Handle edge case of empty or tiny training sets
    if len(x) < 6:
        return np.zeros(6)
        
    # Check time
    if time.time() - start_time > deadline:
        return np.zeros(6)
        
    try:
        # Construct the design matrix for the full degree-five polynomial:
        # [1, x, x^2, x^3, x^4, x^5]
        # We use np.vander with increasing=True to get ascending powers.
        # Note: np.vander(x, N, increasing=True) returns columns x^0, x^1, ..., x^(N-1)
        A = np.vander(x, 6, increasing=True)
        
        # The current best strategy exploits antisymmetry (odd powers only).
        # However, the task asks for a "DIFFERENT strategy". 
        # A robust alternative is Ridge Regression (L2 regularization) on the full basis.
        # This allows the model to learn small even-coefficient contributions if they 
        # exist due to numerical noise or slight deviations from perfect antisymmetry 
        # in the finite sample, while penalizing large coefficients to prevent 
        # overfitting to the training noise.
        
        # To handle potential scaling issues and ensure stability, we can standardize x 
        # or just rely on the fact that Lorenz-96 variables are O(1). 
        # Let's check the scale of x. If x is very large, x^5 will be huge.
        # In L96, x is typically in [-10, 10] range. x^5 can be 100,000.
        # This makes the condition number of A very high.
        
        # Strategy: Use SVD-based Ridge Regression.
        # Minimize ||A @ c - y||^2 + lambda * ||c||^2
        # Solution: c = (A^T A + lambda I)^-1 A^T y
        # Or more stably via SVD of A: A = U S V^T
        # c = V (S (S^2 + lambda I)^-1) U^T y
        
        # Let's perform SVD
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        
        # We need to choose lambda. 
        # A common heuristic is to set lambda relative to the largest singular value.
        # Let's try a few lambdas and pick the one that minimizes a simple validation error 
        # on a holdout set? 
        # The task says "Train on supplied... and minimize... on independent validation".
        # But we don't have validation data in the `construct` function. 
        # We only have training_x and training_subgrid_tendency.
        # So we must pick a fixed lambda or one derived from the data.
        
        # Given the antisymmetry, the even coefficients should be near zero.
        # Ridge regression will naturally shrink them towards zero if they are not significant.
        
        # Let's pick a lambda that balances bias and variance.
        # Since we don't have a validation set, we can use a simple heuristic:
        # lambda = alpha * S_max
        # Let's try alpha = 1e-3, 1e-2, 1e-1 and see? No, we need to return one result.
        
        # Actually, the "odd power" constraint worked well (1.06). 
        # A "different strategy" that might also work well is using the full basis 
        # but with a strong regularization that effectively zeros out the even terms 
        # if they are not supported by the data, or simply using the full basis 
        # with a moderate lambda.
        
        # Let's try a lambda based on the noise level? We don't know the noise.
        # Let's try lambda = 1e-4 * S_max.
        
        S_max = S[0] if len(S) > 0 else 1.0
        lambda_val = 1e-4 * S_max
        
        # Compute the ridge solution
        # c = V @ ( (S * (S^T y / (S^2 + lambda))) )
        # Note: S is 1D, U is (N, K), Vt is (K, K)
        # U^T y
        Ut_y = U.T @ y
        
        # Factor: S / (S^2 + lambda)
        factor = S / (S**2 + lambda_val)
        
        c = Vt.T @ (factor * Ut_y)
        
        # Ensure coefficients are finite
        if not np.all(np.isfinite(c)):
            # Fallback to odd-power least squares if ridge is unstable
            A_odd = np.column_stack((x, x**3, x**5))
            coeffs_odd, _, _, _ = np.linalg.lstsq(A_odd, y, rcond=None)
            c = np.zeros(6)
            c[1] = coeffs_odd[0]
            c[3] = coeffs_odd[1]
            c[5] = coeffs_odd[2]
            
        return c
        
    except Exception:
        # Fallback to odd-power least squares in case of any errors
        try:
            A_odd = np.column_stack((x, x**3, x**5))
            coeffs_odd, _, _, _ = np.linalg.lstsq(A_odd, y, rcond=None)
            c = np.zeros(6)
            c[1] = coeffs_odd[0]
            c[3] = coeffs_odd[1]
            c[5] = coeffs_odd[2]
            return c
        except Exception:
            return np.zeros(6)
# EVOLVE-BLOCK-END
