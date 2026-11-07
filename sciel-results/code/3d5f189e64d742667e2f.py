"""
Chebyshev Spectral Collocation for 1-D Noiseless Symbolic Regression.

Strategy:
1. Map the input domain [x_min, x_max] to the standard Chebyshev interval [-1, 1].
2. Construct the Chebyshev Vandermonde matrix V where V[i, k] = T_k(t_i).
3. Solve for coefficients c_k by minimizing ||V c - y||^2 + lambda * ||c||^2.
   This is a Tikhonov regularized least squares problem.
4. The regularization parameter lambda is selected via a simple heuristic or 
   a small search to avoid overfitting high-frequency noise (though data is noiseless, 
   numerical noise from finite precision or slight non-smoothness can cause Runge's 
   phenomenon if degree is too high).
5. For noiseless smooth data, a moderate degree (e.g., 10-30) often captures the 
   function perfectly without the oscillations of high-degree polynomials.
6. The predictor evaluates the polynomial sum at test points mapped to [-1, 1].

This approach is global, stable, and avoids the local artifacts of splines or the 
search complexity of symbolic regression.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 295.0

def _chebyshev_vandermonde(t, max_deg):
    """
    Compute the Chebyshev Vandermonde matrix.
    t: array of points in [-1, 1]
    max_deg: maximum polynomial degree (inclusive)
    Returns: matrix of shape (len(t), max_deg + 1)
    """
    n = len(t)
    m = max_deg + 1
    V = np.zeros((n, m))
    
    # T_0(t) = 1
    V[:, 0] = 1.0
    if max_deg >= 1:
        # T_1(t) = t
        V[:, 1] = t
        # Recurrence: T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
        for k in range(2, max_deg + 1):
            V[:, k] = 2.0 * t * V[:, k-1] - V[:, k-2]
            
    return V

def fit(X_train, y_train):
    global _DEADLINE
    
    # Ensure 1D input handling
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    x_train = X_train[:, 0]
    y_train = y_train
    
    n = len(x_train)
    if n == 0:
        def predict(X):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return np.zeros(X.shape[0])
        return predict

    x_min = np.min(x_train)
    x_max = np.max(x_train)
    
    # Handle case where all x are the same (degenerate)
    if x_max - x_min < 1e-10:
        y_mean = np.mean(y_train)
        def predict(X):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return np.full(X.shape[0], y_mean)
        return predict

    # Map x to t in [-1, 1]
    # t = 2 * (x - x_min) / (x_max - x_min) - 1
    scale = (x_max - x_min) / 2.0
    center = (x_max + x_min) / 2.0
    
    t_train = (x_train - center) / scale
    
    # Sort by t to ensure stable ordering for potential debugging, though not strictly needed for LS
    sort_idx = np.argsort(t_train)
    t_sorted = t_train[sort_idx]
    y_sorted = y_train[sort_idx]
    
    # We want to find the optimal degree.
    # For noiseless data, we can try a range of degrees and pick the one that minimizes
    # the training error while keeping the coefficient norm small (regularization).
    # Since the data is noiseless, the training error should drop to ~0 quickly.
    # We want the SMALLEST degree that achieves near-zero error to avoid overfitting/oscillation.
    
    # Define a range of degrees to test.
    # 200 points. Degrees up to ~50 are usually sufficient for smooth functions.
    max_deg_to_try = 60
    
    # We will solve the regularized least squares for a few candidate lambdas and degrees.
    # Actually, a simpler robust strategy for noiseless data:
    # 1. Try increasing degrees.
    # 2. For each degree, solve the unregularized LS (or lightly regularized) to get coefficients.
    # 3. Check the residual on the training data.
    # 4. Stop when residual is very small (e.g., < 1e-8 * max_y) or when degree is high.
    # 5. However, unregularized high-degree polynomials can be unstable. 
    #    Using a small lambda is safer.
    
    # Let's use a grid of (degree, lambda) pairs.
    # But to be fast, we can just pick a "sweet spot" degree.
    # Given 200 points, a degree of 20-40 is often a good balance.
    # Let's do a quick CV or just check training error for a few degrees with a fixed small lambda.
    
    best_coeffs = None
    best_deg = 0
    best_lambda = 1e-6
    
    # Try a few representative degrees and lambdas
    # Since we have 200 points, we can afford to solve a few 200xK systems.
    
    candidate_degs = [5, 10, 20, 30, 40, 50, 60]
    candidate_lambdas = [1e-8, 1e-6, 1e-4, 1e-2, 1e0]
    
    # We want to minimize training error, but penalize high degrees.
    # For noiseless data, the true function is likely low-complexity.
    # We can simply find the lowest degree where the fit is "good enough".
    # "Good enough" means residual < 1e-6 * (max|y| + 1).
    
    y_scale = np.max(np.abs(y_train)) + 1.0
    target_residual = 1e-8 * y_scale
    
    found_exact = False
    
    for deg in candidate_degs:
        if time.time() > _DEADLINE - 5:
            break
            
        V = _chebyshev_vandermonde(t_sorted, deg)
        
        # Solve (V^T V + lambda I) c = V^T y
        # Or use np.linalg.lstsq for unregularized, but regularized is more stable.
        # Let's try a small lambda first.
        for lam in candidate_lambdas:
            if time.time() > _DEADLINE - 5:
                break
                
            try:
                # Construct regularized system
                # A = V^T V + lam * I
                # b = V^T y
                # This is efficient for small deg.
                # V is 200 x (deg+1).
                A = V.T @ V
                b = V.T @ y_sorted
                
                # Add regularization to diagonal
                A_reg = A + lam * np.eye(deg + 1)
                
                c = np.linalg.solve(A_reg, b)
                
                # Calculate training residual
                pred = V @ c
                residual = np.mean((pred - y_sorted) ** 2)
                
                if residual < target_residual:
                    # We found a fit that is essentially exact.
                    # Prefer the smallest degree and smallest lambda that achieves this.
                    # Since we iterate deg in increasing order, the first one found is best.
                    best_coeffs = c
                    best_deg = deg
                    best_lambda = lam
                    found_exact = True
                    break # Break lambda loop
                    
            except np.linalg.LinAlgError:
                continue
            except Exception:
                continue
        
        if found_exact:
            break
            
    if not found_exact:
        # If no "exact" fit found (maybe noise or numerical issues), 
        # fall back to the best fit among the tried combinations based on residual.
        # Or just pick a reasonable default like deg=30, lam=1e-6.
        # Let's pick the one with lowest residual from the last deg tried.
        deg = candidate_degs[-1]
        lam = 1e-6
        V = _chebyshev_vandermonde(t_sorted, deg)
        A = V.T @ V + lam * np.eye(deg + 1)
        b = V.T @ y_sorted
        try:
            best_coeffs = np.linalg.solve(A, b)
            best_deg = deg
        except Exception:
            # Fallback to linear regression
            best_coeffs = np.polyfit(t_sorted, y_sorted, 1)
            # Note: np.polyfit returns highest degree first, Chebyshev is T_0, T_1...
            # So we need to reverse if we use polyfit coefficients in Chebyshev basis?
            # No, polyfit is monomial basis. We must stick to Chebyshev basis.
            # Let's just use a very low degree Chebyshev fit.
            deg = 1
            V = _chebyshev_vandermonde(t_sorted, deg)
            A = V.T @ V + 1e-6 * np.eye(deg + 1)
            b = V.T @ y_sorted
            best_coeffs = np.linalg.solve(A, b)
            best_deg = deg

    # Store the model parameters
    model_params = {
        'coeffs': best_coeffs,
        'deg': best_deg,
        'scale': scale,
        'center': center
    }
    
    def predict(X):
        try:
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            x_test = X[:, 0]
            
            # Map to t
            t_test = (x_test - model_params['center']) / model_params['scale']
            
            # Evaluate Chebyshev polynomial
            # Use Clenshaw's algorithm for stability, or just V matrix if n_test is small.
            # n_test is 200, deg is <= 60. V matrix is 200x61. Very fast.
            V_test = _chebyshev_vandermonde(t_test, model_params['deg'])
            preds = V_test @ model_params['coeffs']
            
            # Ensure finite
            preds = np.where(np.isfinite(preds), preds, np.mean(y_train))
            return preds
        except Exception:
            # Fallback: mean
            return np.full(X.shape[0], np.mean(y_train))
    
    return predict
# EVOLVE-BLOCK-END
