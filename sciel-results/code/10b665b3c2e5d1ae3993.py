"""Rewritten program: Tikhonov-regularized degree-5 polynomial fit with internal k-fold CV."""
import numpy as np


def _build_design_matrix(x, degree):
    """Build Vandermonde matrix in ascending powers."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    # Use np.vander with increasing=True for ascending powers
    V = np.vander(x, N=degree + 1, increasing=True)
    return V


def _ridge_solve(V, y, lam, penalize_intercept=False):
    """
    Solve ridge regression: (V^T V + lam * R) w = V^T y
    R is a diagonal matrix. If penalize_intercept is False, R[0,0]=0, else R[0,0]=1.
    """
    n_params = V.shape[1]
    A = V.T @ V
    b = V.T @ y
    
    # Construct regularization matrix R
    # By default, we do NOT penalize the intercept (index 0)
    if penalize_intercept:
        R = np.eye(n_params)
    else:
        R = np.eye(n_params)
        R[0, 0] = 0.0
        
    A_reg = A + lam * R
    
    # Ensure A_reg is well-conditioned
    # If lam is 0 and A is singular, this might fail, but lam > 0 usually fixes it.
    # We use solve for speed.
    try:
        w = np.linalg.solve(A_reg, b)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular
        w = np.linalg.lstsq(A_reg, b, rcond=None)[0]
        
    return w


def construct(training_x, training_subgrid_tendency):
    training_x = np.asarray(training_x, dtype=np.float64)
    training_subgrid_tendency = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    n = len(training_x)
    degree = 5
    
    # Handle edge cases
    if n < degree + 1:
        # Fall back to lower degree or simple mean
        if n == 0:
            return np.zeros(6)
        # Fit a polynomial of degree min(degree, n-1)
        deg_eff = min(degree, n - 1)
        try:
            # Use polyfit for small n, it handles scaling internally somewhat
            coeff = np.polynomial.polynomial.polyfit(training_x, training_subgrid_tendency, deg_eff)
            # Pad to length 6
            full_coeff = np.zeros(6)
            full_coeff[:len(coeff)] = coeff
            return full_coeff
        except:
            return np.zeros(6)

    V = _build_design_matrix(training_x, degree)
    
    # Determine a sensible range for lambda
    # The scale of V^T V is roughly n * Var(x^i).
    # We want lambda to compete with the eigenvalues of V^T V.
    # Let's estimate a characteristic scale.
    # Using the trace of V^T V / n_params as a heuristic for the "size" of the matrix.
    gram_trace = np.trace(V.T @ V)
    scale = gram_trace / (degree + 1) if gram_trace > 0 else 1.0
    
    # Lambda range: from very small (almost OLS) to large (strong regularization)
    # We explore a wide range.
    # log10(lambda) from -10 to +10 relative to scale?
    # Let's try logspace(-8, 8, 40) * scale
    lambdas = np.logspace(-8, 8, 40) * scale
    
    # K-fold cross-validation
    # Use 10 folds for better stability
    k = 10
    if n < k:
        k = max(2, n // 2)
        if k < 2:
            # Just use the whole data with a moderate lambda
            lam = scale * 1e-3
            w = _ridge_solve(V, training_subgrid_tendency, lam, penalize_intercept=False)
            return w
    
    # Create indices
    indices = np.arange(n)
    # Shuffle indices for CV folds to ensure random split
    rng = np.random.default_rng(42) # Fixed seed for determinism
    rng.shuffle(indices)
    
    # Split into k folds
    fold_indices = np.array_split(indices, k)
    
    best_mse = np.inf
    best_w = np.zeros(6)
    best_lam = 0.0
    
    # Optimization: Pre-compute V and y for all folds? 
    # V is n x 6. Subsetting is cheap.
    
    for lam in lambdas:
        mse_sum = 0.0
        for i in range(k):
            val_idx = fold_indices[i]
            train_idx = np.concatenate([fold_indices[j] for j in range(k) if j != i])
            
            V_train = V[train_idx]
            y_train = training_subgrid_tendency[train_idx]
            V_val = V[val_idx]
            y_val = training_subgrid_tendency[val_idx]
            
            w_cv = _ridge_solve(V_train, y_train, lam, penalize_intercept=False)
            pred_val = V_val @ w_cv
            mse = np.mean((pred_val - y_val) ** 2)
            mse_sum += mse
        
        avg_mse = mse_sum / k
        
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_lam = lam
            # Store the solution from the full dataset with this lambda
            # We re-solve here to ensure it's from the full set, 
            # although we could store w_cv from one fold, full set is better.
            best_w = _ridge_solve(V, training_subgrid_tendency, best_lam, penalize_intercept=False)
            
    return best_w
