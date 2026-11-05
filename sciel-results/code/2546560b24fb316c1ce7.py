import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fits a shared degree-5 polynomial to the unresolved subgrid tendency 
    using Chebyshev-basis Ridge regression with cross-validated regularization.
    Does not penalize the intercept (T_0=1) to preserve the mean.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill limit
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Handle edge cases
    if len(x) == 0:
        return np.zeros(6)
    
    # If very few points, fallback to simple linear fit in power basis
    if len(x) < 6:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            return np.pad(coeff, (0, 6 - len(coeff)))
        except Exception:
            return np.zeros(6)

    # 1. Normalize x to [-1, 1] for Chebyshev stability
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    
    if x_range < 1e-12:
        # All x are identical, return constant fit
        mean_y = np.mean(y)
        coeff = np.zeros(6)
        coeff[0] = mean_y
        return coeff
    
    x_norm = 2.0 * (x - x_min) / x_range - 1.0

    # 2. Define Chebyshev basis function
    def chebyshev_basis(x, degree):
        """
        Returns a matrix of Chebyshev polynomials T_0(x) ... T_degree(x)
        """
        n = len(x)
        basis = np.zeros((n, degree + 1))
        if degree >= 0:
            basis[:, 0] = 1.0
        if degree >= 1:
            basis[:, 1] = x
        for k in range(2, degree + 1):
            basis[:, k] = 2.0 * x * basis[:, k - 1] - basis[:, k - 2]
        return basis

    degree = 5
    T = chebyshev_basis(x_norm, degree)
    
    # 3. Cross-Validation to find optimal Ridge parameter lambda
    # We search over a range of lambda values
    # lambda = 0 is least squares
    # Higher lambda shrinks coefficients towards 0
    
    n_splits = 10
    indices = np.arange(len(x))
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = [indices[i::n_splits] for i in range(n_splits)]
    
    # Log-spaced lambda values - wider range
    lambdas = np.logspace(-8, 8, 80)
    
    best_mse = np.inf
    best_lambda = 0.0
    
    # Pre-compute fold indices
    train_folds = []
    val_folds = []
    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        train_folds.append(train_idx)
        val_folds.append(val_idx)
        
    # For efficiency, we can pre-compute T_train and T_val for each fold
    # But T depends on x_norm which is fixed, so we can just slice T
    
    # Pre-compute the identity matrix with first element zeroed (don't penalize intercept)
    reg_eye = np.eye(degree + 1)
    reg_eye[0, 0] = 0.0  # Don't penalize the intercept
    
    for lam in lambdas:
        if time.time() > deadline - 20.0:
            break
            
        fold_mses = []
        for i in range(n_splits):
            train_idx = train_folds[i]
            val_idx = val_folds[i]
            
            T_train = T[train_idx, :]
            y_train = y[train_idx]
            T_val = T[val_idx, :]
            y_val = y[val_idx]
            
            try:
                # Ridge Regression without penalizing the intercept:
                # (T.T @ T + lam * diag([0,1,1,...,1])) @ theta = T.T @ y
                
                A = T_train.T @ T_train
                b = T_train.T @ y_train
                
                # Add lambda to diagonal, but NOT for the first element (intercept)
                reg_matrix = A + lam * reg_eye
                
                # Use SVD-based solve for better numerical stability
                theta = np.linalg.solve(reg_matrix, b)
                
                y_pred = T_val @ theta
                mse = np.mean((y_val - y_pred) ** 2)
                fold_mses.append(mse)
            except Exception:
                fold_mses.append(np.inf)
        
        mean_mse = np.mean(fold_mses)
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_lambda = lam

    # 4. Final Fit with best_lambda on all data
    if time.time() > deadline - 5.0:
        # Emergency fallback: Least Squares
        try:
            theta_cheb = np.linalg.lstsq(T, y, rcond=None)[0]
        except Exception:
            theta_cheb = np.zeros(degree + 1)
    else:
        try:
            A = T.T @ T
            b = T.T @ y
            reg_matrix = A + best_lambda * reg_eye
            theta_cheb = np.linalg.solve(reg_matrix, b)
        except Exception:
            try:
                theta_cheb = np.linalg.lstsq(T, y, rcond=None)[0]
            except Exception:
                theta_cheb = np.zeros(degree + 1)
    
    # 5. High-precision polish: refine using SVD-based solve on the full system
    # This can help with numerical issues from the normal equations
    if time.time() < deadline - 2.0:
        try:
            # Use SVD to solve the regularized system more stably
            # (T.T @ T + lam * reg_eye) @ theta = T.T @ y
            A_full = T.T @ T + best_lambda * reg_eye
            b_full = T.T @ y
            U, s, Vt = np.linalg.svd(A_full)
            # Regularize small singular values
            s_inv = np.where(s > 1e-12, 1.0 / s, 0.0)
            theta_svd = Vt.T @ (s_inv * (U.T @ b_full))
            # Check if this is better (lower training error)
            y_pred_svd = T @ theta_svd
            err_svd = np.sum((y - y_pred_svd) ** 2)
            y_pred_orig = T @ theta_cheb
            err_orig = np.sum((y - y_pred_orig) ** 2)
            if err_svd < err_orig:
                theta_cheb = theta_svd
        except Exception:
            pass

    # 6. Convert Chebyshev coefficients to Power Basis coefficients
    # We need c such that sum(c_i * x^i) approximates the fit.
    # The fit is sum(theta_k * T_k(x_norm)).
    # x_norm = (2*x - (x_min+x_max)) / x_range = alpha * x + beta
    # T_k(x_norm) is a polynomial in x of degree k.
    
    try:
        # cheb2poly converts [c0, c1, ..., cn] (Cheb) to [p0, p1, ..., pn] (Power)
        # such that P(u) = sum c_k T_k(u) = sum p_k u^k
        power_u = np.polynomial.chebyshev.cheb2poly(theta_cheb)
        
        # Now we have P(u) = sum_{k=0}^5 power_u[k] * u^k
        # u = A*x + B
        # We need to expand sum power_u[k] * (A*x + B)^k
        # This is a polynomial in x.
        
        # Construct polynomial for u in terms of x
        A_coeff = 2.0 / x_range
        B_coeff = -1.0 - 2.0 * x_min / x_range
        
        # u(x) = A_coeff * x + B_coeff
        # We want poly_u(u(x)).
        
        # P(x) = sum_{k=0}^5 power_u[k] * (A*x + B)^k
        # (A*x + B)^k = sum_{j=0}^k binom(k, j) * A^j * B^(k-j) * x^j
        
        coeff_x = np.zeros(6)
        for k in range(6):
            pk = power_u[k]
            if pk == 0:
                continue
            # Expand (A*x + B)^k
            # Coefficients for x^j
            for j in range(k + 1):
                # Binomial coefficient C(k, j)
                binom_k_j = 1
                if j > 0:
                    binom_k_j = int(np.prod([k - i + 1 for i in range(1, j+1)])) / int(np.prod([i for i in range(1, j+1)]))
                # Term: pk * C(k, j) * A^j * B^(k-j) * x^j
                term_coeff = pk * binom_k_j * (A_coeff ** j) * (B_coeff ** (k - j))
                coeff_x[j] += term_coeff
                
    except Exception:
        # Fallback: Least squares in power basis directly
        try:
            V = np.vander(x, N=6, increasing=True)
            coeff_x = np.linalg.lstsq(V, y, rcond=None)[0]
        except Exception:
            coeff_x = np.zeros(6)

    # Verify finiteness
    if not np.all(np.isfinite(coeff_x)):
        coeff_x = np.where(np.isfinite(coeff_x), coeff_x, 0.0)

    return coeff_x
