import numpy as np
import time
from itertools import combinations

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 145.0  # Safety margin from 150s
    
    # Convert to numpy arrays
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    # Handle length mismatch
    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)
        
    x = x[:n]
    y = y[:n]
    
    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    
    if n == 0:
        return np.zeros(6, dtype=np.float64)
        
    # If very few points, fall back to lower degree or zero
    if n < 6:
        deg = max(0, n - 1)
        try:
            # Use scaled fit even for small n if possible
            if deg < 5:
                # Pad to 6 later
                # For very small n, scaling might be unstable if std is 0
                mu = np.mean(x)
                sigma = np.std(x)
                if sigma < 1e-10:
                    sigma = 1.0
                x_scaled = (x - mu) / sigma
                V = np.vander(x_scaled, N=deg+1, increasing=True)
                # Ridge with small lambda for stability
                lam = 1e-6
                A = V.T @ V + lam * np.eye(deg+1)
                b = V.T @ y
                coeff_scaled = np.linalg.solve(A, b)
                
                # Transform back
                coeff_orig = np.zeros(6)
                for j in range(deg+1):
                    c_j = 0.0
                    for i in range(j, deg+1):
                        binom = 1.0
                        # Calculate binomial(i, j)
                        # Use formula or precomputed
                        # i is small, so simple loop
                        num = 1
                        den = 1
                        for k in range(j):
                            num *= (i - k)
                            den *= (k + 1)
                        binom = num / den if den != 0 else 1.0
                        
                        term = coeff_scaled[i] * (1.0 / (sigma ** i)) * binom * ((-mu) ** (i - j))
                        c_j += term
                    coeff_orig[j] = c_j
                return coeff_orig
            else:
                coeff = np.polynomial.polynomial.polyfit(x, y, deg)
                full_coeff = np.zeros(6)
                full_coeff[:len(coeff)] = coeff
                return full_coeff
        except Exception:
            return np.zeros(6, dtype=np.float64)

    # Check time
    if time.time() - t_start > deadline:
        return np.zeros(6, dtype=np.float64)

    # Strategy: Scaled Ridge Regression with Cross-Validated Lambda
    # 1. Scale x
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-10:
        sigma = 1.0
    x_scaled = (x - mu) / sigma
    
    # 2. Prepare Vandermonde for degree 5
    V = np.vander(x_scaled, N=6, increasing=True)
    
    # 3. Select lambda via K-Fold CV (K=5)
    # To save time, use a small grid of lambdas
    # Typical range for ridge: 1e-8 to 1e-2
    lambdas = np.logspace(-8, -2, 7)
    
    k_folds = 5
    indices = np.arange(n)
    # Shuffle indices for CV
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    fold_size = n // k_folds
    folds = [indices[i * fold_size : (i + 1) * fold_size] for i in range(k_folds)]
    # Handle remainder
    if len(folds[-1]) < fold_size:
        # Append remainder to last fold
        folds[-1] = np.concatenate([folds[-1], indices[k_folds * fold_size:]])
        
    mse_per_lambda = np.zeros(len(lambdas))
    
    for lam_idx, lam in enumerate(lambdas):
        mse_sum = 0.0
        for i in range(k_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
            
            X_train = V[train_idx]
            y_train = y[train_idx]
            X_val = V[val_idx]
            y_val = y[val_idx]
            
            try:
                # Ridge regression
                A = X_train.T @ X_train + lam * np.eye(6)
                b = X_train.T @ y_train
                coeff_scaled = np.linalg.solve(A, b)
                
                # Predict
                y_pred = X_val @ coeff_scaled
                mse = np.mean((y_val - y_pred) ** 2)
                mse_sum += mse
            except Exception:
                mse_sum = np.inf
                break
        
        mse_per_lambda[lam_idx] = mse_sum / k_folds
        
        # Time check
        if time.time() - t_start > deadline - 5.0:
            # If running out of time, pick a reasonable default lambda
            # and break
            mse_per_lambda[lam_idx+1:] = np.inf
            break

    # Find best lambda
    best_lam_idx = np.argmin(mse_per_lambda)
    best_lam = lambdas[best_lam_idx]
    
    # Final fit with best lambda on all data
    try:
        A = V.T @ V + best_lam * np.eye(6)
        b = V.T @ y
        coeff_scaled = np.linalg.solve(A, b)
    except Exception:
        # Fallback to unregularized or simple polyfit
        try:
            coeff_scaled = np.polynomial.polynomial.polyfit(x_scaled, y, 5)
        except Exception:
            return np.zeros(6, dtype=np.float64)

    # 4. Transform coefficients back to original x scale
    # y = sum_{i=0}^5 c'_i * ((x - mu) / sigma)^i
    #   = sum_{i=0}^5 c'_i * (1/sigma^i) * (x - mu)^i
    # (x - mu)^i = sum_{j=0}^i binom(i, j) * x^j * (-mu)^(i-j)
    # c_j = sum_{i=j}^5 c'_i * (1/sigma^i) * binom(i, j) * (-mu)^(i-j)
    
    coeff_orig = np.zeros(6, dtype=np.float64)
    
    for j in range(6):
        c_j = 0.0
        for i in range(j, 6):
            # Calculate binomial(i, j)
            # i is max 5, j is max 5
            # Use math.comb if available, or manual
            # Python 3.8+ has math.comb
            try:
                import math
                binom = math.comb(i, j)
            except ImportError:
                # Fallback for older Python (unlikely in modern env)
                num = 1
                den = 1
                for k in range(j):
                    num *= (i - k)
                    den *= (k + 1)
                binom = num / den if den != 0 else 1.0
            
            term = coeff_scaled[i] * (1.0 / (sigma ** i)) * binom * ((-mu) ** (i - j))
            c_j += term
        coeff_orig[j] = c_j

    # Final check for finiteness
    if not np.all(np.isfinite(coeff_orig)):
        return np.zeros(6, dtype=np.float64)

    return coeff_orig
