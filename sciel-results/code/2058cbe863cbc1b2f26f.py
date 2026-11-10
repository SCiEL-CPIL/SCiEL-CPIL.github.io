"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _convert_coeffs_to_original(a, mu, sigma, deg=5):
    """
    Convert polynomial coefficients 'a' for the variable z = (x-mu)/sigma
    to coefficients 'b' for the original variable x.
    y = sum(a_j * z^j) = sum(b_i * x^i)
    """
    b = np.zeros(deg + 1)
    # a_j * ((x-mu)/sigma)^j
    # = a_j * sigma^(-j) * (x-mu)^j
    # (x-mu)^j = sum_{k=0}^j binom(j,k) x^k (-mu)^(j-k)
    
    for j in range(deg + 1):
        if a[j] == 0:
            continue
        coeff_a = a[j] / (sigma ** j)
        # Expand (x-mu)^j
        # Term for x^k where k <= j
        for k in range(j + 1):
            # binom(j, k) * x^k * (-mu)^(j-k)
            # We need to accumulate into b[k]
            binom_val = 1.0
            # Calculate binom(j, k)
            # j is small (<=5), so simple calculation is fine
            # binom(j, k) = j! / (k! (j-k)!)
            # Or use iterative update
            # Let's just use scipy.special.comb if available or manual
            # Manual is safer to avoid import issues, though scipy is allowed.
            # Let's use a simple function
            def _nCr(n, r):
                if r < 0 or r > n:
                    return 0
                if r == 0:
                    return 1
                r = min(r, n - r)
                res = 1
                for i in range(r):
                    res = res * (n - i) // (i + 1)
                return res
            
            binom_val = _nCr(j, k)
            term_coeff = coeff_a * binom_val * ((-mu) ** (j - k))
            b[k] += term_coeff
            
    return b

def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin before 120s/150s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Ensure finite inputs
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.all(mask):
        x = x[mask]
        y = y[mask]

    if len(x) < 10:
        # Not enough data, return zeros or simple mean
        return np.array([np.mean(y)] + [0.0]*5)

    # Compute scaling parameters
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-8:
        sigma = 1.0
    z = (x - mu) / sigma

    # Degrees: we are forced to return 6 coeffs, so we fit degree 5.
    # But we can regularize to effectively reduce degree.
    deg = 5

    # Grid of regularization parameters (Lambda for Ridge: ||w||^2 penalty)
    # Note: np.linalg.lstsq or ridge implementation varies. 
    # We use: (V^T V + lambda * I) c = V^T y
    # We want to span a wide range.
    # Since z is standardized, the eigenvalues of V^T V are manageable.
    # lambda range: from very small (least squares) to very large (mean only)
    lambdas = np.logspace(-4, 4, 50)
    
    best_b = np.zeros(6)
    best_cv_mse = np.inf

    # K-Fold Cross Validation for Lambda selection
    n_folds = 5
    n = len(z)
    if n < n_folds * 2:
        n_folds = 2
        if n < 4:
            # Too little data, just fit with high lambda
            best_b = _convert_coeffs_to_original(np.array([np.mean(y)] + [0.0]*5), mu, sigma, deg)
            return best_b

    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    fold_size = n // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        folds.append((indices[start:end], np.concatenate([indices[:start], indices[end:]])))

    # Precompute Vandermonde matrices for training and validation sets for each fold?
    # No, V depends on z. z is fixed. We can precompute V for all points, and slice it.
    V_full = np.polynomial.polynomial.polyvander(z, deg)
    
    for lam in lambdas:
        if time.time() > deadline - 10.0:
            break
            
        cv_mse_sum = 0.0
        valid_folds = 0
        
        for val_idx, train_idx in folds:
            if len(train_idx) < deg + 1:
                continue
                
            V_tr = V_full[train_idx]
            y_tr = y[train_idx]
            V_val = V_full[val_idx]
            y_val = y[val_idx]
            
            # Ridge fit: (V^T V + lam * I) c = V^T y
            # Identity matrix of size deg+1
            I = np.eye(deg + 1)
            try:
                A = V_tr.T @ V_tr + lam * I
                b_vec = V_tr.T @ y_tr
                c = np.linalg.solve(A, b_vec)
                if not np.all(np.isfinite(c)):
                    continue
                
                pred = V_val @ c
                mse = np.mean((pred - y_val) ** 2)
                cv_mse_sum += mse
                valid_folds += 1
            except Exception:
                continue
        
        if valid_folds > 0:
            avg_mse = cv_mse_sum / valid_folds
            if avg_mse < best_cv_mse:
                best_cv_mse = avg_mse
                # Store the best lambda to refit later, or refit now?
                # Refitting on all data with the best lambda is the standard approach.
                # We can refit at the end to save time, or refit now to be safe if we break early.
                # Let's store best_lambda.
                best_lambda = lam
        
    # Refine lambda around the best one found
    if 'best_lambda' in locals():
        # Create a finer grid around the best lambda
        # Range: best_lambda / 1.5 to best_lambda * 1.5
        fine_lambdas = np.linspace(best_lambda / 1.5, best_lambda * 1.5, 20)
        # Ensure positive
        fine_lambdas = fine_lambdas[fine_lambdas > 0]
        
        for lam in fine_lambdas:
            if time.time() > deadline - 5.0:
                break
                
            cv_mse_sum = 0.0
            valid_folds = 0
            
            for val_idx, train_idx in folds:
                if len(train_idx) < deg + 1:
                    continue
                    
                V_tr = V_full[train_idx]
                y_tr = y[train_idx]
                V_val = V_full[val_idx]
                y_val = y[val_idx]
                
                I = np.eye(deg + 1)
                try:
                    A = V_tr.T @ V_tr + lam * I
                    b_vec = V_tr.T @ y_tr
                    c = np.linalg.solve(A, b_vec)
                    if not np.all(np.isfinite(c)):
                        continue
                    
                    pred = V_val @ c
                    mse = np.mean((pred - y_val) ** 2)
                    cv_mse_sum += mse
                    valid_folds += 1
                except Exception:
                    continue
            
            if valid_folds > 0:
                avg_mse = cv_mse_sum / valid_folds
                if avg_mse < best_cv_mse:
                    best_cv_mse = avg_mse
                    best_lambda = lam

    # Refit on all data with best_lambda
    if 'best_lambda' in locals():
        V_all = V_full
        A = V_all.T @ V_all + best_lambda * np.eye(deg + 1)
        b_vec = V_all.T @ y
        try:
            c_std = np.linalg.solve(A, b_vec)
            if np.all(np.isfinite(c_std)):
                b_orig = _convert_coeffs_to_original(c_std, mu, sigma, deg)
                if np.all(np.isfinite(b_orig)):
                    return b_orig
        except Exception:
            pass

    # Fallback: Unregularized fit on scaled data, then convert
    try:
        # Use lstsq for fallback
        c_std, _, _, _ = np.linalg.lstsq(V_full, y, rcond=None)
        if np.all(np.isfinite(c_std)):
            b_orig = _convert_coeffs_to_original(c_std, mu, sigma, deg)
            if np.all(np.isfinite(b_orig)):
                return b_orig
    except Exception:
        pass

    # Final fallback: Mean
    return np.array([np.mean(y)] + [0.0]*5)

# EVOLVE-BLOCK-END
