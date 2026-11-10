"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import comb

def _fit_centered(x, y, degree, alpha):
    """
    Fit a degree-`degree` polynomial in centered/scaled x, with ridge regularization.
    Returns coefficients in the original x basis (ascending powers).
    """
    n = len(x)
    if n == 0:
        return np.zeros(degree + 1)
    
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        sigma = 1.0
    
    xc = (x - mu) / sigma
    
    # Build Vandermonde matrix for centered/scaled variable
    V = np.column_stack([xc**k for k in range(degree + 1)])
    
    A = V.T @ V
    b = V.T @ y
    
    if alpha > 0:
        reg = alpha * np.eye(A.shape[0])
        # Small diagonal for numerical safety
        reg[0, 0] += 1e-12
        try:
            c_centered = np.linalg.solve(A + reg, b)
        except np.linalg.LinAlgError:
            # Fallback to lstsq if solve fails
            c_centered, _, _, _ = np.linalg.lstsq(A + reg, b, rcond=None)
    else:
        try:
            c_centered = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            c_centered, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # Convert from centered/scaled basis to original basis
    # p(x) = sum_{k=0}^{degree} c_k * ((x-mu)/sigma)^k
    # Let z = (x-mu)/sigma => x = sigma*z + mu
    # p(x) = sum c_k * z^k
    # We need to express this in powers of x.
    
    coeff_orig = np.zeros(degree + 1)
    for k in range(degree + 1):
        if abs(c_centered[k]) < 1e-15:
            continue
        
        # Expand (a*x + b)^k where a = 1/sigma, b = -mu/sigma
        # (a*x + b)^k = sum_{i=0}^k binom(k, i) * (a*x)^i * b^(k-i)
        a = 1.0 / sigma
        b_val = -mu / sigma
        
        for i in range(k + 1):
            binom_k_i = comb(k, i)
            term_coeff = binom_k_i * (a ** i) * (b_val ** (k - i))
            coeff_orig[i] += c_centered[k] * term_coeff
            
    return coeff_orig

def _build_physics_basis(x):
    """
    Build a physics-informed sparse basis for Lorenz-96 unresolved forcing.
    The unresolved forcing has structure from cubic advection and mean-field terms.
    Returns a matrix where each column is a basis function evaluated at x.
    Basis: [1, x, x^2, x^3, x^4, x^5] but we'll also consider sparse subsets.
    """
    n = len(x)
    # Standard polynomial basis up to degree 5
    V = np.column_stack([x**k for k in range(6)])
    return V

def _ridge_solve(V, y, alpha):
    """
    Solve ridge regression: min ||V@c - y||^2 + alpha*||c||^2
    Returns coefficients.
    """
    n_features = V.shape[1]
    A = V.T @ V + alpha * np.eye(n_features)
    b = V.T @ y
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return c

def _l1_sparse_selection(V, y, alphas):
    """
    Try L1-like sparse selection by testing different subsets of basis functions.
    For small feature sets, we can enumerate or use a greedy approach.
    Returns best coefficients and selected feature indices.
    """
    n_features = V.shape[1]
    best_mse = np.inf
    best_coeff = np.zeros(n_features)
    best_features = list(range(n_features))
    
    # For 6 features, we can try all subsets of size >= 2
    # But that's 2^6 - 1 - 6 = 57 subsets, which is manageable
    from itertools import combinations
    
    for size in range(2, n_features + 1):
        for feat_idx in combinations(range(n_features), size):
            V_sub = V[:, list(feat_idx)]
            # Use ridge with small alpha for stability
            c_sub = _ridge_solve(V_sub, y, 1e-8)
            y_pred = V_sub @ c_sub
            mse = np.mean((y - y_pred) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_features = list(feat_idx)
                # Map back to full coefficient vector
                best_coeff = np.zeros(n_features)
                best_coeff[list(feat_idx)] = c_sub
    
    return best_coeff, best_features, best_mse

def _cross_validate(x, y, degree, alpha, n_folds=5, seed=42):
    """
    Cross-validate a polynomial fit.
    Returns mean validation MSE.
    """
    n = len(x)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    fold_size = n // n_folds
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(n_folds)]
    
    cv_errors = []
    for i in range(n_folds):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_val = x[val_idx]
        y_val = y[val_idx]
        
        try:
            c = _fit_centered(x_train, y_train, degree, alpha)
            y_pred = np.polynomial.polynomial.polyval(x_val, c)
            mse = np.mean((y_val - y_pred) ** 2)
            cv_errors.append(mse)
        except Exception:
            cv_errors.append(np.inf)
    
    return np.mean(cv_errors)

def _cross_validate_physics(x, y, n_folds=5, seed=42):
    """
    Cross-validate the physics-informed sparse basis fit.
    Returns mean validation MSE.
    """
    n = len(x)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    fold_size = n // n_folds
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(n_folds)]
    
    cv_errors = []
    for i in range(n_folds):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_val = x[val_idx]
        y_val = y[val_idx]
        
        try:
            V_train = _build_physics_basis(x_train)
            c, _, _ = _l1_sparse_selection(V_train, y_train, [1e-8])
            V_val = _build_physics_basis(x_val)
            y_pred = V_val @ c
            mse = np.mean((y_val - y_pred) ** 2)
            cv_errors.append(mse)
        except Exception:
            cv_errors.append(np.inf)
    
    return np.mean(cv_errors)

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    x = np.asarray(training_x, dtype=float).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=float).flatten()
    
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return np.zeros(6)
        
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 6:
        if len(x) >= 2:
            coeff = np.zeros(6)
            try:
                coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            except:
                coeff[0] = np.mean(y)
            return coeff
        elif len(x) >= 1:
            coeff = np.zeros(6)
            coeff[0] = np.mean(y)
            return coeff
        else:
            return np.zeros(6)

    best_coeff = np.zeros(6)
    
    n = len(x)
    if n > 50:
        # Compare two approaches:
        # 1. Standard polynomial with CV-selected degree and alpha
        # 2. Physics-informed sparse basis
        
        # Approach 1: Standard polynomial CV
        best_poly_cv = np.inf
        best_poly_params = (5, 1e-6)
        
        for deg in [5, 4, 3]:
            for alpha in [1e-6, 1e-4, 1e-2, 1.0]:
                cv_score = _cross_validate(x, y, deg, alpha)
                if cv_score < best_poly_cv:
                    best_poly_cv = cv_score
                    best_poly_params = (deg, alpha)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        
        deg, alpha = best_poly_params
        try:
            poly_coeff = _fit_centered(x, y, deg, alpha)
            if len(poly_coeff) < 6:
                padded = np.zeros(6)
                padded[:len(poly_coeff)] = poly_coeff
                poly_coeff = padded
        except Exception:
            poly_coeff = np.zeros(6)
        
        # Approach 2: Physics-informed sparse basis CV
        try:
            phys_cv_score = _cross_validate_physics(x, y)
            V_full = _build_physics_basis(x)
            phys_coeff, _, _ = _l1_sparse_selection(V_full, y, [1e-8])
        except Exception:
            phys_cv_score = np.inf
            phys_coeff = np.zeros(6)
        
        # Choose the better approach based on CV score
        if phys_cv_score < best_poly_cv:
            best_coeff = phys_coeff
        else:
            best_coeff = poly_coeff
    else:
        # Small sample, use a robust fit
        try:
            best_coeff = _fit_centered(x, y, 5, 1e-2)
        except Exception:
            try:
                best_coeff = np.zeros(6)
                best_coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            except:
                best_coeff = np.zeros(6)
                
    # Ensure finite
    if not np.all(np.isfinite(best_coeff)):
        try:
            best_coeff = np.zeros(6)
            best_coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except:
            best_coeff = np.zeros(6)
            
    return best_coeff[:6]
# EVOLVE-BLOCK-END
