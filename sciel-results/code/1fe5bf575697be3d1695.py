import numpy as np
import time
from math import comb

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved subgrid tendency
    using a scaled monomial basis with exact binomial expansion for
    coefficient conversion, avoiding numerical instability.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    
    if n == 0:
        return np.zeros(6)
    if n < 6:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, min(5, n-1))
            return np.pad(coeff, (0, 6 - len(coeff)))
        except:
            return np.zeros(6)
    
    # Scale x to improve conditioning: center and scale
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    
    # Fit in scaled basis: z = (x - x_mean) / x_std
    z = (x - x_mean) / x_std
    
    # Build Vandermonde matrix in z (ascending powers)
    V_z = np.vander(z, 6, increasing=True)
    
    # Tikhonov regularization with structured weights to penalize high-degree terms
    # This provides a smoothness prior without expensive CV
    reg_weights = np.array([1.0, 1.0, 4.0, 16.0, 64.0, 256.0])
    reg_matrix = np.diag(reg_weights)
    
    # Try a small set of regularization strengths
    best_mse = np.inf
    best_coeff_z = None
    
    # Use a holdout set for model selection instead of full CV (faster)
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    n_val = max(1, n // 5)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    
    lambdas = [0.0, 1e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0]
    
    for lam in lambdas:
        if time.time() - start_time > deadline - 10:
            break
        try:
            V_train = V_z[train_idx]
            y_train = y[train_idx]
            V_val = V_z[val_idx]
            y_val = y[val_idx]
            
            if lam == 0.0:
                coeff_z, _, _, _ = np.linalg.lstsq(V_train, y_train, rcond=None)
            else:
                VtV = V_train.T @ V_train
                Vty = V_train.T @ y_train
                A = VtV + lam * reg_matrix
                try:
                    coeff_z = np.linalg.solve(A, Vty)
                except np.linalg.LinAlgError:
                    coeff_z, _, _, _ = np.linalg.lstsq(A, Vty, rcond=None)
            
            if not np.all(np.isfinite(coeff_z)):
                continue
            
            y_pred = V_val @ coeff_z
            mse = np.mean((y_pred - y_val) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_coeff_z = coeff_z.copy()
        except:
            continue
    
    # If selection failed, use OLS on all data
    if best_coeff_z is None:
        try:
            best_coeff_z, _, _, _ = np.linalg.lstsq(V_z, y, rcond=None)
            if not np.all(np.isfinite(best_coeff_z)):
                return np.zeros(6)
        except:
            return np.zeros(6)
    
    # Convert from z-basis to x-basis using exact binomial expansion
    # z = (x - x_mean) / x_std = (1/x_std) * x - (x_mean/x_std)
    # Let p = 1/x_std, q = -x_mean/x_std
    # z = p*x + q
    # z^k = sum_{j=0}^{k} comb(k,j) * (p*x)^j * q^(k-j)
    #      = sum_{j=0}^{k} comb(k,j) * p^j * q^(k-j) * x^j
    #
    # P(x) = sum_{k=0}^{5} c_k * z^k
    #      = sum_{k=0}^{5} c_k * sum_{j=0}^{k} comb(k,j) * p^j * q^(k-j) * x^j
    #      = sum_{j=0}^{5} [sum_{k=j}^{5} c_k * comb(k,j) * p^j * q^(k-j)] * x^j
    
    p = 1.0 / x_std
    q = -x_mean / x_std
    
    coeff_orig = np.zeros(6)
    for j in range(6):
        for k in range(j, 6):
            coeff_orig[j] += best_coeff_z[k] * comb(k, j) * (p ** j) * (q ** (k - j))
    
    # Verify finiteness
    if not np.all(np.isfinite(coeff_orig)):
        # Fallback: direct OLS in monomial basis with scaling
        try:
            V = np.vander(x, 6, increasing=True)
            coeff_orig, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
            if not np.all(np.isfinite(coeff_orig)):
                coeff_orig = np.zeros(6)
        except:
            coeff_orig = np.zeros(6)
    
    if not np.all(np.isfinite(coeff_orig)):
        coeff_orig = np.where(np.isfinite(coeff_orig), coeff_orig, 0.0)
    
    return coeff_orig
