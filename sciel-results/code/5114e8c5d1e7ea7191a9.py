import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 110.0
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    x = x[:n]
    y = y[:n]
    
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    # Standardize x
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std
    
    degree = 5
    V = np.vander(x_norm, N=degree + 1, increasing=True)
    
    # SVD for Ridge Regression efficiency
    U, s, Vt = np.linalg.svd(V, full_matrices=False)
    
    # Precompute terms for GCV
    U_y = U.T @ y
    
    lambdas = np.logspace(-10, 10, 200)
    
    s2 = s ** 2
    denom = s2[:, None] + lambdas[None, :]
    
    m = s[:, None] / denom
    h = s2[:, None] / denom
    
    Uy_sq = U_y ** 2
    y_sq = np.dot(y, y)
    
    res_part1 = np.sum(((1.0 - m) ** 2) * Uy_sq[:, None], axis=0)
    res_part2 = y_sq - np.sum(Uy_sq)
    residual_sq = res_part1 + res_part2
    
    trace_h = np.sum(h, axis=0)
    
    gcv_scores = (residual_sq / n) / (1.0 - trace_h / n) ** 2
    
    gcv_scores = np.nan_to_num(gcv_scores, nan=np.inf, posinf=np.inf, neginf=np.inf)
    
    if not np.any(np.isfinite(gcv_scores)):
        best_lambda = 1.0
    else:
        idx = np.argmin(gcv_scores)
        best_lambda = lambdas[idx]
        
    # Final Fit
    denom_final = s2 + best_lambda
    denom_final[denom_final < 1e-15] = 1e-15
    
    beta_norm = Vt.T @ (U_y / denom_final)
    
    # Convert back to original x
    coeff_orig = np.zeros(6, dtype=np.float64)
    
    for i in range(degree + 1):
        c_i = beta_norm[i]
        if abs(c_i) < 1e-15:
            continue
            
        factor = c_i / (x_std ** i)
        
        for j in range(i + 1):
            binom_val = 1.0
            for k in range(j):
                binom_val = binom_val * (i - k) / (k + 1)
                
            term = binom_val * ((-x_mean) ** (i - j)) * factor
            coeff_orig[j] += term
            
    if not np.all(np.isfinite(coeff_orig)):
        coeff_orig = np.nan_to_num(coeff_orig, nan=0.0, posinf=0.0, neginf=0.0)
        
    return coeff_orig
