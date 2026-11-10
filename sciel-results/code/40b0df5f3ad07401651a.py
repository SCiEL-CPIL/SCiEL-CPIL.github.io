"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Handle constant x
    if np.std(x) < 1e-12:
        return np.array([np.mean(y)] + [0.0]*5)
    
    # Center and scale x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_norm = np.zeros(n)
    else:
        x_norm = (x - x_mean) / x_std
    
    # Build design matrix for degrees 0..5 in normalized basis
    # X_norm[:, k] = x_norm^k
    X_norm = np.vander(x_norm, N=6, increasing=True)
    
    # Estimate noise variance sigma^2 from residuals of a full degree-5 fit
    try:
        coeff_full, residuals, rank, s = np.linalg.lstsq(X_norm, y, rcond=None)
        if rank < 6:
            # Fallback to simpler fit if rank deficient
            coeff_full = np.zeros(6)
            coeff_full[:2] = np.polyfit(x_norm, y, 1)[::-1]
        y_pred_full = X_norm @ coeff_full
        residuals_sq = np.sum((y - y_pred_full)**2)
        dof = n - 6
        if dof > 0:
            sigma2 = residuals_sq / dof
        else:
            sigma2 = np.var(y)
        if sigma2 < 1e-12:
            sigma2 = 1e-12
    except:
        sigma2 = 1.0
    
    # Bayesian Linear Regression with BIC model selection
    # We test all subsets of degrees 0..5 (but always include degree 0)
    # For each model, compute BIC = n*log(SSR/n) + k*log(n)
    # We want to minimize BIC
    
    best_bic = np.inf
    best_coeff_norm = np.zeros(6)
    
    # Precompute X^T X and X^T y for each degree to speed up
    # Actually, for small n and k<=6, we can just do lstsq for each subset
    # But we need to be careful about time.
    # Number of subsets: 2^5 = 32 (excluding empty, and always include 0)
    # Actually, we can iterate over all 31 non-empty subsets of {0,1,2,3,4,5}
    # that include 0. That's 16 subsets.
    
    # Let's iterate over all masks from 1 to 31 (binary 00001 to 11111)
    # But we must ensure bit 0 (degree 0) is always set.
    # So masks: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
    # That's 16 models.
    
    for mask in range(1, 32):
        if mask & 1 == 0:
            continue  # Must include degree 0
        
        if time.time() > deadline:
            break
            
        # Extract columns
        cols = [i for i in range(6) if (mask >> i) & 1]
        k = len(cols)
        
        if k > n:
            continue  # Can't fit more parameters than data points
        
        X_sub = X_norm[:, cols]
        
        try:
            coeff_sub, res, rank_sub, s_sub = np.linalg.lstsq(X_sub, y, rcond=None)
            if rank_sub < k:
                # Rank deficient, skip or use pseudo-inverse
                continue
                
            y_pred = X_sub @ coeff_sub
            ssr = np.sum((y - y_pred)**2)
            
            if ssr < 1e-12:
                bic = 0  # Perfect fit, but unlikely
            else:
                bic = n * np.log(ssr / n) + k * np.log(n)
                
            if bic < best_bic:
                best_bic = bic
                # Map back to full 6-coeff vector
                best_coeff_norm = np.zeros(6)
                for i, col_idx in enumerate(cols):
                    best_coeff_norm[col_idx] = coeff_sub[i]
                    
        except:
            continue
            
    # If no model was selected (shouldn't happen), fallback to linear
    if np.all(best_coeff_norm == 0):
        try:
            # Fit degree 1 in normalized basis
            X_lin = X_norm[:, :2]
            coeff_lin, _, _, _ = np.linalg.lstsq(X_lin, y, rcond=None)
            best_coeff_norm[:2] = coeff_lin
        except:
            best_coeff_norm[0] = np.mean(y)
            
    # Now transform coefficients from normalized basis to original basis
    # We have: y = sum_{k=0}^5 c_k * ((x - x_mean)/x_std)^k
    # We want: y = sum_{i=0}^5 a_i * x^i
    # Let z = (x - x_mean)/x_std = (1/x_std)*x - (x_mean/x_std)
    # This is a polynomial transformation.
    # c_k * z^k = c_k * ( (x - x_mean)/x_std )^k
    # = c_k / x_std^k * (x - x_mean)^k
    # (x - x_mean)^k = sum_{j=0}^k binom(k, j) x^j (-x_mean)^(k-j)
    # So a_i = sum_{k=i}^5 c_k / x_std^k * binom(k, i) * (-x_mean)^(k-i)
    
    a = np.zeros(6)
    for k in range(6):
        c_k = best_coeff_norm[k]
        if c_k == 0:
            continue
        factor = c_k / (x_std ** k)
        for i in range(k + 1):
            # binom(k, i) * (-x_mean)^(k-i)
            binom = 1
            for j in range(i):
                binom = binom * (k - j) / (j + 1)
            # Actually, use math.comb or scipy if available, but let's do it manually
            # binom(k, i) = k! / (i! (k-i)!)
            # Let's compute it properly
            binom_val = 1
            for j in range(i):
                binom_val = binom_val * (k - j) // (j + 1)
            # Wait, integer division might fail. Let's use float.
            binom_val = 1.0
            for j in range(i):
                binom_val = binom_val * (k - j) / (j + 1)
                
            term = factor * binom_val * ((-x_mean) ** (k - i))
            a[i] += term
            
    # Ensure finite
    if not np.all(np.isfinite(a)):
        a = np.zeros(6)
        
    return a

# EVOLVE-BLOCK-END
