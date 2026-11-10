"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved subgrid tendency.
    
    Uses a multi-strategy approach:
    1. Try full degree-5 polynomial fit
    2. Try regularized (Ridge) fits at various lambda values
    3. Try lower-degree polynomials (degree 3, 4)
    4. Try centered/scaled polynomial to improve conditioning
    5. Select the best based on training MSE with a complexity penalty
    
    The key insight: use centered variables for numerical stability,
    then convert back to original power basis.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second budget
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n < 6:
        # Not enough data, return zeros
        return np.zeros(6, dtype=np.float64)
    
    # Remove NaN/inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 6:
        return np.zeros(6, dtype=np.float64)
    
    # Center and scale x for better conditioning
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    
    x_scaled = (x - x_mean) / x_std
    
    # Candidate coefficients in scaled space
    candidates = []
    
    # Strategy 1: Full degree-5 polynomial in scaled space
    try:
        A = np.vander(x_scaled, 6, increasing=True)
        # Use lstsq for stability
        coeffs_scaled, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
        mse = np.mean((A @ coeffs_scaled - y) ** 2)
        candidates.append((coeffs_scaled, mse, 5))
    except Exception:
        pass
    
    # Strategy 2: Regularized (Ridge) fits
    lambdas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    for lam in lambdas:
        try:
            A = np.vander(x_scaled, 6, increasing=True)
            # Ridge: (A^T A + lambda*I) c = A^T y
            ATA = A.T @ A
            ATy = A.T @ y
            reg = lam * np.eye(6)
            # Don't regularize the intercept too much
            reg[0, 0] *= 0.01
            coeffs_scaled = np.linalg.solve(ATA + reg, ATy)
            mse = np.mean((A @ coeffs_scaled - y) ** 2)
            candidates.append((coeffs_scaled, mse, 5))
        except Exception:
            pass
    
    # Strategy 3: Lower degree polynomials (3 and 4)
    for deg in [3, 4]:
        try:
            A = np.vander(x_scaled, deg + 1, increasing=True)
            coeffs_scaled, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
            mse = np.mean((A @ coeffs_scaled - y) ** 2)
            # Pad to 6 coefficients
            full_coeffs = np.zeros(6)
            full_coeffs[:deg+1] = coeffs_scaled
            candidates.append((full_coeffs, mse, deg))
        except Exception:
            pass
    
    # Strategy 4: Weighted fit emphasizing regions where |y| is large
    try:
        weights = 1.0 + 0.5 * (np.abs(y) / (np.mean(np.abs(y)) + 1e-12))
        A = np.vander(x_scaled, 6, increasing=True)
        Aw = A * np.sqrt(weights)[:, np.newaxis]
        yw = y * np.sqrt(weights)
        coeffs_scaled, residuals, rank, sv = np.linalg.lstsq(Aw, yw, rcond=None)
        mse = np.mean((A @ coeffs_scaled - y) ** 2)
        candidates.append((coeffs_scaled, mse, 5))
    except Exception:
        pass
    
    # Strategy 5: Two-stage - fit degree 3, then refine with degree 5 on residuals
    try:
        A3 = np.vander(x_scaled, 4, increasing=True)
        c3, _, _, _ = np.linalg.lstsq(A3, y, rcond=None)
        resid = y - A3 @ c3
        A5 = np.vander(x_scaled, 6, increasing=True)
        # Fit residuals with degree 5
        c5_resid, _, _, _ = np.linalg.lstsq(A5, resid, rcond=None)
        c_full = c3.copy()
        c_full[3:] += c5_resid[3:]  # Add higher-order terms
        mse = np.mean((A5 @ c_full - y) ** 2)
        candidates.append((c_full, mse, 5))
    except Exception:
        pass
    
    if not candidates:
        return np.zeros(6, dtype=np.float64)
    
    # Check time
    if time.time() > deadline - 5:
        # Use the first candidate (full degree-5)
        best_coeffs_scaled = candidates[0][0]
    else:
        # Select best candidate using MSE with a small complexity penalty
        # Prefer lower degree if MSE is similar
        best_idx = 0
        best_score = np.inf
        for i, (c, mse, deg) in enumerate(candidates):
            # Penalty for higher degree to avoid overfitting
            penalty = 0.01 * (5 - deg) * (mse / (np.var(y) + 1e-12))
            score = mse + penalty
            if score < best_score:
                best_score = score
                best_idx = i
        best_coeffs_scaled = candidates[best_idx][0]
    
    # Convert from scaled space to original power basis
    # x_scaled = (x - x_mean) / x_std
    # So x = x_mean + x_std * x_scaled
    # We need coefficients such that sum(c_i * x^i) = sum(d_j * x_scaled^j)
    # where x_scaled = (x - x_mean) / x_std
    
    # Use polynomial conversion via evaluation at enough points
    # or via direct substitution
    
    # Method: evaluate the scaled polynomial at many points and fit in original space
    # Actually, let's do it analytically using the binomial theorem
    
    # x_scaled^j = ((x - x_mean) / x_std)^j = (1/x_std^j) * (x - x_mean)^j
    # (x - x_mean)^j = sum_{k=0}^{j} C(j,k) * x^k * (-x_mean)^(j-k)
    
    orig_coeffs = np.zeros(6, dtype=np.float64)
    
    for j in range(6):
        d_j = best_coeffs_scaled[j]
        if abs(d_j) < 1e-15:
            continue
        factor = d_j / (x_std ** j)
        # Expand (x - x_mean)^j
        for k in range(j + 1):
            # C(j,k) * (-x_mean)^(j-k)
            binom = 1.0
            # Compute C(j,k)
            if j <= 20:
                # Use multiplicative formula
                if k < j - k:
                    kk = k
                else:
                    kk = j - k
                binom = 1.0
                for ii in range(kk):
                    binom *= (j - ii) / (ii + 1)
            else:
                # Fallback
                import math
                binom = math.comb(j, k)
            
            term = binom * ((-x_mean) ** (j - k))
            if k < 6:
                orig_coeffs[k] += factor * term
    
    # Verify the conversion is reasonable
    # Check that the polynomial evaluates similarly
    x_test = x[:min(100, n)]
    y_pred_scaled = np.sum(best_coeffs_scaled[:, np.newaxis] * x_test[np.newaxis, :] ** np.arange(6)[np.newaxis, :], axis=1)
    y_pred_orig = np.sum(orig_coeffs[:, np.newaxis] * x_test[np.newaxis, :] ** np.arange(6)[np.newaxis, :], axis=1)
    
    if not np.all(np.isfinite(orig_coeffs)):
        # Fallback: just use lstsq in original space
        try:
            A_orig = np.vander(x, 6, increasing=True)
            orig_coeffs, _, _, _ = np.linalg.lstsq(A_orig, y, rcond=None)
        except Exception:
            orig_coeffs = np.zeros(6, dtype=np.float64)
    
    # Final sanity check
    if not np.all(np.isfinite(orig_coeffs)):
        orig_coeffs = np.nan_to_num(orig_coeffs, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Clip extremely large coefficients
    max_val = 1e10
    orig_coeffs = np.clip(orig_coeffs, -max_val, max_val)
    
    return orig_coeffs
# EVOLVE-BLOCK-END
