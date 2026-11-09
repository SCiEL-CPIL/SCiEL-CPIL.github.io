"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import comb


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n == 0:
        return np.zeros(6)

    # Remove any non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n == 0:
        return np.zeros(6)

    # Center and scale x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0

    # We'll fit in scaled space: z = (x - x_mean) / x_std
    # Polynomial in z: p(z) = sum c_i_scaled * z^i
    # Then convert back to polynomial in x

    z = (x - x_mean) / x_std

    def check_time():
        return time.time() < deadline - 5

    def convert_to_x_space(c_scaled, degree):
        """Convert polynomial coefficients from z-space to x-space."""
        coeff_x = np.zeros(degree + 1)
        for i in range(degree + 1):
            if c_scaled[i] == 0:
                continue
            ci = c_scaled[i] / (x_std ** i)
            # (x - x_mean)^i = sum_{k=0}^i C(i,k) * x^k * (-x_mean)^(i-k)
            for k in range(i + 1):
                coeff_x[k] += ci * comb(i, k) * ((-x_mean) ** (i - k))
        return coeff_x

    def evaluate_norm_err(coeff_x, x, y):
        """Compute normalized MSE."""
        pred = np.zeros_like(y)
        for i in range(len(coeff_x)):
            pred += coeff_x[i] * (x ** i)
        mse = np.mean((pred - y) ** 2)
        var_y = np.var(y)
        if var_y > 1e-12:
            return mse / var_y
        else:
            return mse

    # Use SVD-based Total Least Squares with truncation
    # This provides principled regularization by keeping only significant singular values
    best_coeff = None
    best_val = np.inf

    # Try different degrees with SVD truncation
    for degree in [3, 4, 5]:
        if not check_time():
            break
        try:
            # Build Vandermonde matrix in scaled space
            V = np.vander(z, N=degree + 1, increasing=True)
            
            # SVD decomposition
            U, s, Vt = np.linalg.svd(V, full_matrices=False)
            
            # Try different truncation levels (number of singular values to keep)
            max_rank = len(s)
            for rank in range(max(1, max_rank - 2), max_rank + 1):
                if not check_time():
                    break
                try:
                    # Truncated SVD: keep only top 'rank' singular values
                    # This acts as regularization - ignores small singular values
                    c_scaled = Vt[:rank, :].T @ (U[:, :rank].T @ y)
                    
                    # Convert to x-space
                    coeff_x = convert_to_x_space(c_scaled, degree)
                    
                    # Evaluate
                    norm_err = evaluate_norm_err(coeff_x, x, y)
                    
                    if np.all(np.isfinite(coeff_x)) and norm_err < best_val:
                        best_val = norm_err
                        best_coeff = np.zeros(6)
                        best_coeff[:degree + 1] = coeff_x[:degree + 1]
                except Exception:
                    continue
        except Exception:
            continue

    # If SVD approach didn't work well, try ridge regression as fallback
    if best_coeff is None or best_val > 1.0:
        for degree in [3, 4, 5]:
            if not check_time():
                break
            try:
                V = np.vander(z, N=degree + 1, increasing=True)
                A = V.T @ V
                b = V.T @ y
                
                # Try a few regularization strengths
                for lam in [1e-6, 1e-4, 1e-2, 0.1]:
                    if not check_time():
                        break
                    try:
                        reg = np.eye(degree + 1) * lam
                        reg[0, 0] = 0
                        c_scaled = np.linalg.solve(A + reg, b)
                        
                        coeff_x = convert_to_x_space(c_scaled, degree)
                        norm_err = evaluate_norm_err(coeff_x, x, y)
                        
                        if np.all(np.isfinite(coeff_x)) and norm_err < best_val:
                            best_val = norm_err
                            best_coeff = np.zeros(6)
                            best_coeff[:degree + 1] = coeff_x[:degree + 1]
                    except Exception:
                        continue
            except Exception:
                continue

    # Ensure all coefficients are finite
    if best_coeff is None or not np.all(np.isfinite(best_coeff)):
        # Fallback to simple polyfit
        try:
            best_coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            if not np.all(np.isfinite(best_coeff)):
                best_coeff = np.zeros(6)
        except Exception:
            best_coeff = np.zeros(6)

    # Final safety check
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = np.zeros(6)

    return best_coeff.astype(np.float64)
# EVOLVE-BLOCK-END
