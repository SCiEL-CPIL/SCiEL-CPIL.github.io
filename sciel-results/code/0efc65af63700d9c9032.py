# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from math import comb

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved subgrid tendency.
    Uses Chebyshev basis for better conditioning and approximation properties.
    
    Args:
        training_x: flattened array of slow-state values
        training_subgrid_tendency: flattened array of corresponding subgrid tendencies
    
    Returns:
        6 finite real coefficients in ascending powers of x
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit (and 150s kill)
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Remove any NaN/Inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    
    if n == 0:
        return np.zeros(6)
    
    # Define mapping from x to t in [-1, 1]
    # Use min/max of training data. 
    # If range is too small, expand it.
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-12:
        x_min -= 1.0
        x_max += 1.0
        
    # Map to [-1, 1]
    t = (2 * x - (x_max + x_min)) / (x_max - x_min)
    
    # Build Chebyshev basis matrix T
    # T[:, k] = T_k(t)
    degree = 5
    T = np.zeros((n, degree + 1))
    T[:, 0] = 1.0
    T[:, 1] = t
    if degree >= 2:
        for k in range(2, degree + 1):
            T[:, k] = 2 * t * T[:, k-1] - T[:, k-2]
            
    # Precompute Gram matrix and T^T y
    G = T.T @ T
    ty = T.T @ y
    
    # Regularization: penalize higher-order Chebyshev terms
    # Chebyshev coefficients decay naturally for smooth functions.
    # Gentle regularization is usually sufficient.
    reg_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    R = np.diag(reg_weights)
    
    # Log-spaced regularization values
    # Wider range to find the optimal bias-variance tradeoff
    lambdas = np.logspace(-10, 5, 150)
    
    best_mse = np.inf
    best_lambda = 0.0
    best_coeffs_cheb = None
    
    for lam in lambdas:
        if time.time() - start_time > deadline - 10:
            break
            
        A = G + lam * R
        
        try:
            A_inv = np.linalg.inv(A)
            
            # Compute coefficients in Chebyshev basis
            coeffs_lam = A_inv @ ty
            
            if not np.all(np.isfinite(coeffs_lam)):
                continue
            
            # Compute fitted values
            y_pred = T @ coeffs_lam
            
            # Compute diagonal of Hat Matrix H = T (G + lambda R)^{-1} T^T
            # H_ii = sum_k (T_ik * (A_inv @ T.T)_ik)
            Temp = T @ A_inv
            H_diag = np.sum(Temp * T, axis=1)
            
            # LOOCV Error Formula
            denom = 1.0 - H_diag
            valid = np.abs(denom) > 1e-12
            if not np.all(valid):
                continue
                
            residuals = y - y_pred
            loocv_errors = (residuals / denom) ** 2
            loocv_mse = np.mean(loocv_errors)
            
            if np.isfinite(loocv_mse) and loocv_mse < best_mse:
                best_mse = loocv_mse
                best_lambda = lam
                best_coeffs_cheb = coeffs_lam.copy()
                
        except np.linalg.LinAlgError:
            continue
        except Exception:
            continue

    if best_coeffs_cheb is None:
        # Fallback to OLS on all data if LOOCV failed
        try:
            coeffs_cheb, _, _, _ = np.linalg.lstsq(T, y, rcond=None)
            if np.all(np.isfinite(coeffs_cheb)):
                best_coeffs_cheb = coeffs_cheb
            else:
                return np.zeros(6)
        except:
            return np.zeros(6)
    
    # Convert Chebyshev coefficients to Monomial coefficients
    # P(x) = sum_{k=0}^5 c_k T_k(t(x))
    # We evaluate this polynomial at 6 distinct points in x-space
    # and solve for the monomial coefficients a_i.
    
    # Choose 6 points in x range, e.g., Chebyshev nodes mapped back to x
    # This ensures good conditioning for the linear system for a_i
    nodes_t = np.cos(np.pi * (2 * np.arange(6) + 1) / (2 * 6))
    nodes_x = 0.5 * (x_max - x_min) * nodes_t + 0.5 * (x_max + x_min)
    
    # Evaluate Chebyshev polynomial at these nodes
    # We can use the recurrence again or just compute T_k(nodes_t)
    T_nodes = np.zeros((6, degree + 1))
    T_nodes[:, 0] = 1.0
    T_nodes[:, 1] = nodes_t
    if degree >= 2:
        for k in range(2, degree + 1):
            T_nodes[:, k] = 2 * nodes_t * T_nodes[:, k-1] - T_nodes[:, k-2]
            
    y_nodes = T_nodes @ best_coeffs_cheb
    
    # Build Vandermonde matrix for monomial basis at nodes_x
    V_nodes = np.zeros((6, degree + 1))
    for k in range(degree + 1):
        V_nodes[:, k] = nodes_x ** k
        
    try:
        coeff_monomial, _, _, _ = np.linalg.lstsq(V_nodes, y_nodes, rcond=None)
    except:
        # Fallback: direct conversion using recurrence is complex, 
        # so if lstsq fails, return zeros or simple fit
        coeff_monomial = np.zeros(6)
    
    # Ensure all coefficients are finite
    if not np.all(np.isfinite(coeff_monomial)):
        return np.zeros(6)
        
    return coeff_monomial
# EVOLVE-BLOCK-END
