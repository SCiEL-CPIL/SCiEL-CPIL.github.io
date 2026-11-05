# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

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
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-12:
        x_min -= 1.0
        x_max += 1.0
        
    # Map to [-1, 1]
    t = (2 * x - (x_max + x_min)) / (x_max - x_min)
    
    # Build Chebyshev basis matrix T
    degree = 5
    T = np.zeros((n, degree + 1))
    T[:, 0] = 1.0
    T[:, 1] = t
    for k in range(2, degree + 1):
        T[:, k] = 2 * t * T[:, k-1] - T[:, k-2]
        
    # Precompute Gram matrix and T^T y
    G = T.T @ T
    ty = T.T @ y
    
    # Regularization: penalize higher-order Chebyshev terms more heavily
    # This reflects the prior that higher-order terms are less reliable
    reg_weights = np.array([1.0, 1.0, 4.0, 16.0, 64.0, 256.0])
    R = np.diag(reg_weights)
    
    # Log-spaced regularization values - wider range
    lambdas = np.logspace(-8, 6, 200)
    
    best_mse = np.inf
    best_lambda = 0.0
    best_coeffs_cheb = None
    
    for lam in lambdas:
        if time.time() - start_time > deadline - 10:
            break
            
        A = G + lam * R
        
        try:
            # Use solve instead of inv for better numerical stability
            coeffs_lam = np.linalg.solve(A, ty)
            
            if not np.all(np.isfinite(coeffs_lam)):
                continue
            
            # Compute fitted values
            y_pred = T @ coeffs_lam
            
            # Compute diagonal of Hat Matrix H = T A^{-1} T^T
            # H_ii = sum_k (T_ik * (A^{-1} T^T)_ik)
            # More stable: compute A^{-1} T^T first, then element-wise product
            AT_inv = np.linalg.solve(A, T.T)  # A^{-1} T^T
            H_diag = np.sum(T * AT_inv.T, axis=1)
            
            # LOOCV Error Formula
            denom = 1.0 - H_diag
            valid = np.abs(denom) > 1e-10
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
    # Using direct algebraic conversion via Chebyshev-to-monomial conversion matrix
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    # T_5(t) = 16t^5 - 20t^3 + 5t
    #
    # t = (2x - (x_max+x_min)) / (x_max-x_min) = alpha*x + beta
    # where alpha = 2/(x_max-x_min), beta = -(x_max+x_min)/(x_max-x_min)
    #
    # We need to express sum_k c_k T_k(alpha*x + beta) in powers of x.
    # This is a linear transformation: monomial_coeffs = M @ cheb_coeffs
    # where M is a 6x6 matrix we can compute numerically.
    
    alpha = 2.0 / (x_max - x_min)
    beta = -(x_max + x_min) / (x_max - x_min)
    
    # Build conversion matrix M such that:
    # P(x) = sum_i m_i * x^i = sum_k c_k * T_k(alpha*x + beta)
    #
    # We evaluate the Chebyshev polynomial at 6 well-chosen points and solve.
    # Use Chebyshev nodes in t-space for good conditioning.
    nodes_t = np.cos(np.pi * (2 * np.arange(6) + 1) / (2 * 6))
    nodes_x = (nodes_t - beta) / alpha  # t = alpha*x + beta => x = (t - beta)/alpha
    
    # Evaluate Chebyshev polynomial at these nodes
    T_nodes = np.zeros((6, degree + 1))
    T_nodes[:, 0] = 1.0
    T_nodes[:, 1] = nodes_t
    for k in range(2, degree + 1):
        T_nodes[:, k] = 2 * nodes_t * T_nodes[:, k-1] - T_nodes[:, k-2]
        
    y_nodes = T_nodes @ best_coeffs_cheb
    
    # Build Vandermonde matrix for monomial basis at nodes_x
    V_nodes = np.zeros((6, degree + 1))
    for k in range(degree + 1):
        V_nodes[:, k] = nodes_x ** k
        
    try:
        coeff_monomial = np.linalg.solve(V_nodes, y_nodes)
    except np.linalg.LinAlgError:
        try:
            coeff_monomial, _, _, _ = np.linalg.lstsq(V_nodes, y_nodes, rcond=None)
        except:
            coeff_monomial = np.zeros(6)
    
    # Ensure all coefficients are finite
    if not np.all(np.isfinite(coeff_monomial)):
        return np.zeros(6)
        
    return coeff_monomial
# EVOLVE-BLOCK-END
