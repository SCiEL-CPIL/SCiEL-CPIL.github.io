"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing
    of a two-scale Lorenz-96 model using Chebyshev basis regression.
    
    Args:
        training_x: Flattened array of slow-state values
        training_subgrid_tendency: Flattened array of unresolved tendencies
    
    Returns:
        Array of 6 finite real polynomial coefficients in ascending powers of x
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    # Filter out any NaN/inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 10:
        return np.zeros(6)
    
    # Determine the range of x to map to [-1, 1] for Chebyshev polynomials
    x_min = np.min(x)
    x_max = np.max(x)
    
    # Avoid division by zero if range is tiny
    if x_max - x_min < 1e-12:
        x_center = x_min
        x_scale = 1.0
    else:
        x_center = (x_max + x_min) / 2.0
        x_scale = (x_max - x_min) / 2.0
    
    # Map x to t in [-1, 1]
    t = (x - x_center) / x_scale
    
    # Generate Chebyshev design matrix T(t) for degree 0 to 5
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    def chebyshev_matrix(t_vals, degree):
        n = len(t_vals)
        T = np.zeros((n, degree + 1))
        T[:, 0] = 1.0
        if degree >= 1:
            T[:, 1] = t_vals
        for k in range(2, degree + 1):
            T[:, k] = 2.0 * t_vals * T[:, k-1] - T[:, k-2]
        return T
    
    degree = 5
    T = chebyshev_matrix(t, degree)
    
    # Solve least squares for Chebyshev coefficients
    # We use np.linalg.lstsq which is robust
    try:
        cheb_coeffs, residuals, rank, sv = np.linalg.lstsq(T, y, rcond=None)
    except Exception:
        # Fallback to simple mean if least squares fails
        return np.array([np.mean(y)] + [0.0]*5)
    
    # Convert Chebyshev coefficients to standard monomial coefficients
    # We want c_i such that sum(c_i * x^i) approximates the function.
    # The fitted function is sum(a_k * T_k(t(x))).
    # We need to express T_k(t(x)) as a polynomial in x.
    # t(x) = (x - center) / scale = alpha * x + beta
    # where alpha = 1/scale, beta = -center/scale.
    
    alpha = 1.0 / x_scale
    beta = -x_center / x_scale
    
    # We will compute the monomial coefficients for each T_k(alpha*x + beta)
    # and then sum them weighted by a_k.
    
    # Recurrence for Chebyshev polynomials in terms of monomials?
    # It's easier to just evaluate the polynomial conversion via basis transformation.
    # Let P(x) = sum_{k=0}^D a_k T_k(t(x)).
    # We can find the monomial coefficients by solving a linear system
    # or by using the property that T_k(z) is a polynomial in z of degree k.
    
    # Let's compute the monomial coefficients of T_k(alpha*x + beta) for k=0..5
    # T_0(z) = 1
    # T_1(z) = z
    # T_k(z) = 2*z*T_{k-1}(z) - T_{k-2}(z)
    
    # We represent each T_k(alpha*x + beta) as a vector of monomial coeffs [c_0, c_1, ..., c_5]
    monomial_coeffs = np.zeros((degree + 1, degree + 1))
    
    # T_0(alpha*x + beta) = 1
    monomial_coeffs[0, 0] = 1.0
    
    # T_1(alpha*x + beta) = alpha*x + beta
    monomial_coeffs[1, 0] = beta
    monomial_coeffs[1, 1] = alpha
    
    for k in range(2, degree + 1):
        # T_k(z) = 2*z*T_{k-1}(z) - T_{k-2}(z)
        # z = alpha*x + beta
        # 2*z*T_{k-1}(z) = 2*(alpha*x + beta) * sum_{j=0}^{k-1} monomial_coeffs[k-1, j] x^j
        #                 = sum_{j=0}^{k-1} 2*alpha*monomial_coeffs[k-1, j] x^{j+1} 
        #                   + sum_{j=0}^{k-1} 2*beta*monomial_coeffs[k-1, j] x^j
        
        # Initialize temp array
        temp = np.zeros(degree + 1)
        
        # Contribution from 2*beta * T_{k-1}(z)
        for j in range(k):
            temp[j] += 2.0 * beta * monomial_coeffs[k-1, j]
        
        # Contribution from 2*alpha * x * T_{k-1}(z)
        for j in range(k):
            temp[j+1] += 2.0 * alpha * monomial_coeffs[k-1, j]
        
        # Subtract T_{k-2}(z)
        for j in range(k-1):
            temp[j] -= monomial_coeffs[k-2, j]
        
        monomial_coeffs[k] = temp
    
    # Now, the total monomial coefficients are:
    # c_total = sum_{k=0}^D a_k * monomial_coeffs[k]
    final_coeffs = np.zeros(degree + 1)
    for k in range(degree + 1):
        final_coeffs += cheb_coeffs[k] * monomial_coeffs[k]
    
    # Check for numerical stability
    if not np.all(np.isfinite(final_coeffs)):
        # If conversion failed due to numerical issues, fall back to a simpler fit
        # Try a lower degree Chebyshev fit
        try:
            T_low = chebyshev_matrix(t, 2)
            cheb_coeffs_low, _, _, _ = np.linalg.lstsq(T_low, y, rcond=None)
            
            # Convert degree 2
            mono_low = np.zeros((3, 3))
            mono_low[0, 0] = 1.0
            mono_low[1, 0] = beta
            mono_low[1, 1] = alpha
            for k in range(2, 3):
                temp = np.zeros(3)
                for j in range(k):
                    temp[j] += 2.0 * beta * mono_low[k-1, j]
                for j in range(k):
                    temp[j+1] += 2.0 * alpha * mono_low[k-1, j]
                for j in range(k-1):
                    temp[j] -= mono_low[k-2, j]
                mono_low[k] = temp
            
            final_coeffs_low = np.zeros(3)
            for k in range(3):
                final_coeffs_low += cheb_coeffs_low[k] * mono_low[k]
            
            final_coeffs = np.zeros(6)
            final_coeffs[:3] = final_coeffs_low
        except Exception:
            final_coeffs = np.zeros(6)
            final_coeffs[0] = np.mean(y)
            
    return final_coeffs
# EVOLVE-BLOCK-END
