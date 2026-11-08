"""
Minimize the normalized maximum of the autoconvolution of a nonnegative step function.
The function is defined by 600 heights on equal intervals covering [-1/4, 1/4].
The normalized maximum is typically defined as max(f*f) / (integral(f))^2.
For nonnegative f, max(f*f) = (f*f)(0) = integral(f^2).
Thus the quantity to minimize is integral(f^2) / (integral(f))^2.
By the Cauchy-Schwarz inequality, this ratio is minimized when f is constant over its support.
To minimize the ratio further, we must maximize the length of the support.
The maximum support length is the full interval [-1/4, 1/4], which has length 0.5.
The minimum value of the ratio is 1/0.5 = 2.
The uniform distribution achieves this minimum.
"""
import time
import numpy as np

N = 600
T = 0.5  # Total length of the interval [-1/4, 1/4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sequence of N nonnegative step heights on equal intervals of [-1/4, 1/4]
    that minimizes the normalized maximum of its autoconvolution.
    
    Structural change: Formulate as a convex optimization problem and solve
    using scipy to verify the uniform solution is optimal. If scipy is available
    and time permits, solve the problem directly. Otherwise, return the analytically
    known optimal uniform solution.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Analytical result: uniform distribution minimizes sum(h^2)/(sum h)^2
    # for nonnegative h, achieving value 1/T = 2.0.
    # The uniform solution h_i = c for all i gives:
    #   sum(h^2) = N*c^2, sum(h) = N*c
    #   ratio = N*c^2 / (N*c)^2 = 1/(N*c) ... wait, need to include delta
    #   
    # Actually: (f*f)(0) = delta * sum(h_i^2), integral(f) = delta * sum(h_i)
    # normalized = delta * sum(h_i^2) / (delta * sum(h_i))^2
    #            = sum(h_i^2) / (delta * (sum h_i)^2)
    # For uniform h_i = c: = N*c^2 / (delta * N^2 * c^2) = 1/(delta*N) = 1/T = 2.0
    
    # Try to verify with scipy optimization if available
    try:
        from scipy.optimize import minimize
        if time.time() - start_time < deadline - 5:
            # Minimize sum(h^2) subject to sum(h) = 1, h >= 0
            # This is a convex QP; solution is h_i = 1/N
            n = N
            h0 = np.ones(n) / n
            
            def objective(h):
                return np.sum(h**2)
            
            def jac(h):
                return 2 * h
            
            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
            bounds = [(0, None)] * n
            
            result = minimize(objective, h0, jac=jac, constraints=constraints,
                             bounds=bounds, method='SLSQP',
                             options={'maxiter': 100, 'ftol': 1e-12})
            
            if result.success and time.time() - start_time < deadline - 2:
                # Verify the solution
                h_opt = np.maximum(result.x, 0)
                h_opt = h_opt / np.sum(h_opt)  # Normalize
                
                # Compute score
                delta = T / N
                sum_h2 = np.sum(h_opt**2)
                sum_h = np.sum(h_opt)
                score = sum_h2 / (delta * sum_h**2)
                
                # Compare with uniform
                h_uniform = np.ones(n)
                sum_h2_u = np.sum(h_uniform**2)
                sum_h_u = np.sum(h_uniform)
                score_uniform = sum_h2_u / (delta * sum_h_u**2)
                
                # Use the better one (should be same or close)
                if score <= score_uniform + 1e-10:
                    # Scale back: the ratio is scale-invariant, so return h_opt scaled
                    # Actually, return the heights as-is (scale doesn't matter for ratio)
                    return [float(x) for x in h_opt]
                else:
                    return [1.0] * N
    except Exception:
        pass
    
    # Fallback: return analytically optimal uniform solution
    return [1.0] * N
# EVOLVE-BLOCK-END
