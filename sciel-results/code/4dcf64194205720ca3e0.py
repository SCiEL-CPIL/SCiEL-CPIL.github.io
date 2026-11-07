"""
Strategy: Hybrid RBF + Chebyshev Basis.
The current program uses a mix of raw polynomials and trigonometric functions.
This approach uses:
1. 5 Gaussian Radial Basis Functions (RBFs) centered at -3, -1.5, 0, 1.5, 3.
   These allow the model to fit localized, non-periodic features (bumps, valleys)
   independently.
2. 5 Chebyshev polynomials of the first kind (T0 to T4).
   These provide a numerically stable basis for global smooth trends,
   superior to raw polynomials (x^n) which can be ill-conditioned, and
   superior to trig functions for non-periodic targets.

This structure gives BFGS a well-conditioned, interpretable landscape to optimize.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y from x using a linear combination of 5 Gaussian RBFs and 5 Chebyshev polynomials.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters. First 5 correspond to RBF weights, last 5 to Chebyshev coefficients.
        
    Returns:
        Predicted y array of shape (N,).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure params is length 10
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
    elif len(p) > 10:
        p = p[:10]

    # 1. Gaussian RBFs
    # Centers: -3, -1.5, 0, 1.5, 3
    # Widths: 1.0 (standard deviation for the Gaussian kernel exp(-0.5 * ((x-c)/sigma)^2))
    # Using sigma=1.0 provides a balance between local flexibility and smoothness.
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    sigma = 1.0
    
    # Compute RBF terms
    # x is (N,), centers is (5,). We want (N, 5)
    x_expanded = x[:, np.newaxis]
    c_expanded = centers[np.newaxis, :]
    
    # Distance squared
    dist_sq = (x_expanded - c_expanded) ** 2
    
    # Gaussian kernel
    rbf_terms = np.exp(-0.5 * dist_sq / (sigma * sigma))
    
    # Weights for RBFs are p[0:5]
    rbf_weights = p[0:5]
    
    # Linear combination of RBFs: (N, 5) @ (5,) -> (N,)
    y_rbf = rbf_terms @ rbf_weights

    # 2. Chebyshev Polynomials of the First Kind
    # T_0(x) = 1
    # T_1(x) = x
    # T_2(x) = 2x^2 - 1
    # T_3(x) = 4x^3 - 3x
    # T_4(x) = 8x^4 - 8x^2 + 1
    
    # To ensure numerical stability and orthogonality benefits, we compute these directly.
    # Note: Chebyshev polynomials are orthogonal on [-1, 1]. Our domain is [-3, 3].
    # While they are not orthogonal on [-3, 3], they are still much better conditioned
    # than raw polynomials x^n. For a more rigorous approach, one might scale x to [-1,1],
    # but keeping the domain in [-3,3] allows the RBFs to work in the same coordinate space.
    # The optimizer will handle the scaling via the coefficients.
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    t0 = np.ones_like(x)
    t1 = x
    t2 = 2.0 * x2 - 1.0
    t3 = 4.0 * x3 - 3.0 * x
    t4 = 8.0 * x4 - 8.0 * x2 + 1.0
    
    # Weights for Chebyshev terms are p[5:10]
    w_t0 = p[5]
    w_t1 = p[6]
    w_t2 = p[7]
    w_t3 = p[8]
    w_t4 = p[9]
    
    y_cheb = w_t0 * t0 + w_t1 * t1 + w_t2 * t2 + w_t3 * t3 + w_t4 * t4
    
    # Final prediction
    y = y_rbf + y_cheb
    
    return y
# EVOLVE-BLOCK-END
