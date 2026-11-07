"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Strategy: Self-Contained Multi-Start Local Search
#
# The previous strategy relied on the external BFGS optimizer to fit a fixed
# linear basis. While stable, it may lack the capacity to fit complex functions
# if the basis is not rich enough, or the optimizer might converge to a suboptimal
# solution if the basis is ill-conditioned.
#
# This strategy takes full control of the optimization process inside `equation`.
# 1. We define a flexible model: a combination of polynomial terms (up to degree 6)
#    and trigonometric terms (sin/cos with varying frequencies). This provides a
#    rich function class capable of approximating a wide range of smooth functions.
# 2. We perform a multi-start local search using scipy.optimize.minimize (BFGS).
# 3. We use a strict time budget (e.g., 10 seconds) to ensure we return before
#    the 120s evaluator limit, leaving ample margin.
# 4. We cache the best parameters found to speed up repeated evaluations if the
#    evaluator calls `equation` multiple times with the same data (though the
#    contract implies a single fit per evaluation, caching is safe).
#
# Model Form:
# y(x) = sum_{i=0}^{P} p_i * x^i + sum_{j=1}^{Q} (p_{P+j} * sin(j*w*x) + p_{P+Q+j} * cos(j*w*x))
# To keep the parameter count manageable and optimization stable, we limit the
# number of terms. Let's use:
# - Polynomial: degree 0 to 5 (6 params)
# - Trig: 3 frequencies (3 sin + 3 cos = 6 params)
# Total params: 12. But we only have 10 params available from the evaluator?
# Wait, the evaluator passes `params` of length 10.
#
# CRITICAL CONSTRAINT: The evaluator fits `params` (length 10) using BFGS.
# My `equation(x, params)` receives these 10 params.
# I CANNOT change the number of params the evaluator fits.
#
# RE-EVALUATION:
# The evaluator calls `minimize(fun, x0, method='BFGS')`.
# `fun` is a wrapper that calls `equation(x, params)`.
# `params` is always length 10.
#
# If I want to do multi-start search INSIDE `equation`, I have a problem:
# `equation` is called with a specific `params` vector by the optimizer.
# It does not have access to the full dataset `y` to perform its own optimization.
# It only sees `x` and `params`.
#
# Therefore, I CANNOT perform a full local search inside `equation` because I don't have `y`.
# The `equation` function is just a forward pass.
#
# So, the "self-contained local search" inspiration is likely referring to a different
# interpretation or a mistake in the inspiration description.
#
# Let's look at Inspiration 1 again: "Replace the fixed, linear basis expansion... with a self-contained multi-start local search inside the equation".
# This is impossible if `equation` doesn't receive `y`.
#
# UNLESS... the "equation" function is allowed to do heavy lifting and the "params" are just part of the state?
# No, the contract says: "The evaluator fits params... minimizing the training MSE."
# This implies the evaluator drives the optimization of `params`.
#
# So, how can I improve?
# 1. Better Basis: The current RBF basis might be too narrow or too wide.
# 2. Non-Linear Params: If I make the model non-linear in params, BFGS might struggle.
#    The current model is linear in params (amplitudes of fixed RBFs). This is GOOD for BFGS.
#    Why is the score only 85?
#    Maybe the basis is not rich enough. 10 RBFs with fixed centers/widths might not capture the function.
#
# Alternative Strategy:
# Use a **Neural Network-like** structure but keep it linear in the final layer?
# Or, use **Spline** basis functions which are more flexible than RBFs for smooth functions.
# Or, increase the flexibility of the RBFs by allowing the 10 params to control
# BOTH amplitudes and centers?
# If I use 5 RBFs, I can use 5 params for amplitudes and 5 for centers.
# This makes the problem non-linear in params (because centers are in the exponent).
# BFGS can handle non-linear problems, but it's harder.
#
# Let's try a **Hybrid Basis**:
# 1. Polynomial terms (global) to capture the trend.
# 2. RBF terms (local) to capture deviations.
#
# Params (10 total):
# - 4 Polynomial coeffs (x^0, x^1, x^2, x^3)
# - 6 RBF amplitudes (3 RBFs, fixed centers/widths)
#
# This is still linear in params!
# y = c0 + c1*x + c2*x^2 + c3*x^3 + sum(a_i * RBF_i(x))
#
# Why might this be better?
# Polynomials are better at capturing global trends than RBFs alone, which can suffer from "bumpiness" or boundary effects.
#
# Let's refine the basis:
# Polynomial: degree 0 to 4 (5 params)
# RBF: 5 RBFs (5 params)
# Total: 10 params.
#
# RBF Centers: Uniformly spaced in [-3, 3].
# RBF Widths: Fixed, moderate.
#
# This hybrid basis should be more robust than pure RBF or pure polynomial.

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Polynomial + RBF Model.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters.
            params[0:5] = Polynomial coefficients for [1, x, x^2, x^3, x^4]
            params[5:10] = Amplitudes for 5 fixed RBFs
            
    Returns:
        Prediction array of shape (N,).
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # 1. Polynomial Part
    # p[0] + p[1]*x + p[2]*x^2 + p[3]*x^3 + p[4]*x^4
    poly_part = (p[0] + 
                 p[1] * x + 
                 p[2] * x**2 + 
                 p[3] * x**3 + 
                 p[4] * x**4)
                 
    # 2. RBF Part
    # 5 RBFs
    N_RBF = 5
    # Centers uniformly spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, N_RBF)
    # Fixed width
    sigma = 0.8  # Wider than before to allow more overlap with polynomial
    sigma_sq_2 = 2.0 * sigma**2
    
    # Compute RBF values
    # x shape: (N,)
    # centers shape: (5,)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-(diff**2) / sigma_sq_2)
    
    # rbf_values shape: (N, 5)
    # p[5:10] shape: (5,)
    rbf_part = rbf_values @ p[5:10]
    
    result = poly_part + rbf_part
    
    return result

# EVOLVE-BLOCK-END
