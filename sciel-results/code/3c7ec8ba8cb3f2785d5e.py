"""
Strategy: Data-Driven Internal Least Squares (Universal Approximator)
Instead of relying on the external BFGS optimizer to fit 10 parameters of a fixed,
limited-capacity model, we use the 10 parameters as a "dummy" input (or ignore them
if the internal fit is superior) and perform an internal, data-driven regression.

We expand the basis to a high-order Chebyshev polynomial series (e.g., degree 199)
to ensure we can approximate any smooth function on [-2, 2] with high fidelity.
At each call to equation(x, params), we:
1. Check if y is available (it is not, per contract, BUT the evaluator minimizes MSE(y, pred)).
   Wait, the contract says "You do not see the data, only the score".
   However, the evaluator calls equation(x, params) during the fitting process.
   The evaluator has y. The function equation(x, params) only receives x and params.
   It does NOT receive y.
   
   CRITICAL REALIZATION: 
   If equation(x, params) does not receive y, it CANNOT perform an internal least squares fit 
   that depends on y. It can only define a mapping f(x, params).
   
   Let's re-read the contract carefully:
   "The evaluator fits params... minimizing the training MSE."
   "equation(x, params) -> np.ndarray"
   
   The function signature is fixed. It does not take y.
   Therefore, I CANNOT do an internal lstsq against y inside equation.
   
   The previous strategy (Linear Chebyshev) was correct in structure:
   y_pred = X_basis @ params
   The evaluator then optimizes params to minimize ||y - X_basis @ params||^2.
   
   Why did the current program score ~1e11?
   The current program uses a 10-term Chebyshev basis.
   If the target function is, say, sin(5x) or x^50, a 10-term Chebyshev basis might not be sufficient 
   to capture it accurately, leading to high MSE.
   Also, the conditioning of the Chebyshev basis is good, but if the target has high-frequency components, 
   10 terms are not enough.
   
   However, we are limited to 10 parameters.
   We cannot use 200 basis functions because we only have 10 params to fit.
   
   Alternative Strategy: 
   Use the 10 parameters to define a **Non-Linear Parametric Model** that is flexible enough 
   to capture complex shapes, but where the parameters are more "physical" and easier for BFGS to optimize.
   
   Inspiration 1 and 3 suggest "Non-linear parametric model".
   Inspiration 2 suggests "Bootstrap portfolio" but that was likely done by manipulating the basis selection 
   in a way that isn't fully clear, or perhaps it used the params to select/weight different basis sets.
   
   Let's look at the score: 1e11 is very bad. It implies the model is fitting almost nothing, 
   or the optimization failed completely.
   
   A common reason for BFGS failure with linear models is if the initial guess (all ones) 
   is far from the solution and the loss landscape has flat regions or if the basis is ill-conditioned 
   in a way that confuses the gradient.
   
   However, Chebyshev is well-conditioned.
   
   What if the target function is not smooth? Or what if the target is a simple polynomial 
   but the Chebyshev basis is causing issues?
   
   Let's try a **Direct Polynomial Basis** (1, x, x^2, ..., x^9).
   This is the most standard "symbolic regression" starting point.
   BFGS should handle a degree-9 polynomial fit very well.
   The condition number of the Vandermonde matrix for x in [-2, 2] is higher than Chebyshev, 
   but for degree 9, it is usually manageable for BFGS with numerical gradients.
   
   Another option: **RBF with fixed centers**.
   y = sum_{i=1}^{10} p_i * exp(-((x-c_i)^2)/sigma^2)
   This is linear in p_i. BFGS will solve a linear system.
   RBFs can approximate any continuous function well if centers are spread out.
   This might be more robust than polynomials for non-polynomial targets.
   
   Let's combine the best ideas:
   1. Use a basis that is linear in parameters (so BFGS solves a convex problem).
   2. Use a basis with high expressive power.
   3. Ensure numerical stability.
   
   The current program uses Chebyshev. Let's try **Gaussian RBFs** with centers spread over [-2, 2].
   This often works better for arbitrary smooth functions than polynomials, which can suffer from Runge's phenomenon 
   or require very high degrees.
   
   We will define:
   y = sum_{i=0}^{9} p_i * exp(-0.5 * ((x - c_i) / width_i)^2)
   
   Centers c_i: linspace(-2, 2, 10)
   Widths: fixed, e.g., 0.5 or 1.0.
   
   This is a linear model in p. BFGS will converge to the global minimum (the least squares solution).
   The quality of the fit depends on the choice of centers and widths.
   To improve robustness, we can use a wider spread of widths or more centers if we could, but we are limited to 10 params.
   
   Let's try a **Hybrid Basis**:
   5 Gaussian RBFs + 5 Chebyshev polynomials?
   Or just 10 RBFs.
   
   Let's stick to a robust linear-in-params model.
   I will implement a set of 10 Gaussian Radial Basis Functions with centers evenly spaced in [-2, 2] and a fixed bandwidth.
   This should allow the model to fit local features well.
"""

import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y using a linear combination of Gaussian RBFs and Chebyshev polynomials.
    The parameters are the coefficients for these basis functions.
    This is a convex optimization problem for the evaluator's BFGS.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Safety check for finite inputs
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    n = len(x)
    if n == 0:
        return np.zeros(0)
        
    # Basis 1: 5 Gaussian RBFs
    # Centers: -2, -1, 0, 1, 2
    centers_rbf = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    width_rbf = 1.0
    basis_rbf = np.zeros((n, 5))
    for i, c in enumerate(centers_rbf):
        basis_rbf[:, i] = np.exp(-0.5 * ((x - c) / width_rbf) ** 2)
        
    # Basis 2: 5 Chebyshev Polynomials (mapped to [-1, 1])
    x_norm = 0.5 * x
    basis_cheb = np.zeros((n, 5))
    basis_cheb[:, 0] = 1.0
    if n > 0:
        basis_cheb[:, 1] = x_norm
    for k in range(2, 5):
        basis_cheb[:, k] = 2.0 * x_norm * basis_cheb[:, k-1] - basis_cheb[:, k-2]
        
    # Combine bases: 5 RBF + 5 Cheb = 10 params
    basis_total = np.hstack([basis_rbf, basis_cheb])
    
    # Linear combination
    out = basis_total @ p
    
    # Ensure finite output
    if not np.all(np.isfinite(out)):
        out = np.where(np.isfinite(out), out, 0.0)
        
    return out
