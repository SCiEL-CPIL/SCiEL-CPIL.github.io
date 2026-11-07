import numpy as np
import time

# Global state to cache the fitted kernel regression solution
# The evaluator calls equation(x, params) multiple times.
# We assume the first call with y (training) allows us to fit, 
# but the interface is equation(x, params) -> pred.
# The contract says: "The evaluator fits params ... minimizing the training MSE."
# This implies the evaluator passes x_train, y_train implicitly via the objective?
# No, the contract says: "The evaluator fits params, an array of length 10, with scipy.optimize.minimize... minimizing the training MSE."
# Wait, the contract says params is length 10. 
# "A prediction of the wrong shape or with non-finite error costs 1e30 during fitting."
# "equation must return an array with one prediction per point."
#
# CRITICAL CONSTRAINT CHECK:
# The evaluator uses scipy.optimize.minimize to fit `params`.
# The length of `params` is FIXED at 10 by the evaluator.
# I CANNOT change the number of parameters. The optimizer will pass an array of length 10.
# Therefore, I CANNOT implement a full kernel regression with 200 parameters because I only receive 10.
#
# RE-EVALUATION:
# The strategy "adaptive kernel regression with 200 basis functions" is impossible because I only get 10 params.
#
# Alternative Structural Change:
# Since I am stuck with 10 parameters, I need a basis that is more flexible than polynomials/trig.
# The current basis is fixed. 
# Can I make the basis functions themselves depend on the data?
# The function signature is equation(x, params). It does NOT receive y during the call.
# So I cannot use the training data to define the basis.
#
# So I must stick to a fixed basis, but perhaps a different one?
# The current basis: 1, x, x^2, x^3, sin, cos, tanh, exp(-x^2), x*sin, x*cos.
# This is a "standard" symbolic regression basis.
#
# What if the target function is not smooth? Or has discontinuities?
# Unlikely, given "Gaussian noise" and typical SR tasks.
#
# Let's look at the score: 4380. This is extremely high (bad). 
# MSE / Var(Y). If Var(Y) is small, this number is huge. 
# If the model is just bad, the MSE is high.
#
# Why is the score so bad?
# Maybe the basis is insufficient. 
# Maybe the optimizer is failing.
#
# Let's try a different structural approach: 
# **Non-linear Basis Functions with Learned Centers/Widths?**
# No, I only have 10 params. I can't learn centers for RBFs.
#
# What if I use a **Piecewise Linear / Spline** basis?
# Fixed knots.
# Or **Chebyshev Polynomials**? They are orthogonal and better conditioned than monomials.
# The current program uses monomials (x, x^2, x^3). These are ill-conditioned for x in [-3,3] compared to Chebyshev.
# Ill-conditioning can cause BFGS to struggle or converge to poor local minima (though for linear basis, MSE is convex, so global min exists, but numerical precision might be an issue).
#
# Let's try **Chebyshev Polynomials** for the polynomial part.
# T0=1, T1=x, T2=2x^2-1, T3=4x^3-3x...
# This is a structural change to the basis representation, improving conditioning.
#
# BUT, is that enough?
#
# Let's consider the "Inspiration" scores. They are all around 4380-4390.
# This suggests the entire population is stuck at a similar performance level.
#
# Is it possible the target function is simply not in the span of these 10 functions?
# If so, no amount of optimization will fix it.
#
# What if I use a **Universal Approximator** style basis?
# With only 10 params, I can't.
#
# Let's look at the "Different Strategy" directive.
# "Rethinking the approach."
#
# What if the issue is that the evaluator's BFGS is not converging well?
# The contract says: "The evaluator fits params... with scipy.optimize.minimize(method='BFGS')".
# I cannot change the optimizer.
#
# However, I can change the **parameterization**.
# Instead of linear combination of basis functions, what if I use a **Non-Linear Parameterization**?
# e.g., y = f(x, params) where f is non-linear in params.
# The MSE would no longer be convex. BFGS is a local optimizer.
# If I make the landscape non-convex, BFGS might get stuck in bad local minima.
# Unless... I design the non-linearity to have a very simple landscape or use a "trick".
#
# Wait, the current program IS linear in params. This is the safest choice for BFGS.
#
# Let's look at the basis again.
# 0: 1
# 1: x
# 2: x^2
# 3: x^3
# 4: sin(x)
# 5: cos(x)
# 6: tanh(x)
# 7: exp(-x^2)
# 8: x*sin(x)
# 9: x*cos(x)
#
# Missing common functions:
# - Higher order trig harmonics (sin(2x), cos(2x))
# - Higher order polynomials (x^4, x^5)
# - Other smooth bumps
#
# If I replace x^3 with x^4, I lose odd symmetry beyond x.
# If I replace sin(x) with sin(2x), I lose the fundamental frequency.
#
# Let's try a **Fourier Series** focused basis.
# If the function is periodic or oscillatory, this is better.
# But inputs are [-3, 3].
#
# Let's try a **Radial Basis Function (RBF)** approach with fixed centers.
# Centers at -3, -1.5, 0, 1.5, 3.
# Basis: exp(-(x-c)^2 / sigma^2).
# With 10 params, I can have 5 centers with 2 params each? No, params are just weights.
# I can have 10 fixed RBFs.
#
# Let's combine the best features:
# 1. Constant
# 2. Linear
# 3. Chebyshev T2 (better conditioned quadratic)
# 4. Chebyshev T3 (better conditioned cubic)
# 5. Sin(x)
# 6. Cos(x)
# 7. Sin(2x)
# 8. Cos(2x)
# 9. Tanh(x)
# 10. Exp(-x^2)
#
# This covers low-order polynomials (conditioned), low-order trigonometrics, and smooth non-periodic features.
#
# Another idea: **Scaled Basis**.
# The values of x^3 at x=3 is 27. 
# The values of exp(-x^2) are small.
# The magnitudes vary.
# If I scale the basis functions to have unit L2 norm on [-3,3], the condition number of the Gram matrix improves.
# This is a structural change to the basis definition.
#
# Let's implement **Normalized Basis Functions**.
# 1. Compute the basis functions.
# 2. Normalize each column of the design matrix so it has unit norm.
# 3. The model is y = sum p_i * (basis_i / ||basis_i||).
# This effectively re-parameterizes the problem. The optimizer will find different p_i values, but the span is the same.
# Does this help BFGS? Yes, better conditioned gradients.
#
# BUT, the span is the same. If the span is insufficient, this won't help.
#
# What if I change the **Family** to include **Fractional Powers**?
# sqrt(|x|), sign(x)*|x|^0.5?
# These are non-smooth at 0. BFGS assumes smoothness. Derivatives might be problematic.
#
# Let's stick to smooth functions.
#
# How about **Gaussian Bumps** at different locations?
# The current exp(-x^2) is only at 0.
# Let's add exp(-(x-1)^2) and exp(-(x+1)^2).
#
# Proposed Basis Set (10 functions):
# 0: 1
# 1: x
# 2: x^2
# 3: x^3
# 4: sin(x)
# 5: cos(x)
# 6: sin(2x)
# 7: cos(2x)
# 8: exp(-(x-1)^2)  (Right bump)
# 9: exp(-(x+1)^2)  (Left bump)
#
# This replaces tanh and the modulated terms with higher frequency trig and localized bumps.
# This covers:
# - Polynomials up to 3
# - Trigonometric up to 2nd harmonic
# - Localized features at +/- 1
#
# This seems more "universal" for a function on [-3,3] than the current set which has x*sin(x) (which is 0 at 0, pi/2, etc) and tanh (which is global).
#
# Let's also consider **Conditioning**.
# I will normalize the basis functions inside the equation.
# pred = sum( p[i] * (basis[i] / norm[basis[i]]) )
# This ensures the gradient with respect to p[i] is well-scaled.
#
# One more thing: The evaluator starts params at all ones.
# If the basis functions are normalized, the initial prediction is sum(norms).
# If the target mean is 0, this might be far off.
# But BFGS should move it.
#
# Let's implement this "Normalized, Expanded Basis" strategy.

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a normalized, expanded basis set.
    
    Strategy:
    1. Use a basis set that covers polynomials, trigonometric harmonics, and localized Gaussian bumps.
    2. Normalize each basis function to unit L2 norm on the input domain to improve the conditioning
       of the linear system, aiding the BFGS optimizer.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    
    # Define Basis Functions
    # 0: Constant
    f0 = np.ones_like(x)
    
    # 1: Linear
    f1 = x
    
    # 2: Quadratic
    f2 = x ** 2
    
    # 3: Cubic
    f3 = x ** 3
    
    # 4: Sine (1st harmonic)
    f4 = np.sin(x)
    
    # 5: Cosine (1st harmonic)
    f5 = np.cos(x)
    
    # 6: Sine (2nd harmonic)
    f6 = np.sin(2 * x)
    
    # 7: Cosine (2nd harmonic)
    f7 = np.cos(2 * x)
    
    # 8: Gaussian bump at x=1
    f8 = np.exp(-((x - 1.0) ** 2))
    
    # 9: Gaussian bump at x=-1
    f9 = np.exp(-((x + 1.0) ** 2))
    
    # Stack basis functions
    # Shape: (10, N)
    basis = np.vstack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9])
    
    # Normalize each basis function (row) to unit norm
    # This prevents large coefficients in the solution if a basis function is large in magnitude
    norms = np.linalg.norm(basis, axis=1, keepdims=True)
    # Avoid division by zero (shouldn't happen for these functions)
    norms = np.where(norms == 0, 1, norms)
    basis_normed = basis / norms
    
    # Compute prediction: p @ basis_normed
    # p is (10,), basis_normed is (10, N) -> result (N,)
    pred = p @ basis_normed
    
    return pred
