"""
Strategy: Parameterized Gaussian Mixture Model.
Instead of a linear combination of fixed basis functions (like polynomials),
we use a model where the parameters define the shape of the basis functions themselves.

We model y(x) as a sum of 3 Gaussian bumps plus a constant offset:
y(x) = C + A1 * exp(-((x - mu1)^2) / (2 * sigma1^2)) + 
       A2 * exp(-((x - mu2)^2) / (2 * sigma2^2)) + 
       A3 * exp(-((x - mu3)^2) / (2 * sigma3^2))

Parameters (length 10):
p[0] = mu1
p[1] = sigma1
p[2] = A1
p[3] = mu2
p[4] = sigma2
p[5] = A2
p[6] = mu3
p[7] = sigma3
p[8] = A3
p[9] = C

To ensure numerical stability and allow BFGS to find valid parameters:
- We map raw parameters to positive sigmas using exp() or softplus to avoid division by zero or negative variance.
- Actually, BFGS starts at 1. If sigma is 1, it's fine. But if BFGS tries negative sigma, sigma^2 is still positive in the denominator, but the derivative might be weird if we don't constrain it.
- Better: Use exp() for sigma to ensure sigma > 0. 
  Let s_i = exp(p[i]). Then sigma_i = s_i.
  However, exp(1) = 2.718. If the true sigma is small, exp() might be too stiff.
  Alternative: sigma = 1 + p[i] if we want to allow small values, but p can be negative.
  Let's use: sigma = abs(p[i]) + 1e-6. Or better, sigma = exp(p[i]).
  If p starts at 1, sigma = e ~ 2.7. This is a reasonable width for [-3, 3].
  If the target is a sharp peak, BFGS can decrease p to make sigma smaller.
  If the target is broad, BFGS can increase p.

Let's refine the parameter mapping for stability:
mu_i = p[3*i]
sigma_i = exp(p[3*i+1])  # Ensures sigma > 0
A_i = p[3*i+2]
C = p[9]

Wait, 3 components * 3 params + 1 constant = 10 params. Perfect.

Why is this better than Chebyshev?
- Chebyshev polynomials are global. They are great for smooth, global trends but can struggle with localized features or functions that are not well-approximated by low-degree polynomials (e.g., sharp transitions, multiple peaks).
- Gaussians are local. A mixture of Gaussians can approximate any continuous function on a compact domain (Universal Approximation Theorem applies to RBF networks). 
- With 3 Gaussians, we can fit functions with up to 3 distinct "modes" or significant curvature changes.
- The non-linear nature of the parameters (mu, sigma) means BFGS is doing a more complex optimization, potentially finding a better fit for non-polynomial targets.

Potential issues:
- BFGS might get stuck in local minima if the initial guess (all ones) is poor.
  - mu = 1, sigma = e ~ 2.7, A = 1.
  - This places a broad bump centered at x=1.
  - If the target is centered at x=0, BFGS must move mu from 1 to 0.
  - If the target is symmetric, we might need two bumps or one centered at 0.
  - Starting all mu at 1 is a bit biased. 
  - Can we initialize better? No, the evaluator starts params at all ones.
  - However, BFGS is quite robust. If the gradient points towards 0, it will move there.
  - To help, we could use a symmetric initialization? No, we can't change the start.
  - But we can make the function less sensitive to initial mu? No.
  - Let's trust BFGS. The domain is small [-3, 3].

- Numerical stability:
  - exp(p) can overflow if p is large. But p is fitted by BFGS, usually staying in a reasonable range.
  - If p is very negative, exp(p) -> 0, sigma -> 0, Gaussian becomes a delta function. This can cause numerical issues in gradients.
  - We can clamp sigma: sigma = np.clip(exp(p), 1e-3, 10.0).
  - This prevents division by zero and extreme narrowness.

Let's implement this.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y using a sum of 3 Gaussian basis functions with learned parameters.
    
    params: np.ndarray of shape (10,)
        p[0], p[1], p[2] -> mu1, log_sigma1, A1
        p[3], p[4], p[5] -> mu2, log_sigma2, A2
        p[6], p[7], p[8] -> mu3, log_sigma3, A3
        p[9]             -> C (offset)
    """
    x = np.asarray(x, dtype=float)
    
    # Unpack parameters
    mu1, log_sig1, A1 = params[0], params[1], params[2]
    mu2, log_sig2, A2 = params[3], params[4], params[5]
    mu3, log_sig3, A3 = params[6], params[7], params[8]
    C = params[9]
    
    # Convert log_sigma to sigma with clipping for stability
    # exp(log_sig) = sigma. 
    # Clip log_sig to avoid overflow/underflow in exp.
    # exp(-100) ~ 0, exp(100) ~ inf.
    # Let's clip log_sig to [-10, 10]. sigma in [4e-5, 22000].
    # This is safe.
    log_sig1 = np.clip(log_sig1, -10.0, 10.0)
    log_sig2 = np.clip(log_sig2, -10.0, 10.0)
    log_sig3 = np.clip(log_sig3, -10.0, 10.0)
    
    sig1 = np.exp(log_sig1)
    sig2 = np.exp(log_sig2)
    sig3 = np.exp(log_sig3)
    
    # Calculate Gaussian components
    # g(x) = A * exp(-0.5 * ((x - mu) / sigma)^2)
    
    term1 = A1 * np.exp(-0.5 * ((x - mu1) / sig1) ** 2)
    term2 = A2 * np.exp(-0.5 * ((x - mu2) / sig2) ** 2)
    term3 = A3 * np.exp(-0.5 * ((x - mu3) / sig3) ** 2)
    
    # Sum and add offset
    y = C + term1 + term2 + term3
    
    return y
# EVOLVE-BLOCK-END
