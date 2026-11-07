"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt
import warnings

_START_TIME = time.time()
_DEADLINE = 145.0

def _mlp_forward(x, params):
    """
    Forward pass for a 2-layer MLP.
    params[0:6]: weights of layer 1 (2 inputs x 3 hidden units? No, x is 1D)
    Let's define the architecture clearly:
    Input: x (N, 1)
    Hidden Layer 1: 5 units, ReLU activation
    Output Layer: 1 unit, Linear activation
    
    We need to map params[0:10] to this architecture.
    Total params needed:
    W1: 1 x 5 = 5 params
    b1: 5 params
    W2: 5 x 1 = 5 params
    b2: 1 param
    Total: 16 params. We only have 10.
    
    Revised Architecture to fit 10 params:
    Input: x (N, 1)
    Hidden Layer 1: 4 units, Sigmoid activation
    Output Layer: 1 unit, Linear activation
    
    W1: 1 x 4 = 4 params (params[0:4])
    b1: 4 params (params[4:8])
    W2: 4 x 1 = 4 params (params[8:12] -> wait, only 2 left)
    
    Let's try:
    Hidden Layer 1: 3 units, Sigmoid
    W1: 3 params (params[0:3])
    b1: 3 params (params[3:6])
    W2: 3 params (params[6:9])
    b2: 1 param (params[9])
    Total: 10 params. Perfect.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    
    # Extract parameters
    W1 = params[0:3]  # (3,)
    b1 = params[3:6]  # (3,)
    W2 = params[6:9]  # (3,)
    b2 = params[9]    # scalar
    
    # Layer 1: (N, 1) x (1, 3) + (3,) -> (N, 3)
    h1 = x * W1 + b1
    h1 = np.tanh(h1)  # Tanh is often better for bounded domains
    
    # Layer 2: (N, 3) x (3, 1) + scalar -> (N, 1)
    # W2 is (3,), so h1 @ W2 gives (N, 1)
    out = h1 @ W2 + b2
    
    return out.flatten()

def _objective(params, x_train, y_train):
    """MSE objective function for the optimizer."""
    try:
        pred = _mlp_forward(x_train, params)
        if not np.all(np.isfinite(pred)):
            return 1e10
        return np.mean((pred - y_train) ** 2)
    except Exception:
        return 1e10

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Neural Network Model with internal optimization.
    The 'params' argument is ignored for the final prediction structure, 
    but we use it to initialize the internal optimization if it looks valid, 
    otherwise we use a default initialization.
    
    Wait, the contract says: "The evaluator fits params... minimizing the training MSE."
    This implies the evaluator calls `equation(x, params)` multiple times with different `params` 
    to evaluate the loss for its own BFGS step.
    
    CRITICAL CONSTRAINT: 
    "The evaluator fits params... with scipy.optimize.minimize(method="BFGS")... 
    A prediction of the wrong shape or with non-finite error costs 1e30 during fitting."
    
    If I implement my own optimizer INSIDE equation, I will be violating the contract's intent 
    that the evaluator controls the fitting process. The evaluator expects `equation` to be 
    a differentiable (or at least smooth) function of `params` so that BFGS can work.
    
    If I run a full optimization inside `equation`, the function `equation(x, params)` 
    becomes a "black box" that returns the *optimized* prediction regardless of the input `params` 
    (or only loosely coupled to it). This breaks the gradient flow expected by BFGS.
    
    Let's re-read carefully: "Write equation(x, params), a closed-form expression... 
    The evaluator fits params... minimizing the training MSE."
    
    This means `equation` MUST be a direct function of `params`. I cannot run an optimizer 
    that ignores the input `params` or changes the landscape discontinuously.
    
    Therefore, the strategy must be a **better fixed basis** that is well-conditioned 
    and expressive enough to be fit by BFGS.
    
    Inspiration 3 (score 74893) used a different approach. 
    Inspiration 1/2 used RBF/Chebyshev.
    
    Let's try a **Scaled Chebyshev Basis** with careful scaling to improve conditioning, 
    plus a **Linear Regression Pre-computation** trick? No, we can't pre-compute for unseen data.
    
    However, we can use **Legendre Polynomials** scaled to [-3, 3]. 
    Standard polynomials are ill-conditioned. Chebyshev are better.
    
    Let's look at the "diff" operator idea from the prompt: 
    "Does operator 'diff' change the chance that a child improves its island?"
    
    Maybe a **Wavelet-like** or **Piecewise Linear** basis?
    Piecewise linear with fixed knots is linear in weights. 
    Let's use **B-Splines** or **Piecewise Linear Basis Functions (PLBF)**.
    
    A PLBF with knots at -3, -2, -1, 0, 1, 2, 3.
    Basis functions: "Hat" functions.
    This is linear in the coefficients.
    The design matrix is sparse and well-conditioned compared to high-degree polynomials.
    
    Let's define 10 basis functions.
    With 7 knots, we have 6 intervals.
    We can define 10 overlapping hat functions.
    
    Actually, a simpler and very robust approach: **Gaussian RBFs with FIXED centers and widths**.
    The previous RBF attempt failed (score 5M) likely due to ill-conditioning or local minima.
    If we fix the centers and widths to be well-spaced and of moderate width, the basis is 
    fixed and linear in weights. BFGS should be able to fit the weights if the basis is 
    reasonably well-conditioned.
    
    Let's use 10 Gaussians centered at:
    -2.7, -2.25, -1.8, -1.35, -0.9, -0.45, 0, 0.45, 0.9, 1.35 ... wait, symmetric?
    Let's spread them evenly from -3 to 3.
    Centers: np.linspace(-3, 3, 10)
    Widths: Fixed at 1.0.
    
    This is a linear model: y = sum(w_i * exp(-(x - c_i)^2 / (2*sigma^2)))
    
    Why did the previous RBF fail? 
    Inspiration 1 says: "fixed set of 9 Gaussian RBFs... narrow width (sigma=0.5)... highly ill-conditioned".
    So, wider sigmas and more spread out centers should help.
    
    Let's try:
    Centers: np.linspace(-3, 3, 10)
    Sigma: 1.5
    
    This should be much better conditioned than sigma=0.5.
    """
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x, dtype=np.float64)
    
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    
    # Fixed centers and widths for well-conditioned linear basis
    centers = np.array([-3.0, -2.333, -1.666, -1.0, -0.333, 0.333, 1.0, 1.666, 2.333, 3.0])
    sigma = 1.2
    
    # Compute basis matrix efficiently
    # diff = x[:, None] - centers[None, :]
    # basis = np.exp(-diff**2 / (2 * sigma**2))
    # return basis @ params
    
    # To save memory/time for large N, we can compute in chunks or just use broadcasting 
    # since N=200 is small.
    diff = x[:, None] - centers[None, :]
    basis = np.exp(-diff**2 / (2.0 * sigma**2))
    
    return basis @ params

# EVOLVE-BLOCK-END
