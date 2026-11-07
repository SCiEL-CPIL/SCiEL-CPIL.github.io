import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Locally Weighted Linear Regression (LWLR) with learned RBF geometry.
    
    We interpret the 10 parameters as:
    - p[0:3]: Centers of 3 Gaussian RBFs (one per input dim x, y, z)
    - p[3:6]: Widths (inverse scales) of the 3 Gaussian RBFs
    - p[6:10]: Coefficients for a local linear model [intercept, coef_x, coef_y, coef_z]
    
    The model computes a weighted average of local linear predictions.
    For each point in the input batch, we calculate the weight of each of the 3 RBFs.
    The prediction is the weighted sum of the local linear functions evaluated at that point,
    normalized by the sum of weights.
    
    Specifically:
    Let centers be c = [cx, cy, cz] and widths be w = [wx, wy, wz].
    We define 3 basis functions (RBFs) B_j for j=0,1,2 corresponding to the 3 dimensions?
    No, a better structure for 10 params to create a flexible surface:
    
    Let's refine the parameter mapping to be more robust for BFGS:
    p[0:3] -> Centers (cx, cy, cz)
    p[3:6] -> Widths (wx, wy, wz)
    p[6:10] -> Linear coefficients (b, mx, my, mz)
    
    Wait, if we only have ONE set of linear coefficients, it's just a global linear model weighted by RBFs.
    If the weights sum to 1, the output is just the global linear model. That's useless.
    
    We need LOCAL linear coefficients. But we only have 10 params.
    
    Alternative Structure:
    Use the 10 params to define a small neural network with a specific activation that allows local linearity.
    
    Let's try:
    p[0:4] -> Weights for hidden layer input 1 (w1, w2, w3, b1)
    p[4:8] -> Weights for hidden layer input 2 (w4, w5, w6, b2)
    p[8:10] -> Output weights (w_out1, w_out2)
    
    Hidden layer activation: ReLU or Tanh.
    This is the "Inspiration 1" which scored 10.78 (bad).
    
    Let's look at the current score 1.30. It's a linear basis: 1, x, y, z, exp(x), exp(y), exp(z), log(x), log(y), log(z).
    
    Why is it stuck at 1.30?
    Maybe the target is a product term like x*y or x*exp(y)?
    The current basis has no interaction terms.
    
    Strategy Change: Include Interaction Terms in the Linear Basis.
    The current basis is purely additive in features (sum of f(x), g(y), h(z)).
    Most functions are not additive.
    
    New Basis (10 terms):
    1. 1
    2. x
    3. y
    4. z
    5. x*y
    6. y*z
    7. z*x
    8. exp(x*y)  -- Strong interaction
    9. log(x*y)  -- Strong interaction
    10. x*z*exp(y) -- Complex interaction
    
    This is still linear in parameters, so BFGS is fast and robust.
    It covers multiplicative interactions which are common in physical/chemical laws.
    
    Let's try a basis that covers:
    - Constant
    - Linear (x, y, z)
    - Pairwise products (xy, yz, zx)
    - One higher-order interaction (xyz)
    - One exponential interaction (exp(x+y+z) or exp(xy))
    - One log interaction (log(x*y*z))
    
    Let's construct:
    b0 = 1
    b1 = x
    b2 = y
    b3 = z
    b4 = x*y
    b5 = y*z
    b6 = z*x
    b7 = x*y*z
    b8 = np.exp(x + y + z)  # Additive exponent
    b9 = np.log(x*y*z)      # Additive log (log x + log y + log z)
    
    This basis is orthogonal to the "additive-only" assumption.
    
    Let's verify stability.
    Inputs in [0.5, 2].
    x*y*z in [0.125, 8]. Log is safe.
    exp(x+y+z) in [exp(1.5), exp(6)] ~ [4.5, 403]. Safe.
    
    This is a structural change from "Additive Non-Polynomial" to "Multiplicative/Interaction Non-Polynomial".
    """
    try:
        n = len(x)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        
        # Clip for safety
        x_safe = np.clip(x, 1e-6, 1e6)
        y_safe = np.clip(y, 1e-6, 1e6)
        z_safe = np.clip(z, 1e-6, 1e6)
        
        p = params
        
        # Basis Functions
        b0 = np.ones_like(x)
        b1 = x
        b2 = y
        b3 = z
        
        # Interaction Terms (Polynomial)
        b4 = x * y
        b5 = y * z
        b6 = z * x
        b7 = x * y * z
        
        # Interaction Terms (Non-Polynomial)
        # exp(sum) captures additive exponential growth
        b8 = np.exp(x_safe + y_safe + z_safe)
        
        # log(product) captures multiplicative logarithmic behavior
        b9 = np.log(x_safe * y_safe * z_safe)
        
        output = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
                  p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
                  p[8] * b8 + p[9] * b9)
        
        if not np.all(np.isfinite(output)):
            return np.zeros(n)
            
        return output

    except Exception:
        try:
            return np.zeros(len(x))
        except:
            return np.array([0.0])
