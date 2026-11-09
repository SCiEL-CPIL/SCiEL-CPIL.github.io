"""
Strategy: Neural Network (MLP) Representation
Instead of a fixed basis of polynomials and trigonometric functions, we use a 
simple 2-layer neural network (input -> hidden -> output) where the 10 parameters 
are the learnable weights. This allows the model to fit complex nonlinear 
relationships directly from the data, bypassing the need for a well-conditioned 
landscape specific to physical intuition. The structure is:
y = sum_j (w_j * tanh(dot(x_j, w_in_j)))
where x_j are the inputs [t, x, v] and w_in_j are fixed random weights, 
and w_j are the 10 fitted parameters. 
Wait, the task says "up to 10 fitted constants". 
If I use a standard MLP, I need more than 10 weights for a decent fit.
With only 10 parameters, a standard MLP is very constrained.
Let's rethink. The current program uses 10 params in a linear combination of features.
The "Inspiration 2" got 1.54 with "Structured multiplicative nonlinear oscillator".
The "Inspiration 3" got 1.16 with "Physics-informed sparse linear model using OMP".

Actually, looking at the inspirations:
- Current: 57.8 (bad)
- Insp 1: 58.0 (bad)
- Insp 2: 1.54 (good, multiplicative)
- Insp 3: 1.16 (good, OMP sparse linear)

The key insight from Insp 2 and 3 is that **linear-in-parameters** models work well 
when the basis functions are chosen correctly, or when the structure is multiplicative 
to capture interactions. 
The current program is linear-in-parameters but uses a broad, potentially redundant 
basis. 
The "Step 1 Strategy" for a MATERIAL DIFFERENCE should be:
**Use a Radial Basis Function (RBF) Network with fixed centers.**
We place 5 fixed centers in the input space (t, x, v) and use 5 parameters for 
amplitudes and 5 parameters for widths? No, widths should be fixed for stability.
Or, better: **Kernel Regression with Fixed Centers**.
Let's use a set of fixed Gaussian RBFs. 
Inputs: t, x, v.
We define 5 centers C_i. 
phi_i(t, x, v) = exp(-gamma_i * || [t, x, v] - C_i ||^2)
Prediction = sum_{i=1}^5 p_i * phi_i + p_6 * t + p_7 * x + p_8 * v + p_9 * t*x + p_10 * x*v
This is still linear in params. 
Is this materially different? 
The current program uses global polynomials/trigs. 
RBFs are local. This changes the inductive bias from "global smoothness" to "local interpolation".
This is a different design family (Local vs Global basis).

Let's refine the RBF approach.
Inputs: t in [30, 50], x in [-0.2, 0.17], v in [-0.34, 0.36].
We need to normalize inputs to make the Gaussian widths meaningful.
Let's define 5 centers spread across the input domain.
Parameters: 5 amplitudes for RBFs, plus 5 linear coefficients for a global trend.
Total 10 params.

Why might this jump?
1. Local basis functions can capture sharp features or non-uniform curvature better than low-order polynomials.
2. The linear part handles the global trend.
3. It avoids the "cancellation" issues of high-order polynomials.
4. It is still linear in params, so BFGS works well, but the feature space is fundamentally different (local vs global).

Let's implement this.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 135.0

# Pre-compute centers and fixed widths for RBFs
# Inputs: t, x, v
# Ranges: t [30, 50], x [-0.2056, 0.1749], v [-0.3391, 0.3624]
# Normalize inputs to [0, 1] range for stable RBF centers?
# Or just use raw values with appropriate widths.
# Let's use normalized inputs for the RBF part to ensure the Gaussian is well-behaved.
# t_norm = (t - 30) / 20
# x_norm = (x - (-0.2056)) / (0.1749 - (-0.2056)) = (x + 0.2056) / 0.3805
# v_norm = (v - (-0.3391)) / (0.3624 - (-0.3391)) = (v + 0.3391) / 0.7015

# Define 5 centers in the normalized [0,1]^3 space
# Spread them out:
# Center 1: Low t, Low x, Low v
# Center 2: Low t, High x, Low v
# Center 3: High t, Low x, High v
# Center 4: High t, High x, High v
# Center 5: Mid t, Mid x, Mid v

C1 = np.array([0.0, 0.0, 0.0])
C2 = np.array([0.0, 1.0, 0.0])
C3 = np.array([1.0, 0.0, 1.0])
C4 = np.array([1.0, 1.0, 1.0])
C5 = np.array([0.5, 0.5, 0.5])

centers = np.array([C1, C2, C3, C4, C5])
# Widths (gamma) in the Gaussian exp(-gamma * dist^2)
# Larger gamma = narrower peak.
# Let's use gamma = 10.0 for all, or vary them.
# Fixed widths to keep params at 10.
gammas = np.array([10.0, 10.0, 10.0, 10.0, 10.0])

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid RBF + Linear Model.
    Params:
    p0-p4: Amplitudes for 5 RBFs
    p5-p9: Coefficients for linear basis [1, t_norm, x_norm, v_norm, t_norm*x_norm]
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs
    t_norm = (t - 30.0) / 20.0
    x_norm = (x + 0.2056) / 0.3805
    v_norm = (v + 0.3391) / 0.7015
    
    # Stack normalized inputs
    # Shape: (N, 3)
    inputs = np.column_stack((t_norm, x_norm, v_norm))
    
    # Compute RBF features
    # dist^2 = sum((inputs - centers)^2, axis=1)
    # Using broadcasting: inputs (N, 1, 3) - centers (1, 5, 3)
    diff = inputs[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # RBF values: exp(-gamma * dist_sq)
    # gammas shape (5,) -> broadcast to (1, 5)
    rbf_vals = np.exp(-gammas[np.newaxis, :] * dist_sq)
    
    # Linear part
    # Basis: [1, t_norm, x_norm, v_norm, t_norm*x_norm]
    bias = np.ones_like(t)
    lin_term1 = p5 * bias
    lin_term2 = p6 * t_norm
    lin_term3 = p7 * x_norm
    lin_term4 = p8 * v_norm
    lin_term5 = p9 * (t_norm * x_norm)
    
    linear_pred = lin_term1 + lin_term2 + lin_term3 + lin_term4 + lin_term5
    
    # RBF part: sum(p_i * rbf_i)
    # rbf_vals shape (N, 5)
    # params p0-p4 shape (5,)
    rbf_amplitudes = np.array([p0, p1, p2, p3, p4])
    rbf_pred = np.sum(rbf_vals * rbf_amplitudes, axis=1)
    
    return linear_pred + rbf_pred
# EVOLVE-BLOCK-END
