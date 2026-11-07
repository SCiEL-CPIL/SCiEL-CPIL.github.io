"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Adaptive Gaussian Process Basis Function.
    
    Uses 5 Gaussian RBF basis functions.
    Parameters:
    - p[0:5] : Centers (c1..c5)
    - p[5:9] : Widths (sigma1..sigma4) -- 4 widths for 5 centers? 
               Actually, let's use 5 widths and 5 centers? 
               We have 10 params.
               Let's use 4 centers and 4 widths + 2 global scales?
               Or 5 centers and 5 widths is 10 params. 
               But BFGS starts at 1. 
               If centers start at 1, they are all clustered. 
               We need to spread them out or let the optimizer find them.
               However, if the target is a polynomial, Gaussians might struggle 
               unless widths are large (approximating constants/linear).
               
               Alternative: Use a fixed set of centers but learn weights? 
               That's the current strategy (linear basis).
               
               New Strategy: Learnable centers and widths for a small number of Gaussians.
               To avoid local minima where all centers collapse, we can initialize 
               centers differently? No, we can't change the optimizer start.
               
               However, we can use a "spreading" mechanism or just rely on the 
               fact that BFGS might find a solution. 
               
               Let's try:
               p[0:5] = centers
               p[5:10] = widths (absolute value to ensure positive)
               
               To help the optimizer, we can add a small regularization or 
               structure. 
               
               Actually, a better "different" strategy:
               Use a **Fourier Series** with learnable frequencies? 
               Or **Polynomial** with learnable roots?
               
               Let's stick to the RBF but make it robust.
               If p[5:10] are widths, we use abs(p[5:10]) + epsilon.
               
               Another option: **Neural Network** like structure?
               equation(x, params) -> sum of sigmas(x*wi + bi) * oi
               10 params -> 2 neurons? 
               Neuron 1: w1, b1, o1
               Neuron 2: w2, b2, o2
               ... 5 neurons? 
               10 params / 5 neurons = 2 params per neuron? 
               Need w, b, out. That's 3. 
               3 neurons * 3 params = 9 params + 1 bias? 
               
               Let's try a **Shallow Neural Network** with 3 hidden units.
               Hidden unit i: tanh(w_i * x + b_i)
               Output: sum(o_i * h_i) + bias
               
               Params:
               w1, b1, o1 (3)
               w2, b2, o2 (3)
               w3, b3, o3 (3)
               bias (1)
               Total: 10.
               
               This is a universal approximator. BFGS might struggle with 
               non-convexity, but it's a different family from the fixed basis.
               The fixed basis is linear in params (convex). The NN is non-convex.
               The task says "BFGS starting from all ones".
               If we use NN, the landscape is rough. 
               
               However, the "Inspiration 2" with score 1.0000 suggests the 
               current linear basis is actually working well for *some* targets 
               or the seed is just the baseline. 
               Wait, the current score is 1.7M (bad). Inspiration 2 is 1.0 (good).
               The current program is the "Initial program". 
               The prompt says "CURRENT PROGRAM (score 1716240.9980)".
               This implies the current linear basis is failing badly on the 
               hidden target. 
               
               Why would a degree 4 poly + trig + tanh fail?
               Maybe the target is high frequency or complex.
               
               A neural network with 3-5 units might capture complex shapes 
               better if the optimizer can escape local minima. 
               But BFGS is sensitive to initialization. All ones is a bad start 
               for NN weights (symmetry breaking is needed).
               
               To break symmetry with fixed initialization (all ones), we can 
               use different basis functions for each "unit" effectively.
               
               Let's go back to **Learnable Centers for Gaussians**.
               If we use:
               y = sum( exp(-((x - c_i)^2) / (2 * s_i^2)) * a_i )
               
               Params: c1, c2, c3, s1, s2, s3, a1, a2, a3, bias.
               
               If c_i start at 1, they are identical. The gradient with respect 
               to c_i might be zero if the function is symmetric or if the 
               loss is flat? No, the loss depends on x. 
               If all c_i are 1, the model is a single Gaussian. 
               BFGS will move them apart if that reduces error.
               
               This seems like a solid "different" strategy: 
               **Adaptive RBF Basis**.
    """
    p = params
    if not np.all(np.isfinite(p)):
        p = np.ones(10)
    
    # Extract parameters
    # 3 centers, 3 widths, 3 amplitudes, 1 bias
    c1, c2, c3 = p[0], p[1], p[2]
    w1, w2, w3 = np.abs(p[3]) + 0.1, np.abs(p[4]) + 0.1, np.abs(p[5]) + 0.1
    a1, a2, a3 = p[6], p[7], p[8]
    bias = p[9]
    
    # Calculate Gaussian terms
    # Clip x to [-10, 10] to prevent overflow in exp if x is huge (though x is in [-2,2])
    x_clipped = np.clip(x, -10, 10)
    
    g1 = np.exp(-((x_clipped - c1) ** 2) / (2 * w1 ** 2))
    g2 = np.exp(-((x_clipped - c2) ** 2) / (2 * w2 ** 2))
    g3 = np.exp(-((x_clipped - c3) ** 2) / (2 * w3 ** 2))
    
    result = a1 * g1 + a2 * g2 + a3 * g3 + bias
    
    # Safety
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
