import numpy as np
import time

def equation(b, s, temp, pH, params):
    """
    E. coli growth rate model using a Multiplicative Monotonic Basis Function (MMBF) approach.
    
    Strategy:
    Instead of an additive sum of piecewise-linear hats, we use a product of 
    smooth, bounded basis functions for each input variable. This allows the 
    model to capture interaction effects between variables (e.g., temperature 
    and pH synergy) which an additive model cannot represent.
    
    The model form is:
    y = p[0] + p[1] * (B_b1 * B_b2) * (B_s1 * B_s2) * (B_t1 * B_t2) * (B_pH1 * B_pH2)
    
    Where each B_ij is a smooth sigmoidal basis function:
    B(x, c, w) = 1 / (1 + exp(-(x - c) / w))
    
    By multiplying two sigmoids with different centers and widths, we can 
    approximate bell-shaped curves (e.g., optimal temperature) or monotonic 
    saturating curves (e.g., substrate concentration).
    
    Parameters (10 total):
    p[0]: Offset (bias)
    p[1]: Global scale factor
    p[2], p[3]: Centers for b (density)
    p[4], p[5]: Centers for s (substrate)
    p[6], p[7]: Centers for temp
    p[8], p[9]: Centers for pH
    
    Widths are fixed to reasonable values relative to input ranges to ensure 
    smoothness and prevent numerical instability, allowing BFGS to focus on 
    optimizing the centers and scale.
    """
    p = params
    
    # Fixed widths for stability, scaled to input ranges
    # b range [1, 5] -> width ~ 1.0
    # s range [1, 5] -> width ~ 1.0
    # temp range [1, 40] -> width ~ 5.0
    # pH range [2, 12] -> width ~ 2.0
    
    w_b = 1.0
    w_s = 1.0
    w_t = 5.0
    w_p = 2.0
    
    # Helper: Stable sigmoid
    def sigmoid(x, c, w):
        z = (x - c) / w
        # Clip z to avoid overflow in exp
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))
    
    # Compute basis functions for each variable
    # We use two sigmoids per variable to allow for more complex shapes
    # e.g., one rising and one falling, or two rising with different centers
    
    # b (density): Often has an optimal range or inhibitory effect at high density
    b1 = sigmoid(b, p[2], w_b)
    b2 = sigmoid(b, p[3], w_b)
    
    # s (substrate): Typically Monod-like saturation
    s1 = sigmoid(s, p[4], w_s)
    s2 = sigmoid(s, p[5], w_s)
    
    # temp (temperature): Bell-shaped, optimal around 37C
    t1 = sigmoid(temp, p[6], w_t)
    t2 = sigmoid(temp, p[7], w_t)
    
    # pH (pH): Bell-shaped, optimal around 7
    p1 = sigmoid(pH, p[8], w_p)
    p2 = sigmoid(pH, p[9], w_p)
    
    # Product of all basis functions creates the interaction surface
    # If all sigmoids are > 0.5, the product is large.
    # To allow for "bell" shapes, we can use the product of a rising and a falling sigmoid.
    # However, with only centers as params, we rely on BFGS to find centers that 
    # create the right shape. 
    # A common trick for a bell shape with sigmoids is: sigmoid(x, c1, w) * sigmoid(-x, c2, w)
    # But our sigmoid is fixed direction. 
    # Let's use a different basis: Gaussian-like via product of two sigmoids?
    # Actually, a simpler robust approach: Use the product of sigmoids as a "window".
    # If p[2] < p[3], b1 rises then b2 rises. 
    # To get a peak, we can use: sigmoid(b, c1, w) * (1 - sigmoid(b, c2, w))
    # This requires an extra param for the sign or a different formulation.
    
    # Let's refine the basis to be more flexible for peaks:
    # Use: B(x) = sigmoid(x, c1, w) * sigmoid(c2 - x, w)
    # This creates a bell shape centered between c1 and c2 if c1 < c2.
    # If c1 > c2, it's a valley.
    # This is a very powerful and smooth basis for univariate functions.
    
    b_bell = sigmoid(b, p[2], w_b) * sigmoid(p[3] - b, w_b)
    s_bell = sigmoid(s, p[4], w_s) * sigmoid(p[5] - s, w_s)
    t_bell = sigmoid(temp, p[6], w_t) * sigmoid(p[7] - temp, w_t)
    p_bell = sigmoid(pH, p[8], w_p) * sigmoid(p[9] - pH, w_p)
    
    # Combine multiplicatively
    # This captures interactions: if temp is suboptimal, the effect of pH is dampened.
    interaction_term = b_bell * s_bell * t_bell * p_bell
    
    # Final prediction
    # p[0] is offset, p[1] is scale
    prediction = p[0] + p[1] * interaction_term
    
    return prediction
