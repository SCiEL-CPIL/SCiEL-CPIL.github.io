"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    n = x.size
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Ensure params is flat and has exactly 10 elements
    if params.size != 10:
        params = np.resize(params, 10)
        
    # Strategy: 
    # Instead of a fixed linear basis (like polynomials or fixed RBFs), 
    # we use the 10 parameters to define a flexible, non-linear model.
    # 
    # Model: Sum of 2 Gaussian "blobs" (centers + widths) + 2 global linear terms.
    # Params mapping (length 10):
    # p[0:3]  -> Center 1 (cx1, cy1, cz1)
    # p[3:6]  -> Center 2 (cx2, cy2, cz2)
    # p[6]    -> Width for Blob 1 (sigma1)
    # p[7]    -> Width for Blob 2 (sigma2)
    # p[8]    -> Linear coefficient for x
    # p[9]    -> Linear coefficient for y
    #
    # Why this might work:
    # 1. BFGS can move the centers to align with peaks/valleys of the true function.
    # 2. BFGS can adjust widths to match the scale of variations.
    # 3. The linear terms handle the global trend.
    # 4. Unlike fixed RBFs, this model is non-linear in params, but the landscape 
    #    is often smoother for "moving peaks" than for "weighting fixed peaks" 
    #    when the fixed peaks are poorly placed.
    #
    # Note on Initialization:
    # BFGS starts from all ones.
    # Centers: (1,1,1) and (1,1,1) -> Overlapping blobs.
    # Widths: 1 and 1 -> Moderate spread.
    # Linear: 1 and 1 -> Slope 1.
    # This is a reasonable start.
    
    # Unpack params
    cx1, cy1, cz1 = params[0], params[1], params[2]
    cx2, cy2, cz2 = params[3], params[4], params[5]
    sigma1 = params[6]
    sigma2 = params[7]
    lx = params[8]
    ly = params[9]
    
    # Ensure sigma is positive to avoid division by zero or negative variance
    # Using softplus-like or simple abs+eps to maintain differentiability and positivity
    # sigma_eff = np.maximum(sigma, 0.1)
    # Better: use exp(sigma) to ensure positivity and smoothness? 
    # If sigma starts at 1, exp(1)=2.7. 
    # Let's use a scaled version to keep it near 1 initially.
    # sigma_eff = 0.5 * np.exp(sigma)
    # If sigma=1, exp(1)=2.718, 0.5*2.718 = 1.35.
    # If sigma=0, exp(0)=1, 0.5*1 = 0.5.
    # This allows BFGS to find optimal widths.
    
    s1 = 0.5 * np.exp(np.clip(params[6], -10, 10))
    s2 = 0.5 * np.exp(np.clip(params[7], -10, 10))
    
    # Calculate squared distances to Center 1
    dx1 = x - cx1
    dy1 = y - cy1
    dz1 = z - cz1
    dist_sq_1 = dx1**2 + dy1**2 + dz1**2
    
    # Calculate squared distances to Center 2
    dx2 = x - cx2
    dy2 = y - cy2
    dz2 = z - cz2
    dist_sq_2 = dx2**2 + dy2**2 + dz2**2
    
    # Gaussian blobs
    # We don't have explicit amplitude params, so we fix amplitudes to 1.0 
    # and rely on the linear terms and the shape to fit. 
    # Wait, if amplitudes are fixed to 1, BFGS can only move/resize, not scale height.
    # This is a strong constraint. 
    # Alternative: Use the linear params also to scale? No, they are for x,y.
    # Let's add a global bias? We only have 10 params.
    # Maybe use p[8] and p[9] for linear, and assume the function is zero-mean?
    # Or, let's map params differently to allow amplitudes.
    
    # Revised Mapping for more flexibility:
    # p[0:3] Center 1
    # p[3:6] Center 2
    # p[6]   Amplitude 1 (can be negative)
    # p[7]   Amplitude 2 (can be negative)
    # p[8]   Shared Width (sigma)
    # p[9]   Linear coefficient for x (simple global trend)
    
    # Let's stick to the previous mapping but realize that fixed amplitude=1 is risky.
    # If the true function has large peaks, we can't scale them.
    # Let's use p[6] and p[7] as amplitudes (a1, a2).
    # And p[8], p[9] as widths? No, widths need to be positive.
    
    # Let's try:
    # p[0:3] C1
    # p[3:6] C2
    # p[6]   A1 (Amplitude 1)
    # p[7]   A2 (Amplitude 2)
    # p[8]   Sigma1 (Width 1) -> exp(scaled)
    # p[9]   Sigma2 (Width 2) -> exp(scaled)
    # This drops linear terms. Is that okay? 
    # If the function is smooth, 2 Gaussians might be enough if placed well.
    # But if it's a plane, Gaussians are bad.
    
    # Hybrid:
    # p[0:3] C1
    # p[3:6] C2
    # p[6]   A1
    # p[7]   A2
    # p[8]   Sigma (Shared width)
    # p[9]   Linear X coeff
    # This is a bit asymmetric.
    
    # Let's go with a "Bump + Linear" model.
    # p[0:3] Center (cx, cy, cz)
    # p[3:6] Amplitudes? No, we only have 1 bump?
    # 2 bumps is better.
    
    # Final Decision: 2 Gaussian Bumps + 1 Linear Term (X) + 1 Bias?
    # We have 10 params.
    # Bump 1: Cx, Cy, Cz, Amp, Sig (5 params)
    # Bump 2: Cx, Cy, Cz, Amp, Sig (5 params)
    # Total 10. No linear term.
    # If the function is global (like x+y+z), this fails.
    
    # Let's look at the inputs: [0.5, 2].
    # A polynomial might be better for global trends.
    # But the prompt says "STAGNATION" and "STRUCTURAL change".
    # The previous linear basis (degree 2) scored 7.4 (bad).
    # The RBF fixed basis scored 0.413 (good, but maybe not great).
    
    # Let's try a **Neural Network with Fixed Weights, Optimized Biases/Output**?
    # No, params are only 10.
    
    # What if we use the 10 params to define a **Rational Function**?
    # f(x,y,z) = (p0 + p1*x + p2*y + p3*z) / (1 + p4*x + p5*y + p6*z)
    # This is non-linear in params.
    # But BFGS might struggle with singularities.
    
    # Let's go back to **Adaptive RBFs** (Moving Centers).
    # To handle global trends, we can include a constant and linear terms in the "basis" 
    # by fixing some params? No, all 10 are optimized.
    
    # How about:
    # pred = p0 * exp(-p1 * (x-cx)^2 ...) 
    # This is getting complicated.
    
    # Let's try a **Simple Feedforward Neural Network** with 1 hidden layer.
    # Input: x, y, z
    # Hidden: 2 neurons
    # Output: 1 neuron
    # 
    # Weights:
    # W1 (3x2) = 6 params
    # b1 (2) = 2 params
    # W2 (2x1) = 2 params
    # Total = 10 params.
    # 
    # Architecture:
    # h1 = tanh(w11*x + w12*y + w13*z + b1)
    # h2 = tanh(w21*x + w22*y + w23*z + b2)
    # out = w31*h1 + w32*h2
    # 
    # This is a universal approximator (with enough neurons, but 2 is few).
    # However, it is non-linear in params.
    # BFGS might get stuck.
    # But the landscape of a small NN is often navigable if initialized well.
    # Init: all ones.
    # w11..w13 = 1, b1 = 1 -> h1 = tanh(x+y+z+1)
    # w21..w23 = 1, b2 = 1 -> h2 = tanh(x+y+z+1)
    # h1 == h2.
    # out = w31*h1 + w32*h2 = (w31+w32) * tanh(x+y+z+1)
    # This is a 1D function of the sum. It can't model x-y dependence initially.
    # But gradients will push them apart.
    
    # Let's implement this NN.
    
    # Unpack params
    w11, w12, w13, w21, w22, w23 = params[0:6]
    b1, b2 = params[6], params[7]
    w31, w32 = params[8], params[9]
    
    # Compute hidden layer
    z1 = w11*x + w12*y + w13*z + b1
    z2 = w21*x + w22*y + w23*z + b2
    
    # Apply activation (tanh is smooth and bounded)
    h1 = np.tanh(z1)
    h2 = np.tanh(z2)
    
    # Output layer (linear)
    pred = w31*h1 + w32*h2
    
    # Clip to prevent overflow in BFGS if params explode
    pred = np.clip(pred, -1e6, 1e6)
    
    return pred
# EVOLVE-BLOCK-END
