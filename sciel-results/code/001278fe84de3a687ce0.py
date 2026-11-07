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
        # Fallback if shape is weird, though contract says length 10
        params = np.resize(params, 10)
        
    # Strategy: 
    # The evaluator uses BFGS to minimize MSE. BFGS is a local optimizer.
    # If we provide a "linear-in-parameters" model, BFGS will converge to the global optimum 
    # (least squares solution) very quickly.
    # If we provide a "nonlinear-in-parameters" model (like a neural net), BFGS might get stuck 
    # in local minima or converge slowly.
    # 
    # However, the current program (score ~7.4) uses a linear basis. 
    # The inspiration 3 (score 1.0) was the seed, likely a simple linear model or even just zeros/ones.
    # Wait, score 1.0 is the baseline. Score 7.39 is WORSE than 1.0?
    # "1.0 equals the seed program and above 1.0 is an improvement"
    # So the current program (7.39) is performing WORSE than the seed (1.0).
    # Why? The seed likely returned something trivial or the basis was poorly scaled/conditioned 
    # causing BFGS to fail or produce bad predictions.
    #
    # Let's look at the current code:
    # It scales inputs to [-1, 1].
    # It uses a linear basis: 1, x, y, z, xy, xz, yz, x^2, y^2, z^2.
    # This is a linear model in params. BFGS should find the optimal params for this basis.
    # If the true function is NOT a quadratic polynomial, this model has high bias.
    # But why is the score 7.4 (bad) instead of 1.0 (baseline)?
    # Maybe the baseline (score 1.0) is just `return np.zeros_like(x)` or similar?
    # If the target has variance 1, and we predict 0, MSE = 1. Normalized MSE = 1.
    # So the seed probably predicts a constant (mean or 0).
    # The current program tries to fit a quadratic. If the fit fails or the quadratic is a bad 
    # approximation of the true function, it can be worse than the constant.
    #
    # Crucial Insight: 
    # The evaluator fits `params` using BFGS.
    # If I make the model VERY flexible (e.g., a deep network), BFGS might struggle.
    # If I make the model LINEAR in params, BFGS will find the exact best fit for that model.
    #
    # The problem is likely that the quadratic basis is not rich enough OR the conditioning 
    # is bad, OR the true function is not quadratic.
    #
    # New Strategy: 
    # Use a Richer Linear Basis.
    # Instead of just up to degree 2, use up to degree 3 or 4, but keep it linear in params.
    # However, we only have 10 params.
    # 
    # Alternative: 
    # What if the "params" are not just weights, but we can use them to construct 
    # adaptive features? No, params are fixed length 10 and optimized by BFGS.
    #
    # Let's reconsider the "Neural Network" idea.
    # If I define:
    # h1 = tanh(W1 @ [x,y,z,1] + b1)
    # h2 = tanh(W2 @ h1 + b2)
    # pred = W3 @ h2
    # Where W1, W2, W3, b1, b2 are FIXED random constants.
    # And `params` are the weights of the final layer or a specific subset?
    # No, `params` has length 10. 
    # If I fix the architecture and random initializations, and let `params` be the ONLY 
    # free variables (e.g., the output layer weights and biases), then the model is 
    # non-linear in params? 
    # Actually, if the hidden layers are fixed, and only the output layer depends on params, 
    # it is linear in params.
    # 
    # Let's try: 
    # 1. Generate a fixed random projection of inputs into a higher dimension (e.g., 50 dims).
    # 2. Apply a fixed non-linearity (e.g., tanh).
    # 3. The output is a linear combination of these 50 features using 10 params? 
    #    No, that would require 50 params. We only have 10.
    #
    # We must map 10 params to a flexible function.
    #
    # Idea: Use the 10 params to define the coefficients of a **Gaussian Process-like** 
    # or **RBF** kernel, but RBF centers are fixed.
    # If centers are fixed, and only weights are free, it's linear in weights.
    # But we only have 10 weights. So we can only have 10 RBF centers.
    #
    # Let's try 10 RBF centers fixed at random locations in [0.5, 2]^3.
    # pred = sum_{i=1 to 10} params[i] * exp(-gamma * ||input - center_i||^2)
    # This is linear in params. BFGS will find the best fit for these 10 fixed RBFs.
    # If the centers are well-placed, this can approximate a wide variety of functions.
    #
    # Why is this different from the quadratic basis?
    # RBFs are local. They can model sharp changes and non-polynomial shapes.
    # The quadratic basis is global.
    #
    # Implementation:
    # 1. Define 10 fixed centers uniformly distributed in [0.5, 2]^3.
    # 2. Compute distances from each input point to each center.
    # 3. Construct the 10 RBF features.
    # 4. Return features @ params.
    #
    # This keeps the model linear in params, ensuring BFGS converges reliably.
    # The richness comes from the non-linearity of the RBF basis.
    #
    # Let's refine the centers. Uniform random might be bad if they cluster.
    # We can use a fixed set of "stratified" centers.
    
    # Fixed random seed for reproducibility of centers
    rng = np.random.RandomState(42)
    
    # Generate 10 centers in [0.5, 2.0]^3
    centers = rng.uniform(0.5, 2.0, size=(10, 3))
    
    # Inputs are (n, 3) effectively, but x, y, z are 1D arrays of length n.
    inputs = np.column_stack([x, y, z])
    
    # Compute squared distances: (n, 10)
    # dist_sq[i, j] = ||inputs[i] - centers[j]||^2
    # Using broadcasting: (n, 1, 3) - (1, 10, 3) -> (n, 10, 3)
    diff = inputs[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # RBF: exp(-gamma * dist_sq)
    # Choose gamma. If inputs are in [0.5, 2], range is 1.5.
    # Max dist_sq is approx 3 * (1.5)^2 = 6.75.
    # If gamma is too small, RBFs are broad (like polynomials).
    # If gamma is too large, RBFs are very narrow (spiky).
    # A moderate gamma, e.g., 1.0 or 2.0, might be good.
    gamma = 2.0
    rbf_features = np.exp(-gamma * dist_sq)
    
    # Linear combination
    pred = rbf_features @ params
    
    # Clip to avoid overflow issues if params are huge (though BFGS should keep them sane)
    pred = np.clip(pred, -1e6, 1e6)
    
    return pred
# EVOLVE-BLOCK-END
