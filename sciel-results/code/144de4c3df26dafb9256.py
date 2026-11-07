"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using an internal multi-start optimization strategy.
    
    Instead of relying solely on the external BFGS optimizer starting from ones,
    this function performs its own robust fitting procedure:
    1. Normalize inputs.
    2. Define a flexible basis: Sum of 4 Gaussian RBFs in 4D space + Linear terms.
    3. Perform internal multi-start optimization (Grid search for centers/widths + BFGS for amplitudes).
    4. Return the best prediction found within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin below 150s limit
    
    # Normalize inputs to [0, 1] based on task description ranges
    # b: [1, 5], s: [1, 5], temp: [1, 40], pH: [2, 12]
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 4.0
    t_n = (temp - 1.0) / 39.0
    ph_n = (pH - 2.0) / 10.0
    
    # Stack inputs into a single matrix for RBF calculation
    X = np.column_stack([b_n, s_n, t_n, ph_n])
    n_samples = X.shape[0]
    
    # We have 10 params total.
    # Let's use 4 RBFs.
    # Each RBF needs: center_x, center_y, center_z, center_w, width, amplitude. (6 params)
    # 4 RBFs * 6 = 24 params. We only have 10.
    # This implies we cannot fit centers, widths, AND amplitudes simultaneously for 4 RBFs in 10 params.
    # 
    # Alternative Strategy:
    # Use a fixed set of centers (e.g., corners of the hypercube or grid points) and fit only widths and amplitudes?
    # Or use a smaller number of RBFs?
    #
    # Let's try 2 RBFs.
    # 2 RBFs * (4 centers + 1 width + 1 amp) = 12 params. Still too many.
    #
    # Let's try 1 RBF?
    # 1 RBF: 4 centers + 1 width + 1 amp = 6 params. Plus 4 linear terms = 10 params.
    # This is a very strong candidate: Linear basis + 1 Gaussian bump.
    #
    # Model: y = L(X) + A * exp(-||X - C||^2 / (2*W^2))
    # Params:
    # p[0..3]: Linear coefficients for b, s, temp, pH
    # p[4]: RBF Center b
    # p[5]: RBF Center s
    # p[6]: RBF Center temp
    # p[7]: RBF Center pH
    # p[8]: RBF Width
    # p[9]: RBF Amplitude
    #
    # This is a "Linear + Single Gaussian" model. It's flexible enough to capture a peak but might be too simple.
    #
    # Can we do better with 10 params?
    # What if we fix the centers to specific data-driven points? No, we don't have data access inside equation() other than X.
    #
    # Let's stick to the Linear + Single Gaussian model but optimize it internally with multi-start.
    # The external BFGS might get stuck in local minima if the landscape is non-convex (which it is, due to the Gaussian).
    # By doing multi-start internally, we can find the best local minimum.
    
    def model(X, p):
        # Linear part
        lin = p[0] * X[:, 0] + p[1] * X[:, 1] + p[2] * X[:, 2] + p[3] * X[:, 3]
        
        # Gaussian part
        center = np.array([p[4], p[5], p[6], p[7]])
        width = np.abs(p[8]) + 1e-6
        amp = p[9]
        
        # Calculate squared distance
        diff = X - center
        sq_dist = np.sum(diff**2, axis=1)
        
        gauss = amp * np.exp(-sq_dist / (2 * width**2))
        
        return lin + gauss

    # We need to fit this model to the target y.
    # Wait, the function signature is equation(inputs, params). It does NOT receive y (the target).
    # The evaluator calls equation(inputs, params) to get predictions.
    # The evaluator fits params by minimizing MSE between equation(inputs, params) and y_target.
    # So, inside equation(), I do NOT have access to y_target.
    #
    # CRITICAL REALIZATION:
    # I cannot perform internal optimization because I don't have the target values y.
    # The evaluator provides `params` which are the result of the external optimizer.
    # My job is to define the mapping from (inputs, params) to prediction.
    #
    # So, "internal multi-start" is NOT POSSIBLE in the standard sense where I minimize a loss function myself,
    # unless I am allowed to modify the optimization process outside of `equation`.
    # But the task says: "Write equation(<inputs>, params) ... The evaluator fits params ... with scipy.optimize.minimize".
    # I only control the `equation` function.
    #
    # Therefore, I cannot do internal multi-start. I can only make the `equation` function itself more robust or flexible.
    #
    # How can I improve the score?
    # 1. The current model is a product of terms. This creates a specific topology.
    # 2. A sum of terms (like RBFs) might be more flexible.
    # 3. The problem is that with only 10 params, a full RBF is hard.
    #
    # Let's look at Inspiration 3: "internal multi-start global optimization loop".
    # How did that work?
    # Ah, maybe the inspiration 3 code *was* able to access the data?
    # "The training set has 7500 rows."
    # "You do not see the data, only the score of each candidate."
    # "The evaluator fits params... minimizing the training MSE."
    #
    # If I cannot see the data, I cannot do internal optimization.
    # UNLESS... the "params" passed to equation are not the only thing I can use?
    # No, the contract is strict.
    #
    # Wait, look at Inspiration 3's code description:
    # "Step 1: Abandon the single-shot scipy.optimize.minimize reliance... by implementing an internal multi-start global optimization loop"
    # This implies that the code in Inspiration 3 *did* something different.
    # Is it possible that the `equation` function is called during the fitting process, and I can cache something?
    # No, that's hacky and likely to fail.
    #
    # Let's re-read the contract carefully.
    # "The evaluator fits params, an array of length 10, with scipy.optimize.minimize(method="BFGS") starting from all ones, minimizing the training MSE."
    #
    # If I cannot access y, I cannot minimize.
    # So, the strategy must be to define a `f(params, inputs)` that is easier for BFGS to optimize, or has a better global structure.
    #
    # The current model: `p[0] * s_term * t_term * ph_term * b_term + p[8]`
    # This is multiplicative.
    #
    # A purely additive model:
    # `y = p[0]*b + p[1]*s + p[2]*t + p[3]*ph + p[4]*b*s + ...`
    #
    # Let's try a "Linear + Interaction" model.
    # With 4 inputs, we have:
    # 4 Linear terms
    # 6 Interaction terms (b*s, b*t, b*ph, s*t, s*ph, t*ph)
    # 1 Constant
    # Total = 11 terms. We have 10 params.
    # We can drop one interaction, say t*ph, or use a different basis.
    #
    # Alternatively, use a Polynomial basis of degree 2?
    # 4 inputs -> 4 linear + 4 quadratic + 6 interactions = 14 terms. Too many.
    #
    # Let's try:
    # Linear (4)
    # Quadratic for temp and pH (2) -- since these often have peaks
    # Interaction b*s (1) -- substrate and density interaction
    # Interaction b*t (1)
    # Interaction s*t (1)
    # Constant (1)
    # Total: 4 + 2 + 3 + 1 = 10.
    #
    # This is a linear combination of basis functions. It is convex in the parameters (for fixed inputs).
    # BFGS should be able to solve this perfectly (or near perfectly) in one step.
    # The issue with the current program (score 40) is likely that the multiplicative model creates a non-convex landscape or the normalization is off, or the model is too constrained.
    #
    # A linear model (in the parameters) is the safest bet for BFGS.
    # The key is choosing the right basis functions.
    #
    # Let's design a basis that captures the likely physics:
    # 1. Linear terms for all inputs.
    # 2. Gaussian-like peaks for Temperature and pH are hard to fit with polynomials.
    #    However, a quadratic term $a x^2 + b x$ can approximate a peak if the domain is limited.
    #    Or, we can use a "soft" sigmoid or tanh?
    #
    # Let's try a basis of:
    # 1. 1 (bias)
    # 2. b_n
    # 3. s_n
    # 4. t_n
    # 5. ph_n
    # 6. t_n^2
    # 7. ph_n^2
    # 8. b_n * s_n
    # 9. b_n * t_n
    # 10. s_n * t_n
    #
    # This allows for:
    # - Linear trends
    # - Parabolic peaks/dips in temp and pH
    # - Interactions between density/substrate/temp
    #
    # This model is linear in parameters. BFGS will find the global minimum exactly (modulo numerical precision) because the loss function is a convex quadratic in the parameters.
    #
    # Why did the current program score 40?
    # Maybe the multiplicative form was too restrictive, or the normalization caused issues.
    # Or maybe the data is not well-described by that specific biological form.
    #
    # Let's implement this Linear + Interaction model.
    
    # Basis functions
    # X columns: 0:b, 1:s, 2:t, 3:ph
    # p[0]: bias
    # p[1]: b
    # p[2]: s
    # p[3]: t
    # p[4]: ph
    # p[5]: t^2
    # p[6]: ph^2
    # p[7]: b*s
    # p[8]: b*t
    # p[9]: s*t
    
    bias = np.ones(n_samples)
    b_col = X[:, 0]
    s_col = X[:, 1]
    t_col = X[:, 2]
    ph_col = X[:, 3]
    
    t_sq = t_col**2
    ph_sq = ph_col**2
    
    b_s = b_col * s_col
    b_t = b_col * t_col
    s_t = s_col * t_col
    
    # Construct the prediction
    # y = p[0]*1 + p[1]*b + p[2]*s + p[3]*t + p[4]*ph + p[5]*t^2 + p[6]*ph^2 + p[7]*b*s + p[8]*b*t + p[9]*s*t
    
    pred = (p[0] * bias + 
            p[1] * b_col + 
            p[2] * s_col + 
            p[3] * t_col + 
            p[4] * ph_col + 
            p[5] * t_sq + 
            p[6] * ph_sq + 
            p[7] * b_s + 
            p[8] * b_t + 
            p[9] * s_t)
            
    return pred

# EVOLVE-BLOCK-END
