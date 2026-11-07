"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF network with 10 parameters as centers.
    Fixed multi-scale widths and amplitudes.
    
    params: array of length 10, interpreted as centers c_0..c_9
    We use 5 basis functions, each with two centers (c_{2k}, c_{2k+1})? 
    No, let's use all 10 parameters. But we have 5 widths.
    
    Let's structure it as:
    We have 10 parameters. Let's use them as centers for 10 RBFs? 
    But the directive says widths and amplitudes are fixed to a multi-scale set.
    If we use 10 RBFs, we need 10 widths. But the set is {0.1, 0.2, 0.5, 1.0, 2.0}.
    We can repeat them or assign them cyclically.
    
    Alternative interpretation:
    "10 parameters are constrained to be the centers of the basis functions"
    This implies there are 10 basis functions.
    "widths and amplitudes are fixed to a multi-scale set (e.g., sigma_k in {0.1, 0.2, 0.5, 1.0, 2.0})"
    This suggests the widths are not parameters. They are fixed.
    So we have 10 centers (params[0..9]).
    We need to assign a width to each of the 10 centers.
    We can cycle through the 5 widths: [0.1, 0.2, 0.5, 1.0, 2.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    And amplitudes? "amplitudes are fixed". Maybe all amplitudes are 1? Or fixed to some values?
    If amplitudes are fixed to 1, the model is:
    y = sum_{i=0}^{9} exp(-0.5 * ((x - c_i) / sigma_i)^2)
    This might not have enough flexibility in amplitude.
    However, the directive says "widths and amplitudes are fixed".
    If amplitudes are fixed, we can't scale the contribution of each RBF.
    But BFGS will optimize the centers.
    If a specific RBF needs to be "turned off", it can move its center far away (e.g., to 100).
    exp(-0.5 * ((x - 100) / 0.1)^2) will be 0 for x in [-2, 2].
    So this is a valid mechanism.
    
    Let's define:
    centers = params
    widths = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 0.1, 0.2, 0.5, 1.0, 2.0])
    amplitudes = np.ones(10)  # Fixed to 1
    
    Wait, if amplitudes are all 1, and we sum 10 RBFs, the max value is 10.
    The target function might have different scales.
    But the scoring is normalized MSE: mean((pred - y)^2) / var(y).
    If the model can't match the scale, it will score poorly.
    However, the parameters are ONLY the centers.
    Is it possible to adjust scale? No.
    Unless... "amplitudes are fixed to a multi-scale set" means something else?
    Or maybe I should interpret "amplitudes" as part of the fixed structure that allows scaling?
    No, "fixed" usually means constant.
    
    Let's reconsider the "amplitudes" part.
    If amplitudes are fixed to 1, the model is quite rigid.
    What if I use a different set of fixed amplitudes?
    e.g., amplitudes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    Let's try a slightly different approach to ensure stability and better fit:
    Use the 10 parameters as centers.
    Use 5 widths, repeating them.
    Use amplitudes = 1.
    
    But wait, if the target function is, say, y = x^2, which has a minimum at 0.
    RBFs are peaked. A sum of RBFs is a sum of peaks.
    To fit a valley (like x^2 near 0 is a minimum, but x^2 goes up),
    actually x^2 is convex. RBFs are convex near the center?
    exp(-x^2) is concave at 0.
    So a sum of RBFs is a sum of "bumps".
    It can fit functions with local maxima.
    It can also fit flat functions (all centers far away -> 0).
    It can fit linear functions? Not really well, but approximations.
    
    Is there a risk of the optimizer getting stuck?
    BFGS starts at all ones.
    centers = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Initially, all RBFs are centered at 1.
    They will likely spread out.
    
    Let's check if we can improve the fixed amplitudes.
    Maybe use alternating signs? No, "amplitudes" usually implies positive scaling.
    
    Let's stick to the directive:
    centers = params
    widths = [0.1, 0.2, 0.5, 1.0, 2.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    amplitudes = np.ones(10)
    
    However, to give the model more power to match the scale of the target,
    perhaps I should not fix amplitudes to 1, but the directive says "fixed".
    If I deviate from the directive, I might violate the "research programme" constraint.
    But the goal is to beat the seed.
    The seed is linear: a*x + b.
    RBFs are strictly more flexible locally.
    
    One potential issue: numerical stability.
    If x and center are far apart, exp(-large) underflows to 0. This is fine.
    If x and center are close, exp(0) = 1.
    
    Let's implement this.
    
    To be safe, I'll use a slightly wider set of widths or ensure they cover the domain [-2, 2] well.
    Widths 0.1 to 2.0 cover local to global features.
    
    Implementation:
    """
    centers = params
    # Fixed widths, cycled to cover local to global features
    base_widths = np.array([0.2, 0.5, 1.0, 2.0, 4.0])
    widths = np.tile(base_widths, 2)
    
    # Fixed amplitudes with alternating signs to allow negative outputs
    amplitudes = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
    
    # Compute RBF outputs
    # x shape: (N,)
    # centers shape: (10,)
    # We want (N, 10) matrix of differences
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Scale by widths
    # widths shape: (10,)
    scaled_diff = diff / widths[np.newaxis, :]
    
    # Compute exponent
    # To avoid overflow/underflow issues, we can clip the exponent, but exp handles 0 fine.
    # exp(-0.5 * scaled_diff^2)
    rbf_values = np.exp(-0.5 * scaled_diff ** 2)
    
    # Weighted sum
    # amplitudes shape: (10,)
    predictions = np.sum(rbf_values * amplitudes[np.newaxis, :], axis=1)
    
    return predictions
# EVOLVE-BLOCK-END
