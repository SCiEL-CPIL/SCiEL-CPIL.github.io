"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation using a 10-parameter logistic mixture model.
    
    Each pair of parameters (slope, intercept) defines a smooth step (logistic) function.
    The superposition of 5 such steps can approximate piecewise-constant or sharply
    varying functions that global polynomials and smooth splines fail to capture.
    
    params: array of length 10.
    - params[0], params[2], params[4], params[6], params[8]: slopes for the 5 logistic steps
    - params[1], params[3], params[5], params[7], params[9]: intercepts (center points)
    
    The output is the sum of the 5 logistic functions:
        f(x) = sum_{i=0}^{4} 1 / (1 + exp(-slope_i * (x - intercept_i)))
    
    This is nonlinear in the slopes but each term is a smooth step. By combining
    multiple steps with different centers and slopes, we can approximate functions
    with sharp transitions, plateaus, or piecewise-constant behavior.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract slopes and intercepts
    slopes = params[0::2]    # params[0], params[2], params[4], params[6], params[8]
    intercepts = params[1::2]  # params[1], params[3], params[5], params[7], params[9]
    
    # Compute each logistic term: 1 / (1 + exp(-slope * (x - intercept)))
    # Shape: (5, N)
    diff = x[np.newaxis, :] - intercepts[:, np.newaxis]  # (5, N)
    scaled = slopes[:, np.newaxis] * diff  # (5, N)
    
    # Use a stable computation to avoid overflow in exp
    # For large positive scaled, exp(-scaled) -> 0, so 1/(1+0) = 1
    # For large negative scaled, exp(-scaled) -> inf, so 1/(1+inf) = 0
    # Use np.where to handle large values
    # Actually, let's use a numerically stable sigmoid
    # sigmoid(z) = 1 / (1 + exp(-z))
    # For z > 0: exp(-z) is small, so sigmoid ≈ 1
    # For z < 0: exp(-z) is large, so sigmoid ≈ 0
    # We can compute as:
    #   sigmoid(z) = exp(-z) / (1 + exp(-z)) for z < 0
    #   sigmoid(z) = 1 / (1 + exp(z)) for z >= 0
    
    # Simpler: use np.where with clipping to avoid overflow
    # Clip scaled to [-500, 500] to avoid overflow in exp
    scaled_clipped = np.clip(scaled, -500, 500)
    
    # Compute logistic
    # 1 / (1 + exp(-scaled_clipped))
    # For very negative scaled_clipped, exp(-scaled_clipped) is huge -> overflow
    # So use the stable form:
    #   where scaled_clipped >= 0: 1 / (1 + exp(-scaled_clipped))
    #   where scaled_clipped < 0: exp(scaled_clipped) / (1 + exp(scaled_clipped))
    
    pos_mask = scaled_clipped >= 0
    neg_mask = ~pos_mask
    
    logits = np.empty_like(scaled_clipped)
    
    # For positive: 1 / (1 + exp(-z))
    exp_neg = np.exp(-scaled_clipped[pos_mask])
    logits[pos_mask] = 1.0 / (1.0 + exp_neg)
    
    # For negative: exp(z) / (1 + exp(z))
    exp_pos = np.exp(scaled_clipped[neg_mask])
    logits[neg_mask] = exp_pos / (1.0 + exp_pos)
    
    # Sum over the 5 logistic terms
    result = np.sum(logits, axis=0)
    
    return result
# EVOLVE-BLOCK-END
