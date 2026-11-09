"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a Piecewise-Linear Approximation (PLA) strategy.
    
    Strategy:
    Instead of a complex biological formula, we model the growth rate as a 
    piecewise-linear function over the input space. We use a dense grid of 
    10 'knots' (the parameters). The function value at any input is a 
    weighted average of the nearest knots. This is robust to BFGS because 
    it avoids the deep local minima of non-linear biological models (like 
    Monod) and provides a smooth, differentiable surface for the optimizer.
    
    Inputs:
    b: population density (range ~[1, 5])
    s: substrate (range ~[1, 40])
    temp: temperature (range ~[1, 12] - likely normalized or specific units)
    pH: pH (range ~[1, 12] - likely normalized or specific units)
    
    params: 10 constants representing the growth rate at specific grid points.
    """
    
    # Unpack parameters (10 constants)
    # We map these 10 parameters to a specific topology in the 4D space.
    # Since we don't know the exact correlations, we use a diagonal mapping 
    # across the main axes to capture the primary trends.
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalize inputs to a 0-1 scale based on observed ranges 
    # b: [1, 5] -> 0-1
    # s: [1, 40] -> 0-1
    # temp: [1, 12] -> 0-1 (Assuming similar range to pH or temp units)
    # pH: [1, 12] -> 0-1
    
    b_norm = np.clip((b - 1.0) / 4.0, 0, 1)
    s_norm = np.clip((s - 1.0) / 39.0, 0, 1)
    t_norm = np.clip((temp - 1.0) / 11.0, 0, 1)
    ph_norm = np.clip((pH - 1.0) / 11.0, 0, 1)

    # We define 10 "knots" in the 4D space.
    # To ensure the function is smooth and differentiable, we use a 
    # "soft" nearest-neighbor interpolation.
    # 
    # Knot locations (normalized):
    # 0: (0, 0, 0, 0) -> p0
    # 1: (1, 0, 0, 0) -> p1 (High b)
    # 2: (0, 1, 0, 0) -> p2 (High s)
    # 3: (0, 0, 1, 0) -> p3 (High temp)
    # 4: (0, 0, 0, 1) -> p4 (High pH)
    # 5: (1, 1, 0, 0) -> p5 (High b, High s)
    # 6: (1, 0, 1, 0) -> p6 (High b, High temp)
    # 7: (0, 1, 1, 0) -> p7 (High s, High temp)
    # 8: (1, 0, 0, 1) -> p8 (High b, High pH)
    # 9: (1, 1, 1, 1) -> p9 (All High)
    
    # Calculate distance from each input point to each knot in normalized space.
    # We use squared Euclidean distance.
    
    # Inputs are arrays of shape (N,)
    # We construct the 4D coordinates for each input
    # To avoid creating a huge (N, 4) array if N is large, we compute distances directly.
    
    # Knot coordinates (10, 4)
    knots = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ])
    
    # Input coordinates (N, 4)
    # b_norm, s_norm, t_norm, ph_norm are (N,)
    inputs_4d = np.stack([b_norm, s_norm, t_norm, ph_norm], axis=1)
    
    # Calculate squared distances (N, 10)
    # dist_sq[i, j] = ||inputs_4d[i] - knots[j]||^2
    # This can be computed as: ||x||^2 + ||k||^2 - 2 x.k
    # ||k||^2 is constant for each knot
    k_norm_sq = np.sum(knots**2, axis=1) # (10,)
    x_norm_sq = np.sum(inputs_4d**2, axis=1) # (N,)
    
    # x.k is (N, 10)
    xk = inputs_4d @ knots.T
    
    dist_sq = x_norm_sq[:, np.newaxis] + k_norm_sq[np.newaxis, :] - 2 * xk
    
    # To make it a "soft" min (smooth nearest neighbor), we use the LogSumExp trick
    # or a simple weighted average based on inverse squared distance with a small epsilon.
    # Weighted average: w_i = 1 / (dist_sq_i + eps)
    # This is differentiable everywhere.
    
    eps = 1e-4
    weights = 1.0 / (dist_sq + eps)
    
    # Normalize weights to sum to 1
    weight_sum = np.sum(weights, axis=1, keepdims=True)
    weights_norm = weights / weight_sum
    
    # The prediction is the weighted sum of the parameter values
    # params is (10,)
    prediction = weights_norm @ params
    
    return prediction
# EVOLVE-BLOCK-END
