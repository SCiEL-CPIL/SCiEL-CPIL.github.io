import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1] based on training ranges
    # b: [1, 5] -> shift by 1, divide by 4 => [0, 1]
    # s: [1, 5] -> shift by 1, divide by 4 => [0, 1]
    # temp: [1, 40] -> shift by 1, divide by 39 => [0, 1]
    # pH: [2, 12] -> shift by 2, divide by 10 => [0, 1]
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    p = params
    
    # Strategy: Fixed-Center Gaussian RBF with Linear Parameters
    # Use 10 parameters as coefficients for 10 fixed radial basis functions.
    # This allows the model to form arbitrary smooth surfaces without
    # constraining the shape of temperature/pH responses to be single-peaked
    # or substrate to be Monod. The centers are fixed to cover the input space.
    
    # Define 10 centers in the 4D normalized space [0,1]^4
    # We want centers that cover the likely regions of high growth.
    # High growth likely at:
    # - Moderate/High substrate (s_norm > 0.5)
    # - Optimal temperature (t_norm ~ 0.5-0.7, i.e. 20-30C)
    # - Optimal pH (ph_norm ~ 0.5-0.7, i.e. 7-9)
    # - Low/Moderate biomass (b_norm < 0.5)
    
    # Let's create a grid of centers or specific points.
    # 10 params -> 10 centers.
    
    # Center 0: Low S, Optimal T, Optimal pH, Low B
    c0 = np.array([0.1, 0.2, 0.6, 0.6])
    # Center 1: High S, Optimal T, Optimal pH, Low B
    c1 = np.array([0.1, 0.8, 0.6, 0.6])
    # Center 2: High S, Optimal T, Optimal pH, High B
    c2 = np.array([0.9, 0.8, 0.6, 0.6])
    # Center 3: Low S, Optimal T, Optimal pH, High B
    c3 = np.array([0.9, 0.2, 0.6, 0.6])
    # Center 4: High S, Low T, Optimal pH, Low B
    c4 = np.array([0.1, 0.8, 0.2, 0.6])
    # Center 5: High S, High T, Optimal pH, Low B
    c5 = np.array([0.1, 0.8, 0.9, 0.6])
    # Center 6: High S, Optimal T, Low pH, Low B
    c6 = np.array([0.1, 0.8, 0.6, 0.2])
    # Center 7: High S, Optimal T, High pH, Low B
    c7 = np.array([0.1, 0.8, 0.6, 0.9])
    # Center 8: Mid S, Optimal T, Optimal pH, Mid B
    c8 = np.array([0.5, 0.5, 0.6, 0.6])
    # Center 9: High S, Optimal T, Optimal pH, Very Low B
    c9 = np.array([0.0, 0.8, 0.6, 0.6])
    
    centers = np.array([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9])
    
    # Width of the RBFs. 
    # We have 10 parameters, so we can't fit widths individually.
    # We must use a fixed width or a width derived from a parameter?
    # If we use a fixed width, we have 10 coeffs for 10 centers.
    # Let's use a fixed width that allows some overlap.
    # Standard deviation in normalized units. 0.3 seems reasonable.
    sigma = 0.3
    sigma_sq_2 = 2.0 * sigma**2
    
    # Compute distances from each input point to each center
    # inputs shape: (N, 4)
    # centers shape: (10, 4)
    # diff shape: (N, 10, 4)
    
    inputs = np.stack([b_norm, s_norm, t_norm, ph_norm], axis=-1)
    
    # diff = inputs[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # dist_sq = np.sum(diff**2, axis=-1)
    # rbf_vals = np.exp(-dist_sq / sigma_sq_2)
    
    # Optimization:
    # diff is (N, 10, 4)
    # dist_sq is (N, 10)
    
    diff = inputs[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    rbf_vals = np.exp(-dist_sq / sigma_sq_2)
    
    # rbf_vals shape: (N, 10)
    # p shape: (10,)
    # y = rbf_vals @ p
    
    y = rbf_vals @ p
    
    return y
