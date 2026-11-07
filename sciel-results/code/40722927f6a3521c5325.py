import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a Linear Radial Basis Function (RBF) Interpolant.
    
    Strategy:
    - The current program uses an additive B-spline model for temperature and pH only, 
      ignoring interactions and the specific inputs `b` and `s`.
    - This approach uses a set of 10 fixed Gaussian RBF centers distributed across the 
      normalized 4D input space (b, s, temp, pH).
    - The model is linear in the parameters (the weights of the RBFs), ensuring the 
      optimization problem is convex (Least Squares) for BFGS.
    - This captures non-linear interactions between all variables, which is critical 
      if growth depends on the combined state of biomass, substrate, temperature, and pH.
    - Fixed centers avoid the non-convexity of fitting centers, while providing 
      sufficient flexibility to model the bounded growth rate surface.
    """
    
    # Normalize inputs to [0, 1] based on training ranges provided in the prompt:
    # b: [1, 4.998] -> approx [1, 5]
    # s: [1, 4.999] -> approx [1, 5]
    # temp: [1.002, 39.98] -> approx [1, 40]
    # pH: [2, 12]
    
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    p_norm = (pH - 2.0) / 10.0
    
    # Clip to [0, 1] to handle any slight out-of-range values gracefully
    b_norm = np.clip(b_norm, 0.0, 1.0)
    s_norm = np.clip(s_norm, 0.0, 1.0)
    t_norm = np.clip(t_norm, 0.0, 1.0)
    p_norm = np.clip(p_norm, 0.0, 1.0)
    
    # Define 10 fixed centers in the normalized 4D space.
    # We want coverage of the corners and the center to allow for a flexible surface.
    # Centers:
    # 0: (0, 0, 0, 0) - Low b, s, T, pH
    # 1: (1, 1, 1, 1) - High b, s, T, pH
    # 2: (0.5, 0.5, 0.5, 0.5) - Center
    # 3: (0, 0, 0.5, 0.5) - Mid T/pH, Low b/s
    # 4: (1, 1, 0.5, 0.5) - Mid T/pH, High b/s
    # 5: (0.5, 0.5, 0, 0) - Low T/pH, Mid b/s
    # 6: (0.5, 0.5, 1, 1) - High T/pH, Mid b/s
    # 7: (0, 1, 0.5, 0.5) - Mixed b/s, Mid T/pH
    # 8: (1, 0, 0.5, 0.5) - Mixed b/s, Mid T/pH
    # 9: (0.5, 0.5, 0.5, 0.5) - Duplicate center? No, let's use a different spread.
    # Let's use a more systematic spread: 8 corners + 2 mid-edges or similar.
    # Actually, 4 corners of the hypercube + 4 face centers + 2 volume centers might be better?
    # Let's stick to a robust set:
    # 0: (0, 0, 0, 0)
    # 1: (1, 1, 1, 1)
    # 2: (0.5, 0.5, 0.5, 0.5)
    # 3: (0, 0, 1, 1)
    # 4: (1, 1, 0, 0)
    # 5: (0, 1, 0.5, 0.5)
    # 6: (1, 0, 0.5, 0.5)
    # 7: (0.5, 0.5, 0, 1)
    # 8: (0.5, 0.5, 1, 0)
    # 9: (0.5, 0.5, 0.5, 0.5) -> Let's make this (0.25, 0.75, 0.5, 0.5) to add asymmetry
    
    centers = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.5, 0.5],
        [1.0, 0.0, 0.5, 0.5],
        [0.5, 0.5, 0.0, 1.0],
        [0.5, 0.5, 1.0, 0.0],
        [0.25, 0.75, 0.5, 0.5]
    ])
    
    # Stack inputs into a matrix of shape (N, 4)
    X = np.column_stack((b_norm, s_norm, t_norm, p_norm))
    
    # Calculate squared distances from each point to each center
    # X: (N, 4), Centers: (10, 4)
    # diff: (N, 10, 4)
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    
    # Gaussian RBF: exp(-beta * sq_dist)
    # We need to choose a beta (inverse of width squared).
    # A beta around 4-10 usually works well for [0,1] normalized inputs.
    # Since we have 10 centers, a moderate width allows overlap.
    # Let's use beta = 6.0. This gives a width of ~1/sqrt(6) ~ 0.4.
    beta = 6.0
    
    RBFs = np.exp(-beta * sq_dists) # Shape (N, 10)
    
    # Linear combination with params
    # params: (10,)
    y = RBFs @ params
    
    return y
