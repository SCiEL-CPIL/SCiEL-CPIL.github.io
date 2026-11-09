import numpy as np

def equation(b, s, temp, pH, params):
    p = params
    
    # Additive RBF model: linear in parameters, convex loss for BFGS
    # 10 parameters: 1 intercept + 2 per variable (center, width)
    # But we need 10 coefficients for basis functions
    # Let's use: intercept + 2 RBFs for s + 2 RBFs for temp + 2 RBFs for pH + 2 RBFs for b
    # That's 1 + 8 = 9... need 10. Let's do:
    # intercept + 3 RBFs for s + 2 RBFs for temp + 2 RBFs for pH + 1 RBF for b = 1+3+2+2+1=9
    # Or: intercept + 2 RBFs per variable = 1 + 8 = 9, plus one more
    # Let's use fixed centers and widths, and 10 coefficients as weights
    
    # Fixed RBF centers and widths based on input ranges
    # b in [1, 4.998], s in [1, 39.98], temp in [1, 4.999] (wait, temp range?)
    # From task: inputs in training range: [1, 4.998], [1, 4.999], [1.002, 39.98], [2, 12]
    # So b in [1, 4.998], s in [1, 4.999], temp in [1.002, 39.98], pH in [2, 12]
    
    # Normalize inputs to [0, 1]
    bn = (b - 1.0) / (4.998 - 1.0)
    sn = (s - 1.0) / (4.999 - 1.0)
    tn = (temp - 1.002) / (39.98 - 1.002)
    pn = (pH - 2.0) / (12.0 - 2.0)
    
    # Define RBF centers (equally spaced) and widths
    # For each variable, use 2 RBFs
    # Total: 1 intercept + 2*4 = 9... need 10
    # Let's use 3 RBFs for s (most important) and 2 for others: 1 + 3 + 2 + 2 + 2 = 10
    
    # RBF centers
    b_centers = np.array([0.25, 0.75])
    s_centers = np.array([0.25, 0.5, 0.75])
    t_centers = np.array([0.25, 0.75])
    p_centers = np.array([0.25, 0.75])
    
    # RBF widths (fixed, moderate)
    width = 0.3
    
    # Compute RBF values
    b_rbf = np.exp(-((bn[:, None] - b_centers[None, :]) ** 2) / (2 * width ** 2))  # (n, 2)
    s_rbf = np.exp(-((sn[:, None] - s_centers[None, :]) ** 2) / (2 * width ** 2))  # (n, 3)
    t_rbf = np.exp(-((tn[:, None] - t_centers[None, :]) ** 2) / (2 * width ** 2))  # (n, 2)
    p_rbf = np.exp(-((pn[:, None] - p_centers[None, :]) ** 2) / (2 * width ** 2))  # (n, 2)
    
    # Parameters: [intercept, b1, b2, s1, s2, s3, t1, t2, p1, p2]
    intercept = p[0]
    b_coeffs = p[1:3]
    s_coeffs = p[3:6]
    t_coeffs = p[6:8]
    p_coeffs = p[8:10]
    
    # Compute prediction
    pred = np.full_like(bn, intercept)
    pred += b_rbf @ b_coeffs
    pred += s_rbf @ s_coeffs
    pred += t_rbf @ t_coeffs
    pred += p_rbf @ p_coeffs
    
    return pred
