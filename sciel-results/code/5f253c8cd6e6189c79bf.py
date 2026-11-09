import numpy as np

def equation(b, s, temp, pH, params):
    p = params
    
    # Ensure inputs are float arrays for safe operations
    b = np.asarray(b, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    pH = np.asarray(pH, dtype=np.float64)
    
    # Normalize inputs to [0, 1] based on training ranges
    # b in [1, 4.998], s in [1, 4.999], temp in [1.002, 39.98], pH in [2, 12]
    bn = (b - 1.0) / (4.998 - 1.0)
    sn = (s - 1.0) / (4.999 - 1.0)
    tn = (temp - 1.002) / (39.98 - 1.002)
    pn = (pH - 2.0) / (12.0 - 2.0)
    
    # Clip normalized inputs to [0, 1] to prevent extreme values in RBFs
    bn = np.clip(bn, 0.0, 1.0)
    sn = np.clip(sn, 0.0, 1.0)
    tn = np.clip(tn, 0.0, 1.0)
    pn = np.clip(pn, 0.0, 1.0)
    
    # Define RBF centers and widths
    # We have 10 parameters total.
    # Strategy: Use a generalized additive model with fixed basis functions.
    # Basis functions:
    # 1. Intercept (1)
    # 2-3. RBFs for b (2 centers)
    # 4-5. RBFs for s (2 centers)
    # 6-7. RBFs for temp (2 centers)
    # 8-9. RBFs for pH (2 centers)
    # 10. Interaction term: temp * pH (or similar cross-term)
    
    # Centers for each variable (equally spaced in [0, 1])
    # Using 2 centers per variable: 0.25 and 0.75
    centers_b = np.array([0.25, 0.75])
    centers_s = np.array([0.25, 0.75])
    centers_t = np.array([0.25, 0.75])
    centers_p = np.array([0.25, 0.75])
    
    # Width for RBFs
    # A width of 0.3 allows for some smoothness and local adaptation
    width = 0.3
    
    # Compute RBF values
    # b_rbf shape: (n, 2)
    b_rbf = np.exp(-((bn[:, None] - centers_b[None, :]) ** 2) / (2 * width ** 2))
    # s_rbf shape: (n, 2)
    s_rbf = np.exp(-((sn[:, None] - centers_s[None, :]) ** 2) / (2 * width ** 2))
    # t_rbf shape: (n, 2)
    t_rbf = np.exp(-((tn[:, None] - centers_t[None, :]) ** 2) / (2 * width ** 2))
    # p_rbf shape: (n, 2)
    p_rbf = np.exp(-((pn[:, None] - centers_p[None, :]) ** 2) / (2 * width ** 2))
    
    # Interaction term
    # The product of the two "optimal" RBFs for temp and pH
    # This captures if the optimum shifts or if there's a synergistic effect
    # f10 = t_rbf[:, 1] * p_rbf[:, 1]  # Interaction of the "high" centers
    # Or simpler: just the product of normalized temp and pH?
    # Let's use the product of the RBFs at the "center" of the physiological range
    # Actually, let's use a simpler interaction: tn * pn
    # But RBF interaction is more flexible. Let's use t_rbf[:, 0] * p_rbf[:, 0]
    inter = t_rbf[:, 0] * p_rbf[:, 0]
    
    # Assemble the design matrix
    # Columns:
    # 0: 1 (intercept)
    # 1: b_rbf[:, 0]
    # 2: b_rbf[:, 1]
    # 3: s_rbf[:, 0]
    # 4: s_rbf[:, 1]
    # 5: t_rbf[:, 0]
    # 6: t_rbf[:, 1]
    # 7: p_rbf[:, 0]
    # 8: p_rbf[:, 1]
    # 9: inter
    
    # We can compute the prediction as a dot product
    # pred = p[0] + p[1]*b_rbf[:,0] + ... + p[9]*inter
    
    # For efficiency, we can stack the basis functions
    # But explicit addition is also fast enough for 7500 rows
    
    pred = np.full_like(bn, p[0])
    pred += b_rbf @ p[1:3]
    pred += s_rbf @ p[3:5]
    pred += t_rbf @ p[5:7]
    pred += p_rbf @ p[7:9]
    pred += inter * p[9]
    
    return pred
