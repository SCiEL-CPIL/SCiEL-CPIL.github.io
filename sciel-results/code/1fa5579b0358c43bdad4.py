import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Universal Smooth Basis Expansion with Fixed Centers.
    
    Instead of assigning specific physical meanings to each parameter (e.g., 
    'Young's Modulus', 'Yield Stress'), we define a fixed set of 10 basis 
    functions B_i(s, t). The output is a linear combination:
    sigma = sum_{i=0}^{9} p_i * B_i(s, t)
    
    Since the basis functions are fixed, the optimization problem for the 
    weights p is linear (if we ignore the fact that BFGS is used, but the 
    gradient with respect to p is linear in the residuals, making the 
    landscape convex in p). This avoids local minima associated with 
    non-linear parameter interactions (like fitting a center and width 
    simultaneously).
    
    The basis functions are chosen to span the typical features of 
    stress-strain curves:
    1. Linear elastic trend: s
    2. Quadratic hardening: s^2
    3. Cubic hardening: s^3
    4. Temperature-dependent linear: t * s
    5. Temperature-dependent quadratic: t * s^2
    6. Localized Gaussian bump at low strain (yield transition): exp(-((s-0.05)/0.05)^2)
    7. Localized Gaussian bump at mid strain: exp(-((s-0.3)/0.1)^2)
    8. Localized Gaussian bump at high strain (plateau/necking): exp(-((s-0.8)/0.1)^2)
    9. Constant offset: 1
    10. Temperature constant offset: t
    
    This approach is robust because:
    - BFGS starting from ones will converge quickly to the least-squares 
      solution for the weights.
    - The fixed centers allow the model to capture non-linear features 
      (like yield points) without the instability of optimizing those 
      centers.
    - It is a "symbolic" regression in the sense that it learns the 
      coefficients of a predefined symbolic structure.
    """
    s = strain
    t = temp
    p = params
    
    # Ensure inputs are 1D arrays for consistent broadcasting if needed, 
    # though they are likely already aligned.
    s = np.asarray(s, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    
    # Basis Functions
    # 1. Linear
    B0 = s
    # 2. Quadratic
    B1 = s * s
    # 3. Cubic
    B2 = s * s * s
    # 4. Temp-Linear
    B3 = t * s
    # 5. Temp-Quadratic
    B4 = t * s * s
    # 6. Gaussian at s=0.05, width=0.05 (Yield region)
    # Clip argument to prevent overflow in exp, though squared values are bounded
    # if s is in [-0.002, 0.96].
    # (s - 0.05) / 0.05 ranges from ~ -1.004 to ~ 18.27. Squared is up to ~334.
    # exp(-334) is 0. Safe.
    B5 = np.exp(-((s - 0.05) / 0.05) ** 2)
    # 7. Gaussian at s=0.3, width=0.1 (Hardening region)
    # (s - 0.3) / 0.1 ranges from ~ -3.3 to ~ 6.6. Squared up to ~44. Safe.
    B6 = np.exp(-((s - 0.3) / 0.1) ** 2)
    # 8. Gaussian at s=0.8, width=0.1 (Plateau region)
    # (s - 0.8) / 0.1 ranges from ~ -8.8 to ~ 1.6. Squared up to ~78. Safe.
    B7 = np.exp(-((s - 0.8) / 0.1) ** 2)
    # 9. Constant
    B8 = np.ones_like(s)
    # 10. Temp Constant
    B9 = t
    
    # Linear Combination
    # sigma = p0*B0 + p1*B1 + ... + p9*B9
    stress = (p[0] * B0 + 
              p[1] * B1 + 
              p[2] * B2 + 
              p[3] * B3 + 
              p[4] * B4 + 
              p[5] * B5 + 
              p[6] * B6 + 
              p[7] * B7 + 
              p[8] * B8 + 
              p[9] * B9)
              
    return stress
