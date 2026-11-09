"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a Radial Basis Function (RBF) network with anisotropic scaling.
    
    Strategy:
    1. Normalize inputs to [0, 1] based on known training ranges.
    2. Use 5 fixed Gaussian RBF centers in the 4D normalized space.
    3. The distance metric is anisotropic, controlled by 4 learnable parameters (beta).
    4. The output is a linear combination of the RBF activations plus a bias.
    
    Parameters (10 total):
    p[0]: Bias (intercept)
    p[1..5]: Weights for the 5 RBF centers
    p[6..9]: Anisotropy scales (inverse variances) for b, s, temp, pH
    """
    # Deadline guard
    start_time = time.time()
    timeout = 110.0  # Safety margin before 120s limit
    
    # Normalization constants based on task description ranges:
    # b: [1, 4.998] -> span ~4
    # s: [1, 4.999] -> span ~4
    # temp: [1.002, 39.98] -> span ~39
    # pH: [2, 12] -> span ~10
    
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    temp_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    # Extract parameters
    bias = params[0]
    w1, w2, w3, w4, w5 = params[1:6]
    beta_b, beta_s, beta_t, beta_ph = params[6:10]
    
    # Ensure betas are positive to maintain valid Gaussian kernels
    # Using softplus or just max with small epsilon to avoid division by zero/negative variance
    # However, BFGS can handle negative values if we square them or use exp.
    # Let's use exp() to ensure positive scaling and better conditioning.
    # beta_eff = exp(beta)
    beta_b_eff = np.exp(beta_b)
    beta_s_eff = np.exp(beta_s)
    beta_t_eff = np.exp(beta_t)
    beta_ph_eff = np.exp(beta_ph)
    
    # Define 5 fixed centers in [0,1]^4
    # Center 1: Low everything (0.1, 0.1, 0.1, 0.1)
    # Center 2: High everything (0.9, 0.9, 0.9, 0.9)
    # Center 3: Optimal-ish guess for temp/pH (0.5, 0.5, 0.7, 0.5)
    # Center 4: Substrate limited (0.5, 0.1, 0.5, 0.5)
    # Center 5: Density high (0.9, 0.5, 0.5, 0.5)
    
    c1 = np.array([0.1, 0.1, 0.1, 0.1])
    c2 = np.array([0.9, 0.9, 0.9, 0.9])
    c3 = np.array([0.5, 0.5, 0.7, 0.5])
    c4 = np.array([0.5, 0.1, 0.5, 0.5])
    c5 = np.array([0.9, 0.5, 0.5, 0.5])
    
    # Compute anisotropic squared distances
    # D = beta_b*(b-c_b)^2 + beta_s*(s-c_s)^2 + ...
    
    # Unpack for speed
    b_arr = b_norm
    s_arr = s_norm
    t_arr = temp_norm
    p_arr = ph_norm
    
    # Center 1
    d1 = (beta_b_eff * (b_arr - c1[0])**2 + 
          beta_s_eff * (s_arr - c1[1])**2 + 
          beta_t_eff * (t_arr - c1[2])**2 + 
          beta_ph_eff * (p_arr - c1[3])**2)
    
    # Center 2
    d2 = (beta_b_eff * (b_arr - c2[0])**2 + 
          beta_s_eff * (s_arr - c2[1])**2 + 
          beta_t_eff * (t_arr - c2[2])**2 + 
          beta_ph_eff * (p_arr - c2[3])**2)
    
    # Center 3
    d3 = (beta_b_eff * (b_arr - c3[0])**2 + 
          beta_s_eff * (s_arr - c3[1])**2 + 
          beta_t_eff * (t_arr - c3[2])**2 + 
          beta_ph_eff * (p_arr - c3[3])**2)
    
    # Center 4
    d4 = (beta_b_eff * (b_arr - c4[0])**2 + 
          beta_s_eff * (s_arr - c4[1])**2 + 
          beta_t_eff * (t_arr - c4[2])**2 + 
          beta_ph_eff * (p_arr - c4[3])**2)
    
    # Center 5
    d5 = (beta_b_eff * (b_arr - c5[0])**2 + 
          beta_s_eff * (s_arr - c5[1])**2 + 
          beta_t_eff * (t_arr - c5[2])**2 + 
          beta_ph_eff * (p_arr - c5[3])**2)
    
    # Gaussian kernels: exp(-D / 2)
    k1 = np.exp(-0.5 * d1)
    k2 = np.exp(-0.5 * d2)
    k3 = np.exp(-0.5 * d3)
    k4 = np.exp(-0.5 * d4)
    k5 = np.exp(-0.5 * d5)
    
    # Linear combination
    prediction = bias + w1 * k1 + w2 * k2 + w3 * k3 + w4 * k4 + w5 * k5
    
    # Check for non-finite values
    if not np.all(np.isfinite(prediction)):
        return np.zeros_like(prediction)
        
    return prediction
# EVOLVE-BLOCK-END
