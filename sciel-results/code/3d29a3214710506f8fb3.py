"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a Mixture of Experts (MoE) model.
    
    Strategy:
    1. Define three fixed linear experts. Each expert models a different regime 
       (e.g., low temp, optimal temp, high temp/death) using linear combinations 
       of the normalized inputs.
    2. The inputs (temp, pH, s) determine a soft-gated convex combination of these experts.
       The gating weights are derived from the inputs and fixed, ensuring the model 
       is smooth but can transition sharply between regimes.
    3. The 10 parameters are the coefficients for the three experts (bias + 3 slopes each = 12, 
       but we constrain to 10 by sharing the bias or fixing some slopes? 
       Actually, let's use 10 params: 
       Expert 1: bias, slope_t, slope_ph, slope_s (4)
       Expert 2: bias, slope_t, slope_ph, slope_s (4)
       Expert 3: bias, slope_t (2) -- assuming pH and s effects are shared or less critical in extreme regimes?
       Better: 
       Let's use a simpler parameterization.
       We have 3 experts. 
       Let's say each expert is a linear function of normalized inputs.
       To fit in 10 params, we can share some coefficients or use a different structure.
       
       Alternative MoE structure with 10 params:
       - 3 experts, each is a linear function: E_k(x) = w_k0 + w_k1*b_n + w_k2*s_n + w_k3*t_n + w_k4*pH_n
       That's 5 params per expert * 3 = 15 params. Too many.
       
       Let's reduce the feature set for experts or share parameters.
       Maybe experts only depend on temp and pH? 
       E_k(t, pH) = w_k0 + w_k1*t_n + w_k2*pH_n
       3 params per expert * 3 = 9 params.
       Plus 1 global scaling factor? Or one more interaction?
       Let's try:
       Params:
       p[0..2]: Expert 1 (bias, t_slope, pH_slope)
       p[3..5]: Expert 2 (bias, t_slope, pH_slope)
       p[6..8]: Expert 3 (bias, t_slope, pH_slope)
       p[9]: Global scaling factor for the whole mixture? Or maybe a slope for 's' shared?
       
       Let's use p[9] as a shared slope for 's' (substrate) applied to the final output.
       So: Prediction = (Mixture of Experts on t, pH) + p[9] * s_norm
       
       Gating:
       The gates should be based on temp and pH to switch between experts.
       Let's use fixed Gaussian-like gates centered at different temp values.
       Gate 1: active at low temp (t_norm ~ 0.2)
       Gate 2: active at mid temp (t_norm ~ 0.5)
       Gate 3: active at high temp (t_norm ~ 0.8)
       
       We need to ensure the gates sum to 1.
       g_k = exp(-((t_norm - c_k)^2) / (2*sigma^2))
       weights = g / sum(g)
       
       This is a structural change from polynomial to MoE.
    """
    # Normalization
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    temp_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    # Extract parameters
    # Expert 1: p[0]=bias, p[1]=t_slope, p[2]=ph_slope
    e1 = params[0] + params[1] * temp_norm + params[2] * ph_norm
    # Expert 2: p[3]=bias, p[4]=t_slope, p[5]=ph_slope
    e2 = params[3] + params[4] * temp_norm + params[5] * ph_norm
    # Expert 3: p[6]=bias, p[7]=t_slope, p[8]=ph_slope
    e3 = params[6] + params[7] * temp_norm + params[8] * ph_norm
    
    # Gating mechanism based on temp_norm
    # Centers: 0.2, 0.5, 0.8
    # Sigma: 0.15 (fixed, sharp transitions)
    sigma = 0.15
    c1, c2, c3 = 0.2, 0.5, 0.8
    
    g1 = np.exp(-((temp_norm - c1) ** 2) / (2 * sigma ** 2))
    g2 = np.exp(-((temp_norm - c2) ** 2) / (2 * sigma ** 2))
    g3 = np.exp(-((temp_norm - c3) ** 2) / (2 * sigma ** 2))
    
    sum_g = g1 + g2 + g3
    # Avoid division by zero
    sum_g = np.where(sum_g == 0, 1e-10, sum_g)
    
    w1 = g1 / sum_g
    w2 = g2 / sum_g
    w3 = g3 / sum_g
    
    # Mixture
    mixture = w1 * e1 + w2 * e2 + w3 * e3
    
    # Add substrate effect (shared slope)
    # p[9] is the slope for s_norm
    prediction = mixture + params[9] * s_norm
    
    return prediction
# EVOLVE-BLOCK-END
