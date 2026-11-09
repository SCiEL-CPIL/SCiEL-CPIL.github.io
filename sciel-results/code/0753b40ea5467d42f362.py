import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a Product-of-Experts (PoE) factorized structure.
    
    The growth rate is modeled as the product of four independent univariate 
    response functions, one for each input variable. This captures multiplicative 
    interactions common in biological systems (e.g., if substrate is limiting, 
    growth is low regardless of other factors).
    
    Each factor is a smooth, bounded non-linear function to ensure numerical 
    stability and help the optimizer navigate the non-convex landscape.
    
    Parameters (10 total):
    - p[0]: Global scaling factor
    - p[1], p[2]: Substrate response (Monod-like or Sigmoidal)
    - p[3], p[4]: Temperature response (Gaussian-like or Sigmoidal)
    - p[5], p[6]: pH response (Gaussian-like or Sigmoidal)
    - p[7], p[8]: Biomass/Density response (Logistic or Sigmoidal)
    - p[9]: Offset or bias term (added to final product to allow negative growth)
    """
    p = params
    
    # Normalize inputs to roughly [-1, 1] or [0, 1] range to help optimization
    # Based on training ranges:
    # b: [1, 5] -> center 3, scale 2
    # s: [1, 40] -> center 10, scale 15 (approx, log scale might be better but let's stick to linear for simplicity first)
    # temp: [1, 39.98] -> center 30, scale 10
    # pH: [1, 12] -> center 7, scale 3
    
    b_n = (b - 3.0) / 2.0
    s_n = (s - 10.0) / 15.0
    t_n = (temp - 30.0) / 10.0
    ph_n = (pH - 7.0) / 3.0
    
    # Factor 1: Substrate (Monod-like: s / (K + s), but using tanh for smoothness)
    # We want it to increase with s. 
    # Using a sigmoid: 1 / (1 + exp(-x))
    # To allow flexibility in slope and center, we use params.
    # However, fixed centers are safer for BFGS from ones.
    # Let's use a simple hyperbolic tangent centered at 0 (which corresponds to s=10)
    # f_s = 0.5 * (1 + tanh(p[1] * s_n + p[2]))
    # If p[1] is positive, it's an increasing sigmoid.
    
    # Factor 2: Temperature (Bell-shaped / Gaussian-like)
    # Growth should peak at an optimal temp and drop on either side.
    # A common form is exp(-(t - t_opt)^2 / sigma^2)
    # Using tanh product for bell shape? Or just a quadratic in the exponent?
    # Let's use a Gaussian: exp(-p[3] * (t_n - p[4])^2)
    # This is non-linear in params p[3], p[4].
    
    # Factor 3: pH (Bell-shaped)
    # Similar to temperature.
    # f_ph = exp(-p[5] * (ph_n - p[6])^2)
    
    # Factor 4: Biomass/Density (Logistic growth constraint)
    # Growth rate often decreases at high density due to crowding.
    # f_b = 1 / (1 + p[7] * exp(p[8] * b_n))  # Logistic function
    # Or simpler: 1 - p[7] * b_n^2 (quadratic decay)
    # Let's stick to a sigmoidal decay: 1 / (1 + exp(p[7] * b_n + p[8]))
    
    # Construct the factors
    # To avoid numerical issues with exp overflow, we clip arguments.
    
    # Substrate: Sigmoidal increase
    arg_s = p[1] * s_n + p[2]
    arg_s = np.clip(arg_s, -50, 50)
    f_s = 1.0 / (1.0 + np.exp(-arg_s))
    
    # Temperature: Gaussian
    # Center is fixed at t_n=0 (temp=30) to aid init, width is p[3]
    # If we allow center to move, it's harder. Let's fix center at 0 for now?
    # No, optimal temp might not be 30. But 30 is a reasonable guess for E. coli.
    # Let's allow a small shift: t_opt_n = p[4] * 0.5 (scaled)
    # Actually, let's keep it simple: Gaussian centered at 0.
    # f_t = exp(-p[3] * t_n * t_n)
    # If p[3] is negative, it explodes. We need p[3] > 0.
    # To ensure positivity without constraints, we can use p[3]^2 or exp(p[3]).
    # Let's use exp(-exp(p[3]) * t_n * t_n) to ensure width is positive.
    width_t = np.exp(p[3])
    f_t = np.exp(-width_t * t_n * t_n)
    
    # pH: Gaussian
    width_ph = np.exp(p[5])
    f_ph = np.exp(-width_ph * ph_n * ph_n)
    
    # Biomass: Logistic decay (growth slows at high b)
    # f_b = 1 / (1 + exp(p[7] * b_n + p[8]))
    # If b is high (b_n positive), and p[7] positive, exp is large, f_b small.
    arg_b = p[7] * b_n + p[8]
    arg_b = np.clip(arg_b, -50, 50)
    f_b = 1.0 / (1.0 + np.exp(arg_b))
    
    # Combine
    # Growth = p[0] * f_s * f_t * f_ph * f_b + p[9]
    # p[9] allows for maintenance energy or negative net growth.
    
    pred = p[0] * f_s * f_t * f_ph * f_b + p[9]
    
    return pred
