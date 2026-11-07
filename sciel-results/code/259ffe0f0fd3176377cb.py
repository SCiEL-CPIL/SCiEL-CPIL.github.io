"""E. coli growth rate model with Fourier-Basis Periodic Environmental Response."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a multiplicative core where temperature and pH
    responses are represented by Fourier series (fixed frequencies, learnable amplitudes/phases).
    
    Structure:
        Prediction = V_max * Monod(s) * TempFourier(T) * pHFourier(pH) * DensityFactor(b) + LinearCorrection
    
    Parameters (10 total):
    p0: log(K_s) for Monod substrate half-saturation
    p1: log(V_max) for maximum growth rate
    p2: log(K_b) for density inhibition half-saturation
    p3: Amplitude of 1st harmonic for Temperature
    p4: Phase of 1st harmonic for Temperature
    p5: Amplitude of 1st harmonic for pH
    p6: Phase of 1st harmonic for pH
    p7: Linear coeff for b (centered at 3)
    p8: Linear coeff for s (centered at 3)
    p9: Linear coeff for temp (centered at 20)
    
    Returns:
        Array of predicted growth rates
    """
    # Deadline guard
    start_time = time.time()
    
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure numerical stability
    eps = 1e-6
    
    # Positive scale parameters via exp
    K_s = np.exp(p0) + eps
    V_max = np.exp(p1) + eps
    K_b = np.exp(p2) + eps
    
    # 1. Substrate: Monod Kinetics
    # s range [1, 5]
    monod = s / (K_s + s)
    
    # 2. Temperature: Fourier Series Response
    # temp range [1, 40]
    # Normalize temp to [0, 1] for stable Fourier arguments
    # Using a range slightly wider than data to avoid edge effects
    t_norm = (temp - 1.0) / 39.0
    
    # First harmonic: sin(2*pi*t + phase)
    # We use a DC offset + 1st harmonic to allow asymmetric peaks
    # The DC offset is fixed to 1.0 to ensure positive values if amplitude is small,
    # but we need it to be learnable or at least positive. 
    # Let's use: 1 + A * sin(2*pi*t + phi)
    # However, if A > 1, it can go negative. Growth rate must be positive.
    # We can use: exp(A * sin(2*pi*t + phi)) which is always positive and smooth.
    # This is a log-periodic response.
    # Let's stick to a safer positive formulation:
    # temp_factor = exp(p3 * sin(2*pi*t_norm + p4))
    # This allows the "peak" to be anywhere and the shape to be asymmetric if we added more harmonics,
    # but with 1 harmonic it's symmetric in log-space.
    # Given only 10 params, we can't fit 2 harmonics for both T and pH easily.
    # Let's use: temp_factor = exp(p3 * sin(2*pi*t_norm + p4))
    # p3 controls the "steepness" or "range" of the response in log-space.
    # p4 controls the location of the peak.
    
    arg_t = 2.0 * np.pi * t_norm + p4
    temp_factor = np.exp(p3 * np.sin(arg_t))
    
    # 3. pH: Fourier Series Response
    # pH range [2, 12]
    # Normalize pH to [0, 1]
    ph_norm = (pH - 2.0) / 10.0
    
    arg_p = 2.0 * np.pi * ph_norm + p6
    ph_factor = np.exp(p5 * np.sin(arg_p))
    
    # 4. Population Density: Inhibition
    # b range [1, 5]
    # Michaelis-Menten style inhibition: K_b / (K_b + b)
    inhib_b = K_b / (K_b + b)
    
    # Combined Core
    core = V_max * monod * temp_factor * ph_factor * inhib_b
    
    # Additive Correction
    # Centered to help BFGS convergence and capture linear residuals
    offset = (p7 * (b - 3.0) + 
              p8 * (s - 3.0) + 
              p9 * (temp - 20.0))
    
    result = core + offset
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    # Check deadline
    if time.time() - start_time > 115:
         # Fallback to simple linear if something went wrong
         return params[0] * b + params[1] * s + params[2] * temp + params[3] * pH + params[4]
         
    return result
# EVOLVE-BLOCK-END
