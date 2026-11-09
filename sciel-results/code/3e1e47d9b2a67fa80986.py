"""
Strategy: Biologically Motivated Mechanistic Model
Instead of a generic neural network, we use a model structure that reflects
the known biology of E. coli growth:
1. Substrate (s) follows Monod kinetics: s / (Ks + s). This captures saturation.
2. Temperature (temp) and pH follow a bell-shaped curve (Gaussian or Haldane-like) 
   to capture the optimum and inhibition at extremes.
3. We include a linear interaction term or direct effects to allow flexibility.

We have 10 parameters. Let's allocate them as follows:
- Monod for substrate: 
  - Ks (half-saturation constant): p[0]
  - Vmax (max growth rate contribution from substrate): p[1]
  - Term: p[1] * s / (p[0] + s)
  
- Temperature effect (Gaussian):
  - Center (optimal temp): p[2]
  - Width (tolerance): p[3]
  - Amplitude (max effect): p[4]
  - Term: p[4] * exp(-0.5 * ((temp - p[2]) / p[3])^2)
  - Note: exp is smooth and well-behaved for BFGS.

- pH effect (Gaussian):
  - Center (optimal pH): p[5]
  - Width (tolerance): p[6]
  - Amplitude (max effect): p[7]
  - Term: p[7] * exp(-0.5 * ((pH - p[5]) / p[6])^2)

- Population density (b) effect:
  - Often growth rate per capita decreases with density (crowding).
  - Let's use a simple linear or logarithmic penalty. 
  - Given limited params, a linear term is safest for convergence.
  - Term: p[8] * b

- Baseline/Intercept:
  - p[9]

Total params: 10.

Equation:
y = p[1] * (s / (p[0] + s)) + p[4] * exp(-0.5 * ((temp - p[2]) / p[3])**2) 
    + p[7] * exp(-0.5 * ((pH - p[5]) / p[6])**2) + p[8] * b + p[9]

This model is strictly non-linear and captures:
- Saturation in substrate (Monod)
- Optima in temperature and pH (Gaussian)
- Linear effect of density

Why this might beat the MLP:
- The MLP has to *learn* the shape of the saturation and optima from scratch.
- This model *assumes* the shape, only fitting the key parameters (center, width, amplitude).
- This reduces the effective dimensionality of the search space for the optimizer.
- BFGS should converge faster and more reliably to the correct values for Ks, T_opt, etc.
"""

import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a Monod-Gaussian mechanistic model.
    
    Inputs:
    b: population density
    s: substrate concentration
    temp: temperature
    pH: pH level
    params: array of 10 floats
        p[0]: Ks (Monod half-saturation)
        p[1]: Vmax_substrate
        p[2]: T_opt (Temperature optimum)
        p[3]: T_width (Temperature tolerance)
        p[4]: A_temp (Temperature amplitude)
        p[5]: pH_opt (pH optimum)
        p[6]: pH_width (pH tolerance)
        p[7]: A_pH (pH amplitude)
        p[8]: W_b (Density weight)
        p[9]: Bias
    
    Returns:
    np.ndarray of predicted growth rates
    """
    # Extract parameters
    Ks = params[0]
    Vmax_s = params[1]
    T_opt = params[2]
    T_width = params[3]
    A_temp = params[4]
    pH_opt = params[5]
    pH_width = params[6]
    A_pH = params[7]
    W_b = params[8]
    Bias = params[9]
    
    # Monod term for substrate
    # Add a small epsilon to denominator to avoid division by zero if Ks goes negative/zero
    # However, Ks should be positive. If BFGS tries negative, the model behaves oddly.
    # To ensure stability, we can use abs(Ks) + epsilon, or just rely on the optimizer.
    # Standard Monod: s / (Ks + s). If Ks < 0, it's problematic.
    # Let's use a safe denominator: np.maximum(Ks, 1e-6)
    # But wait, params are fitted. If the true Ks is small, forcing it to be positive is good.
    # Actually, just using (Ks + s) is standard. If Ks becomes negative, the gradient might be weird.
    # Let's stick to the raw formula but ensure no division by zero.
    denom_s = Ks + s
    # If denom is close to 0, clip it to prevent inf/nan
    denom_s = np.where(np.abs(denom_s) < 1e-8, 1e-8, denom_s)
    monod_s = Vmax_s * (s / denom_s)
    
    # Gaussian term for temperature
    # Ensure width is not zero to avoid division by zero in exponent
    T_width_safe = np.maximum(T_width, 1e-6)
    temp_term = A_temp * np.exp(-0.5 * ((temp - T_opt) / T_width_safe) ** 2)
    
    # Gaussian term for pH
    pH_width_safe = np.maximum(pH_width, 1e-6)
    ph_term = A_pH * np.exp(-0.5 * ((pH - pH_opt) / pH_width_safe) ** 2)
    
    # Linear term for density
    b_term = W_b * b
    
    # Combine
    return monod_s + temp_term + ph_term + b_term + Bias

# EVOLVE-BLOCK-END
