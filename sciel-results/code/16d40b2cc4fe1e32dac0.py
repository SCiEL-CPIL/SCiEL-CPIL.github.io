"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with Monod kinetics, Arrhenius temperature dependence,
    and Gaussian pH sensitivity.
    
    Inputs:
    b: population density (likely 1-5, log scale or similar)
    s: substrate concentration (1-50)
    temp: temperature (1-40 C)
    pH: pH level (2-12)
    
    Params (10 fitted constants):
    p0: max specific growth rate (mu_max)
    p1: half-saturation constant (Ks)
    p2: Hill coefficient (n) - usually ~1 for Monod, but flexible
    p3: activation energy scaling (Ea)
    p4: reference temperature for Arrhenius (Tref)
    p5: optimal pH (pH_opt)
    p6: pH width parameter (sigma)
    p7: inhibition coefficient for high biomass (Kb)
    p8: inhibition strength (alpha)
    p9: baseline offset
    """
    p = params
    
    # Ensure numerical stability for physical parameters
    # Use softplus or exp to keep positive where needed, or just let BFGS handle it if well-scaled
    # Since BFGS starts at 1.0, we can use direct values but need to handle division by zero/negative
    
    # 1. Substrate Kinetics (Monod-like)
    # s is in [1, 50]. Ks should be positive.
    # If p1 is negative, Monod breaks. We can use abs(p1) or exp(p1).
    # Let's use a safe form: s / (s + Ks)
    # To allow BFGS to find Ks, we map p1 to positive: Ks = abs(p1) + 0.1
    Ks = abs(p[1]) + 0.1
    n = abs(p[2]) + 0.1  # Hill coefficient, keep positive
    
    monod_term = (s / (s + Ks)) ** n
    
    # 2. Temperature Dependence (Arrhenius-like)
    # temp is in [1, 40].
    # mu(T) = exp(-Ea * (1/T - 1/Tref))
    # T in Kelvin? Let's assume input is Celsius. Convert to Kelvin for physics.
    T = temp + 273.15
    Tref = abs(p[4]) + 273.15  # Ensure Tref is positive Kelvin
    Ea = p[3]  # Activation energy, can be positive or negative depending on fit
    
    # Arrhenius factor
    # To avoid overflow, clamp the exponent
    arr_exp = -Ea * (1.0 / T - 1.0 / Tref)
    # Clamp to prevent exp overflow/underflow issues during optimization
    arr_exp = np.clip(arr_exp, -50, 50)
    arrhenius_term = np.exp(arr_exp)
    
    # 3. pH Dependence (Gaussian)
    # pH is in [2, 12]. E. coli optimum is around 7.
    pH_opt = p[5]
    sigma = abs(p[6]) + 0.1  # Width, must be positive
    
    ph_term = np.exp(-((pH - pH_opt) ** 2) / (2 * sigma ** 2))
    
    # 4. Biomass/Inhibition Term
    # b is in [1, 5]. High density might inhibit growth.
    # Simple logistic-like inhibition: 1 / (1 + (b/Kb)^alpha)
    Kb = abs(p[7]) + 0.1
    alpha = abs(p[8]) + 0.1
    
    # If b is low (1), inhibition should be low. If b is high, inhibition high.
    # Standard form: 1 - (b/K)^2 or similar.
    # Let's use a simple decay: exp(-b/Kb) or 1/(1 + b/Kb)
    # Since b >= 1, let's use 1 / (1 + (b / Kb) ** alpha)
    inhib_term = 1.0 / (1.0 + (b / Kb) ** alpha)
    
    # Combine
    mu_max = p[0]
    
    growth = mu_max * monod_term * arrhenius_term * ph_term * inhib_term
    
    # Add baseline
    growth += p[9]
    
    return growth
# EVOLVE-BLOCK-END
