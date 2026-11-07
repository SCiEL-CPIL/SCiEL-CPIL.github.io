import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model combining:
    - Monod kinetics for substrate (s)
    - Logistic/self-limiting term for population density (b)
    - Fourier-Basis periodic response for temperature (temp) and pH
    - Linear interactions and offset
    
    params (10 values):
      p0: Monod max growth rate scale
      p1: Monod half-saturation (Ks)
      p2: Temperature Fourier amplitude (1st harmonic)
      p3: Temperature Fourier phase shift
      p4: pH Fourier amplitude (1st harmonic)
      p5: pH Fourier phase shift
      p6: Temperature quadratic coefficient (for asymmetry)
      pH quadratic coefficient (for asymmetry)
      p8: Population density inhibition coefficient
      p9: Offset / intercept
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure numerical stability for Ks (half-saturation constant)
    Ks = np.abs(p1) + 1e-6
    
    # Monod kinetics for substrate
    monod = p0 * s / (Ks + s)
    
    # Population density: logistic-type self-limitation
    # Use a smooth saturation: 1 / (1 + exp(p8 * b))
    # When p8 > 0, high b reduces growth; p8 < 0 allows more growth at high b
    pop_limit = 1.0 / (1.0 + np.exp(np.clip(p8 * b, -50, 50)))
    
    # Temperature response: Fourier basis + quadratic for asymmetry
    # Fixed frequency: 2*pi / range ~ 2*pi / 40
    freq_T = 2.0 * np.pi / 40.0
    T_response = p2 * np.sin(freq_T * temp + p3) + p6 * (temp - 20.0) ** 2
    
    # pH response: Fourier basis + quadratic for asymmetry
    # Fixed frequency: 2*pi / range ~ 2*pi / 10
    freq_pH = 2.0 * np.pi / 10.0
    pH_response = p4 * np.sin(freq_pH * pH + p5) + p7 * (pH - 7.0) ** 2
    
    # Combine: multiplicative for Monod and pop_limit, additive for temp/pH
    # This structure allows the model to capture:
    # - Substrate saturation (Monod)
    # - Density-dependent inhibition
    # - Periodic/asymmetric temperature and pH optima
    
    prediction = monod * pop_limit * (1.0 + T_response + pH_response) + p9
    
    # Clip to avoid extreme values during optimization
    prediction = np.clip(prediction, -1e6, 1e6)
    
    return prediction
