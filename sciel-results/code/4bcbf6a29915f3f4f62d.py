"""Fourier-Basis Periodic Model for E. coli growth rate."""
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using Monod kinetics for substrate,
    Fourier basis for temperature and pH optima,
    and exponential inhibition for population density.
    
    Parameters (10 total):
    p0: max growth rate multiplier
    p1: Monod half-saturation (Ks)
    p2: density inhibition constant
    p3: temp Fourier base
    p4: temp Fourier amplitude
    p5: pH Fourier base
    p6: pH Fourier amplitude
    p7: temp phase shift
    p8: pH phase shift
    p9: bias term
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Monod kinetics for substrate: s / (Ks + s)
    # Ks is parameterized to be positive: exp(p1)
    Ks = np.exp(p1)
    monod = s / (Ks + s)
    
    # Exponential inhibition for population density
    # b is in [1, 5], so we use exp(-p2 * b) to model crowding
    # p2 can be negative to allow growth if needed, but typically positive for inhibition
    density_factor = np.exp(-np.exp(p2) * (b - 1.0))
    
    # Fourier basis for temperature
    # Temp range [1, 40]. We use a single harmonic with fixed frequency.
    # The "optimum" is captured by the phase and amplitude.
    # To allow the response to go from 0 at edges to 1 at optimum, we use:
    # base + amplitude * cos(f * (T - phase))
    # But cos can go negative. We want a positive response.
    # Better: 0.5 * (1 + cos(...)) gives [0, 1].
    # Let's use: temp_resp = p3 + p4 * np.cos(2*np.pi*(temp - p7)/39.0)
    # To ensure positivity, we can clip or use a squared cosine, but let's try a simple shift.
    # Actually, a Gaussian-like peak is often better. But the directive says Fourier.
    # Let's use a sum of two harmonics to allow asymmetry.
    # f1 = 2*pi/39 (fundamental)
    # f2 = 4*pi/39 (second harmonic)
    
    f1_t = 2.0 * np.pi / 39.0
    f2_t = 4.0 * np.pi / 39.0
    temp_resp = p3 + p4 * np.cos(f1_t * (temp - p7)) + 0.5 * p4 * np.cos(f2_t * (temp - p7))
    
    # To ensure the response is non-negative and bounded, we can use a softplus or just let the optimizer handle it.
    # However, if temp_resp becomes negative, it flips the sign of growth, which is bad.
    # Let's constrain temp_resp to be positive by adding a large constant or using exp?
    # No, let's use a formulation that is inherently positive.
    # temp_resp = exp(p3) * (1 + p4 * cos(...))
    # This is still not guaranteed positive if p4 > 1.
    # Let's use: temp_resp = 0.5 + 0.5 * np.cos(...) scaled by amplitude.
    # Let's try: temp_resp = np.exp(p3) * (0.5 * (1 + np.cos(f1_t * (temp - p7))))
    # This is in [0, exp(p3)].
    
    temp_resp = np.exp(p3) * (0.5 * (1.0 + np.cos(f1_t * (temp - p7))))
    
    # Similarly for pH
    # pH range [2, 12], span 10
    f1_p = 2.0 * np.pi / 10.0
    pH_resp = np.exp(p5) * (0.5 * (1.0 + np.cos(f1_p * (pH - p8))))
    
    # Combine
    # Growth = p0 * monod * density_factor * temp_resp * pH_resp + p9
    # p0 is the max growth rate when all factors are 1.
    # p9 is a bias to handle baseline.
    
    growth = p0 * monod * density_factor * temp_resp * pH_resp + p9
    
    return growth
