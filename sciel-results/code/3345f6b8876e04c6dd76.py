import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate using Fourier basis for temp/pH with Monod for substrate."""
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # p0: overall amplitude
    # p1: Monod Ks (half-saturation constant)
    # p2, p3: temperature frequency, phase
    # p4, p5: pH frequency, phase
    # p6: temp sine amplitude
    # p7: pH sine amplitude
    # p8: density inhibition constant
    # p9: baseline offset
    
    # Substrate: Monod term
    monod = s / (s + p1)
    
    # Temperature: sum of sine with learnable freq/phase
    # Use a combination that can create an optimum
    temp_arg = p2 * temp + p3
    temp_resp = p6 * np.sin(temp_arg) + np.cos(temp_arg)  # cos gives the "base" bell-like shape
    
    # pH: similar structure
    ph_arg = p4 * pH + p5
    ph_resp = p7 * np.sin(ph_arg) + np.cos(ph_arg)
    
    # Density: inhibition at high density
    density_resp = 1.0 / (1.0 + b / p8)
    
    # Combine multiplicatively
    result = p0 * monod * temp_resp * ph_resp * density_resp + p9
    
    # Ensure non-negative growth rate (biological constraint)
    # Use softplus-like: we'll let the optimizer handle it, but clip to avoid issues
    # Actually, let's not clip - let BFGS find the right parameters
    # But ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
