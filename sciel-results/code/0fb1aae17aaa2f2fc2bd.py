"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a Fourier basis for temperature and pH
    to capture periodic/oscillatory optima, combined with Monod kinetics for substrate.
    
    Structure:
    Growth = Monod(s) * (1 + A_temp * cos(w_temp * (temp - mu_temp)) + B_temp * sin(...))
                * (1 + A_ph * cos(w_ph * (pH - mu_ph)) + B_ph * sin(...))
                * (1 + c1 * b + c2 * b^2)  # Inhibition/density effect
                * scale
    
    Parameters (10 total):
    p0: Monod half-saturation (Ks)
    p1: Max substrate effect scale (Vmax)
    p2: Temperature frequency (w_temp)
    p3: Temperature phase/offset (mu_temp)
    p4: pH frequency (w_ph)
    p5: pH phase/offset (mu_ph)
    p6: Amplitude of temperature modulation
    p7: Amplitude of pH modulation
    p8: Density inhibition coefficient
    p9: Global scale factor
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Ensure positive parameters where needed via softplus-like transformation or abs
    # For BFGS stability, we use raw values but design the function to be robust.
    # Ks should be positive.
    Ks = np.abs(p0) + 1e-6
    Vmax = np.abs(p1) + 1e-6
    
    # Frequencies and amplitudes
    w_temp = np.abs(p2) + 1e-6
    mu_temp = p3
    w_ph = np.abs(p4) + 1e-6
    mu_ph = p5
    amp_temp = p6
    amp_ph = p7
    
    # Density coefficient
    c_b = p8
    
    # Scale
    scale = p9

    # Monod term for substrate
    # s is in [1, 39.98]
    monod = Vmax * s / (Ks + s)

    # Temperature response: Gaussian-like or Fourier?
    # Directive suggests Fourier. 
    # A single cosine term centered at an optimum can model a peak.
    # cos(w*(x-mu)) is 1 at x=mu.
    # To make it a decaying response, we might need multiple terms or a different form.
    # However, the prompt says "sum-of-sines... learnable frequencies".
    # Let's use a single frequency per variable to stay within parameter budget.
    # Actually, a simple Gaussian is often better for optima. 
    # But let's try the Fourier approach as directed.
    # To create a peak, we can use: 1 + amp * (cos(w*(x-mu)) - 1)
    # This is 1 at x=mu and decreases as we move away, oscillating.
    # If w is small, it's roughly parabolic.
    
    temp_term = 1.0 + amp_temp * (np.cos(w_temp * (temp - mu_temp)) - 1.0)
    ph_term = 1.0 + amp_ph * (np.cos(w_ph * (pH - mu_ph)) - 1.0)

    # Density effect: simple linear inhibition
    # b is in [1, 4.999]
    # If c_b is negative, it's inhibition.
    density_term = 1.0 + c_b * (b - 1.0)

    # Combine multiplicatively
    # Ensure terms don't go negative to avoid log issues if we were taking logs, 
    # but we are not. However, negative growth rates might be unphysical.
    # We'll let the optimizer handle the scale.
    
    growth = scale * monod * temp_term * ph_term * density_term

    return growth
# EVOLVE-BLOCK-END
