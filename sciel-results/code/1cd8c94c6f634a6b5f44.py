"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard (though equation is fast, this satisfies the directive)
_MODULE_START_TIME = time.time()
_DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Hybrid Learnable Fourier Series + Quadratic Polynomial + Sigmoid Correction.
    
    This model combines:
    1. A low-order polynomial (Quadratic) for global trend.
    2. A Fourier series with a learnable fundamental frequency (omega) and 
       two harmonics to capture periodic/oscillatory behavior.
    3. A learnable Sigmoid (fixed steepness) to capture step-like or 
       saturating non-periodic features.
    
    y(x) = p0 + p1*x + p2*x^2 
           + p4*sin(omega*x) + p5*cos(omega*x)
           + p6*sin(2*omega*x) + p7*cos(2*omega*x)
           + p8 * sigmoid(10.0 * (x - p9))
           
    Parameters:
    p0, p1, p2: Polynomial coefficients (constant, linear, quadratic)
    p3: Fundamental frequency omega
    p4, p5: Amplitudes for 1st harmonic (sin, cos)
    p6, p7: Amplitudes for 2nd harmonic (sin, cos)
    p8: Amplitude of Sigmoid correction
    p9: Center of Sigmoid correction
    """
    # Deadline Guard Check
    # If we are close to the timeout, we could theoretically return a simplified model,
    # but since this function is O(N) and fast, we just proceed.
    # This check is here to explicitly satisfy the "deadline guard" directive.
    if time.time() - _MODULE_START_TIME > _DEADLINE_SECONDS:
        # Fallback: return zeros or a simple constant if we are about to be killed
        # This is a safety net. In practice, the evaluator should handle this.
        return np.zeros_like(x, dtype=np.float64)

    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Extract parameters
    p0, p1, p2 = p[0], p[1], p[2]
    omega = p[3]
    A1_s, A1_c = p[4], p[5]
    A2_s, A2_c = p[6], p[7]
    A_sig, c_sig = p[8], p[9]
    
    # 1. Polynomial Part
    poly = p0 + p1 * x + p2 * x**2
    
    # 2. Fourier Part
    # Clip omega to prevent extreme values that might cause precision issues in sin/cos
    # or optimizer instability, though sin/cos are periodic.
    # A range of [-20, 20] covers frequencies up to 10 cycles in [-2,2] range.
    omega_safe = np.clip(omega, -20.0, 20.0)
    
    # 1st Harmonic
    angle1 = omega_safe * x
    h1 = A1_s * np.sin(angle1) + A1_c * np.cos(angle1)
    
    # 2nd Harmonic
    angle2 = 2.0 * omega_safe * x
    h2 = A2_s * np.sin(angle2) + A2_c * np.cos(angle2)
    
    # 3. Sigmoid Part
    # Fixed steepness k=10.0 provides a sharp but differentiable transition.
    k_sig = 10.0
    z_sig = k_sig * (x - c_sig)
    
    # Clip z to avoid overflow in exp
    z_sig = np.clip(z_sig, -40, 40)
    
    # Numerically stable sigmoid
    sig = np.where(z_sig >= 0,
                   1.0 / (1.0 + np.exp(-z_sig)),
                   np.exp(z_sig) / (1.0 + np.exp(z_sig)))
    
    sig_part = A_sig * sig
    
    # Combine
    return poly + h1 + h2 + sig_part
# EVOLVE-BLOCK-END
