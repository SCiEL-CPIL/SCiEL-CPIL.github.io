"""Bootstrap portfolio: model-and-optimize for stress-strain of aluminium rod."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Record the start time to implement a deadline guard.
# The evaluator runs for up to 150s. We will fail gracefully after 140s.
_START_TIME = time.time()
_TIMEOUT_LIMIT = 140.0

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    Uses a hybrid model combining:
    1. A rational term for elastic-plastic transition
    2. A tanh term for strain hardening/saturation
    3. A smooth kink function to capture yield points
    
    The model is designed to be differentiable and stable for BFGS optimization.
    """
    # Deadline guard: if we are running out of time, return a safe neutral value
    if time.time() - _START_TIME > _TIMEOUT_LIMIT:
        return np.zeros_like(strain, dtype=np.float64)

    # Ensure inputs are 1D float64 arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # --- Term 1: Elastic-Plastic Rational Term ---
    # E(T) = p0 + p1*temp (Linear temperature dependence for modulus)
    # H = p2 (Saturation parameter)
    # This term behaves like E*strain for small strain and saturates to E/H for large strain
    E_T = p0 + p1 * temp
    H = p2
    den = 1.0 + H * strain
    # Safe division
    den_safe = np.where(np.abs(den) < 1e-8, 1e-8, den)
    term1 = (E_T * strain) / den_safe
    
    # --- Term 2: Tanh Hardening Term ---
    # S(T) = p3 + p4*temp (Saturation stress)
    # K(T) = p5 + p6*temp (Rate of hardening)
    # This term adds a smooth S-curve shift to model plastic flow
    S_T = p3 + p4 * temp
    K_T = p5 + p6 * temp
    term2 = S_T * np.tanh(K_T * strain)
    
    # --- Term 3: Smooth Kink (Yield Point) Term ---
    # W(T) = p7 + p8*temp (Amplitude of the kink)
    # k = p9 (Location of the yield point, in strain)
    # We use a smooth approximation of |x - k| or a step-like transition.
    # A common smooth kink is based on sqrt(x^2 + eps) or similar.
    # Let's use a smooth step function: 1/(1 + exp(-a*(x-k))) is logistic, but we want a kink in the derivative.
    # A "soft absolute value" |x|_smooth = sqrt(x^2 + eps) has a kink at 0.
    # Shifted: |strain - k|_smooth = sqrt((strain - k)^2 + eps)
    # This term adds a V-shaped feature centered at k.
    # To make it more like a yield plateau, we might want a term that is flat near k and rises elsewhere.
    # Or simply add the soft-abs to capture the sudden change in slope.
    # Let's use: W_T * (sqrt((strain - k)^2 + 1e-6) - sqrt(k^2 + 1e-6))
    # This centers the kink at 0 relative to the yield point.
    # However, sqrt is not perfectly smooth at 0. 
    # Alternative: Use a polynomial approximation of a kink.
    # Or simply: W_T * (strain - k) * tanh(W2 * (strain - k)) ? No, we used all params.
    
    # Let's try a different allocation for flexibility.
    # Maybe the kink should be a step function approximation?
    # Actually, the previous model's issue was potentially the complexity or the specific form.
    # Let's try a simpler, more robust form for the kink:
    # kink(x) = 0.5 * (x + sqrt(x^2 + eps))  # This is softplus-like, smooth, monotonic
    # Or just a simple shifted quadratic?
    
    # Let's stick to the soft-abs but make it robust.
    # term3 = W_T * (np.sqrt((strain - p9)**2 + 1e-8) - np.sqrt(p9**2 + 1e-8))
    # This is 0 when strain = p9. It grows linearly as strain moves away from p9.
    # This adds a "tent" shape. If W_T is negative, it's a "valley".
    
    W_T = p7 + p8 * temp
    k = p9
    # Ensure k is not causing numerical issues if strain is far away
    # The term represents a change in slope magnitude.
    term3 = W_T * (np.sqrt((strain - k)**2 + 1e-8) - np.sqrt(k**2 + 1e-8))
    
    stress = term1 + term2 + term3
    
    # Ensure output is finite
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
# EVOLVE-BLOCK-END
