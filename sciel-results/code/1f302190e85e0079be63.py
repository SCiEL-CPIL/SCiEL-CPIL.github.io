"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function model: y(x) = P_4(x) / Q_4(x)
    
    P_4(x) = p[0] + p[1]*x + p[2]*x^2 + p[3]*x^3 + p[4]*x^4
    Q_4(x) = p[5] + p[6]*x + p[7]*x^2 + p[8]*x^3 + p[9]*x^4
    
    This structure allows modeling functions with poles, asymptotes, or rapid
    local changes that a fixed linear basis or simple polynomial product cannot
    capture efficiently. The non-convex landscape is navigated by BFGS.
    We ensure numerical stability by clipping the denominator to avoid division
    by zero, which would cause non-finite outputs and invalidate the fit.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64)
    
    # Safety check for non-finite parameters
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
    
    # Extract coefficients
    # Numerator P_4(x)
    a0, a1, a2, a3, a4 = p[0], p[1], p[2], p[3], p[4]
    # Denominator Q_4(x)
    b0, b1, b2, b3, b4 = p[5], p[6], p[7], p[8], p[9]
    
    # Compute powers of x for efficiency
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Compute Numerator P_4(x) using Horner's method for stability
    # P(x) = (((a4*x + a3)*x + a2)*x + a1)*x + a0
    num = a0
    num = num + x * (a1 + x * (a2 + x * (a3 + x * a4)))
    
    # Compute Denominator Q_4(x) using Horner's method
    # Q(x) = (((b4*x + b3)*x + b2)*x + b1)*x + b0
    den = b0
    den = den + x * (b1 + x * (b2 + x * (b3 + x * b4)))
    
    # Handle division by zero safely
    # If den is very close to 0, the value blows up. 
    # We clip the denominator to a small epsilon to keep output finite,
    # allowing the optimizer to push parameters away from the singularity.
    eps = 1e-8
    abs_den = np.abs(den)
    # Where abs_den < eps, replace den with eps * sign(den) (or just eps if 0)
    # But sign(0) is 0. So we need to be careful.
    # Simple approach: np.where(abs_den < eps, eps, den)
    # This creates a discontinuity at 0 if den crosses 0, but avoids inf.
    # A smoother approach: den_safe = np.where(abs_den < eps, np.sign(den)*eps + (1-np.sign(den))*eps, den)
    # Actually, np.sign(0)=0, so np.sign(den)*eps is 0. 
    # Let's just use: if |den| < eps, set den to eps (if den>=0) or -eps (if den<0).
    # If den is exactly 0, sign is 0. We can default to eps.
    
    # More robust:
    den_safe = den.copy()
    mask_small = abs_den < eps
    if np.any(mask_small):
        # For small denominators, set to eps preserving sign if possible, else eps
        signs = np.sign(den[mask_small])
        # If sign is 0, replace with 1
        signs[signs == 0] = 1.0
        den_safe[mask_small] = signs * eps
        
    y = num / den_safe
    
    # Clip y to prevent overflow in MSE calculation (y^2)
    # Max float is ~1e308. sqrt is ~1e154.
    y = np.clip(y, -1e150, 1e150)
    
    return y

# EVOLVE-BLOCK-END
