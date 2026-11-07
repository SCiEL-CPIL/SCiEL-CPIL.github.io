import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params.flatten()
    
    # Ensure length is 10
    if len(p) < 10:
        # Fallback to a simple linear model if params are malformed
        # This should rarely happen as the evaluator provides 10 params
        return p[0]*x + p[1]*y + p[2]*z + p[3]
        
    # Numerator: Linear combination of 1, x, y, z, xy, xz, yz
    # Using 7 parameters for the numerator
    num = (p[0] + 
           p[1] * x + 
           p[2] * y + 
           p[3] * z + 
           p[4] * (x * y) + 
           p[5] * (x * z) + 
           p[6] * (y * z))
           
    # Denominator: 1 + linear combination of x, y, z
    # Using 3 parameters for the denominator
    # The constant term is fixed at 1 to provide scale invariance/regularization
    # and to ensure the denominator is non-zero near the origin of the parameter space (all ones).
    # With x,y,z in [0.5, 2], if p[7:10] are small, denom is approx 1.
    # If p[7:10] are large, we risk division by zero, but BFGS should avoid it due to high loss.
    den = 1.0 + p[7] * x + p[8] * y + p[9] * z
    
    # Safety check: replace zeros or very small denominators with a large number
    # to avoid inf/nan which crashes the evaluator.
    # We use a sign-preserving clip.
    den_safe = np.where(np.abs(den) < 1e-6, np.sign(den) * 1e-6 + 1e-6, den)
    # Actually, simpler: just add a tiny epsilon to prevent exact zero division
    # But if den is close to 0, the value blows up, which is correct for MSE.
    # The problem is if den is exactly 0.
    # Let's use np.divide with where clause or just add 1e-10.
    # Adding 1e-10 is safer for numerical stability during optimization steps.
    den = den + 1e-10
    
    return num / den
