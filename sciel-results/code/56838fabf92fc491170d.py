import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 140.0

def equation(b, s, temp, pH, params):
    p = params
    
    # Normalize inputs to roughly [0, 1] based on training ranges
    # b: [1, 4.998] -> center ~3, range ~4
    # s: [1, 4.999] or [1.002, 39.98] -> wait, the prompt says [1, 4.998], [1, 4.999], [1.002, 39.98], [2, 12]
    # Let's look at the input order: b, s, temp, pH
    # The prompt says: "Inputs in training range, in order: [1, 4.998], [1, 4.999], [1.002, 39.98], [2, 12]."
    # So b is in [1, 4.998], s is in [1, 4.999]? No, that's 4 ranges for 4 inputs.
    # Input 1 (b): [1, 4.998]
    # Input 2 (s): [1, 4.999]
    # Input 3 (temp): [1.002, 39.98]
    # Input 4 (pH): [2, 12]
    
    # Let's normalize each to [0, 1]
    b_norm = (b - 1.0) / (4.998 - 1.0)
    s_norm = (s - 1.0) / (4.999 - 1.0)
    temp_norm = (temp - 1.002) / (39.98 - 1.002)
    ph_norm = (pH - 2.0) / (12.0 - 2.0)
    
    # Clip to [0, 1] to avoid extrapolation issues during fitting
    b_norm = np.clip(b_norm, 0, 1)
    s_norm = np.clip(s_norm, 0, 1)
    temp_norm = np.clip(temp_norm, 0, 1)
    ph_norm = np.clip(ph_norm, 0, 1)
    
    # Strategy: Full multivariate polynomial of order 2 or 3?
    # 10 parameters total.
    # A full quadratic in 4 variables has:
    # 1 constant
    # 4 linear
    # 6 quadratic (x^2, y^2, z^2, w^2, xy, xz, xw, yz, yw, zw) -> wait, 4 vars -> 4 squares + 6 cross terms = 10 quadratic terms.
    # Total = 1 + 4 + 10 = 15 terms. Too many for 10 params.
    
    # Let's try a mixed approach or a reduced polynomial.
    # Or, use a neural-network-like structure with 1 hidden layer?
    # Or, stick to the directive: "single, unconstrained 4th-order multivariate polynomial"
    # But 4th order in 4 vars is huge.
    
    # Let's re-read the directive: "Replace the multiplicative Gaussian and Monod terms with a single, unconstrained 4th-order multivariate polynomial in the normalized inputs."
    # This implies using all 10 parameters as coefficients of a selected subset of polynomial terms.
    
    # Which terms?
    # Maybe:
    # p[0]: constant
    # p[1]: b
    # p[2]: s
    # p[3]: temp
    # p[4]: pH
    # p[5]: b*s
    # p[6]: b*temp
    # p[7]: s*temp
    # p[8]: temp*pH
    # p[9]: s*pH
    
    # This is a sparse quadratic model.
    
    result = (p[0] 
              + p[1] * b_norm 
              + p[2] * s_norm 
              + p[3] * temp_norm 
              + p[4] * ph_norm 
              + p[5] * b_norm * s_norm 
              + p[6] * b_norm * temp_norm 
              + p[7] * s_norm * temp_norm 
              + p[8] * temp_norm * ph_norm 
              + p[9] * s_norm * ph_norm)
              
    return result
