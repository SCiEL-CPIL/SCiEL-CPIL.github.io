"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using Gaussian Radial Basis Functions (RBFs).
    
    Instead of hard-coding biological priors (Monod, Gaussian temp/pH), 
    this uses 3 RBFs to approximate the arbitrary function in 4D space.
    The optimizer can position the centers, adjust widths, and set amplitudes.
    
    Inputs:
    b: Population density
    s: Substrate
    temp: Temperature
    pH: pH level
    
    Params (length 10):
    p[0]: Amplitude 1
    p[1]: Amplitude 2
    p[2]: Amplitude 3
    p[3]: Width (shared for stability)
    p[4]: Center b for RBF 1
    p[5]: Center s for RBF 1
    p[6]: Center temp for RBF 1
    p[7]: Center pH for RBF 1
    p[8]: Center b for RBF 2 (RBF 3 is a mirror/shift of RBF 1 to save params)
    p[9]: Center s for RBF 2
    """
    p = params
    
    # Extract parameters
    A1 = p[0]
    A2 = p[1]
    A3 = p[2]
    width = np.abs(p[3]) + 1e-6
    
    # Centers for RBF 1
    c1_b = p[4]
    c1_s = p[5]
    c1_t = p[6]
    c1_ph = p[7]
    
    # Centers for RBF 2 (and 3)
    # To save parameters, we define RBF 2 and 3 relative to RBF 1 or with shared structure.
    # Let's make RBF 2 and 3 have centers derived from p[8], p[9] and fixed offsets or shared other dims.
    # Actually, let's use p[8], p[9] as centers for RBF 2, and assume RBF 2 and 3 share temp/pH centers with RBF 1?
    # No, that restricts flexibility too much.
    
    # Alternative: Use p[8] and p[9] as the b and s centers for RBF 2.
    # For RBF 2 and 3, let's assume they share the temp and pH centers with RBF 1? 
    # This might be too restrictive. 
    # Let's try: 
    # RBF 1: (c1_b, c1_s, c1_t, c1_ph)
    # RBF 2: (c2_b, c2_s, c1_t, c1_ph) -- shares temp/pH
    # RBF 3: (c3_b, c3_s, c1_t, c1_ph) -- shares temp/pH
    # But we only have 2 params left (p[8], p[9]).
    
    # Let's rethink the parameter allocation.
    # 10 params total.
    # 3 Amplitudes: 3
    # 1 Shared Width: 1
    # 4 Centers for RBF 1: 4
    # 2 Centers for RBF 2 (b, s): 2
    # Total: 10.
    # This implies RBF 2 and 3 share the same b,s centers? Or RBF 3 is fixed?
    # If RBF 2 and 3 share centers, they are redundant unless amplitudes differ.
    # Better: 
    # RBF 1: Center (p[4], p[5], p[6], p[7])
    # RBF 2: Center (p[8], p[9], p[6], p[7]) -> Shares temp/pH with RBF 1
    # RBF 3: Center (p[4], p[5], p[8], p[9])? No, that mixes them up.
    
    # Let's try a different structure:
    # 1 Gaussian RBF in 4D.
    # 1 Gaussian RBF in 4D.
    # 1 Gaussian RBF in 4D.
    # Params: 3 Amps, 3 Widths (or 1 shared), 12 Centers. Total 15-18. Too many.
    
    # Constraint: 10 params.
    # Strategy: 2 RBFs.
    # 2 Amps (2)
    # 1 Shared Width (1)
    # 4 Centers RBF1 (4)
    # 4 Centers RBF2 (4)
    # Total: 11. One too many.
    
    # Drop one center? Or use 1 RBF + 1 Polynomial?
    # Let's stick to 3 RBFs but with strong sharing.
    # RBF 1: (p[4], p[5], p[6], p[7])
    # RBF 2: (p[8], p[9], p[6], p[7])  # Shares temp, pH
    # RBF 3: (p[4], p[5], p[8], p[9])  # Shares b, s with RBF 1? No, p[8],p[9] are used above.
    
    # Let's use:
    # RBF 1: (p[4], p[5], p[6], p[7])
    # RBF 2: (p[8], p[9], p[6], p[7])
    # RBF 3: (p[4], p[5], p[6], p[7]) with different amplitude? No, that's just 2 RBFs.
    
    # What if we use 2 RBFs and 2 linear terms?
    # 2 Amps (2)
    # 1 Width (1)
    # 4 Centers RBF1 (4)
    # 4 Centers RBF2 (4)
    # Total 11.
    
    # Let's use 1 RBF and a quadratic polynomial?
    # 1 Amp (1)
    # 1 Width (1)
    # 4 Centers (4)
    # 4 Coeffs for quadratic (b, s, t, ph) + offset? 
    # Quadratic in 4D has 4 linear, 4 quadratic, 6 cross, 1 offset = 15 terms. Too many.
    # Linear in 4D: 4 coeffs + 1 offset = 5.
    # Total: 1+1+4+5 = 11. Still 1 too many.
    
    # Let's use 1 RBF and a linear model with shared width?
    # Params:
    # p[0]: RBF Amplitude
    # p[1]: RBF Width
    # p[2]: RBF Center b
    # p[3]: RBF Center s
    # p[4]: RBF Center temp
    # p[5]: RBF Center pH
    # p[6]: Linear coeff b
    # p[7]: Linear coeff s
    # p[8]: Linear coeff temp
    # p[9]: Linear coeff pH (and offset is 0? or offset is p[9] and pH linear is 0?)
    
    # Let's assume the data is roughly linear + one local bump.
    # This is a very flexible model compared to the fixed biological one.
    
    A_rbf = p[0]
    width = np.abs(p[1]) + 1e-6
    c_b = p[2]
    c_s = p[3]
    c_t = p[4]
    c_ph = p[5]
    
    # Linear terms
    lb = p[6]
    ls = p[7]
    lt = p[8]
    lph = p[9]
    
    # No offset term. The linear terms can act as offset if centers are far away?
    # Or we can force the RBF to have a DC component?
    # Let's add a small offset by making the linear terms include a constant?
    # No, we have exactly 10 params.
    
    # Calculate RBF
    db = b - c_b
    ds = s - c_s
    dt = temp - c_t
    dph = pH - c_ph
    
    dist_sq = db**2 + ds**2 + dt**2 + dph**2
    rbf = A_rbf * np.exp(-dist_sq / (2 * width**2))
    
    # Linear part
    linear = lb * b + ls * s + lt * temp + lph * pH
    
    return rbf + linear
# EVOLVE-BLOCK-END
