import numpy as np

def equation(b, s, temp, pH, params):
    """
    Generalized Additive Model (GAM) with RBF basis functions.
    The model is linear in the fitted parameters (params[4:10] are weights),
    while params[0:4] control the RBF centers. This structure provides
    a smoother optimization landscape for BFGS compared to the multiplicative model.
    
    Inputs:
        b: biomass/density
        s: substrate
        temp: temperature
        pH: pH
    Params:
        p[0]: T_center (normalized)
        p[1]: pH_center (normalized)
        p[2]: s_center (normalized)
        p[3]: b_center (normalized)
        p[4]: w_T
        p[5]: w_pH
        p[6]: w_s
        p[7]: w_b
        p[8]: w_interaction (T-pH)
        p[9]: offset
    """
    p = params
    
    # Normalize inputs to [0, 1]
    xb = (b - 1.0) / 4.0
    xs = (s - 1.0) / 39.0
    xT = (temp - 1.0) / 39.0
    xPH = (pH - 2.0) / 10.0
    
    # RBF centers (tanh constrained to [0, 1])
    c_T = 0.5 * (1.0 + np.tanh(p[0]))
    c_pH = 0.5 * (1.0 + np.tanh(p[1]))
    c_s = 0.5 * (1.0 + np.tanh(p[2]))
    c_b = 0.5 * (1.0 + np.tanh(p[3]))
    
    # Fixed width for RBFs (chosen to cover the range reasonably)
    width = 0.3
    
    # RBF terms
    rbf_T = np.exp(-0.5 * ((xT - c_T) / width)**2)
    rbf_pH = np.exp(-0.5 * ((xPH - c_pH) / width)**2)
    rbf_s = np.exp(-0.5 * ((xs - c_s) / width)**2)
    rbf_b = np.exp(-0.5 * ((xb - c_b) / width)**2)
    
    # Interaction term (Gaussian in 2D for T and pH)
    # This captures the known strong interaction between temp and pH
    rbf_TpH = np.exp(-0.5 * (((xT - c_T) / width)**2 + ((xPH - c_pH) / width)**2))
    
    # Linear combination
    result = (
        p[4] * rbf_T +
        p[5] * rbf_pH +
        p[6] * rbf_s +
        p[7] * rbf_b +
        p[8] * rbf_TpH +
        p[9]
    )
    
    return result
