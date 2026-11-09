"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a decoupled phase-separable model.
    
    Structure:
    - Normalized inputs in [0,1].
    - Separate smooth functions for each input (sin/cos) with learnable frequency and amplitude.
    - A multiplicative coupling term for b and s.
    - Bias term.
    
    Parameters (10 total):
    p[0]: Bias
    p[1]: Amplitude for b
    p[2]: Frequency for b
    p[3]: Amplitude for s
    p[4]: Frequency for s
    p[5]: Amplitude for temp
    p[6]: Frequency for temp
    p[7]: Amplitude for pH
    p[8]: Frequency for pH
    p[9]: Coupling strength for b*s
    """
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    # Normalization to [0, 1]
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    temp_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    # Extract parameters
    bias = params[0]
    
    # Use sin and cos to allow for any phase/shift within the [0,1] range effectively
    # f(x) = amp * sin(freq * x + phase). We don't have phase params, but sin(freq*x) 
    # covers [0, sin(freq)] which can be shifted if we allow negative amplitudes/frequencies?
    # Better: Use a combination of sin and cos? That would double the params needed per feature.
    # With only 2 params per feature (amp, freq), let's stick to:
    # term = amp * cos(freq * x)  => at x=0, term=amp. At x=1, term=amp*cos(freq).
    # This is flexible enough to capture peaks/troughs.
    
    amp_b, freq_b = params[1], params[2]
    amp_s, freq_s = params[3], params[4]
    amp_t, freq_t = params[5], params[6]
    amp_p, freq_p = params[7], params[8]
    coupling = params[9]
    
    # Compute smooth terms
    # Using cos allows the function to be non-zero at x=0, which is important for normalized [0,1] inputs.
    # sin(0)=0 would force the contribution to be 0 at the minimum input, which might be restrictive.
    
    term_b = amp_b * np.cos(freq_b * b_norm)
    term_s = amp_s * np.cos(freq_s * s_norm)
    term_t = amp_t * np.cos(freq_t * temp_norm)
    term_p = amp_p * np.cos(freq_p * ph_norm)
    
    # Interaction term: captures joint effects of biomass and substrate
    # We use the product of the normalized values to keep it bounded and simple
    interaction = coupling * (b_norm * s_norm)
    
    # Sum all components
    prediction = bias + term_b + term_s + term_t + term_p + interaction
    
    # Handle non-finite values
    if not np.all(np.isfinite(prediction)):
        return np.zeros_like(prediction)
        
    return prediction
# EVOLVE-BLOCK-END
