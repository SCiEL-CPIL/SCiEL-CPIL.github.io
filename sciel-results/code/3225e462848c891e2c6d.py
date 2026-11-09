# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate from population density, substrate, temperature and pH.
    
    Uses a sum-of-sines (Fourier) basis.
    Temperature and pH get independent phases to model unimodal optima.
    Substrate uses a sine with fixed phase 0.
    Density uses a scaled sine with fixed frequency to model monotonic effects.
    
    params layout (10 total):
      params[0]: base offset
      params[1]: amplitude for temp
      params[2]: frequency for temp
      params[3]: phase for temp
      params[4]: amplitude for pH
      params[5]: frequency for pH
      params[6]: phase for pH
      params[7]: amplitude for s
      params[8]: frequency for s
      params[9]: amplitude for b
    """
    # Extract parameters
    offset = params[0]
    
    amp_t = params[1]
    freq_t = params[2]
    phase_t = params[3]
    
    amp_ph = params[4]
    freq_ph = params[5]
    phase_ph = params[6]
    
    amp_s = params[7]
    freq_s = params[8]
    
    amp_b = params[9]
    
    # Compute sine-based features
    # Temp and pH: Full flexibility (Amp, Freq, Phase) to capture peaks
    term_t = amp_t * np.sin(freq_t * temp + phase_t)
    term_ph = amp_ph * np.sin(freq_ph * pH + phase_ph)
    
    # Substrate: Fixed phase 0. 
    # term_s = amp_s * np.sin(freq_s * s)
    # If freq_s is small, this is roughly linear.
    term_s = amp_s * np.sin(freq_s * s)
    
    # Density: Fixed frequency (scaled) and phase.
    # Scale b by 0.1 to make sin(0.1*b) approximately linear over [1, 5]
    term_b = amp_b * np.sin(0.1 * b)
    
    return offset + term_b + term_s + term_t + term_ph
# EVOLVE-BLOCK-END
