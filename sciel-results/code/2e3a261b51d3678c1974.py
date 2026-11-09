# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate from population density, substrate, temperature and pH.
    
    Uses a sum-of-sines (Fourier) basis with learnable frequencies for each input.
    Includes an interaction term between temperature and pH to capture non-additive
    effects (e.g., optimal temperature shifting with pH).
    
    params layout (10 total):
      params[0]: base offset
      params[1]: amplitude for b (population density)
      params[2]: frequency for b
      params[3]: amplitude for s (substrate)
      params[4]: frequency for s
      params[5]: amplitude for temp
      params[6]: frequency for temp
      params[7]: amplitude for pH
      params[8]: frequency for pH
      params[9]: amplitude for interaction term (temp * pH)
    """
    # Extract parameters
    offset = params[0]
    amp_b = params[1]
    freq_b = params[2]
    amp_s = params[3]
    freq_s = params[4]
    amp_t = params[5]
    freq_t = params[6]
    amp_ph = params[7]
    freq_ph = params[8]
    amp_inter = params[9]
    
    # Compute sine-based features for each input
    # Using sin(freq * x) to allow periodic behavior
    term_b = amp_b * np.sin(freq_b * b)
    term_s = amp_s * np.sin(freq_s * s)
    term_t = amp_t * np.sin(freq_t * temp)
    term_ph = amp_ph * np.sin(freq_ph * pH)
    
    # Interaction term between temperature and pH
    # Using the same frequencies as the main terms for the phase argument
    # This allows the model to capture that the "peak" of one might shift or
    # modulate the other.
    term_inter = amp_inter * np.sin(freq_t * temp + freq_ph * pH)
    
    # Sum of all terms plus offset
    return offset + term_b + term_s + term_t + term_ph + term_inter
# EVOLVE-BLOCK-END
