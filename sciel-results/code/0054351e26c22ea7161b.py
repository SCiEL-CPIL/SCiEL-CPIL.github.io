"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Minimize value = abs(2*N*max(convolve(h,h)) / (sum(h))^2)
    
    Key facts:
    - sum(convolve(h,h)) = (sum(h))^2
    - There are 2N-1 entries in the convolution
    - So max(conv) >= (sum(h))^2 / (2N-1)
    - Thus value >= 2N/(2N-1) ~ 1.0025
    
    To approach this bound, we want the convolution to be as flat as possible.
    We use Zadoff-Chu sequences with chirp modulation and optimize parameters.
    """
    best_h = np.ones(N)
    best_val = 2.0  # seed value
    
    def evaluate(h):
        if not np.all(np.isfinite(h)):
            return float('inf')
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        conv = np.convolve(h, h)
        m = np.max(conv)
        val = abs(2 * N * m / (s * s))
        return val
    
    # Generate Zadoff-Chu base sequences and modulate with chirp
    t = np.arange(N, dtype=np.float64)
    
    # Try different Zadoff-Chu roots and chirp parameters
    best_candidate = None
    best_candidate_val = float('inf')
    
    for root in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        # Zadoff-Chu sequence
        zc = np.exp(-1j * np.pi * root * t * (t + 1) / N)
        
        # Try different chirp rates for modulation
        for chirp_rate in np.linspace(-0.5, 0.5, 21):
            # Apply quadratic phase modulation (chirp)
            chirp = np.exp(1j * np.pi * chirp_rate * t * t / N)
            modulated = zc * chirp
            
            # Take real part
            h_real = np.real(modulated)
            
            # Also try with a small DC offset to ensure non-zero sum
            for dc_offset in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]:
                h_test = h_real + dc_offset
                val = evaluate(h_test)
                if val < best_candidate_val:
                    best_candidate_val = val
                    best_candidate = h_test.copy()
            
            # Try imaginary part too
            h_imag = np.imag(modulated)
            for dc_offset in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]:
                h_test = h_imag + dc_offset
                val = evaluate(h_test)
                if val < best_candidate_val:
                    best_candidate_val = val
                    best_candidate = h_test.copy()
        
        # Also try the raw Zadoff-Chu real/imag parts
        h_zc_real = np.real(zc)
        h_zc_imag = np.imag(zc)
        for dc_offset in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]:
            h_test = h_zc_real + dc_offset
            val = evaluate(h_test)
            if val < best_candidate_val:
                best_candidate_val = val
                best_candidate = h_test.copy()
            
            h_test = h_zc_imag + dc_offset
            val = evaluate(h_test)
            if val < best_candidate_val:
                best_candidate_val = val
                best_candidate = h_test.copy()
    
    # Also try pure chirp sequences (without Zadoff-Chu)
    for chirp_rate in np.linspace(-1.0, 1.0, 41):
        chirp = np.exp(1j * np.pi * chirp_rate * t * t / N)
        h_real = np.real(chirp)
        h_imag = np.imag(chirp)
        
        for dc_offset in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]:
            h_test = h_real + dc_offset
            val = evaluate(h_test)
            if val < best_candidate_val:
                best_candidate_val = val
                best_candidate = h_test.copy()
            
            h_test = h_imag + dc_offset
            val = evaluate(h_test)
            if val < best_candidate_val:
                best_candidate_val = val
                best_candidate = h_test.copy()
    
    # Try cosine sequences with various frequencies as additional candidates
    for freq in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 75, 100]:
        h_cos = np.cos(2 * np.pi * freq * t / N)
        h_sin = np.sin(2 * np.pi * freq * t / N)
        for dc_offset in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]:
            h_test = h_cos + dc_offset
            val = evaluate(h_test)
            if val < best_candidate_val:
                best_candidate_val = val
                best_candidate = h_test.copy()
            
            h_test = h_sin + dc_offset
            val = evaluate(h_test)
            if val < best_candidate_val:
                best_candidate_val = val
                best_candidate = h_test.copy()
    
    # If we found a better candidate, use it; otherwise use seed
    if best_candidate is not None and best_candidate_val < best_val:
        return best_candidate
    return best_h
# EVOLVE-BLOCK-END
