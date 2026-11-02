"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4]
    to minimize the autoconvolution upper bound C3.
    
    Uses greedy coordinate descent with FFT-based evaluation.
    """
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Initialize with a pseudo-random sign sequence
    rng = np.random.default_rng(42)
    signs = rng.choice([-1.0, 1.0], size=N)
    
    # Precompute indices for faster access
    indices = np.arange(N)
    
    def compute_score(s):
        """Compute the upper bound C3 for a given sign sequence.
        
        The autoconvolution of the step function f(x) = s[i] on interval i
        has peak value related to the maximum of the autocorrelation.
        For signed steps, C3 is proportional to max |autocorr| / (sum |s|)^2
        or similar normalization.
        
        Based on the task: "autoconvolution upper bound" - this is the
        maximum absolute value of the discrete autocorrelation of the sign sequence,
        normalized appropriately.
        """
        # Compute autocorrelation via FFT
        n_fft = 2 * N
        fft_s = np.fft.rfft(s, n=n_fft)
        acf = np.fft.irfft(fft_s * np.conj(fft_s), n=n_fft)[:N]
        
        # The peak sidelobe (excluding the main lobe at lag 0)
        # C3 upper bound is typically the max |acf[k]| for k != 0, normalized
        peak_sidelobe = np.max(np.abs(acf[1:]))
        
        # Normalize by energy
        energy = np.sum(s * s)
        if energy > 0:
            return peak_sidelobe / energy
        return 0.0
    
    # Initial score
    best_signs = signs.copy()
    best_score = compute_score(best_signs)
    
    # Greedy coordinate descent: try flipping each position
    # Use FFT to efficiently evaluate flips
    def score_after_flip(s, pos):
        """Compute score after flipping position pos without full recomputation."""
        new_s = s.copy()
        new_s[pos] *= -1.0
        return compute_score(new_s)
    
    # For efficiency, we'll do multiple passes
    improved = True
    pass_count = 0
    max_passes = 50
    
    while improved and time.time() < deadline and pass_count < max_passes:
        improved = False
        pass_count += 1
        
        # Try flipping each position
        for pos in range(N):
            if time.time() > deadline:
                break
            
            old_val = best_signs[pos]
            best_signs[pos] = -old_val
            new_score = compute_score(best_signs)
            
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                best_signs[pos] = old_val  # Undo
    
    return best_signs
# EVOLVE-BLOCK-END
