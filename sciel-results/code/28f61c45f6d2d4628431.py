"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4]
    to improve an autoconvolution upper bound (C3).
    
    Strategy: 
    1. Design the signal in the frequency domain using a smooth, band-limited window 
       (Raised Cosine) to ensure rapid decay of the autocorrelation sidelobes.
    2. Convert to time domain via IFFT.
    3. Refine the signal using a local search (sign flips and amplitude adjustments) 
       to minimize the L1 norm of the off-diagonal autocorrelation.
    """
    deadline = time.time() + 140  # 140s deadline guard (10s safety margin)
    
    n = N
    
    # Step 1: Frequency Domain Design
    # We want the power spectrum |H(f)|^2 to be smooth and concentrated.
    # Let H(f) be a Raised Cosine window in the frequency domain.
    # This results in a time-domain signal h(t) that is a sinc-like function 
    # with low sidelobes. The autocorrelation of h(t) will have a spectrum 
    # proportional to |H(f)|^2, which is also smooth, leading to rapid decay 
    # in the time-domain autocorrelation sidelobes.
    
    # Create frequency bins
    # We use a symmetric frequency window centered at 0.
    # Bandwidth parameter: fraction of the Nyquist frequency to use.
    # A narrower bandwidth in frequency leads to a wider main lobe in time, 
    # but potentially lower sidelobes if the window is smooth.
    # However, for a fixed N, the "width" is relative.
    
    # Let's try a Kaiser window in the frequency domain, which is known 
    # to have optimal sidelobe suppression for a given main lobe width.
    try:
        from scipy.signal.windows import kaiser
        # Beta parameter for Kaiser window. Higher beta = lower sidelobes.
        # We want the autocorrelation sidelobes to be low.
        # The autocorrelation spectrum is |H|^2. If H is a Kaiser window, 
        # |H|^2 is also a smooth window.
        beta = 10.0
        # Create a Kaiser window of length N in the frequency domain.
        # We place it in the center of the FFT frequency bins.
        H = np.zeros(n, dtype=complex)
        # The main lobe of the Kaiser window is centered at 0.
        # We shift it to the center of the array for IFFT.
        window = kaiser(n, beta)
        # Place the window in the frequency domain.
        # For a real signal, we need symmetry.
        # Let's just use the window directly as the magnitude spectrum.
        H[0] = window[0]
        # Symmetric placement
        # Actually, simpler: Create a time-domain signal that is a Kaiser window, 
        # then check its autocorrelation. 
        # Wait, the directive says "Model ... in the frequency domain ... and converting 
        # it to the time domain via inverse FFT".
        
        # Let's define H(f) as a smooth window.
        # We want H(f) to be real and even to get a real and even h(t).
        # Let H(f) = Kaiser(n, beta) shifted to center.
        # In FFT order, DC is at 0, Nyquist at N/2.
        # We want the "center" of the window to be at DC (0).
        # So we take the Kaiser window and center it.
        
        # Kaiser window is symmetric. 
        # We can create a circularly symmetric window in the frequency domain.
        # Let's just use the Kaiser window as the frequency response magnitude.
        # To ensure h(t) is real, H(-f) = conj(H(f)).
        # If H is real and even, h(t) is real and even.
        
        # Create a frequency grid centered at 0
        # np.fft.ifft expects the input in standard FFT order (0, 1, ..., N/2, -N/2+1, ..., -1)
        # We want a window centered at index 0 (DC).
        # So we take the Kaiser window, which is symmetric, and place it such that 
        # the peak is at DC.
        
        # Standard Kaiser window w[i] is centered at (N-1)/2.
        # We want to shift it so the center is at index 0.
        # This is equivalent to circularly shifting the window by N/2.
        
        # Let's generate the window
        w = kaiser(n, beta)
        
        # Shift w so that its center is at index 0.
        # The center of w is at index n//2 (for even n).
        # We want to move index n//2 to index 0.
        # This is a shift of n//2 to the right (circularly).
        H = np.roll(w, n//2)
        
        # Ensure H is real and even for a real h(t).
        # Since w is real and even, and we shifted by n/2, 
        # H should be real and even around 0.
        
        # Compute time domain signal
        h = np.fft.ifft(H).real
        
    except ImportError:
        # Fallback: Gaussian window in frequency domain
        f = np.arange(n)
        # Center at 0
        f_centered = np.roll(f, n//2)
        sigma = n / 10.0
        H = np.exp(-0.5 * (f_centered / sigma) ** 2)
        h = np.fft.ifft(H).real

    # Normalize h
    norm = np.linalg.norm(h)
    if norm > 0:
        h = h / norm

    best_h = h.copy()
    
    # Define the score function: L1 norm of off-diagonal autocorrelation
    def score(h):
        # Compute autocorrelation
        # Use FFT for efficiency
        fft_size = 2 * n
        h_fft = np.fft.fft(h, fft_size)
        autoconv = np.fft.ifft(h_fft * np.conj(h_fft)).real
        autoconv = autoconv[:n]
        
        # Sum of absolute values of sidelobes
        # Exclude the peak at 0
        sidelobes = np.abs(autoconv[1:])
        # Also consider the negative lags? 
        # The autocorrelation is symmetric for real signals: R[k] = R[-k].
        # The FFT method gives R[0], R[1], ..., R[n-1] (where R[n-k] corresponds to R[-k]).
        # So autoconv[1:] contains R[1]...R[n/2] and R[n/2+1]...R[n-1] (which are R[-n/2+1]...R[-1]).
        # So summing abs(autoconv[1:]) covers all off-diagonal lags.
        return np.sum(sidelobes)

    best_score = score(best_h)
    
    # Local Search: 
    # We can try to flip signs of individual bins or adjust amplitudes.
    # Since it's a "signed step function", flipping signs is a natural move.
    # However, continuous amplitudes might be better.
    # Let's try a combination: small perturbations and sign flips.
    
    rng = np.random.RandomState(42)
    n_iterations = 0
    max_iterations = 50000
    
    # To speed up, we can precompute the FFT of the current signal? 
    # No, if we change one bin, the FFT changes completely.
    # But we can use the linearity of the autocorrelation? 
    # R = IFFT(|FFT(h)|^2). This is non-linear in h.
    # So we must recompute the FFT and IFFT for each candidate.
    # FFT of size 800 is fast. 50000 iterations * ~1ms = 50s. Feasible.
    
    while time.time() < deadline and n_iterations < max_iterations:
        n_iterations += 1
        
        # Generate a candidate
        trial = best_h.copy()
        
        # Strategy: 
        # 1. Flip sign of a random element
        # 2. Add small Gaussian noise to a random element
        # 3. Scale a random element
        
        move_type = rng.randint(3)
        idx = rng.randint(n)
        
        if move_type == 0:
            # Flip sign
            trial[idx] = -trial[idx]
        elif move_type == 1:
            # Small perturbation
            trial[idx] += rng.randn() * 0.01
        else:
            # Scale
            trial[idx] *= (1.0 + rng.randn() * 0.01)
        
        # Recompute score
        trial_score = score(trial)
        
        if trial_score < best_score:
            best_h = trial
            best_score = trial_score

    return best_h
# EVOLVE-BLOCK-END
