"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _spectral_waterfill(h, deadline):
    """
    Structural change: Optimize in the frequency domain.
    The autoconvolution is the inverse FFT of |H(f)|^2.
    To minimize the max absolute value of the autoconvolution, we aim to flatten
    the power spectrum |H(f)|^2.
    
    Algorithm:
    1. Compute FFT of current h.
    2. Compute target magnitude for each bin such that the sum of squares is preserved
       (Rayleigh quotient constraint) but the distribution is as flat as possible.
       Ideally, we want |H_k|^2 to be constant.
    3. Construct a "soft" target signal in time domain by taking the phase of H
       and the target magnitude.
    4. Project this soft signal back to the binary manifold {+1, -1} by thresholding.
    5. Repeat until deadline.
    """
    h = h.copy()
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    best_h = h.copy()
    best_score = _score(h)
    
    # Precompute indices for rfft
    # rfft length is fsize // 2 + 1
    
    while time.time() < deadline:
        # 1. FFT
        H = np.fft.rfft(h, fsize)
        
        # 2. Compute Power Spectrum
        P = np.abs(H)**2
        
        # 3. Determine Target Magnitude
        # We want to flatten P.
        # Constraint: Sum(P) is fixed by Parseval's theorem (Sum(h^2) * 2 or similar depending on normalization).
        # Actually, for real signals, Sum(|H|^2) = fsize * Sum(h^2).
        # Since h is binary ±1, Sum(h^2) = n.
        # So total power is fsize * n.
        # Target power per bin = (fsize * n) / (fsize // 2 + 1)
        
        num_bins = len(H)
        target_power = (fsize * n) / num_bins
        target_mag = np.sqrt(target_power)
        
        # We cannot make it perfectly flat because the DC bin (k=0) is constrained by sum(h).
        # And the Nyquist bin (if exists) is constrained.
        # Also, for real signals, H[k] and H[-k] are conjugates.
        
        # Strategy: Adjust magnitude towards target, but respect DC and Nyquist if possible?
        # Actually, changing magnitude changes the time domain signal, which changes the sum.
        # The sum is the DC component.
        
        # Let's construct a new complex spectrum H_new
        # Phase is preserved from H.
        # Magnitude is set to target_mag, EXCEPT we might want to keep DC fixed?
        # If we change DC, the sum changes, which affects the denominator.
        # The objective is max|ac| / sum^2.
        # Flattening the spectrum reduces max|ac|.
        # Changing sum increases or decreases denominator.
        
        # Let's try to flatten all bins including DC?
        # If we flatten all bins, the time domain signal will be a sinc-like pulse, not binary.
        # We project back to binary.
        
        phases = np.angle(H)
        mags = np.full(num_bins, target_mag)
        
        # Special handling: 
        # If we force all magnitudes equal, the resulting time-domain signal 
        # will have low peak-to-average ratio.
        
        H_new = mags * np.exp(1j * phases)
        
        # Inverse FFT
        h_soft = np.fft.irfft(H_new, fsize)
        
        # Extract first n samples
        h_soft = h_soft[:n]
        
        # Project to binary: 
        # The "best" binary approximation to a real vector x is sign(x).
        # But we must ensure sum != 0.
        h_new = np.sign(h_soft)
        
        # Handle zeros in sign (if any element is exactly 0, sign is 0). 
        # Replace 0 with 1 or -1 randomly or based on previous? 
        # For robustness, replace 0 with 1.
        h_new[h_new == 0] = 1.0
        
        # Check if this is an improvement
        s_new = np.sum(h_new)
        if abs(s_new) < 1e-9:
             # If sum is 0, it's invalid. Flip one bit.
             # Find index with smallest absolute value in h_soft to flip?
             # Or just flip first element.
             h_new[0] = -h_new[0]
             s_new = np.sum(h_new)
             
        new_score = _score(h_new)
        
        if new_score < best_score:
            best_score = new_score
            best_h = h_new.copy()
            h = h_new.copy()
        else:
            # If it didn't improve, maybe we got stuck in a local minimum of the projection.
            # Try a small perturbation? 
            # Or just stop if no improvement for a while?
            # For now, if it doesn't improve, we might oscillate.
            # Let's add a small random perturbation to the phases if stuck.
            # But we don't track "stuck" easily.
            # Let's just keep h as is if it didn't improve? 
            # Actually, if h_new is worse, we should probably not update h, 
            # but the next iteration will use the old h.
            # This might cause oscillation between two states.
            
            # To break ties/oscillation, if new_score is very close to best_score, 
            # we might accept it? No, strict improvement is better for convergence.
            
            # Let's try adding a tiny bit of noise to the phases in the next step if we fail?
            # Simpler: Just don't update h. The loop will just compute the same H 
            # and get the same h_new. This is infinite loop risk.
            
            # Fix: If no improvement, perturb the current h slightly?
            # Or accept if new_score < best_score * 1.01?
            
            # Better: If no improvement, randomly flip a few bits in h to escape local min.
            if time.time() < deadline:
                # Random flip 5 bits
                indices = np.random.choice(n, 5, replace=False)
                h[indices] *= -1
                # Ensure sum != 0
                if abs(np.sum(h)) < 1e-9:
                    h[0] *= -1

    return best_h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N)  # Baseline

    # Evaluate baseline
    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    # Generate a few diverse initial candidates to give the spectral optimizer a head start
    rng = np.random.default_rng(42)
    candidates = []
    
    # 1. Random
    for _ in range(5):
        h = rng.choice([-1.0, 1.0], size=N)
        if abs(np.sum(h)) < 1e-9:
            h[0] *= -1
        candidates.append(h)
        
    # 2. Chirp
    x = np.arange(N)
    h_chirp = np.sign(np.cos(np.pi * x**2 / N))
    if abs(np.sum(h_chirp)) < 1e-9:
        h_chirp[0] *= -1
    candidates.append(h_chirp)
    
    # 3. Alternating
    h_alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    if abs(np.sum(h_alt)) < 1e-9:
        h_alt[0] *= -1
    candidates.append(h_alt)

    # Optimize each candidate with spectral waterfilling
    for h_init in candidates:
        if time.time() > deadline - 5:
            break
            
        h_opt, val_opt = _spectral_waterfill(h_init, deadline - 2)
        
        if val_opt < best_score:
            best_score = val_opt
            best_h = h_opt.copy()

    return best_h

# EVOLVE-BLOCK-END
