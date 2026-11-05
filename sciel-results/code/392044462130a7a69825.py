import numpy as np
import time
import math

def evaluate(coeffs, K=1000):
    n = len(coeffs)
    c_pad = np.zeros(K, dtype=complex)
    c_pad[:n] = coeffs
    P_vals = np.fft.fft(c_pad)
    mags = np.abs(P_vals)
    
    # Find top peaks to refine
    # Get top 10 indices
    top_indices = np.argsort(mags)[-10:]
    
    max_val = 0.0
    js = np.arange(n)
    
    # Precompute coeffs as array for speed
    c_arr = np.array(coeffs)
    
    def P_mag(theta):
        return np.abs(np.sum(c_arr * np.exp(1j * js * theta)))
        
    gr = (np.sqrt(5) - 1) / 2
    
    for idx in top_indices:
        theta_max = 2 * np.pi * idx / K
        delta = 2 * np.pi / K
        a = theta_max - delta/2
        b = theta_max + delta/2
        
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = P_mag(c)
        fd = P_mag(d)
        
        for _ in range(25):
            if fc > fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = P_mag(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = P_mag(d)
                
        peak_val = max(fc, fd)
        if peak_val > max_val:
            max_val = peak_val
            
    return max_val

def construct():
    N = 24
    
    # Spectral Synthesis Initialization
    # We want to create a sequence with low peak-to-average power ratio.
    # This corresponds to a "flat" spectrum in the frequency domain.
    # By taking the IFFT of a flat spectrum (with random phases to break symmetry),
    # we get a complex sequence. We then threshold the real part to get +/- 1.
    # We try multiple random phases and different window functions.
    
    candidates = []
    
    # 1. Golay-Rudin-Shapiro (known good start)
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    candidates.append(p[:N])
    
    # 2. Spectral candidates
    K_spec = 2048 # High resolution for IFFT
    
    # Try different window functions for the spectrum
    # A flat spectrum with random phases
    for _ in range(50):
        # Random phases
        phases = 2 * np.pi * np.random.rand(K_spec)
        spectrum = np.exp(1j * phases)
        
        # Ensure Hermitian symmetry for real output?
        # If we just take IFFT of random complex spectrum, output is complex.
        # We take real part and threshold.
        seq = np.fft.ifft(spectrum).real
        
        # Threshold to +/- 1
        # We want the first N coefficients
        # But the IFFT is periodic. The "center" might be better?
        # Or just take first N.
        # Let's try taking the first N.
        c_candidate = [1 if x > 0 else -1 for x in seq[:N]]
        candidates.append(c_candidate)
        
        # Also try centered (circular shift) to align the "main lobe" differently?
        # Shift by N//2
        seq_shifted = np.roll(seq, N//2)
        c_candidate_shift = [1 if x > 0 else -1 for x in seq_shifted[:N]]
        candidates.append(c_candidate_shift)
        
        # Use a window on the spectrum?
        # E.g., Hamming window to taper the spectrum?
        # This would make the time-domain sequence more localized?
        # Actually, we want the TIME domain to be +/- 1.
        # So we want the FREQUENCY domain to be smooth/flat.
        # Random phases already do this.
        
    # 3. Random candidates
    for _ in range(20):
        c_rand = [1 if np.random.random() < 0.5 else -1 for _ in range(N)]
        candidates.append(c_rand)
        
    # Evaluate all candidates
    best_score = float('inf')
    best_coeffs = None
    
    for c in candidates:
        s = evaluate(c, K=200) # Coarse evaluation for initial selection
        if s < best_score:
            best_score = s
            best_coeffs = c[:]
            
    # Refine the best candidate with Simulated Annealing
    current_coeffs = best_coeffs[:]
    current_score = best_score
    
    # Better SA parameters
    T = 0.1 # Lower initial temperature
    cooling_rate = 0.9999
    start_time = time.time()
    max_time = 60
    
    # Precompute indices for speed
    N = 24
    
    while time.time() - start_time < max_time:
        # Pick a random index to flip
        idx = np.random.randint(N)
        new_coeffs = current_coeffs[:]
        new_coeffs[idx] = -new_coeffs[idx]
        
        new_score = evaluate(new_coeffs, K=200)
        
        delta = new_score - current_score
        if delta < 0:
            current_coeffs = new_coeffs
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_coeffs = new_coeffs[:]
        else:
            if T > 1e-9: # Stop accepting if T is too small
                if np.random.random() < np.exp(-delta / T):
                    current_coeffs = new_coeffs
                    current_score = new_score
            
        T *= cooling_rate
        
    # Final refinement with higher K and local search
    # Use higher K for final local search to ensure we don't miss peaks
    K_final = 500
    
    improved = True
    while improved:
        improved = False
        s_best = evaluate(best_coeffs, K=K_final)
        for i in range(N):
            trial = best_coeffs[:]
            trial[i] = -trial[i]
            s_trial = evaluate(trial, K=K_final)
            if s_trial < s_best:
                best_coeffs = trial
                s_best = s_trial
                improved = True
                break # Restart loop as coeffs changed
                
    return best_coeffs
