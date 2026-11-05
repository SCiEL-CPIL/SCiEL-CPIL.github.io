import numpy as np
import time
import math

def evaluate(coeffs, K=2048):
    """
    Accurately compute the sup norm of P(z) on |z|=1.
    1. Dense sampling via FFT (zero-padded).
    2. Identify local maxima in the magnitude spectrum.
    3. Refine the top peaks using Golden Section Search.
    """
    n = len(coeffs)
    # Ensure coeffs are numpy array of floats for speed in P_mag
    c_arr = np.array(coeffs, dtype=float)
    
    # 1. Dense Sampling
    # P(e^{i theta}) = sum c_j e^{i j theta}
    # np.fft.fft(c_pad) gives sum c_j e^{-i 2 pi j k / K}
    # Magnitude is the same for +i and -i.
    c_pad = np.zeros(K, dtype=complex)
    c_pad[:n] = c_arr
    P_vals = np.fft.fft(c_pad)
    mags = np.abs(P_vals)
    
    # 2. Identify Local Maxima
    # A point k is a local max if mags[k] > mags[k-1] and mags[k] > mags[k+1]
    # Handle circular boundary
    local_max_indices = []
    for k in range(K):
        if mags[k] >= mags[(k-1) % K] and mags[k] >= mags[(k+1) % K]:
            local_max_indices.append(k)
            
    # Sort by magnitude to get the top peaks
    local_max_indices.sort(key=lambda k: mags[k], reverse=True)
    
    # Only refine the top 3 peaks to save time
    top_peaks = local_max_indices[:3]
    
    max_val = 0.0
    
    # 3. Golden Section Refinement
    # Function to evaluate P(e^{i theta})
    def P_mag(theta):
        # Vectorized computation
        # theta is scalar
        js = np.arange(n)
        # c_arr is +/- 1
        # exp(1j * js * theta)
        # sum(c_arr * exp(...))
        return np.abs(np.sum(c_arr * np.exp(1j * js * theta)))
        
    gr = (np.sqrt(5) - 1) / 2
    delta = 2 * np.pi / K
    
    for idx in top_peaks:
        theta_max = 2 * np.pi * idx / K
        a = theta_max - delta/2
        b = theta_max + delta/2
        
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = P_mag(c)
        fd = P_mag(d)
        
        for _ in range(40):
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

def generate_spectral_candidate(N, K_spec, target_shape='flat'):
    """
    Generate a +/-1 sequence using spectral synthesis.
    1. Create target magnitude spectrum.
    2. Assign random phases (Hermitian symmetric).
    3. IFFT.
    4. Threshold to +/-1.
    """
    # Target magnitude
    if target_shape == 'flat':
        mags = np.ones(K_spec)
    elif target_shape == 'taper':
        # Cosine taper to reduce Gibbs phenomenon effects in time domain?
        # Actually, for flat time domain (impulses), we want flat freq.
        # But we are mapping to +/-1, so it's not impulses.
        # Let's try a slightly tapered spectrum to encourage smoothness in time?
        # No, the goal is to minimize peak in freq. 
        # A flat spectrum corresponds to a Dirac comb in time.
        # We want the time domain to be +/-1.
        # Let's just use random phases on a flat spectrum.
        mags = np.ones(K_spec)
    else:
        mags = np.ones(K_spec)
        
    # Random Phases
    phases = np.random.uniform(0, 2*np.pi, K_spec)
    
    # Construct Hermitian Symmetric Spectrum
    # S[k] = mags[k] * exp(i * phases[k])
    # For real time domain, S[K-k] = conj(S[k])
    # So phases[K-k] = -phases[k]
    # And mags[K-k] = mags[k]
    
    S = np.zeros(K_spec, dtype=complex)
    for k in range(K_spec // 2 + 1):
        S[k] = mags[k] * np.exp(1j * phases[k])
        if k > 0 and k < K_spec // 2:
            S[K_spec - k] = np.conj(S[k])
        elif k == 0:
            # DC term must be real
            S[0] = mags[0]
        elif K_spec % 2 == 0 and k == K_spec // 2:
            # Nyquist term must be real
            S[k] = mags[k]
            
    # IFFT
    time_domain = np.fft.ifft(S).real
    
    # Threshold to +/-1
    # To minimize variance, we could adjust the offset, but simple threshold is good for init.
    coeffs = np.sign(time_domain)
    # Handle zeros (unlikely)
    coeffs[coeffs == 0] = 1
    
    # Ensure length N
    # The IFFT gives K_spec points. We only need N=24.
    # Usually, the "main lobe" or the first N points?
    # Actually, spectral synthesis usually implies the time domain is the signal.
    # If we want N coefficients, we should probably design the spectrum such that the time domain is periodic with period N?
    # Or just take the first N samples?
    # If we take the first N samples of an IFFT of length K_spec, it's not a clean polynomial.
    
    # Alternative Interpretation:
    # We want a polynomial of degree N-1.
    # The spectrum of this polynomial is defined on the unit circle.
    # If we synthesize a time-domain signal of length N that minimizes peak magnitude in frequency...
    # This is equivalent to finding a sequence c_0..c_{N-1} in +/-1 such that max |sum c_k e^{i k theta}| is min.
    
    # The spectral synthesis approach described in literature (e.g., for Golay sequences) often involves:
    # 1. Start with a random spectrum.
    # 2. IFFT to get time domain.
    # 3. Project time domain to +/-1.
    # 4. FFT to get new spectrum.
    # 5. Modify spectrum (e.g., flatten magnitude).
    # 6. Repeat.
    
    # Let's implement a few iterations of this "Gerchberg-Saxton" style loop for initialization.
    
    coeffs_init = np.random.choice([-1, 1], size=N)
    
    for _ in range(50):
        # FFT of current coeffs (zero padded to K_spec for smoothness? Or just N?)
        # If we use N points, the spectrum is sampled at N points.
        # If we use K_spec points, we are interpolating.
        # Let's use K_spec for better frequency resolution.
        c_pad = np.zeros(K_spec, dtype=complex)
        c_pad[:N] = coeffs_init
        S = np.fft.fft(c_pad)
        
        # Flatten magnitude of S
        mags = np.abs(S)
        # Target: Constant magnitude?
        # To minimize peak, we want flat magnitude.
        # Set magnitude to mean magnitude
        mean_mag = np.mean(mags)
        new_mags = np.full(K_spec, mean_mag)
        
        # Keep phases of S
        phases = np.angle(S)
        
        # Reconstruct S
        S_new = new_mags * np.exp(1j * phases)
        
        # Enforce Hermitian symmetry for real coefficients
        # S_new[k] should be conj(S_new[K-k])
        # This is automatically true if we derived it from real input, but 
        # after modifying magnitude, we might break it if we don't handle it carefully.
        # Actually, if we just change magnitude to be symmetric, it stays Hermitian.
        # new_mags is symmetric. phases[k] = -phases[K-k] (mod pi) for real signal?
        # Yes, for real signal, S[K-k] = conj(S[k]).
        # So |S[K-k]| = |S[k]| and angle(S[K-k]) = -angle(S[k]).
        # Our new_mags is constant, so symmetric.
        # Phases are from original S, which was Hermitian.
        # So S_new is Hermitian.
        
        # IFFT
        time_domain = np.fft.ifft(S_new).real
        
        # Project to +/-1
        coeffs_init = np.sign(time_domain[:N])
        coeffs_init[coeffs_init == 0] = 1
        
    return coeffs_init.tolist()

def construct():
    N = 24
    K_spec = 1024
    
    # Generate multiple spectral candidates
    candidates = []
    for i in range(10):
        c = generate_spectral_candidate(N, K_spec)
        candidates.append(c)
        
    # Also add a Golay-Rudin-Shapiro sequence as a robust baseline
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    candidates.append(p[:N])
    
    best_coeffs = None
    best_score = float('inf')
    
    start_time = time.time()
    max_time = 110 # Leave 10s buffer
    
    # Local Search for each candidate
    for candidate in candidates:
        if time.time() - start_time > max_time:
            break
            
        current_coeffs = candidate[:]
        current_score = evaluate(current_coeffs, K=512) # Faster eval for local search
        
        # Simulated Annealing or Hill Climbing?
        # Hill Climbing is faster and often sufficient for +/-1 optimization if start is good.
        # Let's do a fast Hill Climbing with occasional random restarts within the candidate's neighborhood?
        # No, just simple bit-flip hill climbing.
        
        improved = True
        while improved and (time.time() - start_time < max_time):
            improved = False
            # Try flipping each bit
            for i in range(N):
                trial = current_coeffs[:]
                trial[i] = -trial[i]
                trial_score = evaluate(trial, K=512)
                
                if trial_score < current_score:
                    current_coeffs = trial
                    current_score = trial_score
                    improved = True
                    break # Restart scan as coeffs changed
                    
        # Final high-accuracy evaluation
        final_score = evaluate(current_coeffs, K=2048)
        
        if final_score < best_score:
            best_score = final_score
            best_coeffs = current_coeffs[:]
            
    # If time remains, do a global refinement on the best candidate with high K
    if best_coeffs is not None and time.time() - start_time < max_time:
        current_coeffs = best_coeffs[:]
        current_score = evaluate(current_coeffs, K=2048)
        
        improved = True
        while improved and (time.time() - start_time < max_time - 5):
            improved = False
            for i in range(N):
                trial = current_coeffs[:]
                trial[i] = -trial[i]
                trial_score = evaluate(trial, K=2048)
                
                if trial_score < current_score:
                    current_coeffs = trial
                    current_score = trial_score
                    improved = True
                    break
                    
        if current_score < best_score:
            best_score = current_score
            best_coeffs = current_coeffs[:]
            
    return best_coeffs
