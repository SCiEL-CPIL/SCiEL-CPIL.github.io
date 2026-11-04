import numpy as np
import time

N = 400

def _autoconvolution_max_abs(f):
    """
    Compute the maximum absolute value of the aperiodic autocorrelation
    for non-zero lags.
    """
    # Zero-pad to 2*N to compute linear autocorrelation via circular convolution
    n = len(f)
    # FFT of f
    F = np.fft.fft(f, 2*n)
    # Power spectrum
    P = F * np.conj(F)
    # Inverse FFT to get autocorrelation
    corr = np.fft.ifft(P).real
    # The autocorrelation is in the first n elements (lags 0 to n-1)
    # and the last n-1 elements (lags -1 to -(n-1))
    # corr[0] is lag 0 (max value, sum of squares)
    # We want max |corr[k]| for k != 0
    # corr[n-1:] corresponds to negative lags? 
    # Actually, np.fft.ifft of F*conj(F) gives circular correlation.
    # For linear correlation, we look at corr[0:n] and corr[n:2n-1]?
    # Let's verify indexing.
    # If f = [1, 1, 0, 0], N=4.
    # corr should be [2, 1, 0, 0, 0, 0, 1, 2]? No.
    # Linear autocorr R[k] = sum f[i] f[i+k].
    # R[0] = 2. R[1] = 1. R[2] = 0. R[3] = 0.
    # R[-1] = 1. R[-2] = 0. R[-3] = 0.
    # np.fft.ifft(fft(f, 2N)**2) gives circular correlation of length 2N.
    # The result array `c` has c[0] = R[0], c[1] = R[1], ..., c[N-1] = R[N-1], 
    # c[N] = R[-N+1] (which is R[N-1] if symmetric? No, R[-k]=R[k] for real f? 
    # Wait, autocorrelation is symmetric: R[k] = R[-k].
    # So c[N] = R[-N+1] = R[N-1].
    # So the peaks are in c[1:N] and c[N+1:2N]?
    # Actually, for real signals, the autocorrelation is symmetric.
    # c[0] is lag 0.
    # c[1] to c[N-1] are positive lags 1 to N-1.
    # c[N] to c[2N-1] are negative lags -N to -1?
    # Let's just take the max of the whole array except index 0.
    # But we must be careful with the wrap-around.
    # The linear autocorrelation is embedded in the circular one.
    # The values c[0:N] are R[0:N-1].
    # The values c[N:2N] are R[-N:-1]? 
    # Since R is symmetric, max(R[1:N-1]) is sufficient.
    
    # To be safe and simple, we can compute linear autocorr directly for N=400?
    # N^2 = 160,000 operations. 
    # np.correlate(f, f, mode='full') does this.
    # Let's use np.correlate for correctness and simplicity, as N is small.
    corr = np.correlate(f, f, mode='full')
    # corr has length 2N-1.
    # corr[N-1] is lag 0.
    # We want max abs of corr excluding the center.
    # lags 1 to N-1 are corr[N:]
    # lags -1 to -(N-1) are corr[:N-1]
    # Since symmetric, just check corr[N:]
    if len(corr) > N:
        return np.max(np.abs(corr[N:]))
    return 0.0

def _legendre_sequence(n):
    """Generate Legendre sequence of length n."""
    def legendre_symbol(a, p):
        ls = pow(a, (p - 1) // 2, p)
        return -1 if ls == p - 1 else ls

    seq = []
    for i in range(1, n + 1):
        # Find smallest prime p > i
        p = i + 1
        while True:
            is_prime = True
            for j in range(2, int(np.sqrt(p)) + 1):
                if p % j == 0:
                    is_prime = False
                    break
            if is_prime:
                break
            p += 1
        # Legendre symbol of i mod p
        if i % p == 0:
            seq.append(0) # Should not happen for i < p
        else:
            seq.append(legendre_symbol(i, p))
    return np.array(seq, dtype=float)

def _m_sequence(n, poly=341):
    """Generate M-sequence of length n using LFSR.
    poly: integer representation of feedback polynomial.
    We use a standard 9-bit LFSR for length 511, then truncate.
    """
    # Primitive polynomial x^9 + x^4 + 1 (octal 51? No, 341 decimal is 101010101 binary)
    # Let's use a known good one: x^9 + x^5 + 1 (octal 125, dec 85? No)
    # x^9 + x^4 + 1 is 100010001 binary = 273 decimal.
    # Let's use x^9 + x^5 + 1 = 100010001? No.
    # Standard 9-bit m-seq: x^9 + x^4 + 1.
    # Feedback taps at 9 and 4.
    # Register size 9.
    reg = 1
    seq = []
    for _ in range(n):
        # Output is MSB or LSB? Let's say LSB.
        seq.append(1 if reg & 1 else -1)
        # Shift left, feedback is XOR of bits at positions 8 and 3 (0-indexed from LSB, so bit 8 and bit 3)
        # Polynomial x^9 + x^4 + 1 means feedback to bit 0 from bit 8 XOR bit 3.
        feedback = (reg >> 8) ^ (reg >> 3)
        reg = ((reg << 1) | feedback) & 0x1FF # Mask to 9 bits
    return np.array(seq, dtype=float)

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    
    best_f = None
    best_val = np.inf
    
    # Generate diverse deterministic seeds
    seeds = []
    
    # 1. Legendre
    try:
        leg = _legendre_sequence(N)
        # Ensure no zeros, replace 0 with 1
        leg[leg == 0] = 1
        seeds.append(leg)
    except Exception:
        pass
        
    # 2. M-sequence
    try:
        mseq = _m_sequence(N)
        seeds.append(mseq)
    except Exception:
        pass
        
    # 3. Random
    np.random.seed(42)
    seeds.append(np.random.choice([-1, 1], size=N))
    
    # 4. Alternating
    alt = np.ones(N)
    alt[::2] = -1
    seeds.append(alt)
    
    # 5. Walsh-Hadamard row (Hadamard matrix row 1)
    # Construct Hadamard matrix H_N (N must be power of 2, 512 > 400)
    # Use recursive construction
    def hadamard(n):
        if n == 1:
            return np.array([[1]])
        H = hadamard(n // 2)
        return np.block([[H, H], [H, -H]])
    
    try:
        H = hadamard(512)
        # Pick a row with good properties, e.g., row 1 (all ones) is bad.
        # Row 2 (1, -1, 1, -1...) is bad.
        # Pick a "complex" row, e.g., index 10
        seeds.append(H[10, :N])
    except Exception:
        pass

    # Simulated Annealing for each seed
    T_start = 10.0
    T_end = 0.01
    alpha = 0.995
    n_iter = 5000
    
    for seed_idx, f in enumerate(seeds):
        if time.time() > deadline - 10:
            break
            
        f = f.copy()
        current_val = _autoconvolution_max_abs(f)
        best_seed_val = current_val
        best_seed_f = f.copy()
        
        if current_val < best_val:
            best_val = current_val
            best_f = f.copy()
            
        T = T_start
        for it in range(n_iter):
            if time.time() > deadline - 5:
                break
                
            # Flip a random bit
            idx = np.random.randint(N)
            f[idx] *= -1
            
            new_val = _autoconvolution_max_abs(f)
            delta = new_val - current_val
            
            if delta < 0 or (T > T_end and np.random.random() < np.exp(-delta / T)):
                current_val = new_val
                if new_val < best_seed_val:
                    best_seed_val = new_val
                    best_seed_f = f.copy()
                    if new_val < best_val:
                        best_val = new_val
                        best_f = best_seed_f.copy()
            else:
                f[idx] *= -1 # Revert
                
            T *= alpha
            
        # Final local polish on the best seed found
        if best_seed_f is not None and time.time() < deadline - 5:
            improved = True
            max_sweeps = 5
            while improved and time.time() < deadline - 5:
                improved = False
                for idx in range(N):
                    if time.time() > deadline - 5:
                        break
                    old_val = best_seed_f[idx]
                    best_seed_f[idx] = -old_val
                    s = _autoconvolution_max_abs(best_seed_f)
                    if s < best_seed_val - 1e-12:
                        best_seed_val = s
                        improved = True
                        if s < best_val:
                            best_val = s
                            best_f = best_seed_f.copy()
                    else:
                        best_seed_f[idx] = old_val

    if best_f is None:
        return np.ones(N)
        
    return best_f
