import numpy as np
import time

def construct():
    N = 24
    M = 8192  # Grid size for FFT
    B = 2048  # Batch size
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Precompute twiddle factors? np.fft does this internally.
    
    best_val = float('inf')
    best_coeffs = None
    top_candidates = [] # List of (val, coeffs)
    
    total_patterns = 1 << N
    
    for start in range(0, total_patterns, B):
        if time.time() > deadline - 5:
            break
            
        end = min(start + B, total_patterns)
        batch_size = end - start
        
        # Generate coefficients for this batch
        # Indices: start to end-1
        indices = np.arange(start, end)
        # Convert to binary bits
        # We want c_k = 1 if bit k is 0, -1 if bit k is 1? Or vice versa.
        # Let's say c_k = 1 - 2*bit_k.
        # bits = (indices >> np.arange(N)) & 1
        # coeffs = 1 - 2 * bits
        # Shape: (batch_size, N)
        
        powers = np.arange(N)
        bits = (indices[:, None] >> powers[None, :]) & 1
        coeffs = 1 - 2 * bits
        coeffs = coeffs.astype(np.float64)
        
        # Pad to M
        # coeffs shape (B, N)
        # Padded shape (B, M)
        padded = np.zeros((batch_size, M), dtype=np.complex128)
        padded[:, :N] = coeffs
        
        # FFT
        # np.fft.fft accepts 2D array, operates along last axis by default?
        # No, axis=-1 is default.
        spectra = np.fft.fft(padded)
        
        # Magnitude
        mags = np.abs(spectra)
        
        # Max magnitude for each polynomial
        max_mags = np.max(mags, axis=1)
        
        # Find best in batch
        local_best_idx = np.argmin(max_mags)
        local_best_val = max_mags[local_best_idx]
        
        if local_best_val < best_val:
            best_val = local_best_val
            best_coeffs = coeffs[local_best_idx].copy()
            
        # Keep top candidates for refinement
        # We can just keep the single best for now, or a few.
        # Given the large search space, the global best is likely unique or close.
        # But to be safe, let's keep a small list.
        # Actually, storing 20 candidates is cheap.
        if len(top_candidates) < 20:
             top_candidates.append((local_best_val, coeffs[local_best_idx].copy()))
        else:
             # Replace worst if better
             worst_idx = np.argmax([v for v, _ in top_candidates])
             if local_best_val < top_candidates[worst_idx][0]:
                 top_candidates[worst_idx] = (local_best_val, coeffs[local_best_idx].copy())
        
        # Optional: Check time and adjust B or M? 
        # If too slow, reduce M.
        
    # Refine top candidates
    # Define accurate sup-norm function
    def get_supnorm_accurate(c):
        # c is array of N floats
        # Sample dense grid
        M_ref = 65536
        c_padded = np.zeros(M_ref, dtype=np.complex128)
        c_padded[:N] = c
        P = np.fft.fft(c_padded)
        mags = np.abs(P)
        
        # Find peak
        # The max on the grid is a lower bound.
        # Refine around the grid peak.
        idx_max = np.argmax(mags)
        # Golden section search around theta_max
        theta_max = 2 * np.pi * idx_max / M_ref
        w = 2 * np.pi / M_ref
        a = theta_max - w
        b = theta_max + w
        
        def f(theta):
            z = np.exp(1j * theta)
            # Horner's rule for speed
            p = 0.0
            for coeff in reversed(c):
                p = p * z + coeff
            return np.abs(p)
        
        # Golden section
        gr = (5 ** 0.5 - 1) / 2
        c_pt = b - gr * (b - a)
        d_pt = a + gr * (b - a)
        fc = f(c_pt)
        fd = f(d_pt)
        
        for _ in range(60):
            if fc < fd:
                b = d_pt
                d_pt = c_pt
                fd = fc
                c_pt = b - gr * (b - a)
                fc = f(c_pt)
            else:
                a = c_pt
                c_pt = d_pt
                fc = fd
                d_pt = a + gr * (b - a)
                fd = f(d_pt)
                
        return max(fc, fd)

    final_best_val = float('inf')
    final_best_coeffs = best_coeffs
    
    for val, c in top_candidates:
        if time.time() > deadline - 2:
            break
        acc_val = get_supnorm_accurate(c)
        if acc_val < final_best_val:
            final_best_val = acc_val
            final_best_coeffs = c

    return [int(x) for x in final_best_coeffs]
