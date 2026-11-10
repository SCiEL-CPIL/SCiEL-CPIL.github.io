import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Initialize with Rudin-Shapiro sequence
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    current = np.array(p[:24], dtype=float)
    
    def evaluate(coeffs):
        # coeffs is a numpy array of +1/-1
        # Evaluate on a grid
        n = len(coeffs)
        # Use FFT for speed. FFT gives values at z = exp(2*pi*i*k/N)
        # We need sup norm over |z|=1.
        # A grid of 4096 points is usually enough for coarse location.
        N_grid = 4096
        # Pad coeffs to N_grid for FFT? No, just evaluate polynomial at grid points.
        # z_k = exp(2*pi*i*k/N_grid)
        # P(z_k) = sum c_j z_k^j
        # This is a DFT of coeffs padded to N_grid.
        padded = np.zeros(N_grid, dtype=complex)
        padded[:n] = coeffs
        fft_vals = np.fft.fft(padded)
        
        # Find max magnitude
        mags = np.abs(fft_vals)
        max_idx = np.argmax(mags)
        
        # Refine the peak around max_idx
        # The peak is between grid points max_idx-1, max_idx, max_idx+1
        # We can use golden section search on the angle.
        # Angle range: [2*pi*(max_idx-1)/N_grid, 2*pi*(max_idx+1)/N_grid]
        
        def poly_val(angle, coeffs):
            z = np.exp(1j * angle)
            # Horner's method or direct sum
            # Direct sum with numpy is fast for n=24
            powers = np.power(z, np.arange(n))
            return np.sum(coeffs * powers)
            
        def poly_mag(angle, coeffs):
            return np.abs(poly_val(angle, coeffs))
            
        # Golden section search
        gr = (np.sqrt(5) - 1) / 2
        a = 2 * np.pi * (max_idx - 1) / N_grid
        b = 2 * np.pi * (max_idx + 1) / N_grid
        # Ensure a < b
        if a > b:
            a, b = b, a
        # Handle wrap around if necessary, but for N=24 and N_grid=4096, peaks are narrow.
        
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = poly_mag(c, coeffs)
        fd = poly_mag(d, coeffs)
        
        for _ in range(50):  # 50 iterations is plenty for 1e-9 accuracy
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = poly_mag(c, coeffs)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = poly_mag(d, coeffs)
                
        peak_val = max(fc, fd)
        
        # Also check neighbors just in case
        # But usually the global max is the one we found.
        # To be safe, check the top 5 peaks from FFT.
        top_indices = np.argsort(mags)[-5:]
        max_refined = 0
        for idx in top_indices:
            a = 2 * np.pi * (idx - 1) / N_grid
            b = 2 * np.pi * (idx + 1) / N_grid
            if a > b:
                a, b = b, a
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc = poly_mag(c, coeffs)
            fd = poly_mag(d, coeffs)
            for _ in range(50):
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = poly_mag(c, coeffs)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = poly_mag(d, coeffs)
            local_peak = max(fc, fd)
            if local_peak > max_refined:
                max_refined = local_peak
                
        return max_refined / np.sqrt(24)

    current_score = evaluate(current)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(24):
            candidate = current.copy()
            candidate[i] *= -1
            cand_score = evaluate(candidate)
            if cand_score < current_score:
                current = candidate
                current_score = cand_score
                improved = True
                if time.time() > deadline:
                    break
                    
    return [int(c) for c in current]
