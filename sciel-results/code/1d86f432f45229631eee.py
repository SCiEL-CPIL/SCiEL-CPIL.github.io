import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    N = 24
    sqrt_N = np.sqrt(N)
    
    # Precompute DFT matrix for fast evaluation
    # Use a fine grid to resolve peaks accurately
    M = 2 ** 15  # 32768 points
    k = np.arange(M)
    n_idx = np.arange(N)
    D = np.exp(2j * np.pi * np.outer(k, n_idx) / M)
    
    def get_score(F):
        return float(np.max(np.abs(F))) / sqrt_N
    
    def flip_bit_F(F, c, idx):
        return F - 2.0 * c[idx] * D[:, idx]
    
    def flip_two_bits_F(F, c, idx1, idx2):
        return F - 2.0 * (c[idx1] * D[:, idx1] + c[idx2] * D[:, idx2])
    
    def local_search_1flip(current_c, current_f, current_max):
        """Thorough 1-flip local search until no improvement."""
        improved = True
        while improved:
            improved = False
            best_flip = -1
            best_max_val = current_max
            for n in range(N):
                f_new = current_f - 2.0 * current_c[n] * D[:, n]
                max_val = np.max(np.abs(f_new))
                if max_val < best_max_val - 1e-12:
                    best_max_val = max_val
                    best_flip = n
            if best_flip != -1:
                current_c[best_flip] *= -1.0
                current_f = D @ current_c
                current_max = best_max_val
                improved = True
        return current_c, current_f, current_max
    
    def local_search_2flip(current_c, current_f, current_max):
        """Try 2-flips to escape local minima."""
        best_flip_2 = (-1, -1)
        best_max_2 = current_max
        for n1 in range(N):
            for n2 in range(n1 + 1, N):
                f_new = current_f - 2.0 * (current_c[n1] * D[:, n1] + current_c[n2] * D[:, n2])
                max_val = np.max(np.abs(f_new))
                if max_val < best_max_2 - 1e-12:
                    best_max_2 = max_val
                    best_flip_2 = (n1, n2)
        if best_flip_2 != (-1, -1):
            n1, n2 = best_flip_2
            current_c[n1] *= -1.0
            current_c[n2] *= -1.0
            current_f = D @ current_c
            current_max = best_max_2
        return current_c, current_f, current_max
    
    def spectral_init(taper_type='rect'):
        """Initialize coefficients from inverse FFT of a target waveform."""
        # Create a target spectrum that is flat or has a specific shape
        if taper_type == 'rect':
            # Flat spectrum -> rectangular in frequency -> sinc-like in time
            target_spec = np.ones(M, dtype=complex)
        elif taper_type == 'cosine':
            # Cosine-tapered spectrum
            t = np.linspace(-1, 1, M)
            target_spec = 0.5 * (1 + np.cos(np.pi * t))
            target_spec[:M//4] = 0
            target_spec[3*M//4:] = 0
        else:
            # Gaussian-tapered
            t = np.linspace(-1, 1, M)
            target_spec = np.exp(-2 * t**2)
            target_spec[:M//4] = 0
            target_spec[3*M//4:] = 0
        
        # Take inverse FFT to get time-domain coefficients
        time_domain = np.fft.ifft(target_spec)
        # Keep only the first N coefficients
        coeffs = np.real(time_domain[:N])
        
        # Normalize and threshold to ±1
        if np.max(np.abs(coeffs)) > 0:
            coeffs = coeffs / np.max(np.abs(coeffs))
        # Threshold: positive -> 1, negative -> -1
        c = np.where(coeffs >= 0, 1.0, -1.0)
        return c
    
    def golay_rudin_shapiro_init():
        """Generate Golay-Rudin-Shapiro sequence."""
        p, q = [1.0], [1.0]
        while len(p) < N:
            p, q = p + q, p + [-c for c in q]
        return np.array(p[:N], dtype=float)
    
    best_score = float('inf')
    best_c = None
    best_F = None
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Golay-Rudin-Shapiro
    candidates.append(golay_rudin_shapiro_init())
    
    # 2. Spectral initializations with different tapers
    candidates.append(spectral_init('rect'))
    candidates.append(spectral_init('cosine'))
    
    # 3. Random sequences
    rng = np.random.RandomState(42)
    for _ in range(3):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Process each candidate
    for c in candidates:
        if time.time() > deadline:
            break
        
        c_arr = c.copy()
        f_arr = D @ c_arr
        max_val = np.max(np.abs(f_arr))
        
        # Do 1-flip local search
        c_arr, f_arr, max_val = local_search_1flip(c_arr, f_arr, max_val)
        
        # Try 2-flip escapes, then re-do 1-flip
        for _ in range(8):
            if time.time() > deadline:
                break
            c_arr, f_arr, max_val = local_search_2flip(c_arr, f_arr, max_val)
            c_arr, f_arr, max_val = local_search_1flip(c_arr, f_arr, max_val)
        
        score = max_val / sqrt_N
        if score < best_score:
            best_score = score
            best_c = c_arr.copy()
            best_F = f_arr.copy()
    
    # Now do iterated local search from best_c with random perturbations
    current_c = best_c.copy()
    current_f = best_F.copy()
    current_max = np.max(np.abs(current_f))
    
    while time.time() < deadline - 5:
        # Try 1-flip
        best_flip_1 = -1
        best_max_1 = current_max
        for n in range(N):
            f_new = current_f - 2.0 * current_c[n] * D[:, n]
            max_val = np.max(np.abs(f_new))
            if max_val < best_max_1 - 1e-12:
                best_max_1 = max_val
                best_flip_1 = n
        
        if best_flip_1 != -1:
            current_c[best_flip_1] *= -1.0
            current_f = D @ current_c
            current_max = best_max_1
            if current_max / sqrt_N < best_score:
                best_score = current_max / sqrt_N
                best_c = current_c.copy()
                best_F = current_f.copy()
            continue
        
        # Try 2-flip
        best_flip_2 = (-1, -1)
        best_max_2 = current_max
        for n1 in range(N):
            for n2 in range(n1 + 1, N):
                f_new = current_f - 2.0 * (current_c[n1] * D[:, n1] + current_c[n2] * D[:, n2])
                max_val = np.max(np.abs(f_new))
                if max_val < best_max_2 - 1e-12:
                    best_max_2 = max_val
                    best_flip_2 = (n1, n2)
        
        if best_flip_2 != (-1, -1):
            n1, n2 = best_flip_2
            current_c[n1] *= -1.0
            current_c[n2] *= -1.0
            current_f = D @ current_c
            current_max = best_max_2
            if current_max / sqrt_N < best_score:
                best_score = current_max / sqrt_N
                best_c = current_c.copy()
                best_F = current_f.copy()
            continue
        
        # Stuck: random restart by flipping 2-3 random bits
        n_flips = rng.choice([2, 3], p=[0.6, 0.4])
        indices = rng.choice(N, n_flips, replace=False)
        current_c[indices] *= -1.0
        current_f = D @ current_c
        current_max = np.max(np.abs(current_f))
        
        # Do a quick 1-flip local search after restart
        current_c, current_f, current_max = local_search_1flip(current_c, current_f, current_max)
        if current_max / sqrt_N < best_score:
            best_score = current_max / sqrt_N
            best_c = current_c.copy()
            best_F = current_f.copy()
    
    # Final high-precision polish: if time allows, do one more round of 1-flip + 2-flip
    if time.time() < deadline - 2:
        current_c = best_c.copy()
        current_f = best_F.copy()
        current_max = np.max(np.abs(current_f))
        c_arr, f_arr, max_val = local_search_1flip(current_c, current_f, current_max)
        for _ in range(3):
            if time.time() > deadline - 1:
                break
            c_arr, f_arr, max_val = local_search_2flip(c_arr, f_arr, max_val)
            c_arr, f_arr, max_val = local_search_1flip(c_arr, f_arr, max_val)
        if max_val / sqrt_N < best_score:
            best_score = max_val / sqrt_N
            best_c = c_arr.copy()
    
    return [int(x) for x in best_c]
