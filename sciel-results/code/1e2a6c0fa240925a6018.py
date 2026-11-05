"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _accurate_supnorm(coeffs):
    """
    Computes the accurate sup norm of P(z) = sum c_k z^k for |z|=1.
    Uses dense FFT sampling to find peaks, then golden-section refinement.
    """
    n = len(coeffs)
    M = 16384
    c_padded = np.zeros(M, dtype=complex)
    c_padded[:n] = coeffs
    P = M * np.fft.ifft(c_padded)
    mags = np.abs(P)
    
    K = 50
    idx = np.argsort(mags)[-K:][::-1]
    theta = 2 * np.pi * np.arange(M) / M
    
    max_val = 0.0
    coeffs_tuple = tuple(coeffs)
    
    for i in idx:
        w = 2 * np.pi / M
        a = theta[i] - w
        b = theta[i] + w
        
        gr = (5 ** 0.5 - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        def f(theta_val):
            z_val = np.exp(1j * theta_val)
            p_val = 0.0
            for coeff in reversed(coeffs_tuple):
                p_val = p_val * z_val + coeff
            return np.abs(p_val)
        
        fc = f(c)
        fd = f(d)
        
        for _ in range(60):
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
            else:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
        
        peak_val = max(fc, fd)
        if peak_val > max_val:
            max_val = peak_val
            
    return max_val

def construct():
    """
    Finds the optimal 24 coefficients in {+1, -1} minimizing the score.
    """
    N = 24
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute basis vectors for fast sup-norm estimation on a grid
    M_grid = 8192
    theta_grid = 2 * np.pi * np.arange(M_grid) / M_grid
    # Z[j, k] = e^{i * theta_grid[j] * k}
    Z = np.exp(1j * np.outer(theta_grid, np.arange(N)))
    
    def fast_supnorm(coeffs):
        """
        Fast approximation of sup norm using precomputed grid.
        """
        P_grid = Z @ coeffs
        return float(np.max(np.abs(P_grid)))
    
    def local_search_vectorized(coeffs, time_limit):
        """
        Local search using 1-flip and 2-flip moves with fast grid evaluation.
        """
        current_coeffs = coeffs.copy()
        current_val = fast_supnorm(current_coeffs)
        best_coeffs = current_coeffs.copy()
        best_val = current_val
        
        # Maintain P_grid
        P_grid = Z @ current_coeffs
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1-flip search
            best_1flip_val = current_val
            best_1flip_idx = -1
            for i in range(N):
                # Z[:, i] is the vector for coefficient i
                P_new = P_grid - 2.0 * current_coeffs[i] * Z[:, i]
                val = float(np.max(np.abs(P_new)))
                if val < best_1flip_val - 1e-12:
                    best_1flip_val = val
                    best_1flip_idx = i
            
            if best_1flip_idx != -1:
                i = best_1flip_idx
                old_c = current_coeffs[i]
                P_grid = P_grid - 2.0 * old_c * Z[:, i]
                current_coeffs[i] = -old_c
                current_val = best_1flip_val
                
                if current_val < best_val - 1e-12:
                    best_val = current_val
                    best_coeffs = current_coeffs.copy()
                improved = True
                continue
            
            # 2-flip search
            best_2flip_val = current_val
            best_2flip = None
            for i in range(N - 1):
                for j in range(i + 1, N):
                    P_new = P_grid - 2.0 * current_coeffs[i] * Z[:, i] - 2.0 * current_coeffs[j] * Z[:, j]
                    val = float(np.max(np.abs(P_new)))
                    if val < best_2flip_val - 1e-12:
                        best_2flip_val = val
                        best_2flip = (i, j)
            
            if best_2flip is not None:
                i, j = best_2flip
                P_grid = P_grid - 2.0 * current_coeffs[i] * Z[:, i] - 2.0 * current_coeffs[j] * Z[:, j]
                current_coeffs[i] *= -1
                current_coeffs[j] *= -1
                current_val = best_2flip_val
                
                if current_val < best_val - 1e-12:
                    best_val = current_val
                    best_coeffs = current_coeffs.copy()
                improved = True
                continue
            
            # 3-flip search (if time allows)
            if time.time() < time_limit - 0.5:
                best_3flip_val = current_val
                best_3flip = None
                for i in range(N - 2):
                    for j in range(i + 1, N - 1):
                        for k in range(j + 1, N):
                            P_new = P_grid - 2.0 * current_coeffs[i] * Z[:, i] - 2.0 * current_coeffs[j] * Z[:, j] - 2.0 * current_coeffs[k] * Z[:, k]
                            val = float(np.max(np.abs(P_new)))
                            if val < best_3flip_val - 1e-12:
                                best_3flip_val = val
                                best_3flip = (i, j, k)
                
                if best_3flip is not None:
                    i, j, k = best_3flip
                    P_grid = P_grid - 2.0 * current_coeffs[i] * Z[:, i] - 2.0 * current_coeffs[j] * Z[:, j] - 2.0 * current_coeffs[k] * Z[:, k]
                    current_coeffs[i] *= -1
                    current_coeffs[j] *= -1
                    current_coeffs[k] *= -1
                    current_val = best_3flip_val
                    
                    if current_val < best_val - 1e-12:
                        best_val = current_val
                        best_coeffs = current_coeffs.copy()
                    improved = True
                    continue
        
        return best_coeffs, best_val

    # --- Initialization ---
    candidates = []
    
    # 1. Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    candidates.append(np.array(p[:N], dtype=float))
    
    # 2. Spectral Synthesis
    for _ in range(60):
        if time.time() > deadline - 15:
            break
        phases = np.random.uniform(0, 2 * np.pi, N)
        spectrum = np.exp(1j * phases)
        c_complex = np.fft.ifft(spectrum)
        c_real = np.real(c_complex)
        c_bin = np.sign(c_real)
        c_bin[c_bin == 0] = 1
        candidates.append(c_bin)
        
        taper = np.cos(np.pi * np.arange(N) / (N - 1)) ** 2
        spectrum2 = taper * np.exp(1j * phases)
        c_complex2 = np.fft.ifft(spectrum2)
        c_real2 = np.real(c_complex2)
        c_bin2 = np.sign(c_real2)
        c_bin2[c_bin2 == 0] = 1
        candidates.append(c_bin2)
        
        candidates.append(np.random.choice([1, -1], N))

    # --- Search ---
    best_overall_coeffs = None
    best_overall_score = float('inf')
    
    # Sort candidates by fast score to prioritize good ones?
    # No, just iterate.
    
    for cand in candidates:
        if time.time() > deadline - 5:
            break
            
        fast_score = fast_supnorm(cand)
        
        # If we are running out of time, just record
        if time.time() > deadline - 10:
            if fast_score < best_overall_score:
                best_overall_score = fast_score
                best_overall_coeffs = cand.copy()
            continue

        # Run local search
        # Time limit per candidate: 0.5s
        local_time_limit = time.time() + 0.5
        optimized_coeffs, opt_score = local_search_vectorized(cand, local_time_limit)
        
        if opt_score < best_overall_score:
            best_overall_score = opt_score
            best_overall_coeffs = optimized_coeffs.copy()

    if best_overall_coeffs is None:
        best_overall_coeffs = np.ones(N)
        
    return [int(c) for c in best_overall_coeffs]

# EVOLVE-BLOCK-END
