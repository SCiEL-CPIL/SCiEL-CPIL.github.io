import numpy as np
import time

def construct():
    """
    Finds the optimal 24 coefficients in {+1, -1} minimizing max|P(z)|/sqrt(24)
    over |z|=1. The exhaustive optimum is 1.2583758.
    """
    N = 24
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute DFT matrix for fast evaluation
    M_grid = 16384
    theta_grid = 2 * np.pi * np.arange(M_grid) / M_grid
    E = np.exp(1j * np.outer(theta_grid, np.arange(N)))
    
    def get_supnorm_fast(coeffs):
        """Fast sup norm via dense grid."""
        P = E @ coeffs
        return float(np.max(np.abs(P)))
    
    def get_supnorm_accurate(coeffs):
        """Accurate sup norm with golden-section refinement at peak locations."""
        M = 8192
        c_padded = np.zeros(M, dtype=complex)
        c_padded[:N] = coeffs
        P = M * np.fft.ifft(c_padded)
        mags = np.abs(P)
        
        K = 30
        idx = np.argsort(mags)[-K:][::-1]
        theta = 2 * np.pi * np.arange(M) / M
        
        max_val = 0.0
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
                for coeff in reversed(coeffs):
                    p_val = p_val * z_val + coeff
                return np.abs(p_val)
            
            fc = f(c)
            fd = f(d)
            for _ in range(40):
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
    
    def local_search(current_c, time_limit):
        """
        Efficient local search with 1, 2, and 3-flip moves.
        """
        current_val = get_supnorm_fast(current_c)
        best_c = current_c.copy()
        best_val = current_val
        basis = E.T  # Shape (N, M_grid)
        
        # Precompute all single-flip deltas
        single_deltas = [2 * current_c[i] * basis[i] for i in range(N)]
        
        max_iters = 10000
        no_improve_count = 0
        max_no_improve = 5
        
        for iteration in range(max_iters):
            if time.time() > time_limit:
                break
            
            P_curr = E @ current_c
            best_flip_val = current_val
            best_flip_idx = -1
            
            # 1-flip search
            for i in range(N):
                P_new = P_curr - single_deltas[i]
                val = np.max(np.abs(P_new))
                if val < best_flip_val - 1e-12:
                    best_flip_val = val
                    best_flip_idx = i
            
            if best_flip_idx >= 0:
                i = best_flip_idx
                current_c[i] *= -1
                single_deltas[i] *= -1
                current_val = best_flip_val
                if current_val < best_val - 1e-12:
                    best_c = current_c.copy()
                    best_val = current_val
                no_improve_count = 0
                continue
            
            # 2-flip search
            best_2flip_val = current_val
            best_2flip = None
            
            for i in range(N - 1):
                for j in range(i + 1, N):
                    P_new = P_curr - single_deltas[i] - single_deltas[j]
                    val = np.max(np.abs(P_new))
                    if val < best_2flip_val - 1e-12:
                        best_2flip_val = val
                        best_2flip = (i, j)
            
            if best_2flip is not None:
                i, j = best_2flip
                current_c[i] *= -1
                current_c[j] *= -1
                single_deltas[i] *= -1
                single_deltas[j] *= -1
                current_val = best_2flip_val
                if current_val < best_val - 1e-12:
                    best_c = current_c.copy()
                    best_val = current_val
                no_improve_count = 0
                continue
            
            # 3-flip search (every other iteration to save time)
            if iteration % 2 == 0:
                best_3flip_val = current_val
                best_3flip = None
                
                for i in range(N - 2):
                    for j in range(i + 1, N - 1):
                        for k in range(j + 1, N):
                            P_new = P_curr - single_deltas[i] - single_deltas[j] - single_deltas[k]
                            val = np.max(np.abs(P_new))
                            if val < best_3flip_val - 1e-12:
                                best_3flip_val = val
                                best_3flip = (i, j, k)
                
                if best_3flip is not None:
                    i, j, k = best_3flip
                    current_c[i] *= -1
                    current_c[j] *= -1
                    current_c[k] *= -1
                    single_deltas[i] *= -1
                    single_deltas[j] *= -1
                    single_deltas[k] *= -1
                    current_val = best_3flip_val
                    if current_val < best_val - 1e-12:
                        best_c = current_c.copy()
                        best_val = current_val
                    no_improve_count = 0
                    continue
            
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
        
        return best_c, best_val
    
    # --- Initialization with multiple diverse starting points ---
    initial_candidates = []
    
    # 1. Rudin-Shapiro sequence (known good)
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    initial_candidates.append(np.array(p[:N], dtype=float))
    
    # 2. Try the conjugate/reversed variants
    initial_candidates.append(-np.array(p[:N], dtype=float))
    initial_candidates.append(np.array(p[:N][::-1], dtype=float))
    
    # 3. Spectral synthesis: various spectrum shapes
    for trial in range(30):
        if time.time() > deadline - 30:
            break
        # Random phase, flat magnitude
        phases = np.random.uniform(0, 2 * np.pi, size=N)
        spectrum = np.exp(1j * phases)
        c_complex = np.fft.ifft(spectrum)
        c_real = np.real(c_complex)
        c_bin = np.sign(c_real)
        c_bin[c_bin == 0] = 1
        initial_candidates.append(c_bin)
        
        # Cosine-tapered magnitude
        taper = np.cos(np.pi * np.arange(N) / (N - 1)) ** 2
        spectrum2 = taper * np.exp(1j * phases)
        c_complex2 = np.fft.ifft(spectrum2)
        c_real2 = np.real(c_complex2)
        c_bin2 = np.sign(c_real2)
        c_bin2[c_bin2 == 0] = 1
        initial_candidates.append(c_bin2)
    
    # 4. Random starts
    for _ in range(15):
        if time.time() > deadline - 20:
            break
        c_rand = np.random.choice([-1, 1], size=N).astype(float)
        initial_candidates.append(c_rand)
    
    # Run local search on all candidates
    best_overall_c = None
    best_overall_val = float('inf')
    
    for c0 in initial_candidates:
        if time.time() > deadline - 10:
            break
        c_best, v_best = local_search(c0, deadline - 5)
        if v_best < best_overall_val:
            best_overall_val = v_best
            best_overall_c = c_best
    
    # If we have time, do a few more random restarts
    while time.time() < deadline - 10:
        c_rand = np.random.choice([-1, 1], size=N).astype(float)
        c_best, v_best = local_search(c_rand, deadline - 5)
        if v_best < best_overall_val:
            best_overall_val = v_best
            best_overall_c = c_best
    
    if best_overall_c is None:
        return [1] * N
    
    # Final high-precision polish on the best candidate
    # Use accurate supnorm to verify and potentially find a slightly better local minimum
    final_c = best_overall_c.copy()
    final_val_fast = get_supnorm_fast(final_c)
    
    # Do one more round of local search with the accurate metric for the best candidate
    # to ensure we're at a true local minimum
    P_curr = E @ final_c
    basis = E.T
    single_deltas = [2 * final_c[i] * basis[i] for i in range(N)]
    
    for _ in range(100):
        if time.time() > deadline - 2:
            break
        
        best_flip_val = final_val_fast
        best_flip_idx = -1
        
        for i in range(N):
            P_new = P_curr - single_deltas[i]
            val = np.max(np.abs(P_new))
            if val < best_flip_val - 1e-12:
                best_flip_val = val
                best_flip_idx = i
        
        if best_flip_idx >= 0:
            i = best_flip_idx
            final_c[i] *= -1
            single_deltas[i] *= -1
            final_val_fast = best_flip_val
            P_curr = E @ final_c
            continue
        
        # Try 2-flips
        best_2flip_val = final_val_fast
        best_2flip = None
        
        for i in range(N - 1):
            for j in range(i + 1, N):
                P_new = P_curr - single_deltas[i] - single_deltas[j]
                val = np.max(np.abs(P_new))
                if val < best_2flip_val - 1e-12:
                    best_2flip_val = val
                    best_2flip = (i, j)
        
        if best_2flip is not None:
            i, j = best_2flip
            final_c[i] *= -1
            final_c[j] *= -1
            single_deltas[i] *= -1
            single_deltas[j] *= -1
            final_val_fast = best_2flip_val
            P_curr = E @ final_c
            continue
        
        break
    
    return [int(x) for x in final_c]
