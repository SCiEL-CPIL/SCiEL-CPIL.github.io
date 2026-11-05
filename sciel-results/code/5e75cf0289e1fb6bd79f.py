import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    N = 24
    M = 2**15  # FFT size for DFT grid
    sqrt_N = np.sqrt(N)

    # Precompute DFT matrix D where F = D @ c
    k = np.arange(M)
    n = np.arange(N)
    D = np.exp(-2j * np.pi * (k[:, None] * n[None, :] / M))

    def get_score(F):
        return np.max(np.abs(F)) / sqrt_N

    def flip_bit_F(F, c, idx):
        return F - 2.0 * c[idx] * D[:, idx]

    def flip_two_bits_F(F, c, idx1, idx2):
        return F - 2.0 * (c[idx1] * D[:, idx1] + c[idx2] * D[:, idx2])

    def flip_three_bits_F(F, c, idx1, idx2, idx3):
        return F - 2.0 * (c[idx1] * D[:, idx1] + c[idx2] * D[:, idx2] + c[idx3] * D[:, idx3])

    # Smooth surrogate: soft-max of |F| using log-sum-exp
    def soft_max_score(F, beta=80.0):
        abs_F = np.abs(F)
        max_val = np.max(abs_F)
        shifted = abs_F - max_val
        # Clip to avoid underflow in exp
        shifted_clipped = np.clip(shifted, -50.0, 0.0)
        log_sum = np.log(np.sum(np.exp(beta * shifted_clipped))) / beta
        return (max_val + log_sum) / sqrt_N

    def refine_peak(coeffs, theta0):
        """Golden section refinement around theta0 to find local max of |P(e^{i*theta})|."""
        def f(theta):
            z = np.exp(1j * theta)
            val = 0.0
            for kk in range(N - 1, -1, -1):
                val = val * z + coeffs[kk]
            return abs(val)
        
        delta = 2.0 / M
        a = theta0 - delta
        b = theta0 + delta
        gr = (np.sqrt(5) - 1) / 2
        
        x1 = b - gr * (b - a)
        x2 = a + gr * (b - a)
        f1 = f(x1)
        f2 = f(x2)
        
        for _ in range(60):
            if f1 < f2:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + gr * (b - a)
                f2 = f(x2)
            else:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - gr * (b - a)
                f1 = f(x1)
        
        return max(f1, f2)

    def get_score_refined(coeffs):
        """Accurate sup norm with golden-section peak refinement."""
        c_padded = np.zeros(M, dtype=float)
        c_padded[:N] = coeffs
        F = np.fft.fft(c_padded)
        absF = np.abs(F)
        top_k = 8
        idx = np.argsort(absF)[-top_k:]
        max_val = 0.0
        for i in idx:
            theta = 2.0 * np.pi * i / M
            val = refine_peak(coeffs, theta)
            if val > max_val:
                max_val = val
        return max_val / sqrt_N

    rng = np.random.RandomState(42)

    best_score = float('inf')
    best_c = None

    # Known good starting points
    # Golay sequence of length 24
    golay_24 = np.array([1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1], dtype=float)
    
    # Additional known good candidates from literature
    candidates = [
        golay_24.copy(),
        np.array([1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1], dtype=float),
        np.array([1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1], dtype=float),
        np.array([1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1], dtype=float),
        np.array([1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1], dtype=float),
        np.ones(N),
        np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]),
    ]
    
    # Add random starts
    for _ in range(8):
        candidates.append(rng.choice([-1.0, 1.0], N))

    for init_c in candidates:
        if time.time() > deadline - 15.0:
            break
        
        c = init_c.copy()
        F = D @ c
        current_score = get_score(F)
        
        if current_score < best_score:
            best_score = current_score
            best_c = c.copy()
        
        # Simulated Annealing with smooth surrogate
        T = 0.08
        alpha = 0.9997
        T_min = 1e-7
        
        current_smooth = soft_max_score(F)
        
        while time.time() < deadline - 20.0 and T > T_min:
            # Perturb: 1, 2, or 3 flips
            r = rng.rand()
            if r < 0.55:
                # 1 flip
                idx = rng.randint(N)
                F_new = flip_bit_F(F, c, idx)
                c_new = c.copy()
                c_new[idx] *= -1
            elif r < 0.90:
                # 2 flip
                idx1, idx2 = rng.choice(N, 2, replace=False)
                F_new = flip_two_bits_F(F, c, idx1, idx2)
                c_new = c.copy()
                c_new[idx1] *= -1
                c_new[idx2] *= -1
            else:
                # 3 flip
                idx1, idx2, idx3 = rng.choice(N, 3, replace=False)
                F_new = flip_three_bits_F(F, c, idx1, idx2, idx3)
                c_new = c.copy()
                c_new[idx1] *= -1
                c_new[idx2] *= -1
                c_new[idx3] *= -1
            
            new_smooth = soft_max_score(F_new)
            delta_smooth = new_smooth - current_smooth
            
            if delta_smooth < 0 or rng.rand() < np.exp(-delta_smooth / T):
                F = F_new
                c = c_new
                current_smooth = new_smooth
                current_score = get_score(F)
                
                if current_score < best_score:
                    best_score = current_score
                    best_c = c.copy()
            
            T *= alpha

    # Final Local Search on best_c
    if best_c is not None:
        c = best_c.copy()
        F = D @ c
        current_score = get_score(F)
        
        # 1-flip local search
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            best_delta = 0
            best_idx = -1
            best_F_candidate = None
            for idx in range(N):
                F_candidate = flip_bit_F(F, c, idx)
                new_score = get_score(F_candidate)
                delta = new_score - current_score
                if delta < best_delta - 1e-14:
                    best_delta = delta
                    best_idx = idx
                    best_F_candidate = F_candidate
            
            if best_idx != -1:
                c[best_idx] *= -1
                F = best_F_candidate
                current_score += best_delta
                if current_score < best_score:
                    best_score = current_score
                    best_c = c.copy()
                improved = True
        
        # 2-flip local search with multiple passes
        for _ in range(3):
            if time.time() > deadline - 8.0:
                break
            improved = True
            while improved and time.time() < deadline - 8.0:
                improved = False
                best_delta = 0
                best_idx1 = -1
                best_idx2 = -1
                best_F_candidate = None
                for n1 in range(N):
                    for n2 in range(n1+1, N):
                        F_candidate = flip_two_bits_F(F, c, n1, n2)
                        new_score = get_score(F_candidate)
                        delta = new_score - current_score
                        if delta < best_delta - 1e-14:
                            best_delta = delta
                            best_idx1 = n1
                            best_idx2 = n2
                            best_F_candidate = F_candidate
                
                if best_idx1 != -1:
                    c[best_idx1] *= -1
                    c[best_idx2] *= -1
                    F = best_F_candidate
                    current_score += best_delta
                    if current_score < best_score:
                        best_score = current_score
                        best_c = c.copy()
                    improved = True
                
                # Re-run 1-flip after 2-flip
                if time.time() < deadline - 8.0:
                    improved2 = True
                    while improved2 and time.time() < deadline - 8.0:
                        improved2 = False
                        best_delta2 = 0
                        best_idx2 = -1
                        best_F_candidate2 = None
                        for idx in range(N):
                            F_candidate = flip_bit_F(F, c, idx)
                            new_score = get_score(F_candidate)
                            delta = new_score - current_score
                            if delta < best_delta2 - 1e-14:
                                best_delta2 = delta
                                best_idx2 = idx
                                best_F_candidate2 = F_candidate
                        
                        if best_idx2 != -1:
                            c[best_idx2] *= -1
                            F = best_F_candidate2
                            current_score += best_delta2
                            if current_score < best_score:
                                best_score = current_score
                                best_c = c.copy()
                            improved2 = True

    # Final high-precision evaluation with golden-section refinement
    if best_c is not None:
        refined_score = get_score_refined(best_c)
        if refined_score < best_score:
            best_score = refined_score

    if best_c is None:
        return [1]*24
        
    return [int(x) for x in best_c]
