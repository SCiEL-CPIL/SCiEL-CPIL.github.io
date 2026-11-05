"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _eval_norm(coeffs):
    """
    Computes the sup norm of the polynomial P(z) = sum c_k z^k for |z|=1.
    Uses dense sampling and golden section refinement for accuracy.
    Returns max |P(z)|.
    """
    n = len(coeffs)
    # Dense sampling
    N = 4096
    z = np.exp(2j * np.pi * np.arange(N) / N)
    # Evaluate polynomial
    # coeffs shape (n,). 
    # Use matrix multiplication for speed: P = coeffs @ powers
    # powers[k, j] = z_j^k
    # Since n is small (24), we can compute powers efficiently
    # z_j = exp(i * theta_j)
    # z_j^k = exp(i * k * theta_j)
    # We can use FFT? No, direct is fine for N=4096, n=24.
    # Let's use a faster method:
    # P(z_j) = sum_{k=0}^{n-1} c_k z_j^k
    # This is a polynomial evaluation.
    
    # Vectorized evaluation:
    # Create matrix of powers
    # k = np.arange(n)[:, None]
    # theta = np.arange(N)[None, :] * 2 * np.pi / N
    # powers = np.exp(1j * k * theta)
    # P = powers @ coeffs
    
    # Even faster: use FFT if we pad coeffs?
    # No, direct is probably fast enough for 4096 points.
    k = np.arange(n)
    theta = 2 * np.pi * np.arange(N) / N
    # powers shape (N, n)
    # powers[j, k] = exp(i * j * 2pi/N * k)
    # This is a Vandermonde matrix.
    # We can compute P using:
    # P_j = sum_k c_k (z_j)^k
    # Let's just use np.vander or similar.
    # np.vander(z, n, increasing=True) gives rows as z^0, z^1, ..., z^{n-1}
    # So P = vander(z, n, increasing=True) @ coeffs
    v = np.vander(z, n, increasing=True)
    P = v @ coeffs
    mags = np.abs(P)
    
    # Find peaks
    # Get indices of top K peaks
    K = 20
    idx = np.argsort(mags)[-K:][::-1]
    
    max_val = 0.0
    for i in idx:
        # Golden section refinement around theta_i
        # theta_i = 2 * pi * i / N
        # Search in [theta_i - 2pi/N, theta_i + 2pi/N]
        # Actually, the peak might be slightly off.
        # Let's refine in a small window.
        # Window size: maybe 2 * 2pi/N
        w = 2 * np.pi / N
        a = theta[i] - w
        b = theta[i] + w
        
        # Golden section search for max of |P(z)|
        # f(theta) = |P(e^{i theta})|
        gr = (5 ** 0.5 - 1) / 2  # golden ratio
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        # Evaluate at c and d
        def f(theta_val):
            z_val = np.exp(1j * theta_val)
            # P(z_val) = sum c_k z_val^k
            # Use dot product
            # z_powers = z_val ** np.arange(n)
            # return np.abs(np.dot(coeffs, z_powers))
            # Faster:
            # Use recurrence?
            # Or just compute directly
            # Since n is small, direct is fine.
            # But we call this many times.
            # Let's precompute coeffs as complex? No, they are real.
            # z_val ** k
            # We can use Horner's method?
            # P(z) = c_0 + z(c_1 + z(c_2 + ...))
            # This is efficient.
            p_val = 0.0
            for coeff in reversed(coeffs):
                p_val = p_val * z_val + coeff
            return np.abs(p_val)
        
        fc = f(c)
        fd = f(d)
        
        # Iterate
        for _ in range(50):
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

def _score(coeffs):
    """
    Computes the score: max |P(z)| / sqrt(24)
    """
    n = len(coeffs)
    norm = _eval_norm(coeffs)
    return norm / np.sqrt(n)

def construct():
    """
    Finds the optimal 24 coefficients in {+1, -1} minimizing the score.
    Uses Simulated Annealing with a good initial guess.
    """
    N = 24
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, leaving 10s buffer
    
    # Initial guess: Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    coeffs = np.array(p[:N], dtype=float)
    
    best_coeffs = coeffs.copy()
    best_score = _score(best_coeffs)
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.99
    min_T = 1e-6
    
    # Precompute some things for speed?
    # _score is the bottleneck.
    
    # To speed up _score, we can cache or use a faster evaluation.
    # Let's try to optimize _eval_norm.
    
    # Instead of calling _eval_norm which does vander (expensive),
    # we can use FFT for the dense part?
    # P(z_j) = sum c_k z_j^k
    # If we pad coeffs to length N_fft, we can use IFFT?
    # No, this is polynomial evaluation, not convolution.
    # However, for roots of unity, it's related to DFT.
    # P(z_j) where z_j = exp(2pi i j / M)
    # This is the M-point DFT of the sequence c_0, ..., c_{N-1}, 0, ..., 0
    # if M >= N.
    # So we can use FFT!
    
    def _eval_norm_fast(coeffs):
        n = len(coeffs)
        M = 4096
        # Pad coeffs to length M
        c_padded = np.zeros(M, dtype=complex)
        c_padded[:n] = coeffs
        # IFFT gives values at roots of unity?
        # DFT: X_k = sum x_n e^{-2pi i n k / M}
        # IFFT: x_n = 1/M sum X_k e^{2pi i n k / M}
        # We want P(z_j) = sum c_k z_j^k = sum c_k e^{2pi i j k / M}
        # This is M * IFFT(c_padded)_j
        P = M * np.fft.ifft(c_padded)
        mags = np.abs(P)
        
        # Refine top peaks
        K = 20
        idx = np.argsort(mags)[-K:][::-1]
        
        max_val = 0.0
        for i in idx:
            w = 2 * np.pi / M
            a = 2 * np.pi * i / M - w
            b = 2 * np.pi * i / M + w
            
            def f(theta_val):
                z_val = np.exp(1j * theta_val)
                # Horner
                p_val = 0.0
                for coeff in reversed(coeffs):
                    p_val = p_val * z_val + coeff
                return np.abs(p_val)
            
            gr = (5 ** 0.5 - 1) / 2
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc = f(c)
            fd = f(d)
            
            for _ in range(30):  # Fewer iterations for speed
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
    
    def _score_fast(coeffs):
        norm = _eval_norm_fast(coeffs)
        return norm / np.sqrt(N)
    
    # Re-initialize best score with fast evaluator
    best_score = _score_fast(best_coeffs)
    
    # Local search first (steepest descent)
    improved = True
    while improved:
        improved = False
        for i in range(N):
            # Flip bit i
            coeffs[i] *= -1
            new_score = _score_fast(coeffs)
            if new_score < best_score:
                best_score = new_score
                best_coeffs = coeffs.copy()
                improved = True
            else:
                # Revert
                coeffs[i] *= -1
                # If we improved, coeffs is already flipped.
                # If not, we reverted.
                # Wait, if we improved, we set best_coeffs, but coeffs is flipped.
                # We should keep the flip.
                # So if new_score < best_score, we keep the flip.
                # Else we revert.
                # The logic above:
                # coeffs[i] *= -1
                # if new_score < best_score:
                #    ...
                # else:
                #    coeffs[i] *= -1
                # This is correct.
    
    # Simulated Annealing
    T = 1.0
    while T > min_T and time.time() < deadline:
        # Pick random index
        i = np.random.randint(N)
        coeffs[i] *= -1
        new_score = _score_fast(coeffs)
        delta = new_score - best_score
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            best_score = new_score
            best_coeffs = coeffs.copy()
        else:
            coeffs[i] *= -1
        
        T *= cooling_rate
    
    # Final local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            coeffs[i] *= -1
            new_score = _score_fast(coeffs)
            if new_score < best_score:
                best_score = new_score
                best_coeffs = coeffs.copy()
                improved = True
            else:
                coeffs[i] *= -1
    
    # Return list of ints
    return [int(c) for c in best_coeffs]

# EVOLVE-BLOCK-END
