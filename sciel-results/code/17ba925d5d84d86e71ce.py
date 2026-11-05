import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    N = 24
    sqrtN = np.sqrt(N)
    
    # Fast sup norm evaluation using FFT
    # We need enough points to resolve the peaks. For N=24, the polynomial
    # has degree 23. The sup norm is achieved at points where the derivative
    # is zero. Using a dense grid of M points on the unit circle.
    # M should be large enough. 2^15 = 32768 points gives ~1e-4 angular resolution.
    # For accuracy ~1e-9, we'll refine peaks with golden section.
    
    M = 1 << 15  # 32768
    
    def get_supnorm(coeffs):
        """Compute max |P(z)| for |z|=1 using FFT on dense grid."""
        c_padded = np.zeros(M, dtype=float)
        c_padded[:N] = coeffs
        F = np.fft.fft(c_padded)
        return float(np.max(np.abs(F)))
    
    def refine_peak(coeffs, theta0):
        """Golden section refinement around theta0 to find local max of |P(e^{iθ})|."""
        def f(theta):
            z = np.exp(1j * theta)
            # Horner's method or direct evaluation
            # P(z) = sum c_k z^k
            val = 0.0
            for k in range(N - 1, -1, -1):
                val = val * z + coeffs[k]
            return abs(val)
        
        # Golden section search in [theta0 - delta, theta0 + delta]
        delta = 2.0 / M  # one grid spacing
        a = theta0 - delta
        b = theta0 + delta
        gr = (np.sqrt(5) - 1) / 2  # golden ratio conjugate
        
        x1 = b - gr * (b - a)
        x2 = a + gr * (b - a)
        f1 = f(x1)
        f2 = f(x2)
        
        for _ in range(50):  # ~50 iterations gives ~1e-9 accuracy
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
    
    def get_supnorm_refined(coeffs):
        """Sup norm with peak refinement for accuracy."""
        c_padded = np.zeros(M, dtype=float)
        c_padded[:N] = coeffs
        F = np.fft.fft(c_padded)
        absF = np.abs(F)
        # Find top few peaks and refine them
        # Get indices of largest values
        top_k = 5
        idx = np.argsort(absF)[-top_k:]
        max_val = 0.0
        for i in idx:
            theta = 2.0 * np.pi * i / M
            val = refine_peak(coeffs, theta)
            if val > max_val:
                max_val = val
        return max_val
    
    # Initial solution: Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:N], dtype=float)
    best_score = get_supnorm_refined(best_coeffs) / sqrtN
    
    # Simulated Annealing
    random.seed(42)
    np.random.seed(42)
    
    T_start = 0.5
    T_end = 0.001
    cooling_rate = 0.999
    
    current_coeffs = best_coeffs.copy()
    current_score = best_score
    
    while time.time() < deadline:
        T = T_start * (T_end / T_start) ** ((time.time() - start_time) / (deadline - start_time))
        if T < T_end:
            T = T_end
        
        # Random perturbation: flip 1-3 coefficients
        n_flips = random.randint(1, 3)
        new_coeffs = current_coeffs.copy()
        for _ in range(n_flips):
            idx = random.randint(0, N - 1)
            new_coeffs[idx] = -new_coeffs[idx]
        
        new_score = get_supnorm(new_coeffs) / sqrtN  # Use fast version for SA
        
        if new_score < current_score:
            current_coeffs = new_coeffs
            current_score = new_score
            if new_score < best_score:
                best_coeffs = new_coeffs.copy()
                best_score = new_score
        else:
            delta = new_score - current_score
            if random.random() < np.exp(-delta / T):
                current_coeffs = new_coeffs
                current_score = new_score
        
        # Periodically refine the best solution
        if random.random() < 0.01:
            refined = get_supnorm_refined(best_coeffs) / sqrtN
            if refined < best_score:
                best_score = refined
    
    # Final refinement of the best solution
    final_score = get_supnorm_refined(best_coeffs) / sqrtN
    if final_score < best_score:
        best_score = final_score
    
    return [int(c) for c in best_coeffs]
