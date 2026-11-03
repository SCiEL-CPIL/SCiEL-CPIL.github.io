"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard
DEADLINE = 150.0
START_TIME = time.time()

def time_remaining():
    return DEADLINE - (time.time() - START_TIME)

def compute_overlap(h):
    """
    Compute the maximum overlap of h with shifted 1-h.
    For a circular shift by k, overlap = sum_i h[i] * (1 - h[(i+k) % N])
    We want to minimize the max over k.
    """
    one_minus_h = 1.0 - h
    # Compute all circular cross-correlations
    # overlap[k] = sum_i h[i] * one_minus_h[(i+k) % N]
    # This is the circular cross-correlation of h and one_minus_h
    # Using FFT for efficiency
    H = np.fft.fft(h)
    O = np.fft.fft(one_minus_h)
    # Cross-correlation: ifft(fft(h) * conj(fft(one_minus_h)))
    # But we need sum_i h[i] * one_minus_h[(i+k)%N]
    # This is the circular correlation.
    # corr[k] = sum_i h[i] * one_minus_h[(i+k)%N]
    # In frequency domain: IFFT(FFT(h) * conj(FFT(one_minus_h))) gives sum_i h[i+k] * one_minus_h[i]
    # Let's be careful.
    # Let a = h, b = one_minus_h.
    # We want c[k] = sum_i a[i] * b[(i+k) % N]
    # This is the cross-correlation of a and b.
    # np.correlate(a, b, 'full') is not circular.
    # Using FFT:
    # C[k] = IFFT( A * conj(B) )[k] where A=FFT(a), B=FFT(b)
    # Let's verify: IFFT(A * conj(B))[k] = 1/N sum_m A[m] conj(B[m]) exp(2*pi*i*m*k/N)
    # A[m] = sum_j a[j] exp(-2*pi*i*m*j/N)
    # conj(B[m]) = sum_l conj(b[l]) exp(2*pi*i*m*l/N) = sum_l b[l] exp(2*pi*i*m*l/N) (since b is real)
    # So product = sum_j sum_l a[j] b[l] exp(-2*pi*i*m*j/N) exp(2*pi*i*m*l/N)
    # = sum_j sum_l a[j] b[l] exp(2*pi*i*m*(l-j)/N)
    # IFFT: 1/N sum_m [sum_j sum_l a[j] b[l] exp(2*pi*i*m*(l-j)/N)] exp(2*pi*i*m*k/N)
    # = 1/N sum_j sum_l a[j] b[l] [sum_m exp(2*pi*i*m*(l-j+k)/N)]
    # The inner sum is N if (l-j+k) % N == 0, else 0.
    # So l - j + k ≡ 0 (mod N) => l ≡ j - k (mod N)
    # So result = sum_j a[j] b[(j-k) % N]
    # We want sum_i h[i] * b[(i+k)%N]. Let i=j, then we want sum_j a[j] b[(j+k)%N].
    # The formula gives sum_j a[j] b[(j-k)%N].
    # So if we compute C[k] = IFFT(A * conj(B))[k], we get sum_j a[j] b[(j-k)%N].
    # We want D[k] = sum_j a[j] b[(j+k)%N].
    # Note D[k] = C[-k] = C[(N-k)%N].
    # So we can compute C and then reverse it (with index 0 fixed).
    
    A = np.fft.fft(h)
    B = np.fft.fft(one_minus_h)
    C = np.fft.ifft(A * np.conj(B)).real
    # C[k] = sum_j h[j] * (1-h)[(j-k)%N]
    # We want D[k] = sum_j h[j] * (1-h)[(j+k)%N] = C[-k mod N]
    D = C[np.mod(-np.arange(N), N)]
    # D[k] is the overlap for shift k.
    # Note: D[0] = sum h[i](1-h[i]) = sum h[i] - sum h[i]^2 = N/2 - sum h[i]^2
    return np.max(D)

def smooth_max(x, alpha=100.0):
    """Soft-max approximation using log-sum-exp."""
    # To avoid overflow, shift by max
    x_shifted = x - np.max(x)
    return np.max(x) + np.log(np.sum(np.exp(alpha * x_shifted))) / alpha

def objective_spectral(coeffs):
    """
    Objective function for spectral coefficients.
    coeffs[0] is DC (fixed to ensure sum=N/2, so we don't optimize it or fix it).
    Actually, let's fix coeffs[0] = N/2.
    The other coefficients are real and imaginary parts of DFT modes.
    We use a small number of low-frequency modes.
    """
    # coeffs: [a1_r, a1_i, a2_r, a2_i, ..., am_r, am_i]
    m = len(coeffs) // 2
    # Reconstruct h
    h = np.zeros(N)
    # DC component
    h += N / 2.0 / N  # This is 0.5, so sum is N/2
    # Wait, if we define h as the inverse DFT of the coefficients...
    # Let's use a simpler parameterization:
    # h[i] = 0.5 + sum_{k=1}^m (A_k cos(2*pi*k*i/N + phi_k))
    # This ensures h is real and sum is N/2.
    # Coeffs: [A_1, phi_1, A_2, phi_2, ...]
    # But A_k can be negative, so let's use [r_1, s_1, r_2, s_2, ...]
    # h[i] = 0.5 + sum_{k=1}^m (r_k cos(2*pi*k*i/N) + s_k sin(2*pi*k*i/N))
    
    h = np.full(N, 0.5)
    for k in range(1, m + 1):
        r_k = coeffs[2*(k-1)]
        s_k = coeffs[2*(k-1)+1]
        h += r_k * np.cos(2 * np.pi * k * np.arange(N) / N)
        h += s_k * np.sin(2 * np.pi * k * np.arange(N) / N)
    
    # Clip to [0, 1]
    h = np.clip(h, 0.0, 1.0)
    
    # Compute overlap
    overlap = compute_overlap(h)
    return overlap

def objective_spectral_smooth(coeffs, alpha=50.0):
    """
    Smooth version of the objective for better gradient-based optimization.
    """
    m = len(coeffs) // 2
    h = np.full(N, 0.5)
    for k in range(1, m + 1):
        r_k = coeffs[2*(k-1)]
        s_k = coeffs[2*(k-1)+1]
        h += r_k * np.cos(2 * np.pi * k * np.arange(N) / N)
        h += s_k * np.sin(2 * np.pi * k * np.arange(N) / N)
    
    # Clip to [0, 1]
    h = np.clip(h, 0.0, 1.0)
    
    # Compute overlaps
    one_minus_h = 1.0 - h
    A = np.fft.fft(h)
    B = np.fft.fft(one_minus_h)
    C = np.fft.ifft(A * np.conj(B)).real
    D = C[np.mod(-np.arange(N), N)]
    
    return smooth_max(D, alpha)

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Number of spectral modes to use
    m = 5  # Start with 5 modes (10 parameters)
    dim = 2 * m
    
    # Initial guess: all zeros (h = 0.5)
    x0 = np.zeros(dim)
    
    # Best result so far
    best_h = np.full(N, 0.5)
    best_score = compute_overlap(best_h)  # 0.5 * 0.5 * N / N = 0.25? No, max overlap.
    # For h=0.5, overlap[k] = sum 0.5 * 0.5 = N * 0.25 = 25. Max is 25.
    # Wait, the task says "minimize the maximum overlap".
    # Let's check the scale. The evaluator measures the raw value.
    # "score is the upper bound on C5"
    # "seed/value"
    # If value is 25, and seed is 25, score is 1.0.
    
    # Try a few different initializations and optimization strategies
    strategies = [
        ("Nelder-Mead", objective_spectral),
        ("Powell", objective_spectral),
        ("BFGS", objective_spectral_smooth),
    ]
    
    best_overall = best_score
    best_h_overall = best_h.copy()
    
    for name, obj_func in strategies:
        if time_remaining() < 10:
            break
            
        try:
            if name == "BFGS":
                # BFGS requires gradient, but we can use it with a smooth objective
                # and hope it works, or use a finite difference gradient.
                # Let's just use Nelder-Mead for simplicity and robustness.
                result = minimize(obj_func, x0, method=name, 
                                  options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            else:
                result = minimize(obj_func, x0, method=name, 
                                  options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if result.success or result.fun < best_overall:
                x_opt = result.x
                # Reconstruct h
                h_opt = np.full(N, 0.5)
                for k in range(1, m + 1):
                    r_k = x_opt[2*(k-1)]
                    s_k = x_opt[2*(k-1)+1]
                    h_opt += r_k * np.cos(2 * np.pi * k * np.arange(N) / N)
                    h_opt += s_k * np.sin(2 * np.pi * k * np.arange(N) / N)
                h_opt = np.clip(h_opt, 0.0, 1.0)
                
                score = compute_overlap(h_opt)
                if score < best_overall:
                    best_overall = score
                    best_h_overall = h_opt.copy()
                    
        except Exception as e:
            pass
    
    # If we have time, try a full-space local optimization starting from best_h_overall
    if time_remaining() > 20:
        try:
            # Define objective for full space
            def obj_full(h):
                return compute_overlap(h)
            
            # Constraint: 0 <= h <= 1, sum(h) = N/2
            # This is tricky for SLSQP. Let's just use the spectral result.
            pass
        except:
            pass
    
    return best_h_overall

# EVOLVE-BLOCK-END
