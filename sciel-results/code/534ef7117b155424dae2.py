import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import time

N = 600
NFFT = 2048  # next power of 2 >= 2*N - 1 = 1199

def _fft_convolve(h):
    """FFT-based autoconvolution for speed."""
    H = fft(h, n=NFFT)
    conv = np.real(ifft(H * H, n=NFFT))[:2 * N - 1]
    return conv

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    total = np.sum(h)
    if total == 0:
        return 0.0
    conv = _fft_convolve(h)
    return np.max(conv) / (total * total)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s deadline, leaving 5s margin
    
    # Initial guess: all ones (this gives score = 1.0)
    h = np.ones(N)
    current_score = compute_autoconv_max(h)
    best_h = h.copy()
    best_score = current_score
    
    # Soft-max objective for smooth optimization with adaptive temperature (FFT-accelerated)
    def soft_objective(x, T=0.01):
        h = np.abs(x)
        conv = _fft_convolve(h)
        total = np.sum(h)
        if total < 1e-10:
            return 1e10
        conv_max = np.max(conv)
        # Use log-sum-exp for numerical stability with smaller T
        shifted = (conv - conv_max) / T
        # Clip to avoid overflow
        shifted = np.clip(shifted, -100, 100)
        soft_max = conv_max + T * np.log(np.sum(np.exp(shifted)))
        return soft_max / (total * total)
    
    # Try multiple strategies with different initial conditions
    np.random.seed(42)
    
    # More diverse restarts (increased from 12 to 20 due to FFT speedup)
    for restart in range(20):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            # Uniform
            h0 = np.ones(N)
        elif restart == 1:
            # Slight cosine perturbation
            h0 = 1.0 + 0.15 * np.cos(2 * np.pi * np.linspace(0, 1, N, endpoint=False))
            h0 = np.maximum(h0, 0.01)
        elif restart == 2:
            # Slight sine perturbation
            h0 = 1.0 + 0.15 * np.sin(2 * np.pi * np.linspace(0, 1, N, endpoint=False))
            h0 = np.maximum(h0, 0.01)
        elif restart == 3:
            # Two-frequency perturbation
            t = np.linspace(0, 1, N, endpoint=False)
            h0 = 1.0 + 0.08 * np.cos(2 * np.pi * t) + 0.08 * np.cos(4 * np.pi * t)
            h0 = np.maximum(h0, 0.01)
        elif restart == 4:
            # Random small perturbation
            h0 = np.ones(N) + 0.08 * np.random.randn(N)
            h0 = np.maximum(h0, 0.01)
        elif restart == 5:
            # Edge-enhanced
            h0 = 1.0 + 0.25 * np.abs(np.linspace(-1, 1, N))
            h0 = np.maximum(h0, 0.01)
        elif restart == 6:
            # Center-enhanced
            t = np.linspace(-1, 1, N)
            h0 = 1.0 + 0.25 * np.exp(-t**2 * 5)
            h0 = np.maximum(h0, 0.01)
        elif restart == 7:
            # Three-frequency perturbation
            t = np.linspace(0, 1, N, endpoint=False)
            h0 = 1.0 + 0.05 * np.cos(2 * np.pi * t) + 0.05 * np.cos(4 * np.pi * t) + 0.05 * np.cos(6 * np.pi * t)
            h0 = np.maximum(h0, 0.01)
        elif restart == 8:
            # Larger random perturbation
            h0 = np.ones(N) + 0.15 * np.random.randn(N)
            h0 = np.maximum(h0, 0.01)
        elif restart == 9:
            # Random with different seed
            np.random.seed(restart * 100 + 42)
            h0 = np.random.rand(N) + 0.1
            h0 = h0 / np.mean(h0)
        elif restart == 10:
            # Parabolic shape
            t = np.linspace(-1, 1, N)
            h0 = 1.0 + 0.3 * (1 - t**2)
            h0 = np.maximum(h0, 0.01)
        elif restart == 11:
            # Random with another seed
            np.random.seed(restart * 200 + 7)
            h0 = np.random.rand(N) + 0.05
            h0 = h0 / np.mean(h0)
        elif restart == 12:
            # Four-frequency perturbation
            t = np.linspace(0, 1, N, endpoint=False)
            h0 = 1.0 + 0.04 * np.cos(2 * np.pi * t) + 0.04 * np.cos(4 * np.pi * t) + \
                  0.04 * np.cos(6 * np.pi * t) + 0.04 * np.cos(8 * np.pi * t)
            h0 = np.maximum(h0, 0.01)
        elif restart == 13:
            # Gaussian mixture-like shape
            t = np.linspace(-1, 1, N)
            h0 = 0.5 + 0.5 * np.exp(-((t - 0.3)**2) * 10) + 0.5 * np.exp(-((t + 0.3)**2) * 10)
            h0 = np.maximum(h0, 0.01)
        elif restart == 14:
            # Cosine window
            t = np.linspace(0, np.pi, N)
            h0 = 0.5 * (1 + np.cos(t))
            h0 = np.maximum(h0, 0.01)
        elif restart == 15:
            # Random with yet another seed
            np.random.seed(restart * 300 + 13)
            h0 = np.random.rand(N) + 0.2
            h0 = h0 / np.mean(h0)
        elif restart == 16:
            # Quadratic phase modulation
            t = np.linspace(0, 1, N, endpoint=False)
            h0 = 1.0 + 0.1 * np.cos(2 * np.pi * (t + 0.5 * t**2))
            h0 = np.maximum(h0, 0.01)
        elif restart == 17:
            # Bump function
            t = np.linspace(-1, 1, N)
            h0 = 1.0 + 0.4 * np.exp(-t**2 * 8)
            h0 = np.maximum(h0, 0.01)
        elif restart == 18:
            # Random with larger variance
            np.random.seed(restart * 400 + 23)
            h0 = np.random.rand(N) + 0.3
            h0 = h0 / np.mean(h0)
        else:
            # Final random
            np.random.seed(restart * 500 + 37)
            h0 = np.random.rand(N) + 0.15
            h0 = h0 / np.mean(h0)
        
        h0 = np.abs(h0)
        
        # Stage 1: Coarse optimization with soft-max (higher T for exploration)
        try:
            result1 = optimize.minimize(
                lambda x: soft_objective(x, T=0.03), h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() > deadline - 15:
                break
            
            # Stage 2: Fine optimization with lower T (exploitation)
            if result1.fun < best_score * 1.02:
                result2 = optimize.minimize(
                    lambda x: soft_objective(x, T=0.005), result1.x,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                # Evaluate both stage 1 and stage 2 with exact objective
                for candidate in [result1.x, result2.x]:
                    h_new = np.abs(candidate)
                    actual_score = compute_autoconv_max(h_new)
                    if actual_score < best_score:
                        best_score = actual_score
                        best_h = h_new.copy()
                        
        except Exception:
            pass
    
    # Final polish: try to improve the best solution found with very low T (increased iterations)
    if time.time() < deadline - 5:
        try:
            for _ in range(15):  # increased from 8 to 15
                h_perturbed = best_h + 0.005 * np.random.randn(N)
                h_perturbed = np.maximum(h_perturbed, 0)
                
                result = optimize.minimize(
                    lambda x: soft_objective(x, T=0.002), h_perturbed,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}  # tighter tolerances
                )
                
                h_new = np.abs(result.x)
                actual_score = compute_autoconv_max(h_new)
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = h_new.copy()
                    
        except Exception:
            pass
    
    return list(best_h)
