import numpy as np
from scipy.optimize import differential_evolution
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Define grid
    x = np.linspace(-0.25, 0.25, N)
    dx = x[1] - x[0]
    
    # Compute autoconvolution efficiently using FFT
    def compute_score(params):
        a, b, c = params
        # f(x) = 1 + a*cos(2πx) + b*cos(4πx) + c*cos(6πx)
        f = 1.0 + a * np.cos(2 * np.pi * x) + b * np.cos(4 * np.pi * x) + c * np.cos(6 * np.pi * x)
        
        # Ensure nonnegativity - if violated, return large score
        if np.any(f < -1e-10):
            return 1e10
        
        f = np.maximum(f, 0.0)  # Clamp small negatives
        
        # Compute autoconvolution via FFT
        # Need to zero-pad to avoid circular convolution
        n_fft = 2 * N
        F = np.fft.fft(f, n=n_fft)
        conv = np.fft.ifft(F * F).real
        
        # The autoconvolution is conv * dx
        conv *= dx
        
        # Normalization: max(conv) / (sum(f)*dx)^2
        integral = np.sum(f) * dx
        if integral < 1e-10:
            return 1e10
        
        score = np.max(conv) / (integral * integral)
        return score
    
    # Try a few initial points and use differential evolution
    best_score = 1.0  # Uniform function baseline
    best_f = np.ones(N)
    
    # First try some handcrafted shapes
    # Raised cosine
    f_cos = 0.5 * (1 + np.cos(4 * np.pi * x))  # One period of cosine over [-1/4,1/4]
    # Wait, cos(4πx) has period 1/2, so on [-1/4,1/4] it goes from cos(-π)= -1 to cos(0)=1 to cos(π)=-1
    # That's not right. Let me use cos(2πx) which has period 1, so on [-1/4,1/4] it's from cos(-π/2)=0 to cos(0)=1 to cos(π/2)=0
    
    # Actually, let me just use differential evolution directly
    bounds = [(-0.9, 0.9), (-0.9, 0.9), (-0.9, 0.9)]
    
    try:
        result = differential_evolution(
            compute_score,
            bounds=bounds,
            maxiter=100,
            popsize=15,
            tol=1e-4,
            seed=42,
            polish=True
        )
        
        if result.fun < best_score:
            best_score = result.fun
            a, b, c = result.x
            best_f = 1.0 + a * np.cos(2 * np.pi * x) + b * np.cos(4 * np.pi * x) + c * np.cos(6 * np.pi * x)
            best_f = np.maximum(best_f, 0.0)
        
        elapsed = time.time() - start_time
        if elapsed < deadline - 10:
            # Try a second round with more harmonics
            def compute_score2(params):
                a, b, c, d = params
                f = 1.0 + a * np.cos(2 * np.pi * x) + b * np.cos(4 * np.pi * x) + \
                    c * np.cos(6 * np.pi * x) + d * np.cos(8 * np.pi * x)
                if np.any(f < -1e-10):
                    return 1e10
                f = np.maximum(f, 0.0)
                n_fft = 2 * N
                F = np.fft.fft(f, n=n_fft)
                conv = np.fft.ifft(F * F).real * dx
                integral = np.sum(f) * dx
                if integral < 1e-10:
                    return 1e10
                return np.max(conv) / (integral * integral)
            
            bounds2 = [(-0.8, 0.8), (-0.8, 0.8), (-0.8, 0.8), (-0.8, 0.8)]
            result2 = differential_evolution(
                compute_score2,
                bounds=bounds2,
                maxiter=80,
                popsize=12,
                tol=1e-4,
                seed=123,
                polish=True
            )
            
            if result2.fun < best_score:
                best_score = result2.fun
                a, b, c, d = result2.x
                best_f = 1.0 + a * np.cos(2 * np.pi * x) + b * np.cos(4 * np.pi * x) + \
                    c * np.cos(6 * np.pi * x) + d * np.cos(8 * np.pi * x)
                best_f = np.maximum(best_f, 0.0)
        
    except Exception:
        pass
    
    # Return the best function found
    return list(best_f)
