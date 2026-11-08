"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 180.0
    
    def H0(x):
        return np.ones_like(x)
    
    def H4(x):
        return 16*x**4 - 48*x**2 + 12
    
    def H8(x):
        return 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
    
    def H12(x):
        return 4096*x**12 - 73728*x**10 + 460800*x**8 - 1290240*x**6 + 1574400*x**4 - 725760*x**2 + 1034880
    
    def g(x, c0, c4, c8):
        c12 = -(c0 + 12*c4 + 1680*c8) / 1034880.0
        return (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * np.exp(-x**2/2.0)
    
    def compute_C4_v1(c0, c4, c8):
        """L1 norm product"""
        x = np.linspace(-10, 10, 10000)
        dx = x[1] - x[0]
        f = g(x, c0, c4, c8)
        n = len(f)
        nfft = 1 << (n-1).bit_length()
        f_padded = np.zeros(nfft)
        f_padded[:n] = f
        F = np.fft.fft(f_padded) * dx
        L1_f = np.sum(np.abs(f)) * dx
        L1_F = np.sum(np.abs(F)) * dx
        return L1_f * L1_F
    
    def compute_C4_v2(c0, c4, c8):
        """Variance-based uncertainty"""
        x = np.linspace(-10, 10, 10000)
        dx = x[1] - x[0]
        f = g(x, c0, c4, c8)
        f2 = f**2
        norm = np.sum(f2) * dx
        if norm < 1e-15:
            return 1e10
        x_mean = np.sum(x * f2) * dx / norm
        x_var = np.sum((x - x_mean)**2 * f2) * dx / norm
        n = len(f)
        nfft = 1 << (n-1).bit_length()
        f_padded = np.zeros(nfft)
        f_padded[:n] = f
        F = np.fft.fft(f_padded) * dx
        F2 = np.abs(F)**2
        norm_F = np.sum(F2) * dx
        if norm_F < 1e-15:
            return 1e10
        k = np.fft.fftfreq(nfft, d=dx/(2*np.pi*nfft))
        k_mean = np.sum(k * F2) * dx / norm_F
        k_var = np.sum((k - k_mean)**2 * F2) * dx / norm_F
        return np.sqrt(x_var * k_var)
    
    def compute_C4_v3(c0, c4, c8):
        """L1 weighted moment product"""
        x = np.linspace(-10, 10, 10000)
        dx = x[1] - x[0]
        f = g(x, c0, c4, c8)
        abs_f = np.abs(f)
        norm_f = np.sum(abs_f) * dx
        if norm_f < 1e-15:
            return 1e10
        A_f = np.sum(np.abs(x) * abs_f) * dx / norm_f
        n = len(f)
        nfft = 1 << (n-1).bit_length()
        f_padded = np.zeros(nfft)
        f_padded[:n] = f
        F = np.fft.fft(f_padded) * dx
        abs_F = np.abs(F)
        norm_F = np.sum(abs_F) * dx
        if norm_F < 1e-15:
            return 1e10
        k = 2*np.pi*np.fft.fftfreq(nfft, d=dx)
        A_F = np.sum(np.abs(k) * abs_F) * dx / norm_F
        return A_f * A_F
    
    # Test all versions with seed
    seed_coeffs = np.array([-113/100, 1/25, 1/3240])
    c4_v1 = compute_C4_v1(*seed_coeffs)
    c4_v2 = compute_C4_v2(*seed_coeffs)
    c4_v3 = compute_C4_v3(*seed_coeffs)
    
    # Pick the version that seems most reasonable (smallest or most standard)
    # Use v1 as primary, but we can try others
    compute_C4 = compute_C4_v1
    
    best_coeffs = seed_coeffs.copy()
    best_C4 = compute_C4(*seed_coeffs)
    
    # Local optimization
    try:
        def objective(coeffs):
            return compute_C4(coeffs[0], coeffs[1], coeffs[2])
        
        result = minimize(objective, seed_coeffs, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-9, 'fatol': 1e-11})
        
        if time.time() - start_time < deadline - 15:
            if result.fun < best_C4:
                best_C4 = result.fun
                best_coeffs = result.x
    except Exception:
        pass
    
    # Try COBYLA as well
    if time.time() - start_time < deadline - 30:
        try:
            result2 = minimize(objective, best_coeffs, method='COBYLA',
                              options={'maxiter': 5000})
            if result2.fun < best_C4:
                best_C4 = result2.fun
                best_coeffs = result2.x
        except Exception:
            pass
    
    # Random restarts
    np.random.seed(123)
    for i in range(10):
        if time.time() - start_time > deadline - 25:
            break
        
        scale = 0.05 * (1 + i*0.5)
        perturbation = np.random.randn(3) * scale
        trial_coeffs = seed_coeffs + perturbation
        
        try:
            trial_C4 = compute_C4(*trial_coeffs)
            if trial_C4 < best_C4:
                best_C4 = trial_C4
                best_coeffs = trial_coeffs.copy()
                
                # Refine
                result = minimize(objective, best_coeffs, method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11})
                if result.fun < best_C4:
                    best_C4 = result.fun
                    best_coeffs = result.x
        except Exception:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
