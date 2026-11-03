import numpy as np
import time
import random
from scipy.optimize import minimize_scalar

N = 600
TIME_LIMIT = 140  # Safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Use FFT for convolution to be fast, though N=600 is small
    # np.convolve is fine, but let's stick to standard robust method
    conv = np.convolve(f, f, mode='full')
    return np.max(conv) / (s * s)

def spectral_to_step(coeffs, beta=1.0):
    """
    Convert DFT coefficients to a non-negative step function.
    coeffs: complex array of shape (N,)
    We use a soft-plus projection to ensure non-negativity:
    f = softplus(FFT(coeffs)) / scale
    Actually, simpler: f = max(0, FFT(coeffs)). 
    But max(0, x) is non-smooth. 
    Let's use f = np.log1p(np.exp(FFT(coeffs))) / N. This is always positive and smooth.
    Wait, the task asks for a step function with nonnegative heights.
    The "soft-plus" idea from the prompt: "enforcing non-negativity via a soft-plus projection".
    
    Let z = np.fft.ifft(coeffs)
    f = np.log1p(np.exp(z)) / N  # This ensures f > 0 and smooth dependence on coeffs
    """
    z = np.fft.ifft(coeffs)
    # Soft plus: log(1 + exp(x))
    # To avoid overflow, use stable version
    z_real = z.real
    # f = np.logaddexp(0, z_real) / N
    # Actually, just clipping is often sufficient for discrete steps if we treat it as a projection,
    # but the prompt suggests soft-plus. Let's try the smooth positive version.
    # If we use f = max(0, ifft(coeffs)), the objective becomes non-smooth in coeffs.
    # Using softplus makes it smooth.
    f = np.logaddexp(0.0, z_real) / N
    return f

def objective_spectral(coeffs, beta=1.0):
    f = spectral_to_step(coeffs, beta)
    return compute_score(f)

def coordinate_descent_spectral(coeffs0, time_limit):
    """
    Perform coordinate descent on the real and imaginary parts of the DFT coefficients.
    """
    coeffs = coeffs0.copy()
    n_coeffs = len(coeffs)
    start_time = time.time()
    
    # Pre-compute indices to iterate
    indices = np.arange(n_coeffs)
    random.shuffle(indices)
    
    # We will iterate over a subset of frequencies or all of them depending on time
    # N=600 is small, so we can try a few passes over all coefficients.
    
    # To make it faster, we might only optimize the first half (symmetry) or just random subsets.
    # Let's try optimizing all 600 coefficients in a few passes.
    
    for iteration in range(20): # Max passes
        if time.time() - start_time > time_limit - 10:
            break
            
        for idx in indices:
            if time.time() - start_time > time_limit - 10:
                return coeffs
            
            # Optimize real part of coeffs[idx]
            def obj_real(x):
                c = coeffs.copy()
                c[idx].real = x
                return objective_spectral(c)
            
            res = minimize_scalar(obj_real, bounds=(-10.0, 10.0), method='bounded', options={'xatol': 1e-6, 'maxiter': 50})
            if res.success:
                coeffs[idx].real = res.x
                
            # Optimize imaginary part of coeffs[idx]
            def obj_imag(x):
                c = coeffs.copy()
                c[idx].imag = x
                return objective_spectral(c)
            
            res = minimize_scalar(obj_imag, bounds=(-10.0, 10.0), method='bounded', options={'xatol': 1e-6, 'maxiter': 50})
            if res.success:
                coeffs[idx].imag = res.x
                
    return coeffs

def construct():
    start_time = time.time()
    
    # Initialize with a constant function in time domain -> Delta in frequency domain?
    # No, constant in time is a single spike in frequency.
    # Let's start with a "flat" spectrum (white noise) or a low-pass spectrum.
    
    # Strategy: Start with a Gaussian in time domain, transform to frequency, then optimize.
    # Or start with a simple frequency shape.
    
    # Let's try starting with a few different initializations.
    best_coeffs = None
    best_score = np.inf
    
    # Initialization 1: Gaussian in time
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f_init = np.exp(-20 * x**2)
    coeffs_init = np.fft.fft(f_init)
    
    # Initialization 2: Box in time
    f_init2 = np.zeros(N)
    f_init2[100:500] = 1.0
    coeffs_init2 = np.fft.fft(f_init2)
    
    # Initialization 3: Low pass filter (smooth spectrum)
    coeffs_init3 = np.zeros(N, dtype=complex)
    for i in range(N):
        if i < 10 or (i > N - 10 and i < N): # Low frequencies
             coeffs_init3[i] = 1.0
    
    inits = [coeffs_init, coeffs_init2, coeffs_init3]
    
    for c0 in inits:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
            
        # Run coordinate descent
        c_opt = coordinate_descent_spectral(c0, time_limit=TIME_LIMIT - 20)
        
        # Evaluate
        f_opt = spectral_to_step(c_opt)
        score = compute_score(f_opt)
        
        if score < best_score:
            best_score = score
            best_coeffs = c_opt
            
    # If time remains, try random restarts in frequency domain
    restarts = 10
    for r in range(restarts):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
            
        # Random low-pass spectrum
        c_rand = np.zeros(N, dtype=complex)
        # Add random noise to low frequencies
        low_freqs = np.random.randint(0, N//4, size=10)
        for i in low_freqs:
            c_rand[i] = complex(np.random.randn(), np.random.randn())
            # Ensure conjugate symmetry if we want real time domain?
            # np.fft.ifft will handle complex input. If we want real f, we need Hermitian symmetry.
            # But spectral_to_step uses .real part of ifft, so it handles complex input by taking real part.
            # This effectively projects to the real part of the inverse transform.
            # It's better to maintain Hermitian symmetry to avoid wasting capacity on imaginary parts that get discarded?
            # Actually, if f is real, F must be Hermitian.
            # Let's enforce Hermitian symmetry in the initialization and updates?
            # The coordinate descent updates real and imag separately.
            # If we don't enforce symmetry, the "time domain" function is the real part of the IFFT.
            # This is a valid non-negative function.
            
        c_opt = coordinate_descent_spectral(c_rand, time_limit=TIME_LIMIT - 10)
        f_opt = spectral_to_step(c_opt)
        score = compute_score(f_opt)
        
        if score < best_score:
            best_score = score
            best_coeffs = c_opt
            
    # Final conversion
    if best_coeffs is not None:
        f_final = spectral_to_step(best_coeffs)
    else:
        # Fallback to constant
        f_final = np.ones(N)
        
    # Ensure non-negative
    f_final = np.maximum(f_final, 0.0)
    
    # One last check: compare with a simple optimized time-domain function?
    # The prompt asks for a different strategy. We stick to the spectral result.
    # But we can compare with the constant function just in case.
    f_const = np.ones(N)
    s_const = compute_score(f_const)
    
    if s_const < best_score:
        f_final = f_const
        
    return [float(x) for x in f_final]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
