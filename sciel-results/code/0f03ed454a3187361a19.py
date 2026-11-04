"""
Strategy: Fractional Fourier Transform (FRFT) based design.
We model the sequence x as the output of a fractional-order Fourier transform
applied to a base signal with a flat magnitude spectrum (random phases).
The fractional order alpha and the phases are optimized to minimize the
maximum absolute autoconvolution.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400

# Precompute FFT size for autoconvolution
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    return float(np.sum(x))

def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _frft(x, alpha):
    """
    Compute the Fractional Fourier Transform of x.
    We use the chirp multiplication method (three FFTs) for efficiency.
    alpha is in radians.
    """
    n = len(x)
    # Handle trivial cases
    if abs(np.sin(alpha)) < 1e-12:
        return x.copy()
    if abs(np.cos(alpha)) < 1e-12:
        # alpha = pi/2, standard FFT
        # Note: Standard DFT definition vs FRFT definition might differ by constants/scaling.
        # For our purpose, we just want a unitary-ish rotation in signal space.
        # We use the standard FFT but normalized to preserve energy or similar.
        # Actually, let's stick to the continuous formula approximation or a discrete equivalent.
        # A simpler robust implementation for discrete FRFT:
        pass
    
    # Discrete FRFT via chirp multiplication
    # x(k) -> exp(-i pi k^2 cot alpha) * x(k) -> FFT -> exp(-i pi k^2 tan alpha)
    
    k = np.arange(n)
    # To avoid large exponents, we can reduce alpha modulo pi
    # But cot and tan have singularities.
    
    if abs(np.sin(alpha)) < 1e-12:
        return x.copy()
        
    if abs(np.cos(alpha)) < 1e-12:
        # alpha = pi/2
        return np.fft.fft(x) / np.sqrt(n) # Normalize to keep energy comparable? 
        # Actually, let's just return FFT. The optimization will handle scaling.
        # But wait, if we mix different alphas, the scale might vary wildly.
        # Let's normalize the output of FRFT to have unit L2 norm or similar?
        # No, we want to control the integral S.
        # Let's just compute it.
        
    # General case
    # We need to be careful with the phase terms to avoid overflow/underflow
    # and to ensure the transform is invertible.
    
    # Let's use a slightly more stable formulation if possible, or just careful numpy.
    # cot(alpha) = cos/sin
    # tan(alpha) = sin/cos
    
    # If alpha is close to 0 or pi, use series expansion? 
    # For now, assume alpha is not too close to singularities, or handle it.
    
    # To ensure stability, we can restrict alpha to [0, pi/2] and use symmetry?
    # FRFT(alpha) = FRFT(pi/2 - alpha)^* ... no, it's complex.
    
    # Let's just implement the 3-FFT method.
    # h1 = exp(-i * pi * k^2 * cot(alpha))
    # h2 = exp(-i * pi * k^2 * tan(alpha))
    
    # To prevent huge numbers in exp, we can work with phases modulo 2*pi?
    # np.exp handles large imaginary parts poorly (loss of precision).
    # Let's reduce k^2 * cot(alpha) modulo 2*pi.
    
    k2 = k * k
    cot_a = np.cos(alpha) / np.sin(alpha)
    tan_a = np.sin(alpha) / np.cos(alpha)
    
    # Reduce phase to [-pi, pi] for numerical stability
    phase1 = (-np.pi * k2 * cot_a) % (2 * np.pi)
    phase2 = (-np.pi * k2 * tan_a) % (2 * np.pi)
    
    h1 = np.exp(1j * phase1)
    h2 = np.exp(1j * phase2)
    
    # Apply
    y = x * h1
    y = np.fft.fft(y)
    y = y * h2
    
    # Scaling factor for FRFT
    # C = sqrt(1 - i*cot(alpha)) * exp(i*pi*alpha/4) ... 
    # For our purpose of minimizing peak, the global scale doesn't matter much 
    # because we divide by S^2. But relative scaling between frequency bins matters.
    # The 3-FFT method above is a valid FRFT up to a constant factor.
    # Let's normalize the result to have the same L2 norm as input to keep things stable?
    # Or just return y.
    
    return y

def _generate_base_signal(rng, N):
    """
    Generate a base signal with flat magnitude spectrum (random phases).
    x(t) = IDFT( exp(i * phi_k) )
    """
    # Generate random phases
    # We want the signal to be real? 
    # If we want real x, we need Hermitian symmetry in frequency domain.
    # X[k] = conj(X[N-k])
    
    phases = rng.uniform(0, 2 * np.pi, size=N)
    # Make it Hermitian
    # X[0] is real -> phase 0 or pi
    # X[N/2] is real if N even -> phase 0 or pi
    # X[k] and X[N-k] are conjugates.
    
    X = np.zeros(N, dtype=complex)
    X[0] = 1.0 # DC component, let's fix it for now or optimize it?
    # Let's optimize the DC phase too? Or just keep it 1.
    
    for k in range(1, N // 2 + 1):
        if k == N - k: # Nyquist
            X[k] = np.exp(1j * phases[k])
        else:
            X[k] = np.exp(1j * phases[k])
            X[N - k] = np.conj(X[k])
            
    # IDFT to get time domain signal
    x = np.fft.ifft(X).real
    return x

def _optimize_frft_phases(x_init, alpha_init, deadline, rng):
    """
    Optimize the phases of the frequency domain representation 
    and the fractional order alpha.
    """
    best_x = x_init.copy()
    best_val = _metric(best_x)
    
    # We will optimize the phases in the frequency domain of the BASE signal,
    # and the alpha.
    # Let's parameterize the base signal by its frequency phases.
    
    # Initial phases from x_init
    X_init = np.fft.fft(x_init)
    # Extract phases
    # To maintain real-valued output, we enforce Hermitian symmetry in the optimization variables.
    # Let's represent the signal by the first N/2+1 phases.
    
    n_phases = N // 2 + 1
    # Initial phases
    init_phases = np.angle(X_init[:n_phases])
    # Handle DC and Nyquist specially (they should be real)
    # We can just optimize them as phases and then take real part, 
    # but that breaks the Hermitian constraint if we don't map back correctly.
    
    # Better: Optimize the complex values X[0..N/2] directly?
    # No, let's stick to phases for stability.
    
    # Objective function
    def obj(params):
        alpha = params[0]
        phases = params[1:]
        
        # Construct X
        X = np.zeros(N, dtype=complex)
        X[0] = np.exp(1j * phases[0])
        
        for k in range(1, n_phases):
            if k == N - k:
                X[k] = np.exp(1j * phases[k])
            else:
                X[k] = np.exp(1j * phases[k])
                X[N - k] = np.conj(X[k])
                
        # IDFT to get base signal
        x_base = np.fft.ifft(X).real
        
        # Apply FRFT
        x_frft = _frft(x_base, alpha)
        
        # Take real part? 
        # FRFT of a real signal with Hermitian spectrum is not necessarily real.
        # The task asks for "signed step heights", which implies real values.
        # So we must take the real part of the FRFT output.
        x_out = np.real(x_frft)
        
        s = _integral(x_out)
        if abs(s) < 1e-12:
            return 1e10
        
        conv = _autoconv_max_abs(x_out)
        return conv / (s * s)
    
    # Gradient descent or L-BFGS-B
    # We can use L-BFGS-B on the parameters [alpha, phases...]
    # Note: alpha is periodic. Let's constrain it to [0, pi/2].
    
    x0 = np.concatenate(([alpha_init], init_phases))
    
    bounds = [(0, np.pi/2)] + [(0, 2*np.pi)] * (n_phases - 1)
    
    result = minimize(
        obj,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'maxfun': 5000}
    )
    
    if result.fun < best_val:
        alpha_opt = result.x[0]
        phases_opt = result.x[1:]
        
        # Reconstruct best_x
        X = np.zeros(N, dtype=complex)
        X[0] = np.exp(1j * phases_opt[0])
        for k in range(1, n_phases):
            if k == N - k:
                X[k] = np.exp(1j * phases_opt[k])
            else:
                X[k] = np.exp(1j * phases_opt[k])
                X[N - k] = np.conj(X[k])
                
        x_base = np.fft.ifft(X).real
        x_out = np.real(_frft(x_base, alpha_opt))
        
        best_x = x_out
        best_val = result.fun
        
    return best_x, best_val

def _local_search_phases(x, deadline, rng):
    """
    Local search on the phases of the FRFT output directly?
    Or on the original signal?
    Let's do a simple local search on the signal x itself, 
    but using the FRFT structure as a restart mechanism.
    """
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    step_scale = 0.1
    
    for round_idx in range(100000):
        if time.time() > deadline:
            break
        
        if no_improve > 100:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.001:
                break
        
        k = int(rng.integers(1, 5))
        indices = rng.choice(N, size=k, replace=False)
        
        best_delta = 0
        best_idx = None
        for trial in range(2):
            x_test = best_x.copy()
            for idx in indices:
                x_test[idx] += rng.normal(0, step_scale)
            v = _metric(x_test)
            if v < best_val + best_delta:
                best_delta = v - best_val
                best_idx = x_test
        
        if best_idx is not None and best_delta < 0:
            best_val += best_delta
            best_x = best_idx
            no_improve = 0
        else:
            no_improve += 1
            
    return best_x, best_val

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(2024)
    
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Try several random initializations for the FRFT approach
    for i in range(5):
        if time.time() > deadline - 20:
            break
            
        # Generate base signal
        x_base = _generate_base_signal(rng, N)
        
        # Try a few random alphas
        for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2]:
            if time.time() > deadline - 10:
                break
                
            x_frft = np.real(_frft(x_base, alpha))
            v = _metric(x_frft)
            if v < best_val:
                best_val = v
                best_x = x_frft.copy()
                
    # Optimize phases and alpha for the best candidate found so far
    # We need to recover the phases from best_x
    # This is tricky because best_x is the output of FRFT.
    # Let's just re-run the optimization starting from a fresh random base 
    # but using the best alpha found? 
    # Or better, optimize the parameters of the FRFT model directly.
    
    # Let's run the optimizer with the best_x as a guide for initialization?
    # The optimizer in _optimize_frft_phases expects an x_init to derive phases.
    # If we pass best_x, it will derive phases of best_x, apply FRFT, which is not 
    # the same as the original generation process.
    
    # Instead, let's pick the best base signal and alpha from the loop above 
    # and refine them.
    
    # We didn't store the best base signal. Let's re-generate and pick one, 
    # or just run the optimizer on a new random base.
    
    # Let's run the optimizer on a new random base signal.
    x_base_new = _generate_base_signal(rng, N)
    # Use a random alpha
    alpha_init = rng.uniform(0.1, 1.4)
    
    if time.time() < deadline - 30:
        x_opt, val_opt = _optimize_frft_phases(x_base_new, alpha_init, deadline - 30, rng)
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt
    
    # Final local search
    if time.time() < deadline - 5:
        best_x, best_val = _local_search_phases(best_x, deadline - 5, rng)
        
    return best_x

# EVOLVE-BLOCK-END
