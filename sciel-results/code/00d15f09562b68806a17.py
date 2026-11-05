import numpy as np
import time

N = 400
DEADLINE = 145.0  # 5s safety margin before 150s kill

def _evaluate(h):
    """Compute the objective value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _dct2(x):
    """Compute DCT-II of a real vector."""
    N = len(x)
    n = np.arange(N)
    # DCT-II: X[k] = sum_{n=0}^{N-1} x[n] * cos(pi * (2n+1) * k / (2N))
    # Use numpy for efficiency
    return np.fft.fft(np.concatenate([x, x]))[:N].real * 2 / N  # Approximate, not exact DCT-II

def construct():
    start_time = time.time()
    best_h = np.ones(N)
    best_val = _evaluate(best_h)  # Should be 2.0
    
    # Try various structured initializations
    initializations = []
    
    # 1. Constant (baseline)
    initializations.append(np.ones(N))
    
    # 2. Linear ramp
    initializations.append(np.linspace(0.5, 1.5, N))
    
    # 3. Cosine bump
    t = np.linspace(0, 1, N)
    initializations.append(1.0 + 0.5 * np.cos(2 * np.pi * t))
    
    # 4. Sinc-like
    initializations.append(np.sinc((t - 0.5) * 2) + 0.5)
    
    # 5. Alternating with decay
    initializations.append(np.ones(N) * (1.0 + 0.3 * np.cos(4 * np.pi * t)))
    
    # 6. Gaussian bump
    initializations.append(np.exp(-20 * (t - 0.5)**2) + 0.5)
    
    # 7. Square wave
    initializations.append(np.sign(np.sin(2 * np.pi * t)) * 0.5 + 1.0)
    
    # 8. DCT-based: emphasize low frequencies
    # Create a spectrum with energy concentrated at low freqs
    for freq_scale in [1, 2, 5, 10]:
        spectrum = np.exp(-np.arange(N) / freq_scale)
        # Inverse DCT approximation via IFFT
        h_dct = np.real(np.fft.ifft(spectrum * 2))[:N]
        h_dct = h_dct - np.mean(h_dct) + 1.0  # Ensure positive mean
        initializations.append(h_dct)
    
    # Evaluate all initializations
    for h_init in initializations:
        h_init = h_init.astype(float)
        if not np.all(np.isfinite(h_init)) or np.sum(h_init) == 0:
            continue
        val = _evaluate(h_init)
        if val < best_val:
            best_val = val
            best_h = h_init.copy()
        
        if time.time() > DEADLINE:
            break
    
    # Gradient descent refinement on the best candidate
    if time.time() < DEADLINE - 10:
        h = best_h.copy()
        lr = 0.01
        for step in range(2000):
            if time.time() > DEADLINE - 5:
                break
            
            s = np.sum(h)
            if s == 0:
                break
            conv = np.convolve(h, h)
            max_val = np.max(conv)
            val = abs(2.0 * N * max_val / (s * s))
            
            # Numerical gradient
            grad = np.zeros(N)
            eps = 1e-6
            for i in range(0, N, 10):  # Sample gradient to save time
                h_plus = h.copy()
                h_plus[i] += eps
                s_plus = np.sum(h_plus)
                if s_plus == 0:
                    continue
                conv_plus = np.convolve(h_plus, h_plus)
                val_plus = abs(2.0 * N * np.max(conv_plus) / (s_plus * s_plus))
                grad[i] = (val_plus - val) / eps
            
            # Scale gradient for all elements based on sampled ones
            if np.any(grad != 0):
                scale = np.mean(np.abs(grad[grad != 0]))
                if scale > 0:
                    grad = grad / scale
            
            h = h - lr * grad
            h = np.clip(h, -10, 10)  # Prevent divergence
            
            val_new = _evaluate(h)
            if val_new < best_val:
                best_val = val_new
                best_h = h.copy()
                lr = max(lr * 0.95, 0.001)
            else:
                lr = max(lr * 0.5, 0.0001)
    
    # Final local search with random perturbations
    if time.time() < DEADLINE - 5:
        h = best_h.copy()
        for _ in range(500):
            if time.time() > DEADLINE - 2:
                break
            h_trial = h + np.random.randn(N) * 0.1
            val_trial = _evaluate(h_trial)
            if val_trial < best_val:
                best_val = val_trial
                best_h = h_trial.copy()
                h = best_h.copy()
    
    return best_h
