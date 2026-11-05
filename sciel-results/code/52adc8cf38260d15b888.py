import numpy as np
from scipy.optimize import minimize

N = 400

def _evaluate_h(h):
    S = np.sum(h)
    if np.abs(S) < 1e-12:
        return np.inf
    C = np.convolve(h, h)
    M = np.max(C)
    return abs(2 * N * M / (S * S))

def construct():
    best_h = np.ones(N)
    best_val = _evaluate_h(best_h)
    
    # Strategy: Use a parameterization based on a low-order polynomial or
    # a combination of smooth functions.
    # Let's try h_i = 1 + sum_{k=1}^K (a_k * sin(2*pi*k*i/N) + b_k * cos(2*pi*k*i/N))
    # This ensures the signal is band-limited, which often leads to smoother autocorrelations.
    
    # Alternative: Directly optimize h with a smoothness penalty?
    # No, the objective is non-convex.
    
    # Let's stick to the harmonic approach but improve it.
    # The previous code used random frequencies.
    # Better: Use fixed, low frequencies (harmonics of the fundamental) to ensure smoothness.
    
    i = np.arange(N)
    
    # Try different numbers of harmonics
    for K in [1, 2, 3, 4, 5]:
        # Parameters: a0, a1, b1, a2, b2, ..., aK, bK
        # Total params: 1 + 2*K
        n_params = 1 + 2 * K
        
        # Basis functions
        # phi_0 = 1
        # phi_{2j-1} = sin(2*pi*j*i/N)
        # phi_{2j} = cos(2*pi*j*i/N)
        
        basis = np.ones((N, 1))
        for j in range(1, K + 1):
            sin_part = np.sin(2 * np.pi * j * i / N)
            cos_part = np.cos(2 * np.pi * j * i / N)
            basis = np.column_stack((basis, sin_part, cos_part))
        
        # Objective: h = basis @ p
        # We want to minimize _evaluate_h(h)
        
        def objective(p):
            h = basis @ p
            return _evaluate_h(h)
        
        # Random restarts
        for _ in range(50):
            # Initialize parameters
            # Keep a0 positive and large to ensure S is not zero
            a0 = np.random.uniform(0.5, 1.5)
            # Small random coefficients for harmonics
            coeffs = np.random.uniform(-0.1, 0.1, size=2 * K)
            p0 = np.concatenate(([a0], coeffs))
            
            try:
                # Use Nelder-Mead or L-BFGS-B
                # L-BFGS-B might be better if we can provide gradients, but we don't have them.
                # Nelder-Mead is derivative-free.
                
                # To help the optimizer, we can normalize the output h such that sum(h) = 1
                # But the objective function handles scaling invariantly?
                # V = 2N * max(h*h) / (sum h)^2
                # If h -> c*h, V -> 2N * c^2 * max(h*h) / (c * sum h)^2 = V.
                # So the value is scale invariant.
                # This means the optimizer might wander in scale.
                # We can add a constraint or just normalize inside objective?
                # Normalizing inside objective changes the landscape but keeps the value.
                
                def objective_norm(p):
                    h = basis @ p
                    S = np.sum(h)
                    if np.abs(S) < 1e-6:
                        return 1e10
                    h_norm = h / S
                    C = np.convolve(h_norm, h_norm)
                    M = np.max(C)
                    return abs(2 * N * M)
                
                res = minimize(objective_norm, p0, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
                
                if res.fun < 1e9:
                    h_res = basis @ res.x
                    val = _evaluate_h(h_res)
                    if val < best_val:
                        best_val = val
                        best_h = h_res
                        
            except Exception:
                pass

    # Try a few random smooth signals as well
    for _ in range(50):
        # Generate a smooth random signal using low-pass filtering of white noise
        noise = np.random.normal(0, 1, N)
        # Apply a simple low-pass filter in frequency domain
        fft_noise = np.fft.fft(noise)
        # Keep only low frequencies
        mask = np.zeros(N)
        mask[:10] = 1
        mask[-10:] = 1
        fft_filtered = fft_noise * mask
        h = np.real(np.fft.ifft(fft_filtered))
        
        # Ensure sum is not zero
        S = np.sum(h)
        if np.abs(S) < 1e-6:
            continue
            
        val = _evaluate_h(h)
        if val < best_val:
            best_val = val
            best_h = h

    return best_h
