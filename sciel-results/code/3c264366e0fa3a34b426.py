import numpy as np
import time
import scipy.optimize as opt

N = 400

def construct():
    deadline = time.time() + 115  # Safety margin
    
    def score(h):
        s = np.sum(h)
        if s == 0 or not np.isfinite(s):
            return float('inf')
        # np.convolve(h, h) is linear convolution
        # For efficiency in the inner loop, we can use FFT, 
        # but the evaluator uses np.convolve. 
        # We must ensure the returned h yields the same value.
        # The metric is: abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))

    # Strategy: Represent h as a truncated Fourier series.
    # h[n] = sum_{k=0}^{K} (a_k * cos(2*pi*k*n/N) + b_k * sin(2*pi*k*n/N))
    # This allows for periodic structures with multiple peaks/valleys.
    # The central peak of the auto-convolution of a "smooth" function is high.
    # By introducing higher frequency components, we can make h oscillate,
    # causing the convolution to spread out or have cancellations in the center.
    
    # We optimize the coefficients.
    # To ensure non-zero sum, we fix the DC term (a_0) to be large enough or optimize it.
    # Let's include a_0 in the optimization but constrain it or start with a good guess.
    
    # Number of harmonics to include. 
    # Too many -> overfitting to noise/local minima, hard to optimize.
    # Too few -> limited flexibility.
    # Let's try K=5 to 10 harmonics.
    
    K = 8
    # Parameters: a_0, a_1...a_K, b_1...b_K
    # Total params: 1 + 2*K
    n_params = 1 + 2 * K
    
    def build_h_from_coeffs(coeffs):
        a_0 = coeffs[0]
        a = coeffs[1:K+1]
        b = coeffs[K+1:2*K+1]
        
        n = np.arange(N)
        h = np.full(N, a_0)
        
        for k in range(1, K+1):
            h += a[k-1] * np.cos(2 * np.pi * k * n / N)
            h += b[k-1] * np.sin(2 * np.pi * k * n / N)
            
        return h

    def objective(coeffs):
        h = build_h_from_coeffs(coeffs)
        s = np.sum(h)
        if abs(s) < 1e-6:
            return 1e6
        # We want to minimize the value
        val = score(h)
        return val

    best_h = np.ones(N)
    best_score = score(best_h)
    
    # Initial guesses
    # 1. All zeros except DC (which is just the seed, score 2.0)
    # 2. Small random perturbations
    # 3. Structured guesses: e.g., strong 2nd harmonic
    
    guesses = []
    
    # Seed-like
    guesses.append(np.array([1.0] + [0.0]* (2*K)))
    
    # Random small
    for _ in range(5):
        guesses.append(np.random.normal(1.0, 0.1, n_params))
        
    # Structured: Strong 2nd harmonic (cosine)
    g = np.zeros(n_params)
    g[0] = 1.0
    g[2] = 0.5 # a_1? No, index 1 is a_1. 
    # coeffs: [a0, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8]
    # a1 is at index 1.
    g[1] = 0.5
    guesses.append(g)
    
    # Structured: Strong 3rd harmonic
    g = np.zeros(n_params)
    g[0] = 1.0
    g[2] = 0.5 # a2? No, a_k is at index k. 
    # Index 0: a0
    # Index 1: a1
    # Index 2: a2
    # ...
    # Index K: aK
    # Index K+1: b1
    # ...
    g[2] = 0.5 
    guesses.append(g)
    
    # Random larger perturbations
    for _ in range(10):
        guesses.append(np.random.normal(1.0, 0.5, n_params))

    for guess in guesses:
        if time.time() > deadline - 10:
            break
        
        try:
            # Nelder-Mead is good for low-dim non-differentiable (due to max/abs)
            res = opt.minimize(objective, guess, method='Nelder-Mead', 
                               options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.fun < best_score and np.isfinite(res.fun):
                h_opt = build_h_from_coeffs(res.x)
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-6:
                    val = score(h_opt)
                    if val < best_score:
                        best_score = val
                        best_h = h_opt
        except Exception:
            pass

    # If the Fourier approach didn't yield a significant improvement, 
    # try a "Wavelet-like" or "Multi-scale" approach using a different basis?
    # Or simply more random restarts with the Fourier basis.
    
    if time.time() < deadline - 20:
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            guess = np.random.normal(1.0, 1.0, n_params)
            try:
                res = opt.minimize(objective, guess, method='Nelder-Mead', 
                                   options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < best_score and np.isfinite(res.fun):
                    h_opt = build_h_from_coeffs(res.x)
                    s_opt = np.sum(h_opt)
                    if abs(s_opt) > 1e-6:
                        val = score(h_opt)
                        if val < best_score:
                            best_score = val
                            best_h = h_opt
            except Exception:
                pass

    return best_h
