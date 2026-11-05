import numpy as np
import time
from scipy.optimize import differential_evolution

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # We optimize in the frequency domain.
    # h is real, so we can represent it by its real and imaginary parts of the FFT.
    # However, enforcing reality and the DC constraint (sum(h) != 0) is easier in time domain 
    # or by fixing the DC component.
    # 
    # Let's stick to time domain but use differential_evolution on a subset of "important" 
    # parameters? No, the directive asks for a different design family. 
    # The "smooth basis" family is different from "arbitrary high-dim" or "spectral".
    # 
    # Actually, 400 dimensions is too high for standard DE within 150s if we use default settings.
    # We need a smarter approach or a reduced dimensionality that isn't just "low freq Chebyshev".
    # 
    # Alternative Structural Change: 
    # The current program uses Chebyshev polynomials. 
    # Let's use a **Sine Basis** (Fourier series) but optimized with **Differential Evolution**.
    # Why? Chebyshev is optimal for uniform approximation, but the objective involves 
    # autocorrelation (convolution). In the frequency domain, convolution becomes multiplication.
    # 
    # Let H(k) be the DFT of h. 
    # The autocorrelation peak is related to the max of the inverse DFT of |H|^2.
    # Minimizing the max autocorrelation peak is equivalent to minimizing the max of the 
    # aperiodic autocorrelation, which is a known problem (Low Peak-to-Average Power Ratio).
    #
    # However, optimizing 400 real numbers is hard.
    # Let's try a **Hybrid Approach**:
    # 1. Use a **Gaussian Mixture** or **Bump Function** representation? 
    #    The inspirations mention "bump functions".
    #    Let's represent h as a sum of a few Gaussian bumps.
    #    h(t) = sum_{j=1}^K A_j * exp(-0.5 * ((t - mu_j) / sigma_j)^2)
    #    This is a very different design family from global polynomials.
    #    It allows for localized structures that might cancel peaks in autocorrelation 
    #    more effectively than global smooth waves.
    #
    # Parameters: K bumps. Each has A, mu, sigma.
    # Total params: 3*K.
    # If K=10, we have 30 params. DE is feasible.
    
    K = 8  # Number of Gaussian bumps
    # Parameters: A (1), mu (1), sigma (1) for each bump
    # Bounds:
    # A: [-5, 5] (allow negative bumps for cancellation)
    # mu: [-1, 1] (grid is [-1, 1] effectively, mapped from indices)
    # sigma: [0.01, 1.0] (must be positive)
    
    # We will map the index i in [0, N-1] to x in [-1, 1]
    x = np.linspace(-1, 1, N)
    
    def evaluate_params(params):
        # params: [A0, mu0, sig0, A1, mu1, sig1, ...]
        h = np.zeros(N)
        for j in range(K):
            A = params[3*j]
            mu = params[3*j + 1]
            sig = params[3*j + 2]
            
            # Add Gaussian bump
            h += A * np.exp(-0.5 * ((x - mu) / sig) ** 2)
        
        s = np.sum(h)
        if s == 0 or not np.isfinite(s) or np.any(~np.isfinite(h)):
            return float('inf')
        
        # Compute autocorrelation using FFT
        n_fft = 1 << (2 * N - 1).bit_length()
        H = np.fft.rfft(h, n_fft)
        # Power spectrum
        P = H * np.conj(H)
        # Inverse FFT to get autocorrelation
        # Note: np.fft.irfft of P gives circular autocorrelation if n_fft >= 2N-1?
        # Actually, linear convolution of h with h is equivalent to circular convolution 
        # if zero-padded to length >= 2N-1.
        # The result of irfft(|H|^2) is the circular autocorrelation of the zero-padded h.
        # This matches the linear autocorrelation for the first N-1 lags if padding is sufficient.
        # Wait, np.convolve(h, h, 'full') has length 2N-1.
        # The DC term (lag 0) is sum(h^2).
        # The value is abs(2 * N * max(acf) / s^2)
        
        acf = np.fft.irfft(P, n_fft)[:2*N-1]
        
        # The max is taken over signed entries.
        # For a real signal, acf is real.
        max_acf = np.max(acf)
        
        return abs(2 * N * max_acf / (s * s))

    # Define bounds for Differential Evolution
    # A: [-10, 10]
    # mu: [-1, 1]
    # sig: [0.05, 1.5] (avoid extremely narrow or wide)
    
    bounds = []
    for _ in range(K):
        bounds.append((-10, 10))  # A
        bounds.append((-1, 1))    # mu
        bounds.append((0.05, 1.5)) # sig
    
    # Initial population size and max iterations
    # DE can be slow. We need to be careful with the budget.
    # popsize=15, maxiter=200 might be too slow for 24 dims?
    # Let's use a smaller population and fewer iterations, but multiple restarts?
    # Or just one run with tight constraints.
    
    # To speed up, we can use `polish=True` which does a local refinement at the end.
    
    best_result = None
    best_score = float('inf')
    
    # We will run DE. If it times out, we return the best found so far?
    # DE doesn't easily support callbacks to stop mid-run and return partial best 
    # in a way that is safe for the 150s limit without risking hanging.
    # However, we can estimate time.
    # 24 dimensions, popsize 10, maxiter 50.
    # Evaluations: 10 * 50 = 500.
    # Each eval: FFT of size 1024 (since 2*400=800, next pow2 is 1024).
    # FFT is very fast. 500 FFTs is negligible (<0.1s).
    # The overhead of DE is the Python loop.
    # Let's try popsize=20, maxiter=100.
    
    try:
        # Use differential_evolution
        # tol=1e-8, atol=1e-8
        result = differential_evolution(
            evaluate_params,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            polish=True,
            init='sobol'  # Sobol sequence for better coverage
        )
        
        if result.success or result.fun < best_score:
            best_score = result.fun
            best_result = result.x
            
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 5:
        if best_result is None:
            return np.ones(N)
        # Construct h from best_result
        h = np.zeros(N)
        for j in range(K):
            A = best_result[3*j]
            mu = best_result[3*j + 1]
            sig = best_result[3*j + 2]
            h += A * np.exp(-0.5 * ((x - mu) / sig) ** 2)
        
        if not np.all(np.isfinite(h)) or np.sum(h) == 0:
            return np.ones(N)
        return h

    # If we have time left, try a second round with different seeds or more iterations?
    # Or refine the local neighborhood of the best result?
    
    # Let's do a local refinement using L-BFGS-B on the best result found by DE.
    if best_result is not None:
        # L-BFGS-B needs gradient or finite difference.
        # We can use scipy.optimize.minimize with method='L-BFGS-B'
        # It will use finite differences by default.
        
        # We need to handle the case where sum(h) becomes 0 during optimization.
        # The objective returns inf, which should guide it away.
        
        try:
            res_local = minimize(
                evaluate_params,
                best_result,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res_local.fun < best_score:
                best_score = res_local.fun
                best_result = res_local.x
                
        except Exception:
            pass
    
    # Construct final h
    if best_result is None:
        return np.ones(N)
        
    h = np.zeros(N)
    for j in range(K):
        A = best_result[3*j]
        mu = best_result[3*j + 1]
        sig = best_result[3*j + 2]
        h += A * np.exp(-0.5 * ((x - mu) / sig) ** 2)
    
    # Sanity check
    if not np.all(np.isfinite(h)) or np.sum(h) == 0:
        return np.ones(N)
        
    return h

# Need to import minimize if used in the second phase
from scipy.optimize import minimize
