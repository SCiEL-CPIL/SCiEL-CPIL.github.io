import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Strategy:
    # 1. Parameterize h as a sum of cosines: h[i] = A * sum_j cos(2*pi*f_j*i/N + phi_j)
    #    This ensures h is real and band-limited.
    #    By choosing frequencies and phases, we can shape the power spectrum |H(k)|^2.
    #    The autocorrelation is the inverse FFT of |H(k)|^2.
    #    Minimizing the max of the inverse FFT of |H|^2 is equivalent to minimizing PAPR.
    #    Signals with "flat" spectra (like chirps or random phases) tend to have lower peaks.
    #    We optimize the phases and frequencies.
    
    # We use a mix of a few low-frequency components to allow some shape flexibility
    # while maintaining the "structured" nature that avoids sharp time-domain spikes.
    # Actually, pure cosines are smooth. The autocorrelation of a sum of cosines
    # is a sum of cosines (beats).
    # Let's try K components.
    # Parameters: for each component j: frequency f_j (in cycles/sample, 0.5 to N-0.5? No, 0 to 0.5), phase phi_j, amplitude A_j.
    # To ensure sum(h) != 0, we add a DC term A_0.
    
    K = 6  # Number of AC components
    # Total params: 1 (DC) + K * 3 (freq, phase, amp) = 1 + 18 = 19 params
    
    x = np.arange(N)
    
    def evaluate_params(params):
        # params: [A0, f1, phi1, A1, f2, phi2, A2, ...]
        h = np.full(N, params[0]) # DC term
        
        for j in range(K):
            f = params[1 + 3*j]
            phi = params[2 + 3*j]
            A = params[3 + 3*j]
            # Frequency f is in [0, 0.5] cycles/sample
            # We use 2*pi*f*i
            h += A * np.cos(2 * np.pi * f * x + phi)
        
        s = np.sum(h)
        if s == 0 or not np.isfinite(s) or not np.all(np.isfinite(h)):
            return float('inf')
        
        # Compute autocorrelation via FFT
        # Zero pad to next power of 2 >= 2N-1 to compute linear convolution via circular convolution
        n_fft = 1 << (2 * N - 1).bit_length()
        H = np.fft.rfft(h, n_fft)
        P = H * np.conj(H)
        acf = np.fft.irfft(P, n_fft)[:2*N-1]
        
        max_acf = np.max(acf)
        
        # Objective: abs(2 * N * max_acf / s^2)
        val = abs(2 * N * max_acf / (s * s))
        
        return val

    # Bounds:
    # A0 (DC): [-10, 10]
    # f_j: [0.0, 0.5]
    # phi_j: [0, 2*pi]
    # A_j: [-10, 10]
    
    bounds = [(-10, 10)] # DC
    for _ in range(K):
        bounds.append((0.0, 0.5))   # freq
        bounds.append((0.0, 2*np.pi)) # phase
        bounds.append((-10, 10))    # amp
    
    best_result = None
    best_score = float('inf')
    
    # Run Differential Evolution
    try:
        result = differential_evolution(
            evaluate_params,
            bounds,
            seed=42,
            maxiter=150,
            popsize=15,
            tol=1e-9,
            polish=False, # We will do our own polish
            init='sobol',
            workers=1
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_result = result.x
            
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        if best_result is None:
            return np.ones(N)
        # Construct h
        h = np.full(N, best_result[0])
        for j in range(K):
            f = best_result[1 + 3*j]
            phi = best_result[2 + 3*j]
            A = best_result[3 + 3*j]
            h += A * np.cos(2 * np.pi * f * x + phi)
        
        if not np.all(np.isfinite(h)) or np.sum(h) == 0:
            return np.ones(N)
        return h

    # Local Refinement with L-BFGS-B
    if best_result is not None:
        try:
            res_local = minimize(
                evaluate_params,
                best_result,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            if res_local.fun < best_score:
                best_score = res_local.fun
                best_result = res_local.x
                
        except Exception:
            pass
            
    # Check time again
    if time.time() > deadline - 5:
        if best_result is None:
            return np.ones(N)
        h = np.full(N, best_result[0])
        for j in range(K):
            f = best_result[1 + 3*j]
            phi = best_result[2 + 3*j]
            A = best_result[3 + 3*j]
            h += A * np.cos(2 * np.pi * f * x + phi)
        
        if not np.all(np.isfinite(h)) or np.sum(h) == 0:
            return np.ones(N)
        return h

    # If we have time, try a second DE run with different seed or more iterations?
    # Or try a different K?
    # Let's try a second run with K=10 to see if more components help, using remaining time.
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        K2 = 10
        bounds2 = [(-10, 10)]
        for _ in range(K2):
            bounds2.append((0.0, 0.5))
            bounds2.append((0.0, 2*np.pi))
            bounds2.append((-10, 10))
            
        def evaluate_params2(params):
            h = np.full(N, params[0])
            for j in range(K2):
                f = params[1 + 3*j]
                phi = params[2 + 3*j]
                A = params[3 + 3*j]
                h += A * np.cos(2 * np.pi * f * x + phi)
            
            s = np.sum(h)
            if s == 0 or not np.isfinite(s) or not np.all(np.isfinite(h)):
                return float('inf')
            
            n_fft = 1 << (2 * N - 1).bit_length()
            H = np.fft.rfft(h, n_fft)
            P = H * np.conj(H)
            acf = np.fft.irfft(P, n_fft)[:2*N-1]
            max_acf = np.max(acf)
            return abs(2 * N * max_acf / (s * s))

        try:
            result2 = differential_evolution(
                evaluate_params2,
                bounds2,
                seed=123,
                maxiter=100,
                popsize=12,
                tol=1e-9,
                polish=False,
                init='sobol'
            )
            
            if result2.fun < best_score:
                best_score = result2.fun
                best_result = result2.x
                # Mark that we used K2 structure
                # We need to reconstruct h with K2
                h = np.full(N, best_result[0])
                for j in range(K2):
                    f = best_result[1 + 3*j]
                    phi = best_result[2 + 3*j]
                    A = best_result[3 + 3*j]
                    h += A * np.cos(2 * np.pi * f * x + phi)
                
                if np.all(np.isfinite(h)) and np.sum(h) != 0:
                    return h
        except Exception:
            pass

    # Final construction from best_result (K=6 case if K2 didn't win or wasn't run)
    if best_result is None:
        return np.ones(N)
        
    # Determine if best_result corresponds to K or K2
    # Since we overwrote best_result in K2 block only if it was better, 
    # and we returned immediately in that block if it was valid, 
    # if we are here, best_result is from K=6 (or K2 if K2 was better but returned early? 
    # Wait, in K2 block I did "return h" if valid. So if we are here, best_result is K=6).
    # Actually, if K2 was better, I returned. If K2 was not better, best_result is still K=6.
    # So we just construct with K=6.
    
    h = np.full(N, best_result[0])
    for j in range(K):
        f = best_result[1 + 3*j]
        phi = best_result[2 + 3*j]
        A = best_result[3 + 3*j]
        h += A * np.cos(2 * np.pi * f * x + phi)
    
    if not np.all(np.isfinite(h)) or np.sum(h) == 0:
        return np.ones(N)
        
    return h
