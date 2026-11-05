"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _value(h):
    """Compute the evaluator's value: abs(2 * N * max(autoconv) / sum(h)^2)"""
    s = np.sum(h)
    if s == 0:
        return np.inf
    ac = np.convolve(h, h, mode='full')
    peak = np.max(ac)
    return abs(2 * N * peak / (s ** 2))

def _softmax_obj(coeffs, beta, x_grid, freqs):
    """
    Smooth surrogate: minimize abs(2*N*softmax(ac)/s^2) using log-sum-exp trick.
    coeffs: Fourier coefficients (real, even symmetry assumed for real h)
    x_grid: precomputed linspace(-1,1,N)
    freqs: frequency indices [1, 3, 5, ...] (odd harmonics for antisymmetric? No, we want general)
          Actually, let's use a mix of cos and sin? 
          Simpler: h = sum c_k * cos(k * pi * t) + d_k * sin(k * pi * t)?
          Or just use complex FFT coefficients?
          Let's stick to real-valued basis: h[t] = sum_{k=1}^{M} a_k cos(k pi t) + b_k sin(k pi t)
          This ensures h is smooth and captures oscillatory behavior.
    """
    M = len(freqs)
    # Reconstruct h from coefficients
    # coeffs layout: [a_1, b_1, a_2, b_2, ...]
    h = np.zeros(N)
    for i, k in enumerate(freqs):
        a = coeffs[2*i]
        b = coeffs[2*i+1]
        h += a * np.cos(k * np.pi * x_grid) + b * np.sin(k * np.pi * x_grid)
        
    if not np.all(np.isfinite(h)):
        return 1e10
        
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
        
    # Compute autoconvolution via FFT for speed
    # Zero-pad to avoid circular convolution issues
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    ac = np.fft.irfft(H * H, n=n_fft)
    
    # The full autoconvolution length is 2*N-1.
    # np.convolve(h,h,'full') gives indices 0 to 2N-2.
    # FFT circular conv of length 2N gives indices 0 to 2N-1.
    # The linear conv part we care about is roughly the middle.
    # Specifically, np.convolve(h,h,'full')[k] corresponds to ac[k] for k=0..N-1?
    # Let's check:
    # h = [h0, h1]
    # full = [h0^2, 2h0h1, h1^2]
    # fft len 4: H=[h0+h1, ...], ac = ifft(H^2)
    # ac[0] = h0^2 + h1^2 + 2h0h1? No.
    # ac[0] = sum h_i h_{(0-i)%4}.
    # It is safer to just use np.convolve for the exact peak if N is small enough?
    # N=400, convolve is O(N^2) = 160k ops. Very fast.
    # FFT is O(N log N). 
    # Let's use np.convolve for exactness in the objective, it's fast enough for 400.
    ac_linear = np.convolve(h, h, mode='full')
    
    # Soft-max of ac_linear
    # We want to approximate max(ac_linear)
    # log-sum-exp: max + log(sum(exp(ac - max)))
    # To avoid overflow, subtract max first
    ac_max = np.max(ac_linear)
    # If ac_max is very negative, exp(ac - ac_max) is in (0,1]
    # If ac_max is very positive, same.
    # However, if beta is large, exp(beta * (ac - ac_max)) can underflow for small differences, which is good.
    # We want to minimize the peak, so we want to suppress the maximum.
    # The soft-max is an upper bound on the max.
    # We minimize the soft-max value.
    
    # Note: The evaluator uses np.max(ac) directly.
    # If we minimize a smooth upper bound, we might drive the max down.
    
    # Compute log-sum-exp safely
    # To prevent overflow in exp, we scale by beta.
    # log(sum(exp(beta * ac))) = beta * ac_max + log(sum(exp(beta * (ac - ac_max))))
    # The term inside log is sum of exps of negative or zero numbers.
    
    scaled = beta * (ac_linear - ac_max)
    # exp(scaled) is <= 1
    exp_sum = np.sum(np.exp(scaled))
    
    # soft_max = ac_max + (1/beta) * log(exp_sum)
    soft_max = ac_max + (1.0 / beta) * np.log(exp_sum)
    
    val = abs(2.0 * N * soft_max / (s ** 2))
    
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Basis: Low frequency Fourier modes
    # We want to allow negative heights to create destructive interference in autoconv.
    # Use frequencies 1 to M.
    M = 12  # 12 cos + 12 sin = 24 parameters
    freqs = list(range(1, M + 1))
    
    x_grid = np.linspace(-1, 1, N)
    
    # Initial best
    best_h = np.ones(N)
    best_val = _value(best_h)
    
    # Random initializations
    np.random.seed(42)
    
    # List of (coeffs, val) to keep track of best candidates
    candidates = []
    
    # Try several random starts
    num_starts = 20
    beta_initial = 50.0
    
    for i in range(num_starts):
        if time.time() > deadline - 20:
            break
            
        # Random initial coefficients
        # Scale: amplitudes should be comparable to 1
        init_coeffs = np.random.uniform(-1, 1, 2 * M)
        
        # L-BFGS-B optimization
        try:
            res = minimize(
                _softmax_obj,
                init_coeffs,
                args=(beta_initial, x_grid, freqs),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < best_val:
                # Reconstruct h
                h = np.zeros(N)
                for j, k in enumerate(freqs):
                    a = res.x[2*j]
                    b = res.x[2*j+1]
                    h += a * np.cos(k * np.pi * x_grid) + b * np.sin(k * np.pi * x_grid)
                
                if np.all(np.isfinite(h)) and abs(np.sum(h)) > 1e-10:
                    exact_val = _value(h)
                    if exact_val < best_val:
                        best_h = h
                        best_val = exact_val
                        candidates.append(res.x.copy())
        except Exception:
            pass

    # Polish pass: Use the best candidates found and refine with higher beta
    # Higher beta makes the soft-max closer to the hard max, allowing finer tuning of the peak.
    if candidates:
        # Take the top 3 candidates
        # (We didn't store all, so we re-run the best one found in the loop? 
        #  Actually, let's just take the current best_h and re-optimize its coefficients?
        #  We don't have the coeffs for best_h directly unless it was from a successful res.
        #  Let's assume the last successful res that updated best_val is the best.
        #  To be safe, let's do a local polish on the best_h by converting it back to coeffs?
        #  Inverse problem is hard.
        
        # Alternative: Keep the best res.x from the loop.
        # Let's modify the loop to store the best res.x
        pass # We'll implement a simpler polish: just re-optimize from the best coeffs found.
        
    # Let's restructure: Keep track of the best coeffs found.
    best_coeffs = None
    best_val_coeffs = np.inf
    
    # Re-run search to capture coeffs? No, that wastes time.
    # Let's modify the first loop to store coeffs.
    
    # Since I can't easily re-run, I will implement the polish on the `best_h` by 
    # approximating its Fourier coefficients and then optimizing those.
    if time.time() < deadline - 10:
        # Compute Fourier coeffs of best_h
        # h ~ sum a_k cos + b_k sin
        # a_k = (2/N) sum h_i cos(k pi x_i)
        # b_k = (2/N) sum h_i sin(k pi x_i)
        # Note: x_grid is -1 to 1.
        
        coeffs_best = np.zeros(2 * M)
        for j, k in enumerate(freqs):
            coeffs_best[2*j] = (2.0 / N) * np.sum(best_h * np.cos(k * np.pi * x_grid))
            coeffs_best[2*j+1] = (2.0 / N) * np.sum(best_h * np.sin(k * np.pi * x_grid))
            
        # Polish with higher beta
        beta_polish = 200.0
        try:
            res_polish = minimize(
                _softmax_obj,
                coeffs_best,
                args=(beta_polish, x_grid, freqs),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            
            if res_polish.fun < best_val:
                h = np.zeros(N)
                for j, k in enumerate(freqs):
                    a = res_polish.x[2*j]
                    b = res_polish.x[2*j+1]
                    h += a * np.cos(k * np.pi * x_grid) + b * np.sin(k * np.pi * x_grid)
                
                if np.all(np.isfinite(h)) and abs(np.sum(h)) > 1e-10:
                    exact_val = _value(h)
                    if exact_val < best_val:
                        best_h = h
                        best_val = exact_val
        except Exception:
            pass
            
    return best_h

# EVOLVE-BLOCK-END
