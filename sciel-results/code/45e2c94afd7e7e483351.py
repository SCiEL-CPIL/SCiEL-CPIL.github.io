"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def construct():
    t_start = time.time()
    deadline = t_start + 130.0
    
    # Strategy: Use a smooth surrogate objective (log-sum-exp of squared autoconv)
    # to allow gradient-based optimization (L-BFGS-B) in the continuous domain.
    # This finds better basins than discrete search. Then, project to binary signs
    # and refine with discrete local search.
    
    # Precompute FFT size
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
        
    # 1. Define smooth surrogate objective
    # We want to minimize max_k |ac_k| / (sum h)^2
    # Let S = sum h.
    # We can minimize log(sum_k exp(beta * ac_k^2)) - 2*log(S)
    # This is differentiable.
    # Note: ac_k can be negative, but we square it.
    # Actually, max |x| ~ (1/beta) log(sum exp(beta |x|)) is hard due to abs.
    # But max x^2 is equivalent to max |x| in terms of which index is the max.
    # So we can use log-sum-exp on ac_k^2.
    
    beta = 50.0  # Sharpness of the max approximation
    
    def smooth_objective(h_flat):
        h = h_flat.reshape(N)
        H = np.fft.rfft(h, fsize)
        ac = np.fft.irfft(H * H, fsize)
        # Ensure we only look at the valid part of the autoconvolution?
        # Linear autoconvolution length is 2N-1.
        # irfft returns fsize/2 + 1 points? No, irfft of size fsize returns fsize/2+1 if input is rfft?
        # np.fft.irfft(H*H, fsize) returns an array of length fsize/2 + 1.
        # Wait, rfft returns fsize//2 + 1 elements.
        # irfft with n=fsize returns fsize elements?
        # Let's check: np.fft.irfft(y, n) returns n elements.
        # So ac has length fsize.
        # The valid autoconvolution is the first 2N-1 elements.
        # The rest are wrap-around if we considered circular, but rfft/irfft with n=fsize > 2N-1 
        # effectively computes linear convolution if we zero-padded correctly?
        # Yes, rfft(h, fsize) zero-pads h to fsize.
        # So H is the DFT of the zero-padded h.
        # H*H is the DFT of the circular convolution of zero-padded h with itself.
        # Since fsize >= 2N-1, the circular convolution is the same as linear convolution.
        # So ac[0:2N-1] is the linear autoconvolution.
        
        ac_valid = ac[:size]
        ac_sq = ac_valid * ac_valid
        
        # Log-sum-exp for numerical stability
        m = np.max(ac_sq)
        if m == 0:
            lse = 0
        else:
            lse = m + np.log(np.sum(np.exp((ac_sq - m) * beta)))
            
        s = np.sum(h)
        if abs(s) < 1e-6:
            return 1e10
        log_s_sq = 2 * np.log(abs(s))
        
        return lse / beta - log_s_sq

    def smooth_gradient(h_flat):
        # We could implement gradient, but L-BFGS-B can use numerical gradient.
        # Given N=400, numerical gradient is 400 evaluations of the objective.
        # Each objective is one FFT. This is fast.
        return None # Let scipy handle it

    # 2. Generate diverse initial points
    rng = random.Random(42)
    
    inits = []
    # All ones
    inits.append(np.ones(N))
    # Random
    for _ in range(5):
        inits.append(np.array([1.0 if rng.random() > 0.5 else -1.0 for _ in range(N)]))
    # Chirp
    x = np.arange(N)
    inits.append(np.sign(np.cos(np.pi * x**2 / N)))
    
    best_h = inits[0].copy()
    best_val = _score(best_h)
    
    # 3. Optimize each initial point using L-BFGS-B
    from scipy.optimize import minimize
    
    for h_init in inits:
        if time.time() > deadline - 20:
            break
            
        # Flatten for scipy
        x0 = h_init.copy().flatten()
        
        # Bounds: Let's allow values in [-10, 10] to let it explore magnitudes,
        # though the objective is scale invariant, the smoothness might benefit from it?
        # Actually, scale invariance means the gradient along the radial direction is zero?
        # No, the denominator (sum h)^2 changes scale.
        # If we scale h by c, ac scales by c^2, sum h scales by c.
        # Ratio is (c^2 * ac) / (c^2 * sum^2) = invariant.
        # So the objective is constant along rays from origin.
        # L-BFGS-B might struggle with flat directions.
        # However, the "max" part is not strictly scale invariant in the log-sum-exp approximation
        # unless we are very careful.
        # Let's fix the norm or sum to break symmetry?
        # Or just let it run. The flat direction might cause issues, but L-BFGS-B is robust.
        
        try:
            res = minimize(
                smooth_objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=[(-10, 10)] * N, 
                options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            
            h_opt = res.x.reshape(N)
            
            # Project to binary signs
            h_bin = np.sign(h_opt)
            # Handle zeros
            h_bin[h_bin == 0] = 1.0
            
            # Ensure non-zero sum
            if np.sum(h_bin) == 0:
                h_bin[0] = 1.0
                if np.sum(h_bin) == 0: h_bin[1] = 1.0

            val = _score(h_bin)
            if val < best_val:
                best_val = val
                best_h = h_bin.copy()
                
        except Exception as e:
            pass
            
    # 4. Refine the best binary solution with discrete local search
    # Use the same efficient FFT update logic as before
    h = best_h.copy()
    H = np.fft.rfft(h, fsize)
    k_indices = np.arange(fsize // 2 + 1)
    twiddles = np.exp(-1j * 2 * np.pi * np.outer(np.arange(N), k_indices) / fsize)
    
    current_val = _score(h)
    
    # Greedy hill climb
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            old_val = h[i]
            new_val = -old_val
            h[i] = new_val
            delta = -2 * old_val
            H += delta * twiddles[i]
            
            ac = np.fft.irfft(H * H, fsize)
            new_ac_max = np.max(np.abs(ac[:size]))
            s = np.sum(h)
            if abs(s) < 1e-9:
                new_score = 1e10
            else:
                new_score = new_ac_max / (s * s)
            
            if new_score < current_val:
                current_val = new_score
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
                improved = True
            else:
                h[i] = old_val
                H -= delta * twiddles[i]
                
    # 5. Final Simulated Annealing for further refinement if time permits
    if time.time() < deadline - 10:
        h = best_h.copy()
        H = np.fft.rfft(h, fsize)
        current_val = _score(h)
        T = 0.5
        for _ in range(5000):
            if time.time() > deadline - 2:
                break
            i = rng.randint(0, N - 1)
            old_val = h[i]
            new_val = -old_val
            h[i] = new_val
            delta = -2 * old_val
            H += delta * twiddles[i]
            
            ac = np.fft.irfft(H * H, fsize)
            new_ac_max = np.max(np.abs(ac[:size]))
            s = np.sum(h)
            if abs(s) < 1e-9:
                new_score = 1e10
            else:
                new_score = new_ac_max / (s * s)
            
            delta_score = new_score - current_val
            if delta_score < 0 or rng.random() < np.exp(-delta_score / T):
                current_val = new_score
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                h[i] = old_val
                H -= delta * twiddles[i]
            T *= 0.999
            
    return best_h

# EVOLVE-BLOCK-END
