import numpy as np
import scipy.optimize
import time

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin from 150s limit
    
    N = 100
    target_sum = N / 2.0
    
    def get_correlations_linear(h):
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero
    
    def score_h(h):
        c_all = get_correlations_linear(h)
        return np.max(c_all)
    
    def lse_smooth_max(c, beta):
        max_c = np.max(c)
        if max_c > 500:
            max_c = 500
        return max_c + np.log(np.sum(np.exp(beta * (c - max_c)))) / beta
    
    def objective(h, beta, lambda_sum):
        c_all = get_correlations_linear(h)
        lse_val = lse_smooth_max(c_all, beta)
        sum_diff = np.sum(h) - target_sum
        pen = lambda_sum * sum_diff**2
        return lse_val + pen

    def optimize_instance(h0, beta, lambda_sum, max_iter):
        bounds = [(0, 1) for _ in range(N)]
        
        def obj(h):
            return objective(h, beta, lambda_sum)
        
        try:
            res = scipy.optimize.minimize(
                obj,
                h0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10, 'maxls': 50}
            )
            h_opt = res.x
        except Exception:
            h_opt = h0.copy()
        
        h_opt = np.clip(h_opt, 0, 1)
        
        # Correct sum to be exactly N/2
        for _ in range(20):
            current_sum = np.sum(h_opt)
            diff = target_sum - current_sum
            if abs(diff) < 1e-14:
                break
            h_opt += diff / N
            h_opt = np.clip(h_opt, 0, 1)
        
        return h_opt

    # Phase 1: Combinatorial Search (Simulated Annealing) on Binary Sequences
    # This finds a good structural "skeleton" with low max overlap.
    
    rng = np.random.default_rng(42)
    
    # Initialize with a random binary sequence with exactly N/2 ones
    h_bin = np.zeros(N)
    indices = rng.permutation(N)
    h_bin[indices[:N//2]] = 1.0
    
    best_h_bin = h_bin.copy()
    best_score_bin = score_h(h_bin)
    
    # Simulated Annealing
    T = 10.0
    cooling_rate = 0.995
    min_T = 0.1
    
    while T > min_T and time.time() - start_time < time_limit - 30:
        # Pick a 1 and a 0 to swap
        ones = np.where(h_bin == 1.0)[0]
        zeros = np.where(h_bin == 0.0)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            break
            
        i = ones[rng.integers(len(ones))]
        j = zeros[rng.integers(len(zeros))]
        
        # Swap
        h_bin[i], h_bin[j] = h_bin[j], h_bin[i]
        
        new_score = score_h(h_bin)
        delta = new_score - best_score_bin
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            best_h_bin = h_bin.copy()
            best_score_bin = new_score
        else:
            # Revert
            h_bin[i], h_bin[j] = h_bin[j], h_bin[i]
            
        T *= cooling_rate
        
    # Phase 2: Continuous Refinement
    # Use the best binary sequence as a starting point for L-BFGS-B
    # This allows the values to move away from 0/1 to potentially lower the max peak further.
    
    h0 = best_h_bin.copy()
    
    # Run a few refinements with different beta
    best_h = h0
    best_score = best_score_bin
    
    for beta in [10.0, 20.0, 50.0]:
        if time.time() - start_time > time_limit - 10:
            break
            
        h_opt = optimize_instance(h0, beta, 1000.0, 200)
        
        if np.isclose(np.sum(h_opt), target_sum, atol=1e-8) and \
           np.all(h_opt >= -1e-10) and np.all(h_opt <= 1 + 1e-10):
            
            current_score = score_h(h_opt)
            if current_score < best_score:
                best_score = current_score
                best_h = h_opt.copy()
                h0 = best_h.copy() # Use improved solution for next refinement
    
    # Phase 3: High-precision polish pass
    # Use higher beta and tighter tolerances to refine further
    if time.time() - start_time < time_limit - 5:
        for beta in [100.0, 200.0]:
            if time.time() - start_time > time_limit - 3:
                break
                
            h_opt = optimize_instance(best_h, beta, 10000.0, 500)
            
            if np.isclose(np.sum(h_opt), target_sum, atol=1e-8) and \
               np.all(h_opt >= -1e-10) and np.all(h_opt <= 1 + 1e-10):
                
                current_score = score_h(h_opt)
                if current_score < best_score:
                    best_score = current_score
                    best_h = h_opt.copy()
    
    # Phase 4: DFT-based spectral refinement
    # Parameterize h via inverse DFT of complex spectrum with fixed DC
    # Minimize peak correlation via gradient descent on spectral phases
    
    if time.time() - start_time < time_limit - 2:
        # Get current best as starting point for spectral optimization
        h_start = best_h.copy()
        
        # Compute DFT
        H = np.fft.fft(h_start)
        
        # DC component is fixed: sum(h) = N/2 => H[0] = N/2
        H_dc = H[0]
        
        # Extract phases and magnitudes for non-DC components
        # For real h, H[k] = conj(H[N-k]), so we optimize phases of first N/2 components
        # H[0] is real and positive (DC)
        # H[N/2] is real (Nyquist)
        
        # We'll optimize the phases of H[1:N//2] and keep magnitudes fixed initially
        # Then allow magnitudes to vary too, with constraint that h stays in [0,1]
        
        n_half = N // 2
        phases = np.angle(H[1:n_half])
        mags = np.abs(H[1:n_half])
        nyq_real = H[n_half].real  # Nyquist is real
        
        def h_from_spectrum(ph, mg, nyq_r):
            """Reconstruct h from spectral parameters"""
            H_spec = np.zeros(N, dtype=complex)
            H_spec[0] = target_sum  # DC = sum = N/2
            for k in range(1, n_half):
                H_spec[k] = mg[k-1] * np.exp(1j * ph[k-1])
                H_spec[N-k] = np.conj(H_spec[k])
            H_spec[n_half] = nyq_r
            h_rec = np.fft.ifft(H_spec).real
            return h_rec
        
        # Gradient descent on phases and magnitudes
        # Objective: LSE smooth max of correlations + penalty for out-of-bounds
        
        def spectral_objective(params, beta):
            """params = [phases..., mags..., nyq_real]"""
            ph = params[:n_half-1]
            mg = params[n_half-1:2*(n_half-1)]
            nyq_r = params[2*(n_half-1)]
            
            h_rec = h_from_spectrum(ph, mg, nyq_r)
            
            # Penalty for out-of-bounds
            bound_pen = np.sum(np.maximum(0, h_rec - 1)**2) + np.sum(np.maximum(0, -h_rec)**2)
            
            # Correlation objective
            c_all = get_correlations_linear(h_rec)
            lse_val = lse_smooth_max(c_all, beta)
            
            return lse_val + 100.0 * bound_pen
        
        # Use scipy minimize on spectral parameters
        params0 = np.concatenate([phases, mags, [nyq_real]])
        
        try:
            res_spec = scipy.optimize.minimize(
                lambda p: spectral_objective(p, 50.0),
                params0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            params_opt = res_spec.x
            ph_opt = params_opt[:n_half-1]
            mg_opt = params_opt[n_half-1:2*(n_half-1)]
            nyq_opt = params_opt[2*(n_half-1)]
            
            h_spec = h_from_spectrum(ph_opt, mg_opt, nyq_opt)
            h_spec = np.clip(h_spec, 0, 1)
            
            # Correct sum
            for _ in range(20):
                diff = target_sum - np.sum(h_spec)
                if abs(diff) < 1e-14:
                    break
                h_spec += diff / N
                h_spec = np.clip(h_spec, 0, 1)
            
            current_score = score_h(h_spec)
            if current_score < best_score:
                best_score = current_score
                best_h = h_spec.copy()
                
        except Exception:
            pass
    
    # Final constraint check and adjustment
    best_h = np.clip(best_h, 0, 1)
    for _ in range(10):
        diff = target_sum - np.sum(best_h)
        if abs(diff) < 1e-14:
            break
        best_h += diff / N
        best_h = np.clip(best_h, 0, 1)
        
    return best_h
