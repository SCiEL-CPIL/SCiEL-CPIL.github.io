import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 400

def _compute_c3_upper_bound(s):
    """Compute the C3 upper bound: max |autoconvolution| / (sum |s_i|)^2"""
    # Linear autocorrelation
    ac = np.correlate(s, s, mode='full')
    # ac has length 2N-1, center at N-1
    # Peak sidelobe (excluding lag 0)
    center = N - 1
    ac_sidelobes = np.concatenate([ac[:center], ac[center+1:]])
    peak = np.max(np.abs(ac_sidelobes))
    # Normalization: (sum |s_i|)^2 = N^2 for ±1 sequence
    norm = N * N
    return peak / norm

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin (150s limit)
    
    def time_remaining():
        return deadline - (time.time() - start_time)
    
    best_s = None
    best_score = np.inf
    
    # Phase 1: Spectral shaping - optimize power spectrum to minimize peak
    # Use log-sum-exp approximation of max for smooth optimization
    
    def spectral_objective(params):
        # params: real vector representing log-amplitude of power spectrum
        # We optimize over the amplitude of the FFT coefficients
        # For a real sequence, FFT is conjugate symmetric
        # params[0] = DC, params[N//2] = Nyquist (if N even)
        # params[1:N//2] determine the rest via symmetry
        
        if len(params) < N:
            return 1e10
        
        # Reconstruct power spectrum from params
        # params[i] for i in 0..N-1 represents log(|F[i]|^2) + 1
        # But we need to ensure conjugate symmetry for real signal
        
        # Simpler: optimize a continuous signal x in [-1,1]^N directly
        # then threshold
        
        x = np.tanh(params)  # Map to (-1, 1)
        
        # Compute circular autocorrelation via FFT
        fx = np.fft.fft(x)
        ac = np.real(np.fft.ifft(fx * fx))
        
        # ac[0] is the energy, we want to minimize peak of ac[1:]
        # Use log-sum-exp for smooth max approximation
        # max(x) ≈ (1/β) * log(sum(exp(β*x)))
        beta = 50.0
        sidelobes = np.abs(ac[1:])
        # Clip to avoid overflow
        sidelobes_clipped = np.clip(sidelobes, -100, 100)
        log_sum_exp = (1.0/beta) * np.log(np.sum(np.exp(beta * sidelobes_clipped)))
        
        return log_sum_exp
    
    def spectral_gradient(params):
        x = np.tanh(params)
        fx = np.fft.fft(x)
        ac = np.real(np.fft.ifft(fx * fx))
        
        beta = 50.0
        sidelobes = np.abs(ac[1:])
        sidelobes_clipped = np.clip(sidelobes, -100, 100)
        
        # Gradient of log-sum-exp
        exp_beta = np.exp(beta * sidelobes_clipped)
        sum_exp = np.sum(exp_beta)
        weights = exp_beta / sum_exp  # softmax weights
        
        # d(log_sum_exp)/d(sidelobes_clipped) = weights
        # d(sidelobes_clipped)/d(ac) = sign(ac[1:])
        # d(ac)/d(x) via FFT
        
        # ac[k] = sum_j x[j] * x[j-k] (circular)
        # d(ac[k])/d(x[i]) = x[i-k] + x[i+k] (for k != 0)
        
        # This is complex; use numerical gradient if analytical is too slow
        # For now, use a simple approach: finite differences for a few coordinates
        
        # Actually, let's use a different approach: optimize x directly with L-BFGS-B
        # and use a smooth max objective
        
        # For gradient, we need d/dx_i of sum_k |ac[k]| (smoothed)
        # This is expensive analytically. Use finite differences sparingly.
        
        # Alternative: just use the objective with L-BFGS-B's numerical gradient
        # But that's slow. Let's try a different strategy.
        
        return None  # Force numerical gradient
    
    # Phase 1: Try spectral optimization with multiple restarts
    num_restarts = 8
    for r in range(num_restarts):
        if time_remaining() < 30:
            break
        
        # Random continuous initialization
        np.random.seed(42 + r)
        x0 = np.random.uniform(-0.5, 0.5, N)
        
        try:
            res = minimize(
                spectral_objective, 
                x0, 
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50}
            )
            
            if time_remaining() < 20:
                break
            
            # Threshold to ±1
            s = np.sign(res.x)
            s[s == 0] = 1
            
            score = _compute_c3_upper_bound(s)
            
            if score < best_score:
                best_score = score
                best_s = s.copy()
                
        except Exception as e:
            pass
    
    # Phase 2: Local search refinement on the best discrete solution
    if best_s is None:
        best_s = np.random.choice([-1, 1], N).astype(float)
        best_score = _compute_c3_upper_bound(best_s)
    
    s = best_s.copy()
    current_score = best_score
    
    # Local search: try flipping bits to reduce the C3 upper bound
    max_local_iters = 500
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_local_iters:
        improved = False
        iter_count += 1
        
        if time_remaining() < 15:
            break
        
        # Try flipping each bit
        for i in range(N):
            if time_remaining() < 10:
                break
            
            # Flip bit i
            s[i] *= -1
            
            # Compute new score
            new_score = _compute_c3_upper_bound(s)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                # Revert
                s[i] *= -1
            
            if improved:
                # Continue scanning (don't restart) to find more improvements
                pass
            else:
                # If no improvement after trying all bits, break
                if i == N - 1:
                    break
    
    best_s = s
    best_score = current_score
    
    return best_s
