import numpy as np
import time
import scipy.optimize as opt

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    def compute_metric(h):
        """Compute normalized max of autoconvolution: max(f*f) / (sum(h))^2"""
        h = np.asarray(h, dtype=np.float64)
        total = np.sum(h)
        if total == 0:
            return float('inf')
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        acorr = np.fft.irfft(H * np.conj(H), n=size)
        peak = np.max(acorr)
        return peak / (total * total)

    def spectral_to_heights(cutoff, window_coeffs):
        """
        Convert spectral parameters to step function heights.
        cutoff: normalized cutoff frequency [0, 0.5]
        window_coeffs: coefficients for spectral window (low-order polynomial)
        """
        # Frequency bins for rfft of length N
        freqs = np.fft.rfftfreq(N) * 2  # normalized to [0, 1] where 0.5 is Nyquist
        
        # Base spectral shape: sinc-like function truncated at cutoff
        # Ideal low-pass has |H(f)| = 1 for |f| < cutoff, 0 otherwise
        # Smooth it with a window function
        base_spectrum = np.ones_like(freqs)
        
        # Apply cutoff with smooth transition
        transition_width = 0.05
        smooth_cutoff = np.clip((cutoff - freqs) / transition_width + 0.5, 0, 1)
        base_spectrum *= smooth_cutoff
        
        # Apply spectral window (polynomial in frequency)
        if len(window_coeffs) > 0:
            # window_coeffs[0] + window_coeffs[1]*f + window_coeffs[2]*f^2 + ...
            f = freqs
            window_val = np.ones_like(f)
            for i, c in enumerate(window_coeffs):
                window_val = window_val * (1 + c * f**i)
            base_spectrum *= window_val
        
        # Ensure non-negative spectrum
        base_spectrum = np.maximum(base_spectrum, 0)
        
        # Inverse FFT to get time-domain heights
        # Need to construct full FFT from rfft
        H_full = np.zeros(N, dtype=complex)
        H_full[:len(base_spectrum)] = base_spectrum
        # Make it conjugate symmetric
        if N % 2 == 0:
            H_full[N//2:] = np.conj(H_full[N//2:0:-1])
        else:
            H_full[(N+1)//2:] = np.conj(H_full[(N+1)//2:0:-1])
        
        # Inverse FFT
        h = np.fft.ifft(H_full).real
        h = np.maximum(h, 0)  # Ensure non-negative
        
        # Normalize
        total = np.sum(h)
        if total > 0:
            h = h / total
        else:
            h = np.ones(N) / N
        
        return h

    def spectral_objective(params, alpha):
        """
        Objective function in spectral domain.
        params[0]: cutoff frequency
        params[1:]: window coefficients
        """
        cutoff = np.clip(params[0], 0.05, 0.5)
        window_coeffs = params[1:]
        
        h = spectral_to_heights(cutoff, window_coeffs)
        
        # Compute auto-correlation
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        power = np.abs(H) ** 2
        acorr = np.fft.irfft(power, n=size)
        
        # Smooth max using log-sum-exp
        max_acorr = np.max(acorr)
        log_sum_exp = max_acorr + (1.0 / alpha) * np.log(np.sum(np.exp(alpha * (acorr - max_acorr))))
        
        value = log_sum_exp
        
        # Gradient computation (numerical for stability)
        eps = 1e-7
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            val_plus = spectral_objective_value_only(params_plus, alpha)
            grad[i] = (val_plus - value) / eps
        
        return value, grad

    def spectral_objective_value_only(params, alpha):
        """Value only, no gradient."""
        cutoff = np.clip(params[0], 0.05, 0.5)
        window_coeffs = params[1:]
        
        h = spectral_to_heights(cutoff, window_coeffs)
        
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        power = np.abs(H) ** 2
        acorr = np.fft.irfft(power, n=size)
        
        max_acorr = np.max(acorr)
        log_sum_exp = max_acorr + (1.0 / alpha) * np.log(np.sum(np.exp(alpha * (acorr - max_acorr))))
        
        return log_sum_exp

    def local_refine(h_init):
        """Local refinement in time domain using L-BFGS-B."""
        def time_objective(x, alpha):
            h = np.exp(x)
            h = h / np.sum(h)
            
            size = 2 * N
            H = np.fft.rfft(h, n=size)
            power = np.abs(H) ** 2
            acorr = np.fft.irfft(power, n=size)
            
            max_acorr = np.max(acorr)
            log_sum_exp = max_acorr + (1.0 / alpha) * np.log(np.sum(np.exp(alpha * (acorr - max_acorr))))
            
            # Numerical gradient
            eps = 1e-8
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                h_plus = np.exp(x_plus)
                h_plus = h_plus / np.sum(h_plus)
                H_plus = np.fft.rfft(h_plus, n=size)
                acorr_plus = np.fft.irfft(np.abs(H_plus)**2, n=size)
                max_plus = np.max(acorr_plus)
                lse_plus = max_plus + (1.0/alpha) * np.log(np.sum(np.exp(alpha*(acorr_plus - max_plus))))
                grad[i] = (lse_plus - log_sum_exp) / eps
            
            return log_sum_exp, grad
        
        x0 = np.log(h_init + 1e-10)
        res = opt.minimize(
            time_objective,
            x0,
            args=(100,),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 30, 'ftol': 1e-10}
        )
        h_opt = np.exp(res.x)
        h_opt = h_opt / np.sum(h_opt)
        return h_opt

    # Spectral optimization
    np.random.seed(42)
    
    best_h = None
    best_m = float('inf')
    
    # Try different initial spectral configurations
    n_window_coeffs = 3
    alphas = [20, 50, 100]
    
    initial_configs = [
        [0.3, 0, 0, 0],  # Moderate cutoff
        [0.2, 0, 0, 0],  # Lower cutoff
        [0.4, 0, 0, 0],  # Higher cutoff
        [0.3, -1, 0.5, 0],  # With window
        [0.25, 0.5, -0.3, 0.1],  # Different window
    ]
    
    for config in initial_configs:
        if time.time() > deadline - 30:  # Leave time for refinement
            break
        
        params = np.array(config, dtype=np.float64)
        
        for alpha in alphas:
            if time.time() > deadline - 30:
                break
            
            try:
                res = opt.minimize(
                    spectral_objective,
                    params,
                    args=(alpha,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0.05, 0.5)] + [(-5, 5)] * n_window_coeffs,
                    options={'maxiter': 40, 'ftol': 1e-10}
                )
                
                cutoff = np.clip(res.x[0], 0.05, 0.5)
                window_coeffs = res.x[1:]
                h_candidate = spectral_to_heights(cutoff, window_coeffs)
                
                # Local refinement
                h_refined = local_refine(h_candidate)
                m_refined = compute_metric(h_refined)
                
                if m_refined < best_m:
                    best_m = m_refined
                    best_h = h_refined.copy()
                
            except Exception as e:
                continue
    
    # Fallback to uniform if optimization fails
    uniform_h = np.ones(N) / N
    uniform_m = compute_metric(uniform_h)
    
    if best_h is None or uniform_m < best_m:
        return list(uniform_h)
        
    return list(best_h)

# EVOLVE-BLOCK-END
