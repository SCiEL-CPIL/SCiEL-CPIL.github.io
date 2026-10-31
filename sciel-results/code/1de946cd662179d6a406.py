import numpy as np
import time
from scipy import optimize
from scipy.fft import fft, ifft

N = 600

def _fft_convolve(h):
    """FFT-based autoconvolution for speed."""
    n = len(h)
    size = 2 * n
    H = fft(h, size)
    conv = np.real(ifft(H * H, size)[:2*n-1])
    return conv

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    conv = _fft_convolve(h)
    total = np.sum(h)
    if total < 1e-10:
        return 0.0
    return float(np.max(conv) / (total * total))

def _soft_max_obj_grad(h, beta):
    """
    Compute smooth max autoconvolution value and gradient w.r.t. h.
    Uses log-sum-exp with temperature 1/beta.
    """
    h = np.maximum(h, 1e-10)
    total = np.sum(h)
    if total < 1e-10:
        return 1e10, np.zeros_like(h)
    
    # Normalize
    h_norm = h / total
    conv = _fft_convolve(h_norm)
    
    m = np.max(conv)
    shifted = (conv - m) * beta
    shifted = np.clip(shifted, -50, 50)
    exp_conv = np.exp(shifted)
    sum_exp = np.sum(exp_conv)
    weights = exp_conv / sum_exp
    
    soft_max_val = m + np.log(sum_exp) / beta
    
    # Gradient of soft_max w.r.t. h_norm via FFT
    size = 2 * N
    W = fft(weights, size)
    Hn = fft(h_norm, size)
    grad_norm = np.real(ifft(W * Hn, size))[:N]
    grad_norm *= 2.0
    
    # Chain rule for f(h) = soft_max(conv(h/S))
    sum_grad_u = np.dot(grad_norm, h_norm)
    grad_h = (grad_norm - sum_grad_u) / total
    
    return soft_max_val, grad_h

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def time_left():
        return deadline - time.time()

    best_score = np.inf
    best_h = np.ones(N)
    best_score = compute_autoconv_max(best_h)
    
    # Strategy 1: GMM-based global search
    K = 4
    
    def gmm_to_h(params):
        """Convert GMM parameters to discrete heights."""
        h = np.zeros(N)
        i = np.arange(N, dtype=float)
        for k in range(K):
            w_k = max(params[3*k], 0.01)
            mu_k = params[3*k+1]
            sig_k = max(params[3*k+2], 1.0)
            h += w_k * np.exp(-((i - mu_k)**2) / (2 * sig_k**2))
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s > 1e-10:
            h /= s
        else:
            h = np.ones(N) / N
        return h

    def gmm_objective(params):
        h = gmm_to_h(params)
        return compute_autoconv_max(h)

    if time_left() > 50:
        bounds = []
        for k in range(K):
            bounds.append((0.1, 5.0))
            bounds.append((0, N-1))
            bounds.append((5.0, N/3.0))
        
        try:
            max_iter_de = int(min(40, time_left() // 2.0))
            if max_iter_de < 10:
                max_iter_de = 10
            
            de_result = optimize.differential_evolution(
                gmm_objective,
                bounds=bounds,
                seed=42,
                maxiter=max_iter_de,
                popsize=12,
                tol=1e-5,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube',
                updating='deferred'
            )
            if de_result.fun < best_score:
                h_de = gmm_to_h(de_result.x)
                best_score = de_result.fun
                best_h = h_de.copy()
        except Exception:
            pass
    
    # Strategy 2: Annealed L-BFGS-B with analytic gradient
    if time_left() > 25 and best_h is not None:
        betas = [10.0, 50.0, 200.0, 500.0, 1000.0, 2000.0]
        
        current_h = best_h.copy()
        current_val = best_score
        
        for beta in betas:
            if time_left() < 5:
                break
            
            try:
                res = optimize.minimize(
                    lambda h: _soft_max_obj_grad(h, beta)[0],
                    current_h,
                    jac=lambda h: _soft_max_obj_grad(h, beta)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 300}
                )
                
                final_h = np.maximum(res.x, 0)
                s = np.sum(final_h)
                if s > 1e-10:
                    final_h /= s
                    final_val = compute_autoconv_max(final_h)
                    if final_val < current_val:
                        current_val = final_val
                        current_h = final_h.copy()
            except Exception:
                pass
        
        if current_val < best_score:
            best_score = current_val
            best_h = current_h.copy()
    
    # Strategy 3: Multi-restart with diverse initializations
    if time_left() > 15:
        np.random.seed(123)
        
        initializations = []
        
        # Raised cosine
        t = np.linspace(0, 1, N, endpoint=False)
        initializations.append(np.maximum(0.5 + 0.5 * np.cos(2 * np.pi * t), 0.01))
        
        # Triangular
        t2 = np.linspace(-1, 1, N)
        initializations.append(np.maximum(1 - np.abs(t2), 0.01))
        
        # Gaussian
        initializations.append(np.exp(-t2**2 * 2))
        
        # Two-lobe
        initializations.append(np.maximum(0.5 + 0.4 * np.cos(4 * np.pi * t), 0.01))
        
        # Random
        initializations.append(np.random.rand(N) + 0.1)
        
        n_restarts = int(min(len(initializations), time_left() // 4))
        
        for h0 in initializations[:n_restarts]:
            if time_left() < 6:
                break
            
            h0 = np.maximum(h0, 0.001)
            s = np.sum(h0)
            if s > 1e-10:
                h0 /= s
            
            try:
                for beta in [50.0, 200.0]:
                    if time_left() < 4:
                        break
                    res = optimize.minimize(
                        lambda h: _soft_max_obj_grad(h, beta)[0],
                        h0,
                        jac=lambda h: _soft_max_obj_grad(h, beta)[1],
                        method='L-BFGS-B',
                        options={'maxiter': 40, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 100}
                    )
                    h_new = np.maximum(res.x, 0)
                    s = np.sum(h_new)
                    if s > 1e-10:
                        h_new /= s
                        score_new = compute_autoconv_max(h_new)
                        if score_new < best_score:
                            best_score = score_new
                            best_h = h_new.copy()
            except Exception:
                pass
    
    # Final high-precision polish
    if time_left() > 8:
        try:
            res = optimize.minimize(
                lambda h: _soft_max_obj_grad(h, 5000.0)[0],
                best_h,
                jac=lambda h: _soft_max_obj_grad(h, 5000.0)[1],
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 200}
            )
            h_final = np.maximum(res.x, 0)
            s = np.sum(h_final)
            if s > 1e-10:
                h_final /= s
                score_final = compute_autoconv_max(h_final)
                if score_final < best_score:
                    best_score = score_final
                    best_h = h_final.copy()
        except Exception:
            pass
    
    return list(best_h)
