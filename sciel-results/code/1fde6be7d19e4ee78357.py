import numpy as np
import time
import scipy.optimize as opt

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

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

    def smooth_max(arr, alpha):
        """Log-sum-exp smooth maximum."""
        m = np.max(arr)
        return m + (1.0 / alpha) * np.log(np.sum(np.exp(alpha * (arr - m))))

    def parametric_to_heights(params):
        """
        params: [sigma, a0, a1, a2, ..., aK]
        h(x) = exp(-x^2/(2*sigma^2)) * (a0 + a1*cos(2*pi*x) + a2*cos(4*pi*x) + ...)
        evaluated on N equal intervals on [-1/4, 1/4]
        Returns nonnegative normalized heights.
        """
        sigma = np.clip(params[0], 0.01, 0.5)
        coeffs = params[1:]  # a0, a1, ..., aK
        
        # Grid points (interval centers)
        x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
        
        # Gaussian envelope
        gauss = np.exp(-x**2 / (2.0 * sigma**2))
        
        # Even cosine polynomial
        # cos(2*pi*k*x) for k = 0, 1, 2, ...
        # For k=0: cos(0) = 1
        # For k>=1: cos(2*pi*k*x)
        poly = np.ones_like(x)
        for k, a in enumerate(coeffs):
            if k == 0:
                poly = poly + a  # a0 * 1
            else:
                poly = poly + a * np.cos(2.0 * np.pi * k * x)
        
        h = gauss * poly
        h = np.maximum(h, 0.0)  # Ensure non-negative
        
        total = np.sum(h)
        if total > 0:
            h = h / total
        else:
            h = np.ones(N) / N
        
        return h

    def objective(params, alpha):
        """
        Objective: smooth_max(autoconvolution) with log-sum-exp.
        params: [sigma, a0, a1, ..., aK]
        Returns (value, gradient) via numerical differentiation.
        """
        h = parametric_to_heights(params)
        
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        power = np.abs(H) ** 2
        acorr = np.fft.irfft(power, n=size)
        
        val = smooth_max(acorr, alpha)
        
        # Numerical gradient (central difference)
        eps = 1e-6
        grad = np.zeros_like(params)
        for i in range(len(params)):
            p_plus = params.copy()
            p_plus[i] += eps
            h_plus = parametric_to_heights(p_plus)
            H_plus = np.fft.rfft(h_plus, n=size)
            acorr_plus = np.fft.irfft(np.abs(H_plus)**2, n=size)
            val_plus = smooth_max(acorr_plus, alpha)
            
            p_minus = params.copy()
            p_minus[i] -= eps
            h_minus = parametric_to_heights(p_minus)
            H_minus = np.fft.rfft(h_minus, n=size)
            acorr_minus = np.fft.irfft(np.abs(H_minus)**2, n=size)
            val_minus = smooth_max(acorr_minus, alpha)
            
            grad[i] = (val_plus - val_minus) / (2.0 * eps)
        
        return val, grad

    def gradient_descent(params_init, alpha, lr, max_iter, tol=1e-10):
        """Simple gradient descent with momentum."""
        params = params_init.copy()
        momentum = np.zeros_like(params)
        best_params = params.copy()
        best_val = float('inf')
        v = np.zeros_like(params)
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            val, grad = objective(params, alpha)
            
            if val < best_val:
                best_val = val
                best_params = params.copy()
            
            # Momentum update
            v = 0.9 * v - lr * grad
            params = params + v
            
            # Clip sigma to valid range
            params[0] = np.clip(params[0], 0.01, 0.5)
            
            if np.max(np.abs(grad)) < tol:
                break
        
        return best_params, best_val

    def local_refine_time_domain(h_init, max_iter=50):
        """
        Local refinement in time domain using L-BFGS-B on log-softmax parameterization.
        h = softmax(x) ensures non-negativity and sums to 1.
        """
        def time_objective(x, alpha):
            h = np.exp(x - np.max(x))
            h = h / np.sum(h)
            
            size = 2 * N
            H = np.fft.rfft(h, n=size)
            power = np.abs(H) ** 2
            acorr = np.fft.irfft(power, n=size)
            
            val = smooth_max(acorr, alpha)
            
            # Numerical gradient
            eps = 1e-7
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                h_plus = np.exp(x_plus - np.max(x_plus))
                h_plus = h_plus / np.sum(h_plus)
                H_plus = np.fft.rfft(h_plus, n=size)
                acorr_plus = np.fft.irfft(np.abs(H_plus)**2, n=size)
                val_plus = smooth_max(acorr_plus, alpha)
                grad[i] = (val_plus - val) / eps
            
            return val, grad
        
        x0 = np.log(h_init + 1e-12)
        res = opt.minimize(
            time_objective,
            x0,
            args=(200,),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': 1e-12}
        )
        h_opt = np.exp(res.x - np.max(res.x))
        h_opt = h_opt / np.sum(h_opt)
        return h_opt

    # === Main optimization loop ===
    np.random.seed(42)
    
    best_h = None
    best_m = float('inf')
    
    # Number of cosine coefficients
    K = 4  # a0 through a4
    
    # Initial configurations: [sigma, a0, a1, a2, a3, a4]
    # sigma controls Gaussian width; a's control cosine shape
    initial_configs = [
        [0.15, 1.0, 0.0, 0.0, 0.0, 0.0],   # Pure Gaussian
        [0.12, 1.0, 0.5, 0.0, 0.0, 0.0],   # Gaussian + cos term
        [0.18, 1.0, -0.3, 0.2, 0.0, 0.0],  # Gaussian + alternating
        [0.10, 1.0, 0.8, -0.4, 0.3, 0.0],  # More complex
        [0.20, 1.0, 0.3, 0.3, -0.2, 0.1],  # Broader Gaussian
        [0.13, 1.0, 0.6, 0.2, 0.2, -0.1],  # Moderate
    ]
    
    alphas = [50, 100, 200]
    
    for config in initial_configs:
        if time.time() > deadline - 40:
            break
        
        for alpha in alphas:
            if time.time() > deadline - 30:
                break
            
            try:
                params_init = np.array(config, dtype=np.float64)
                
                # Gradient descent with multiple learning rates
                for lr in [0.001, 0.005, 0.01]:
                    if time.time() > deadline - 25:
                        break
                    
                    params_opt, val_opt = gradient_descent(
                        params_init, alpha, lr, max_iter=200
                    )
                    
                    h_candidate = parametric_to_heights(params_opt)
                    m_candidate = compute_metric(h_candidate)
                    
                    if m_candidate < best_m:
                        best_m = m_candidate
                        best_h = h_candidate.copy()
                
                # Local refinement in time domain
                if time.time() > deadline - 20:
                    continue
                    
                h_refined = local_refine_time_domain(h_candidate, max_iter=30)
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
