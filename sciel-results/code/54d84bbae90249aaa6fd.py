import numpy as np
import time
from scipy.optimize import differential_evolution

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    def autoconv_lin(x):
        # Linear autoconvolution: x * x (full)
        # np.correlate(x, x, 'full') gives [x0*x0, x0*x1+..., ...]
        # Actually, autoconvolution is usually x * x.
        # np.convolve(x, x, 'full') is equivalent to correlation if x is symmetric?
        # No. Correlation: sum x[i] y[i+k]. Convolution: sum x[i] y[k-i].
        # For auto-correlation/convolution of a real signal, they are related by reversal.
        # The task says "autoconvolution".
        # Let's check the seed: `np.correlate(x, x, mode='full')`.
        # So we must use correlation to match the seed's metric definition.
        return np.correlate(x, x, mode='full')

    def metric(x):
        s = np.sum(x)
        if s == 0:
            return float('inf')
        c = autoconv_lin(x)
        max_c = np.max(np.abs(c))
        return max_c / (s * s)

    # 1. Baseline: All ones
    x_ones = np.ones(N)
    best_x = x_ones.copy()
    best_score = metric(x_ones)
    
    # 2. Spectral Parameterization:
    # Represent x(t) = a0 + sum_{k=1}^{M} [ a_k cos(2*pi*k*t/N) + b_k sin(2*pi*k*t/N) ]
    # This ensures x is a sum of low-frequency sinusoids.
    # Low-frequency signals have slow variation, leading to high integral (sum) 
    # relative to local peaks? 
    # Actually, we want to maximize |sum x| / max|autocorr|.
    # For x = 1 + small perturbations, sum ~ N, autocorr peak ~ N + small.
    # The ratio is ~ 1.
    # If we use a chirp or flat spectrum, we might reduce the peak of autocorrelation 
    # relative to the square of the sum?
    # Let's try optimizing the coefficients of a low-frequency basis.
    
    # We will use differential evolution to optimize the coefficients.
    # Dimensions: 1 (DC) + 2*M (AC).
    # Let M = 10. Total dims = 21.
    M = 10
    dims = 1 + 2 * M
    
    def eval_params(params):
        a0 = params[0]
        coeffs = params[1:]
        t = np.arange(N)
        x = np.full(N, a0)
        for k in range(1, M + 1):
            a_k = coeffs[2*(k-1)]
            b_k = coeffs[2*(k-1)+1]
            x += a_k * np.cos(2*np.pi*k*t/N) + b_k * np.sin(2*np.pi*k*t/N)
        
        # Normalize to avoid overflow/underflow and fix scale invariance?
        # The metric is scale invariant: metric(c*x) = metric(x).
        # So we can fix the norm or the sum.
        # Let's fix sum(x) = 1 to remove the scaling degree of freedom.
        s = np.sum(x)
        if abs(s) < 1e-12:
            return 1e10
        x = x / s
        # Now sum(x) = 1.
        # Metric: max|autocorr| / 1^2 = max|autocorr|.
        c = autoconv_lin(x)
        return np.max(np.abs(c))

    # Bounds for coefficients
    # a0 should be around 1/N if we want sum=1? No, we normalize after.
    # So coefficients can be in [-1, 1] roughly.
    bounds = [(-5, 5)] + [(-1, 1)] * (2 * M)
    
    # Run Differential Evolution
    try:
        result = differential_evolution(
            eval_params, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-4, 
            seed=42,
            polish=True,
            # Callback to check time
            callback=lambda xk, convergence: (time.time() > deadline)
        )
        
        if time.time() < deadline:
            if result.fun < best_score:
                # Reconstruct x
                a0 = result.x[0]
                coeffs = result.x[1:]
                t = np.arange(N)
                x_new = np.full(N, a0)
                for k in range(1, M + 1):
                    a_k = coeffs[2*(k-1)]
                    b_k = coeffs[2*(k-1)+1]
                    x_new += a_k * np.cos(2*np.pi*k*t/N) + b_k * np.sin(2*np.pi*k*t/N)
                s_new = np.sum(x_new)
                if abs(s_new) > 1e-12:
                    x_new = x_new / s_new
                    score_new = metric(x_new)
                    if score_new < best_score:
                        best_score = score_new
                        best_x = x_new
    except Exception as e:
        pass

    # 3. Local Search / Polish on the best found so far
    # Use L-BFGS-B on the continuous parameters if we have a good start.
    # Or simple random restarts with local optimization.
    
    # Let's try a few random spectral seeds and polish them.
    if time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            np.random.seed(restart)
            init_params = np.random.uniform(-1, 1, size=dims)
            init_params[0] = 1.0 # Start with significant DC
            
            # Use L-BFGS-B
            from scipy.optimize import minimize
            
            def objective_and_grad(params):
                # For L-BFGS-B we need gradient.
                # Numerical gradient is expensive.
                # Let's just use Nelder-Mead or Powell which don't need gradients.
                return eval_params(params)
            
            res = minimize(
                objective_and_grad, 
                init_params, 
                method='Powell', 
                options={'maxiter': 200, 'xatol': 1e-6}
            )
            
            if time.time() > deadline - 2:
                break
                
            if res.fun < best_score:
                a0 = res.x[0]
                coeffs = res.x[1:]
                t = np.arange(N)
                x_new = np.full(N, a0)
                for k in range(1, M + 1):
                    a_k = coeffs[2*(k-1)]
                    b_k = coeffs[2*(k-1)+1]
                    x_new += a_k * np.cos(2*np.pi*k*t/N) + b_k * np.sin(2*np.pi*k*t/N)
                s_new = np.sum(x_new)
                if abs(s_new) > 1e-12:
                    x_new = x_new / s_new
                    score_new = metric(x_new)
                    if score_new < best_score:
                        best_score = score_new
                        best_x = x_new

    return best_x
