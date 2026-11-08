"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 600
DELTA = 0.5 / N  # width of each interval

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Helper to compute the exact normalized maximum of autoconvolution.
    # The score is the upper bound on C1.
    # For step functions with heights h[i] on intervals of width DELTA:
    # ||f||_1 = DELTA * sum(h)
    # The continuous autoconvolution (f*f)(t) is a piecewise linear function.
    # Its maximum occurs at the vertices of the convolution support.
    # For a step function defined on N intervals, the convolution is supported
    # on 2N-1 intervals (if we consider the discrete convolution of the heights).
    # However, the continuous convolution of two step functions of width DELTA
    # can be computed efficiently.
    #
    # Let's use the discrete convolution as a proxy but scaled correctly.
    # Actually, the exact maximum of the continuous autoconvolution of a step
    # function is NOT simply the max of the discrete convolution.
    #
    # Let's derive the exact formula.
    # f(x) = h[i] for x in [i*DELTA, (i+1)*DELTA).
    # (f*f)(t) = integral f(t-s) f(s) ds.
    # This is a piecewise linear function. The vertices are at t = k*DELTA.
    # The value at the vertices can be computed from the discrete convolution.
    # Let c[k] = sum_{i} h[i] h[i-k] be the discrete autoconvolution.
    # The continuous autoconvolution at t = k*DELTA is approximately:
    # (f*f)(k*DELTA) = DELTA * c[k] ? No.
    #
    # Let's test with uniform h=1.
    # N=600, DELTA=1/1200.
    # f(x)=1 on [-0.25, 0.25].
    # (f*f)(0) = integral_{-0.25}^{0.25} 1*1 dx = 0.5.
    # Discrete conv c[0] = sum h[i]h[i] = N = 600.
    # DELTA * c[0] = 0.5. This matches.
    #
    # What about the peaks between vertices?
    # The continuous convolution of two step functions is a "hat" function 
    # convolved with itself? No, it's the convolution of two piecewise constants.
    # The result is piecewise linear. The maximum must be at a vertex.
    # Why? The derivative is piecewise constant. The function is convex in each 
    # segment? No.
    # Actually, the convolution of two step functions is a piecewise linear 
    # function whose slopes are constant on each interval.
    # The maximum of a piecewise linear function is at a vertex.
    # So the maximum of (f*f)(t) is the maximum of its values at t = k*DELTA.
    #
    # Therefore, max_continuous = DELTA * max(discrete_autoconvolution).
    #
    # Let's verify with a simple case.
    # h = [1, 0]. N=2. DELTA=0.25.
    # f(x) = 1 on [-0.25, 0], 0 on [0, 0.25].
    # (f*f)(t) = integral f(t-s)f(s) ds.
    # Support of f is [-0.25, 0]. Support of f*f is [-0.5, 0].
    # (f*f)(t) is a triangle from -0.5 to 0, peak at -0.25?
    # f(t-s) is 1 if t-s in [-0.25, 0] => t in [s-0.25, s].
    # f(s) is 1 if s in [-0.25, 0].
    # Overlap length is the integral.
    # For t in [-0.5, -0.25]: overlap is t+0.5.
    # For t in [-0.25, 0]: overlap is -t.
    # Peak at t=-0.25, value 0.25.
    # Discrete conv c = [1, 0, 0]. Max c = 1.
    # DELTA * max(c) = 0.25 * 1 = 0.25. Matches.
    #
    # So the formula is:
    # max_continuous = DELTA * max(discrete_autoconvolution(h))
    # ||f||_1 = DELTA * sum(h)
    # Ratio = (DELTA * max_conv) / (DELTA * sum(h))^2
    #        = max_conv / (DELTA * (sum(h))^2)
    
    def objective_exact(h):
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
        # Compute discrete autoconvolution
        # Use FFT for speed
        n = len(h)
        size = 2 * n
        f = np.zeros(size)
        f[:n] = h
        F = np.fft.fft(f)
        autocorr = np.fft.ifft(F * np.conj(F)).real
        # The full autocorrelation is of length 2n-1, but we only need the positive lags
        # Actually, the autocorrelation computed this way gives lags from -(n-1) to n-1
        # We need the max over all lags.
        max_conv = np.max(autocorr)
        # Note: autocorr[0] is the sum of squares.
        # The formula derived: Ratio = max_conv / (DELTA * s^2)
        return max_conv / (DELTA * s * s)
    
    def soft_max(arr, beta):
        arr = np.asarray(arr, dtype=float)
        m = np.max(arr)
        if m < 1e-10:
            return m
        return m + (1.0/beta) * np.log(np.sum(np.exp(beta * (arr - m))))
    
    def objective_smooth(h, beta):
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
        n = len(h)
        size = 2 * n
        f = np.zeros(size)
        f[:n] = h
        F = np.fft.fft(f)
        autocorr = np.fft.ifft(F * np.conj(F)).real
        # Use soft max to smooth the max operation
        max_conv_smooth = soft_max(autocorr, beta)
        return max_conv_smooth / (DELTA * s * s)

    best_h = np.ones(N)
    best_score = objective_exact(best_h)
    
    bounds = [(0, None)] * N
    np.random.seed(42)
    
    # Phase 1: Optimization with smooth surrogate
    # Use increasing beta values for annealing
    betas = [10.0, 50.0, 200.0]
    
    for beta in betas:
        if time.time() > deadline - 20:
            break
        
        for restart in range(3):
            if time.time() > deadline - 20:
                break
            
            if restart == 0:
                x0 = np.ones(N)
            elif restart == 1:
                x0 = np.random.exponential(1.0, N)
            else:
                # Cosine-shaped
                x0 = np.cos(np.linspace(0, np.pi, N)) ** 2
                x0 = np.maximum(x0, 0)
            
            try:
                obj = lambda h: objective_smooth(h, beta)
                result = minimize(obj, x0, method='L-BFGS-B', 
                                 bounds=bounds, options={'maxiter': 50, 'maxfun': 150})
                
                if time.time() > deadline - 20:
                    break
                
                exact_score = objective_exact(result.x)
                if exact_score < best_score:
                    best_score = exact_score
                    best_h = np.maximum(result.x, 0)
            except Exception:
                continue
    
    # Phase 2: Final polish with high precision on the best candidate
    if time.time() < deadline - 5:
        try:
            result = minimize(objective_exact, best_h, method='L-BFGS-B', 
                             bounds=bounds, options={'maxiter': 200, 'maxfun': 500, 
                                                    'ftol': 1e-12, 'gtol': 1e-12})
            exact_score = objective_exact(result.x)
            if exact_score < best_score:
                best_score = exact_score
                best_h = np.maximum(result.x, 0)
        except Exception:
            pass
    
    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
