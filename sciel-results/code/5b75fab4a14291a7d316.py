import numpy as np
import scipy.optimize
import time

N = 600
DEADLINE = 145.0

def compute_score(heights):
    """
    Computes the normalized maximum of the autoconvolution.
    """
    p = np.asarray(heights)
    if np.sum(p) == 0:
        return 1e9
    c = np.fft.irfft(np.fft.rfft(p) ** 2, n=2 * N)
    max_c = np.max(c)
    norm = np.sum(p) ** 2
    return max_c / norm

def spectral_objective(a):
    """
    Objective: minimize max of autoconvolution given spectral amplitudes a.
    p = |IDFT(a * exp(i*phi))|^2, then autoconv = |FFT(p)|^2 = |a|^4 (elementwise)
    Wait, that's not right. Let me reconsider.
    
    If p = |q|^2 where q = IDFT(Q), and Q = a * exp(i*phi),
    then FFT(p) = FFT(|q|^2). This is NOT simply related to Q.
    
    Let me use a different approach: optimize p directly but use a 
    better initialization and optimization strategy.
    """
    # Convert amplitudes to time-domain signal
    # q = IDFT(a * exp(i*phi)) where phi are fixed random phases
    # p = |q|^2
    # But this makes p non-negative automatically.
    
    # Actually, let me just optimize p directly with a better method.
    # The issue with the current approach is that softmax constrains p to sum to 1,
    # but we can scale p freely since the score is scale-invariant.
    
    # Let me try a different strategy: use scipy's minimize on p directly
    # with a smooth max objective, but with better initialization.
    
    raise NotImplementedError  # Placeholder, will be replaced

def construct():
    start_time = time.time()
    
    # Strategy: Use a "flat-top" spectral design.
    # The autoconvolution peak is minimized when the spectral energy is spread 
    # as evenly as possible, but with careful phase alignment to reduce time-domain peaks.
    
    # Key insight: For a step function p with autoconvolution c,
    # the peak of c is at most (sum p)^2 / N (by averaging), but we can do better
    # by shaping p to reduce the peak.
    
    # Approach: Start with a spectral mask that has a "flat" magnitude response
    # in the passband, which gives a sinc-like autocorrelation with reduced sidelobes.
    
    # Let's try: p_i proportional to 1 + epsilon * cos(2*pi*i/N) with small epsilon,
    # then optimize locally.
    
    # Better: Use the fact that minimizing max(c) where c = p*p (convolution)
    # is equivalent to making the "energy" of p as flat as possible in frequency.
    
    # I'll use a direct optimization on p with a log-sum-exp smooth max,
    # but with multiple restarts and a spectral initialization.
    
    def smooth_max(values, beta=200.0):
        v_max = np.max(values)
        return v_max + np.log(np.sum(np.exp(beta * (values - v_max)))) / beta
    
    def objective_and_grad(x):
        """
        x: unconstrained parameters, p = exp(x) / sum(exp(x)) to ensure non-negativity and sum=1
        """
        x_shifted = x - np.max(x)
        p = np.exp(x_shifted)
        p /= np.sum(p)
        
        # Autoconvolution via FFT
        P = np.fft.rfft(p)
        C = P ** 2
        c = np.fft.irfft(C, n=2 * N)
        
        # Smooth max of c
        beta = 200.0
        c_max = np.max(c)
        exp_c = np.exp(beta * (c - c_max))
        sum_exp = np.sum(exp_c)
        w = exp_c / sum_exp
        
        # Gradient of smooth max w.r.t. c is w
        # Gradient of c w.r.t. p: c = p * p (convolution)
        # dL/dp = 2 * (w * p) where * is convolution
        # Using FFT: FFT(dL/dp) = 2 * FFT(w) * P
        W = np.fft.rfft(w)
        dL_dp = np.fft.irfft(2 * W * P, n=N)
        
        # Gradient w.r.t. x (softmax)
        dot = np.sum(p * dL_dp)
        grad_x = p * (dL_dp - dot)
        
        obj_val = c_max + np.log(sum_exp) / beta
        return obj_val, grad_x
    
    def optimize_from(x_init):
        try:
            res = scipy.optimize.minimize(
                objective_and_grad,
                x_init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            return res.x
        except Exception:
            return x_init
    
    # Multiple initializations
    best_x = None
    best_score = 1e9
    
    initializations = []
    
    # 1. Uniform
    initializations.append(np.zeros(N))
    
    # 2. Cosine window (raised cosine) - reduces spectral leakage
    i = np.arange(N)
    initializations.append(0.5 * (1 + np.cos(2 * np.pi * i / N)))
    
    # 3. Hann window
    initializations.append(0.5 * (1 - np.cos(2 * np.pi * i / N)))
    
    # 4. Gaussian
    center = N / 2
    sigma = N / 6
    initializations.append(np.exp(-(i - center) ** 2 / (2 * sigma ** 2)))
    
    # 5. Flat-top window (for spectral flatness)
    # Flat-top: 0.21557 - 0.41663*cos(2pi*i/N) + 0.27726*cos(4pi*i/N) - 0.08357*cos(6pi*i/N)
    flat_top = (0.21557 - 0.41663 * np.cos(2 * np.pi * i / N) 
                + 0.27726 * np.cos(4 * np.pi * i / N) 
                - 0.08357 * np.cos(6 * np.pi * i / N))
    initializations.append(np.maximum(flat_top, 0.01))
    
    # 6. Random with small perturbation
    initializations.append(0.1 * np.random.randn(N))
    
    # 7. Two-lobe: high at edges, low in middle
    initializations.append(1.0 - 0.8 * np.exp(-(i - N/2) ** 2 / (2 * (N/4) ** 2)))
    
    # 8. Spectral flatness initialization: start with flat spectrum
    # This means p is a delta function, which is bad. Instead, start with 
    # a signal whose FFT has flat magnitude.
    # A random phase signal has flat spectrum on average.
    q_random = np.random.randn(N) + 1j * np.random.randn(N)
    p_from_flat_spec = np.abs(np.fft.irfft(q_random, n=N)) ** 2
    p_from_flat_spec = p_from_flat_spec / np.sum(p_from_flat_spec)
    initializations.append(np.log(p_from_flat_spec + 1e-10))
    
    for x_init in initializations:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        x_opt = optimize_from(x_init)
        p_opt = np.exp(x_opt - np.max(x_opt))
        p_opt /= np.sum(p_opt)
        score = compute_score(p_opt)
        
        if score < best_score:
            best_score = score
            best_x = x_opt
    
    # If we have time, do a few more random restarts
    if best_x is None or time.time() - start_time < DEADLINE - 30:
        for _ in range(5):
            if time.time() - start_time > DEADLINE - 10:
                break
            x_init = 0.5 * np.random.randn(N)
            x_opt = optimize_from(x_init)
            p_opt = np.exp(x_opt - np.max(x_opt))
            p_opt /= np.sum(p_opt)
            score = compute_score(p_opt)
            if score < best_score:
                best_score = score
                best_x = x_opt
    
    if best_x is None:
        best_x = np.zeros(N)
    
    p_final = np.exp(best_x - np.max(best_x))
    p_final /= np.sum(p_final)
    return list(p_final)
