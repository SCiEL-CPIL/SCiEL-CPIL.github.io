import numpy as np
import scipy.optimize
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    n = np.arange(N)
    dx = 1.0 / (2 * N)  # Width of each interval in [-1/4, 1/4]
    # x_i = -1/4 + (i + 0.5) * dx
    x = -0.25 + (n + 0.5) * dx

    def compute_psr(h):
        h = np.asarray(h)
        if np.any(h < -1e-10):
            return 1e10
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s == 0:
            return 1e10
        h_norm = h / s
        
        # Compute autoconvolution
        size = 2 * N
        H = np.fft.rfft(h_norm, n=size)
        c = np.fft.irfft(H * np.conj(H), n=size)
        
        c0 = c[0]
        if c0 < 1e-10:
            return 1e10
            
        # Max sidelobe
        max_side = np.max(c[1:N])
        
        return max_side / c0

    def smooth_max_obj(params, beta):
        # params: [sigma, a1, a2, ..., aK]
        sigma = params[0]
        coeffs = params[1:]
        
        # Compute h
        # Gaussian part
        g = np.exp(-0.5 * (x / sigma) ** 2)
        
        # Cosine polynomial part
        # h(x) = g(x) * (1 + sum_{k=1}^K a_k cos(2 pi k x / dx)) ? 
        # Wait, the domain is [-1/4, 1/4]. Length is 1/2.
        # Let's use frequencies relative to the domain length L=0.5.
        # Basis: cos(2 pi k x / L) = cos(4 pi k x)
        # Or just use frequencies k * 2*pi / (2*N) ? No.
        # Let's stick to the previous parametric family but generalized.
        # Previous: 1 + a cos(2 pi i / N) + b cos(4 pi i / N)
        # This corresponds to frequencies 1 and 2 in "bins".
        # Let's use: P(x) = 1 + sum_{k=1}^K a_k cos(4 pi k x)
        # Since x in [-0.25, 0.25], 4*pi*x covers [-pi, pi].
        # So k=1 is fundamental frequency of the interval.
        
        P = np.ones(N)
        for k in range(len(coeffs)):
            P += coeffs[k] * np.cos(4 * np.pi * (k + 1) * x)
            
        h = g * P
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s == 0:
            return 1e10
        h_norm = h / s
        
        # Compute autoconvolution
        size = 2 * N
        H = np.fft.rfft(h_norm, n=size)
        c = np.fft.irfft(H * np.conj(H), n=size)
        
        c0 = c[0]
        if c0 < 1e-10:
            return 1e10
        
        sides = c[1:N]
        
        # Smooth max
        max_side = np.max(sides)
        exp_sides = np.exp(beta * (sides - max_side))
        sum_exp = np.sum(exp_sides)
        M_smooth = max_side + np.log(sum_exp) / beta
        
        return M_smooth / c0

    # Initial guess
    # sigma ~ 0.1 (width of gaussian)
    # coeffs: Blackman-like
    # Blackman: 0.42 - 0.5 cos + 0.08 cos(2)
    # In our basis: 1 + a1 cos(4 pi x) + a2 cos(8 pi x)
    # Note: The previous parametric_h used cos(2 pi i / N) which is cos(2 pi x * 2N / 2) = cos(2 pi x N)?
    # Let's check mapping.
    # i = 0..N-1. x_i = -0.25 + (i+0.5)/2N.
    # cos(2 pi i / N) = cos(2 pi (2N x_i - 1) / 2) = cos(2 pi N x_i - pi) = -cos(2 pi N x_i).
    # This is a high frequency.
    # Let's just use the same structure as before but with Gaussian envelope.
    
    K = 4 # Number of cosine coefficients
    
    def init_params():
        sigma = 0.1
        # Try to match Blackman shape
        a1 = -0.5
        a2 = 0.08
        a3 = 0.0
        a4 = 0.0
        return np.array([sigma, a1, a2, a3, a4])
    
    best_params = init_params()
    
    # Annealed optimization
    # Start with low beta (smooth max) to explore, then increase beta to sharpen.
    
    best_val = 1e10
    
    # Phase 1: Coarse search with smooth max
    beta_start = 5.0
    beta_end = 50.0
    
    # Use Nelder-Mead on the smooth objective
    # To handle annealing, we can run multiple minimizations with increasing beta.
    
    current_beta = beta_start
    current_params = best_params.copy()
    
    # Run a sequence of minimizations
    # Each step increases beta slightly and refines params
    
    steps = 10
    for i in range(steps):
        if time.time() > deadline - 10:
            break
            
        current_beta = beta_start + (beta_end - beta_start) * (i / max(1, steps - 1))
        
        def obj_wrapper(params):
            return smooth_max_obj(params, current_beta)
        
        # Nelder-Mead
        # Initial simplex: perturb current params
        res = scipy.optimize.minimize(
            obj_wrapper, 
            current_params, 
            method='Nelder-Mead', 
            options={'maxiter': 50, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True}
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x.copy()
            
        current_params = res.x.copy()
        
        # Check time
        if time.time() > deadline - 5:
            break

    # Final evaluation with exact max (beta -> inf)
    def exact_obj(params):
        sigma = params[0]
        coeffs = params[1:]
        g = np.exp(-0.5 * (x / sigma) ** 2)
        P = np.ones(N)
        for k in range(len(coeffs)):
            P += coeffs[k] * np.cos(4 * np.pi * (k + 1) * x)
        h = g * P
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s == 0:
            return 1e10
        h_norm = h / s
        size = 2 * N
        H = np.fft.rfft(h_norm, n=size)
        c = np.fft.irfft(H * np.conj(H), n=size)
        c0 = c[0]
        if c0 < 1e-10:
            return 1e10
        return np.max(c[1:N]) / c0

    final_val = exact_obj(best_params)
    if final_val < best_val:
        best_val = final_val
    
    # Construct final h
    sigma = best_params[0]
    coeffs = best_params[1:]
    g = np.exp(-0.5 * (x / sigma) ** 2)
    P = np.ones(N)
    for k in range(len(coeffs)):
        P += coeffs[k] * np.cos(4 * np.pi * (k + 1) * x)
    h = g * P
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s > 0:
        h = h / s
    else:
        # Fallback to uniform
        h = np.ones(N) / N
        
    return list(h)

if __name__ == "__main__":
    h = construct()
    print("Constructed h")
