"""
Strategy: Spectral Basis Optimization (DCT)
Represents the step function as a linear combination of the first M Discrete Cosine Transform (DCT) modes.
This reduces the 600-dimensional optimization problem to a M-dimensional one (M=20).
We use a smooth-max surrogate (log-sum-exp) to handle the non-differentiable max operation in the objective.
We optimize the coefficients using L-BFGS-B, enforcing non-negativity of the resulting step function.
This allows us to find global structures (like flattened peaks) that might be missed by local coordinate descent on the full 600D space.
"""
import time
import numpy as np
from scipy import optimize
from scipy.fft import dct, idct

N = 600
M = 20  # Number of DCT modes to use

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    # Use FFT for speed
    size = 2 * N
    f = np.fft.rfft(h, size)
    conv = np.fft.irfft(f * f, size)[:2*N-1]
    max_conv = np.max(conv)
    return max_conv / (s * s)

def _dct_to_h(coeffs):
    """
    Convert DCT coefficients to step function heights.
    coeffs: array of length M.
    Returns array of length N.
    """
    # Pad coefficients to length N
    full_coeffs = np.zeros(N)
    full_coeffs[:M] = coeffs
    # Inverse DCT (Type II)
    h = idct(full_coeffs, type=2, norm='ortho')
    # Enforce non-negativity
    h = np.maximum(h, 0)
    return h

def _smoothed_objective(coeffs, beta):
    """
    Compute the smooth approximation of the score and its gradient.
    coeffs: array of length M.
    beta: temperature for log-sum-exp.
    """
    h = _dct_to_h(coeffs)
    s = np.sum(h)
    if s == 0:
        return 1e10, np.zeros(M)
    
    # Autoconvolution via FFT
    size = 2 * N
    f = np.fft.rfft(h, size)
    conv = np.fft.irfft(f * f, size)[:2*N-1]
    
    # Smooth max of conv
    # conv_scaled = conv / beta
    # lse = beta * (max(conv/beta) + log(sum(exp(conv/beta - max(conv/beta)))))
    # To avoid overflow, we can just use the standard trick.
    c_max = np.max(conv)
    if c_max < 0:
        c_max = 0
    conv_shifted = conv - c_max
    lse = c_max + beta * np.log(np.sum(np.exp(conv_shifted / beta)))
    
    score = lse / (s * s)
    
    # Gradient calculation
    # We need d(score)/d(coeffs) = d(score)/d(h) * d(h)/d(coeffs)
    # d(score)/d(h_i) = (1/s^2) * d(lse)/d(h_i) - 2*lse/s^3
    # d(lse)/d(h_i) = sum_j w_j * d(conv_j)/d(h_i)
    # where w_j = exp((conv_j - c_max)/beta) / sum_k exp((conv_k - c_max)/beta)
    
    weights = np.exp(conv_shifted / beta)
    weights /= np.sum(weights)
    
    # d(conv_j)/d(h_i)
    # conv_j = sum_k h_k h_{j-k}
    # This is a linear operator. We can compute the gradient of lse w.r.t h
    # by "backpropagating" the weights through the convolution.
    # grad_lse_h = convolve(weights, h) + convolve(h, weights) ?
    # Let's check.
    # lse = log(sum(exp(conv/beta))) * beta
    # d lse / d h_i = sum_j (1/beta) * exp(conv_j/beta) * (conv_j - max) ... no
    # d lse / d h_i = sum_j weights_j * d conv_j / d h_i
    # d conv_j / d h_i = h_{j-i} + h_{-i-j} (with appropriate indexing/periodicity)
    # Effectively, grad_lse_h = 2 * convolve(h, weights, mode='same')? 
    # Actually, conv is full. weights are length 2N-1.
    # Let's use the property that grad of sum(w_j * conv_j) w.r.t h is the convolution of w with h.
    # Since conv_j = <h, shift_j(h)>, grad is sum_j w_j * shift_j(h).
    # This is equivalent to convolving h with the "impulse train" defined by weights.
    
    # Construct the gradient of lse w.r.t h
    # We need to map weights (length 2N-1) to a filter.
    # The convolution h*h has length 2N-1.
    # The gradient of (h*h)_k w.r.t h_i is h_{k-i} + h_{-k+i} (roughly).
    # So grad_lse_h[i] = sum_k weights[k] * (h[k-i] + h[-k+i])
    # This is 2 * (weights * h) at index i?
    # Let's just compute it numerically if M is small? No, M=20, N=600.
    # Numerical gradient w.r.t coeffs (M=20) is cheap enough for L-BFGS-B if we do it once per eval?
    # L-BFGS-B calls the function many times. Numerical gradient of 20D is 40 evals.
    # Each eval is an FFT (fast).
    # Let's try analytical gradient w.r.t h first, then chain rule.
    
    # grad_lse_h = np.convolve(h, weights, mode='full')? No.
    # Let's use the fact that grad of max(conv) is the normalized weights convolved with h.
    # Specifically, if we view conv as a quadratic form, the gradient is 2 * A^T * h.
    # Here A is the convolution matrix.
    # Let's just do numerical gradient w.r.t coeffs. It's robust and M is small.
    
    # But wait, I need to return the gradient for L-BFGS-B.
    # If I provide numerical gradient, L-BFGS-B will use it.
    # Let's compute numerical gradient w.r.t coeffs here to be safe and simple.
    # However, doing this inside the objective function called by L-BFGS-B might be slow if it calls it too often.
    # L-BFGS-B usually estimates gradient if not provided.
    # I will provide the objective only, and let L-BFGS-B estimate the gradient numerically.
    # Since M=20, this is 41 evaluations per gradient step.
    # Each evaluation is ~1ms. So 41ms per gradient step.
    # 100 steps = 4s. This is acceptable.
    
    # So I will just return the score.
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def check_time():
        return time.time() > deadline

    # Initialize best solution with uniform
    best_h = np.ones(N)
    best_score = compute_score(best_h)
    
    # DCT basis: The coefficients for a uniform vector in DCT domain
    # DCT of ones is a delta at 0.
    # c_0 = sqrt(2/N) * sum(1) = sqrt(2/N)*N = sqrt(2N)
    # c_k = 0 for k>0
    init_coeffs = np.zeros(M)
    init_coeffs[0] = np.sqrt(2.0 * N) # For orthonormal DCT-II, DC component is sqrt(2/N)*sum(h)
    # Wait, scipy.idct(dct(x)) = x.
    # Let's just initialize coeffs from the uniform h.
    h_uniform = np.ones(N)
    coeffs_uniform = dct(h_uniform, type=2, norm='ortho')
    
    # We will optimize coeffs[1:M] (AC components) and keep DC fixed to maintain scale?
    # Or optimize all, but the scale is invariant?
    # Score is scale invariant: max(c*h * c*h) / (sum(c*h))^2 = max(h*h)/sum(h)^2.
    # So we can fix the L1 norm or just let the optimizer find the shape.
    # However, if all coeffs go to 0, score is undefined.
    # Let's fix coeffs[0] to the value corresponding to sum(h)=N (or similar).
    # Actually, let's just optimize all coeffs but penalize if sum(h) is too small.
    # Or simpler: Fix coeffs[0] = coeffs_uniform[0]. Optimize coeffs[1:].
    
    fixed_dc = coeffs_uniform[0]
    # Initial AC coeffs are 0.
    init_ac = np.zeros(M-1)
    
    def objective(ac_coeffs):
        if check_time():
            return 1e10
        coeffs = np.zeros(M)
        coeffs[0] = fixed_dc
        coeffs[1:] = ac_coeffs
        
        h = _dct_to_h(coeffs)
        s = np.sum(h)
        if s < 1e-6:
            return 1e10
            
        # Compute score
        size = 2 * N
        f = np.fft.rfft(h, size)
        conv = np.fft.irfft(f * f, size)[:2*N-1]
        max_conv = np.max(conv)
        
        # Use smooth max for better gradient estimation by L-BFGS-B?
        # L-BFGS-B uses finite differences. A smooth function helps.
        # Let's use a soft-max.
        beta = 10.0
        c_max = np.max(conv)
        if c_max < 0: c_max = 0
        conv_shifted = conv - c_max
        lse = c_max + beta * np.log(np.sum(np.exp(conv_shifted / beta)))
        
        return lse / (s * s)

    # Run L-BFGS-B
    # We might need multiple restarts or annealing beta.
    # Start with high beta (smoother) and decrease?
    # Or just use a moderate beta.
    
    betas = [50.0, 20.0, 10.0, 5.0]
    current_ac = init_ac.copy()
    
    for beta in betas:
        if check_time():
            break
            
        # Redefine objective with current beta
        def obj(x, b=beta):
            if check_time():
                return 1e10
            coeffs = np.zeros(M)
            coeffs[0] = fixed_dc
            coeffs[1:] = x
            
            h = _dct_to_h(coeffs)
            s = np.sum(h)
            if s < 1e-6:
                return 1e10
                
            size = 2 * N
            f = np.fft.rfft(h, size)
            conv = np.fft.irfft(f * f, size)[:2*N-1]
            
            c_max = np.max(conv)
            if c_max < 0: c_max = 0
            conv_shifted = conv - c_max
            lse = c_max + b * np.log(np.sum(np.exp(conv_shifted / b)))
            
            return lse / (s * s)

        try:
            res = optimize.minimize(
                obj,
                current_ac,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            current_ac = res.x
        except Exception:
            pass
        
        # Evaluate actual score with the found coeffs
        coeffs = np.zeros(M)
        coeffs[0] = fixed_dc
        coeffs[1:] = current_ac
        h_cand = _dct_to_h(coeffs)
        
        if np.sum(h_cand) > 1e-6:
            score_cand = compute_score(h_cand)
            if score_cand < best_score:
                best_score = score_cand
                best_h = h_cand

    # Also try a few random initializations for AC coeffs
    np.random.seed(42)
    for _ in range(5):
        if check_time():
            break
        rand_ac = np.random.normal(0, 0.1, M-1)
        
        # Quick optimization
        def obj_rand(x, b=10.0):
            if check_time():
                return 1e10
            coeffs = np.zeros(M)
            coeffs[0] = fixed_dc
            coeffs[1:] = x
            
            h = _dct_to_h(coeffs)
            s = np.sum(h)
            if s < 1e-6:
                return 1e10
                
            size = 2 * N
            f = np.fft.rfft(h, size)
            conv = np.fft.irfft(f * f, size)[:2*N-1]
            
            c_max = np.max(conv)
            if c_max < 0: c_max = 0
            conv_shifted = conv - c_max
            lse = c_max + b * np.log(np.sum(np.exp(conv_shifted / b)))
            
            return lse / (s * s)

        try:
            res = optimize.minimize(
                obj_rand,
                rand_ac,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            rand_ac_opt = res.x
            coeffs = np.zeros(M)
            coeffs[0] = fixed_dc
            coeffs[1:] = rand_ac_opt
            h_cand = _dct_to_h(coeffs)
            
            if np.sum(h_cand) > 1e-6:
                score_cand = compute_score(h_cand)
                if score_cand < best_score:
                    best_score = score_cand
                    best_h = h_cand
        except Exception:
            pass

    return best_h.tolist()
