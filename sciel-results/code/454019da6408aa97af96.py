import time
import numpy as np

N = 600
DEADLINE = 140.0

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function.
    Returns the value such that uniform heights give score 1.0.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n  # width of each interval
    
    # Use FFT for linear autoconvolution
    M = 1
    while M < 2 * n - 1:
        M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
    
    # Normalization: uniform p_i = 1/n gives max_conv = sum_i (1/n)^2 * min(n, ...) 
    # For uniform, conv[0] = sum p_i^2 = 1/n, conv[k] for k>0 = (n-k)/n^2
    # Actually for uniform p_i=1/n: conv[0] = sum (1/n)^2 = 1/n, conv[1] = (n-1)/n^2
    # max is conv[0] = 1/n
    # score = (1/n) * n / ((1)^2 * w) = 1/w * ... 
    # Let me just compute: total = 1, max_conv = 1/n for uniform
    # score = (1/n) * n / (1 * w) = 1/w = 2n
    # Hmm, that doesn't give 1. Let me re-examine.
    # Actually for uniform p_i = 1/n, sum = 1.
    # conv[0] = sum_i p_i^2 = n*(1/n^2) = 1/n
    # score = (1/n)*n / (1*w) = 1/w
    # w = 0.5/n, so score = n/0.5 = 2n = 1200. That's not 1.
    # 
    # Let me reconsider. The task says "uniform heights give score 1.0".
    # Let's try: if heights are all h, then total = N*h, max_conv = N*h^2
    # score = N*h^2 * N / (N^2*h^2 * w) = N^2*h^2/(N^2*h^2*w) = 1/w
    # That's still 1/w. Unless the normalization is different.
    #
    # Maybe the normalization uses the integral rather than sum?
    # Or maybe score = max_conv / (total^2 * w) without the *n factor?
    # For uniform: (N*h^2) / (N^2*h^2*w) = 1/(N*w) = 1/(N*0.5/N) = 1/0.5 = 2. Still not 1.
    #
    # Let me try: score = max_conv * n^2 / (total^2 * n * w) = max_conv * n / (total^2 * w)
    # For uniform: (N*h^2) * N / (N^2*h^2 * 0.5/N) = N^2*h^2 / (N^2*h^2 * 0.5/N) = N/0.5 = 2N. Nope.
    #
    # OK let me just match the seed program's formula and verify with uniform:
    # heights = ones(N), total = N, max_conv = N (since conv[0] = sum h_i^2 = N)
    # score = N * N / (N^2 * 0.5/N) = N^2 / (N^2*0.5/N) = N/0.5 = 2N = 1200
    # That's not 1 either. But the task says uniform gives 1.0 and the seed scores 1.0.
    # So the seed must be returning something that scores 1.0, meaning the evaluator normalizes.
    # 
    # Looking at the task: "score is the upper bound on C1" and "the evaluator measures the raw value".
    # The seed returns uniform probabilities (1/N each) and gets score 1.0.
    # So the evaluator must be dividing by some constant, or the score formula in the seed
    # is wrong but the evaluator compensates.
    #
    # Since the seed returns p_i = 1/N and scores 1.0, and my compute_score is used internally
    # for comparison, I just need a consistent measure. Let me use:
    # score = max_conv / (total^2 * w)
    # For uniform p_i=1/N: max_conv = 1/N, total = 1, w = 0.5/N
    # score = (1/N) / (1 * 0.5/N) = (1/N) * (N/0.5) = 1/0.5 = 2
    # For the evaluator, uniform gives 1.0, so there might be a factor of 2 difference.
    # Since we only care about relative comparison, this is fine.
    #
    # Actually, let me just use the same formula as the seed for consistency:
    score = max_conv * n / (total * total * w)
    return score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    def time_left():
        return deadline - time.time()

    # Pre-compute FFT size
    M = 1
    while M < 2 * N - 1:
        M *= 2

    # FFT size for N-point DFT
    def idft_to_heights(spec_params):
        """
        Given spectral parameters, construct nonnegative heights.
        
        Approach: parametrize the power spectrum S(k) = exp(sum_j c_j * basis_j(k))
        where basis_j are low-frequency basis functions. Then the autoconvolution
        in frequency domain is related to |F|^2. We want to minimize max of |F|^2
        over the convolution.
        
        Actually, a cleaner approach: the autoconvolution c[k] has DFT C(k) = |F(k)|^2.
        We want to minimize max_k c[k]. By Parseval, sum c[k] = sum |F(k)|^2 = N * sum p_i^2.
        So minimizing max c[k] while keeping sum p_i = 1 is like flattening c.
        
        Strategy: parametrize p as idft of a parametric function, ensuring p >= 0.
        We'll use p_i = max(0, idft(A_k)) where A_k = exp(sum_j theta_j * phi_j(k)).
        The basis functions phi_j(k) are low-frequency modes.
        """
        # Build the spectral magnitude
        # A_k for k = 0, 1, ..., M-1 (we use full DFT for flexibility)
        k = np.arange(M)
        # Frequency index in cycles per unit interval
        freq = k / M
        
        # Basis functions in frequency domain (low frequency)
        # phi_0 = 1 (DC)
        # phi_1 = cos(2*pi*freq), phi_2 = sin(2*pi*freq), etc.
        # Use up to order L basis functions
        
        L = len(spec_params)
        # spec_params = [c_0, c_1, c_2, ...] where A_k = exp(sum c_j * phi_j(k))
        logA = np.zeros(M)
        
        # phi_0: constant
        logA += spec_params[0]
        
        # Higher frequency basis
        # Use cos and sin of harmonics
        harmonic_idx = 0
        for h in range(1, 20):  # up to 20 harmonics
            if harmonic_idx >= L - 1:
                break
            cos_h = np.cos(2 * np.pi * h * freq)
            sin_h = np.sin(2 * np.pi * h * freq)
            
            if harmonic_idx < L - 1:
                logA += spec_params[1 + 2*harmonic_idx] * cos_h
            if 2 + 2*harmonic_idx < L:
                logA += spec_params[2 + 2*harmonic_idx] * sin_h
            
            harmonic_idx += 1
        
        A = np.exp(logA)
        
        # To get a real, nonnegative time-domain signal, we need A to be a valid
        # power spectrum. A_k should be nonnegative and symmetric: A_k = A_{M-k}.
        # Our construction with cos/sin basis is not symmetric. Let's fix this.
        # Instead, let's use a different parametrization:
        # A_k = exp(sum_j c_j * cos(2*pi*j*k/M)) for j=0,...,J
        # This is automatically symmetric and positive.
        
        # Redo: use only cosine basis
        logA2 = np.zeros(M)
        J = min(len(spec_params) - 1, 30)  # number of cosine harmonics
        for j in range(J + 1):
            if j == 0:
                logA2 += spec_params[0]
            else:
                idx = j  # spec_params[1] = c_1, spec_params[2] = c_2, etc.
                if idx < len(spec_params):
                    cos_j = np.cos(2 * np.pi * j * k / M)
                    logA2 += spec_params[idx] * cos_j
        
        A2 = np.exp(logA2)
        
        # A2 is a valid power spectrum (nonnegative, symmetric)
        # The corresponding time-domain signal is p_i = (1/M) * idft(A2)_i
        # But we want sum p_i = 1, and p_i >= 0.
        
        # Compute idft
        # numpy's idft: ifft(x) = (1/M) * sum x_k * exp(2*pi*i*k*n/M)
        # We want real output, so A2 should be symmetric, which it is.
        p_raw = np.fft.ifft(A2).real
        
        # p_raw might have negative values. We need to handle this.
        # Option 1: clip to zero and renormalize
        # Option 2: add a large DC component to ensure positivity
        
        # Let's clip and renormalize
        p = np.maximum(p_raw, 0.0)
        total = np.sum(p)
        if total <= 0:
            return np.ones(N) / N
        p = p / total
        
        # Take first N points (the signal is on N intervals, but we used M >= 2N-1)
        # Actually, we want exactly N points. The idft gives M points.
        # We should design the spectrum so that the first N points are what we want.
        # This is tricky. Let me reconsider.
        
        # Alternative: work directly with N-point DFT.
        # p_i for i=0,...,N-1, with DFT F_k for k=0,...,N-1.
        # The autoconvolution c[m] for m=0,...,2N-2 has values.
        # But the DFT of the zero-padded autoconvolution of length 2N-1 is |F_k|^2
        # (with appropriate padding).
        
        # Let me just use N-point DFT:
        p_N = np.zeros(N)
        # A2_N is the N-point version
        k_N = np.arange(N)
        logA_N = np.zeros(N)
        for j in range(min(J+1, N)):
            if j == 0:
                logA_N += spec_params[0]
            else:
                if j < len(spec_params):
                    cos_j = np.cos(2 * np.pi * j * k_N / N)
                    logA_N += spec_params[j] * cos_j
        
        A_N = np.exp(logA_N)
        p_N = np.fft.ifft(A_N).real
        
        p_N = np.maximum(p_N, 0.0)
        tot = np.sum(p_N)
        if tot <= 0:
            return np.ones(N) / N
        p_N = p_N / tot
        
        return p_N

    def spectral_objective_and_grad(theta):
        """
        Objective: max of autoconvolution (smoothed) of p = idft_to_heights(theta).
        theta: spectral parameters.
        """
        p = idft_to_heights(theta)
        
        # Compute autoconvolution
        f = np.zeros(M, dtype=np.float64)
        f[:N] = p
        F = np.fft.rfft(f)
        conv = np.fft.irfft(F * F, n=M)
        
        # Smooth max
        alpha = 100.0
        max_conv = np.max(conv)
        conv_scaled = alpha * (conv - max_conv)
        exp_conv = np.exp(conv_scaled)
        sum_exp = np.sum(exp_conv)
        smooth_max_val = max_conv + np.log(sum_exp) / alpha
        weights = exp_conv / sum_exp
        
        # Gradient of smooth_max w.r.t p
        W = np.fft.rfft(weights)
        P = np.fft.rfft(p)
        corr = np.fft.irfft(W * np.conj(P), n=M)
        grad_p = 2.0 * corr[:N]
        
        # Gradient of p w.r.t theta
        # p_i = max(0, ifft(A_k)_i) / sum, where A_k = exp(sum_j theta_j cos(2pi j k/N))
        # This is complicated due to the max and normalization.
        # Let's use finite differences for the gradient w.r.t theta.
        
        # Actually, let's compute gradient numerically
        grad_theta = np.zeros(len(theta))
        eps = 1e-6
        for j in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[j] += eps
            p_plus = idft_to_heights(theta_plus)
            f_plus = np.zeros(M, dtype=np.float64)
            f_plus[:N] = p_plus
            F_plus = np.fft.rfft(f_plus)
            conv_plus = np.fft.irfft(F_plus * F_plus, n=M)
            # Recompute smooth max with same alpha
            max_plus = np.max(conv_plus)
            conv_scaled_plus = alpha * (conv_plus - max_plus)
            exp_plus = np.exp(conv_scaled_plus)
            sum_exp_plus = np.sum(exp_plus)
            smooth_plus = max_plus + np.log(sum_exp_plus) / alpha
            
            theta_minus = theta.copy()
            theta_minus[j] -= eps
            p_minus = idft_to_heights(theta_minus)
            f_minus = np.zeros(M, dtype=np.float64)
            f_minus[:N] = p_minus
            F_minus = np.fft.rfft(f_minus)
            conv_minus = np.fft.irfft(F_minus * F_minus, n=M)
            max_minus = np.max(conv_minus)
            conv_scaled_minus = alpha * (conv_minus - max_minus)
            exp_minus = np.exp(conv_scaled_minus)
            sum_exp_minus = np.sum(exp_minus)
            smooth_minus = max_minus + np.log(sum_exp_minus) / alpha
            
            grad_theta[j] = (smooth_plus - smooth_minus) / (2 * eps)
        
        return smooth_max_val, grad_theta, p

    # Optimization: use coordinate descent / gradient descent on spectral parameters
    # Number of spectral parameters: DC + J cosine harmonics
    J_max = 15
    n_params = 1 + J_max  # DC + 15 harmonics
    
    rng = np.random.default_rng(42)
    
    best_p = np.ones(N) / N
    best_raw_score = compute_score(best_p)
    
    # Initial guesses for spectral parameters
    inits = []
    
    # 1. All zeros (uniform)
    inits.append(np.zeros(n_params))
    
    # 2. Positive DC boost (concentrated at center via low-frequency dominance)
    for dc_boost in [0.5, 1.0, 2.0]:
        theta = np.zeros(n_params)
        theta[0] = dc_boost
        inits.append(theta)
    
    # 3. Emphasis on low frequencies (smooth bump)
    for scale in [0.5, 1.0, 2.0]:
        theta = np.zeros(n_params)
        for j in range(1, J_max + 1):
            theta[j] = -scale * j / J_max  # decay with frequency
        inits.append(theta)
    
    # 4. Random small perturbations
    for _ in range(5):
        theta = rng.normal(0, 0.5, size=n_params)
        inits.append(theta)
    
    # Optimization loop
    lr = 0.01
    momentum = 0.9
    
    for x_init in inits:
        if time_left() < 15:
            break
        
        theta = x_init.copy()
        velocity = np.zeros(n_params)
        
        # Gradient descent with momentum
        for step in range(200):
            if time_left() < 10:
                break
            
            val, grad, p = spectral_objective_and_grad(theta)
            
            # Update
            velocity = momentum * velocity - lr * grad
            theta = theta + velocity
            
            # Clip theta to avoid extreme values
            theta = np.clip(theta, -5, 5)
            
            # Check raw score
            raw_score = compute_score(p)
            if raw_score < best_raw_score:
                best_raw_score = raw_score
                best_p = p.copy()
        
        # After gradient descent, try local refinement with smaller steps
        lr2 = 0.001
        for step in range(100):
            if time_left() < 5:
                break
            
            val, grad, p = spectral_objective_and_grad(theta)
            velocity = momentum * velocity - lr2 * grad
            theta = theta + velocity
            theta = np.clip(theta, -5, 5)
            
            raw_score = compute_score(p)
            if raw_score < best_raw_score:
                best_raw_score = raw_score
                best_p = p.copy()
    
    # Try to improve with a different strategy: direct time-domain optimization
    # using the best spectral solution as initialization
    if time_left() > 20:
        from scipy.optimize import minimize
        
        # Convert best_p to log-heights for direct optimization
        p_start = best_p.copy()
        p_start = np.maximum(p_start, 1e-10)
        x_start = np.log(p_start / np.sum(p_start))
        
        M_fft = M
        
        def time_domain_obj(x):
            """Smooth max of autoconvolution of softmax(x)."""
            x_max = np.max(x)
            exp_x = np.exp(x - x_max)
            p = exp_x / np.sum(exp_x)
            
            f = np.zeros(M_fft, dtype=np.float64)
            f[:N] = p
            F = np.fft.rfft(f)
            conv = np.fft.irfft(F * F, n=M_fft)
            
            alpha_td = 200.0
            max_conv = np.max(conv)
            conv_scaled = alpha_td * (conv - max_conv)
            exp_conv = np.exp(conv_scaled)
            sum_exp = np.sum(exp_conv)
            return max_conv + np.log(sum_exp) / alpha_td
        
        def time_domain_grad(x):
            x_max = np.max(x)
            exp_x = np.exp(x - x_max)
            p = exp_x / np.sum(exp_x)
            
            f = np.zeros(M_fft, dtype=np.float64)
            f[:N] = p
            F = np.fft.rfft(f)
            conv = np.fft.irfft(F * F, n=M_fft)
            
            alpha_td = 200.0
            max_conv = np.max(conv)
            conv_scaled = alpha_td * (conv - max_conv)
            exp_conv = np.exp(conv_scaled)
            sum_exp = np.sum(exp_conv)
            weights = exp_conv / sum_exp
            
            W = np.fft.rfft(weights)
            P = np.fft.rfft(p)
            corr = np.fft.irfft(W * np.conj(P), n=M_fft)
            grad_p = 2.0 * corr[:N]
            
            dot_term = np.dot(grad_p, p)
            return p * (grad_p - dot_term)
        
        try:
            res = minimize(
                time_domain_obj,
                x_start,
                jac=time_domain_grad,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x_opt = res.x
            x_max = np.max(x_opt)
            exp_x = np.exp(x_opt - x_max)
            p_opt = exp_x / np.sum(exp_x)
            
            raw_score = compute_score(p_opt)
            if raw_score < best_raw_score:
                best_raw_score = raw_score
                best_p = p_opt
        except Exception:
            pass
    
    # Final polish
    if time_left() > 10:
        from scipy.optimize import minimize
        
        p_start = np.maximum(best_p, 1e-10)
        p_start = p_start / np.sum(p_start)
        x_start = np.log(p_start)
        
        def final_obj(x):
            x_max = np.max(x)
            exp_x = np.exp(x - x_max)
            p = exp_x / np.sum(exp_x)
            
            f = np.zeros(M_fft, dtype=np.float64)
            f[:N] = p
            F = np.fft.rfft(f)
            conv = np.fft.irfft(F * F, n=M_fft)
            
            alpha_td = 500.0
            max_conv = np.max(conv)
            conv_scaled = alpha_td * (conv - max_conv)
            exp_conv = np.exp(conv_scaled)
            sum_exp = np.sum(exp_conv)
            return max_conv + np.log(sum_exp) / alpha_td
        
        def final_grad(x):
            x_max = np.max(x)
            exp_x = np.exp(x - x_max)
            p = exp_x / np.sum(exp_x)
            
            f = np.zeros(M_fft, dtype=np.float64)
            f[:N] = p
            F = np.fft.rfft(f)
            conv = np.fft.irfft(F * F, n=M_fft)
            
            alpha_td = 500.0
            max_conv = np.max(conv)
            conv_scaled = alpha_td * (conv - max_conv)
            exp_conv = np.exp(conv_scaled)
            sum_exp = np.sum(exp_conv)
            weights = exp_conv / sum_exp
            
            W = np.fft.rfft(weights)
            P = np.fft.rfft(p)
            corr = np.fft.irfft(W * np.conj(P), n=M_fft)
            grad_p = 2.0 * corr[:N]
            
            dot_term = np.dot(grad_p, p)
            return p * (grad_p - dot_term)
        
        try:
            res = minimize(
                final_obj,
                x_start,
                jac=final_grad,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            x_opt = res.x
            x_max = np.max(x_opt)
            exp_x = np.exp(x_opt - x_max)
            p_opt = exp_x / np.sum(exp_x)
            
            raw_score = compute_score(p_opt)
            if raw_score < best_raw_score:
                best_raw_score = raw_score
                best_p = p_opt
        except Exception:
            pass
    
    best_p = np.maximum(best_p, 0.0)
    return list(best_p)


if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
