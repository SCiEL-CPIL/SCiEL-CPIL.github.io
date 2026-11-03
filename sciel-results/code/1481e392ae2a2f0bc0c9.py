import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 140.0  # safety margin before 150s kill

def compute_autoconv(h):
    """Compute linear autoconvolution coefficients (unnormalized) using FFT."""
    # Pad to avoid circular convolution
    size = 2 * N
    fft_h = np.fft.rfft(h, n=size)
    autoconv = np.fft.irfft(fft_h * np.conj(fft_h), n=size)
    # Linear autoconvolution: autoconv[0:N] gives shifts 0 to N-1
    # By symmetry, negative shifts mirror positive
    return autoconv[:N]

def smooth_max(x, beta=50.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    # Numerically stable log-sum-exp
    if x_max < -100:
        return x_max
    exp_x = np.exp(x - x_max)
    return x_max + np.log(np.sum(exp_x)) / beta

def objective_and_grad(h, beta=50.0):
    """Compute smooth max of autoconvolution and its gradient.
    
    h is normalized to sum to 1 before evaluation.
    Returns (value, gradient) where gradient is w.r.t. h (unnormalized).
    """
    h_sum = np.sum(h)
    if h_sum <= 0:
        return 1e10, np.zeros_like(h)
    
    # Normalize
    hn = h / h_sum
    
    # Compute autoconvolution coefficients
    autoconv = compute_autoconv(hn)
    
    # The normalized max is max(autoconv) since we normalized sum to 1
    # (the dx factors cancel in the ratio)
    # Actually let me re-derive:
    # mass = dx * sum(h)
    # autoconv[m] = dx * sum_i h[i]h[i+m]
    # normalized = max_m autoconv[m] / mass^2 = max_m [dx * sum h[i]h[i+m]] / [dx^2 * (sum h)^2]
    #            = max_m [sum h[i]h[i+m]] / [dx * (sum h)^2]
    # With dx = 1/(2N):
    #            = 2N * max_m [sum h[i]h[i+m]] / (sum h)^2
    # If we normalize sum h = 1: = 2N * max_m [sum h[i]h[i+m]]
    # But wait, let me just compute the actual quantity.
    
    dx = 1.0 / (2.0 * N)
    mass = dx * h_sum
    if mass <= 0:
        return 1e10, np.zeros_like(h)
    
    # autoconv[m] as computed is sum_i hn[i]*hn[i+m] (with hn = h/h_sum)
    # The actual autoconv value is dx * sum_i h[i]*h[i+m] = dx * h_sum^2 * sum_i hn[i]*hn[i+m]
    # = dx * h_sum^2 * autoconv[m]
    # normalized = max_m [dx * h_sum^2 * autoconv[m]] / (dx * h_sum)^2
    #            = max_m autoconv[m] / dx
    #            = 2N * max_m autoconv[m]
    
    # For the smooth version:
    smax = smooth_max(autoconv, beta)
    value = 2.0 * N * smax
    
    # Gradient of smooth_max w.r.t. autoconv:
    # d(smooth_max)/d(autoconv[m]) = exp((autoconv[m] - max)/beta) / (beta * sum exp((autoconv[j]-max)/beta))
    # = softmax(autoconv/beta)[m] / beta * beta = softmax(autoconv/beta)[m]
    # More precisely: d/dx_m [max + log(sum exp((x_j - max)/beta))/beta]
    # = exp((x_m - max)/beta) / (beta * sum_j exp((x_j - max)/beta))
    
    x_max = np.max(autoconv)
    exp_x = np.exp((autoconv - x_max) / beta)
    sum_exp = np.sum(exp_x)
    weights = exp_x / (beta * sum_exp)  # d(smooth_max)/d(autoconv[m])
    
    # Now d(autoconv[m])/d(hn[i]) where hn = h/h_sum
    # autoconv[m] = sum_i hn[i] * hn[i+m] (for m >= 0, with appropriate bounds)
    # This is a quadratic form. The gradient w.r.t. hn is:
    # d(autoconv[m])/d(hn[k]) = hn[k-m] + hn[k+m] (with appropriate boundary handling)
    # For the full autoconv vector, the gradient w.r.t. hn is:
    # grad_autoconv = 2 * autoconv (due to symmetry) ... no, let me think more carefully.
    
    # autoconv[m] = sum_i hn[i] * hn[i+m] for m=0,...,N-1
    # d(autoconv[m])/d(hn[k]) = sum_i [delta_{i,k} * hn[i+m] + hn[i] * delta_{i+m,k}]
    #                          = hn[k+m] (if k+m < N) + hn[k-m] (if k-m >= 0)
    #                          = (hn shifted by m)[k] + (hn shifted by -m)[k]
    
    # The gradient of the smooth max w.r.t. hn:
    # grad_hn = sum_m weights[m] * d(autoconv[m])/d(hn)
    #         = sum_m weights[m] * (shift(hn, m) + shift(hn, -m))
    
    # This can be computed efficiently using FFT!
    # sum_m weights[m] * shift(hn, m) is a convolution of weights with hn
    # Specifically: (weights * hn)[k] = sum_m weights[m] * hn[k-m]
    # And sum_m weights[m] * shift(hn, -m)[k] = sum_m weights[m] * hn[k+m]
    
    # Let's compute:
    # A = sum_m weights[m] * hn[k+m] for each k  -> this is cross-correlation
    # B = sum_m weights[m] * hn[k-m] for each k  -> this is also related
    
    # Actually: sum_m weights[m] * (hn[k+m] + hn[k-m])
    # = sum_m weights[m]*hn[k+m] + sum_m weights[m]*hn[k-m]
    
    # The first term: for fixed k, sum_m weights[m]*hn[k+m]
    # The second term: for fixed k, sum_m weights[m]*hn[k-m]
    
    # Both can be computed via convolution.
    # Let's use FFT:
    # conv_w_hn[k] = sum_m weights[m] * hn[k-m]  (standard convolution)
    # We need sum_m weights[m] * hn[k+m] = sum_m weights[m] * hn[-(m-k)] 
    # = sum_m weights[m] * hn_flipped[k-m] where hn_flipped[j] = hn[-j]... 
    
    # Simpler: just compute both convolutions.
    size = 2 * N
    fft_weights = np.fft.rfft(weights, n=size)
    fft_hn = np.fft.rfft(hn, n=size)
    
    # conv1[k] = sum_m weights[m] * hn[k-m] (for k=0..N-1)
    conv1 = np.fft.irfft(fft_weights * fft_hn, n=size)[:N]
    
    # conv2[k] = sum_m weights[m] * hn[k+m]
    # = sum_m weights[m] * hn[k+m]
    # Let j = m, then we want sum_j weights[j] * hn[k+j]
    # This is the cross-correlation: sum_j weights[j] * hn[k+j]
    # = (weights reversed * hn)[k+N-1] ... 
    # Alternatively: sum_j weights[j] * hn[k+j] = sum_j weights[j] * hn_flipped[N-1-k-j]
    # where hn_flipped[i] = hn[N-1-i]
    # This is getting complicated. Let me just use the fact that:
    # sum_m weights[m] * hn[k+m] = (weights * hn)[k] if we define convolution appropriately
    # Actually, standard convolution: (w * h)[k] = sum_j w[j] * h[k-j]
    # We want: sum_j w[j] * h[k+j] = sum_j w[j] * h[-(k-j)] 
    # Let h_rev[i] = h[N-1-i], then h[k+j] = h_rev[N-1-k-j]
    # sum_j w[j] * h_rev[N-1-k-j] = (w * h_rev)[N-1-k]
    
    hn_rev = hn[::-1]
    fft_hn_rev = np.fft.rfft(hn_rev, n=size)
    conv2_full = np.fft.irfft(fft_weights * fft_hn_rev, n=size)
    conv2 = conv2_full[N-1-N+1:N]  # This is messy
    
    # Let me just compute it directly for clarity, using vectorized operations
    # Since N=600, we can afford O(N^2) for the gradient if needed, but let's try FFT.
    
    # Actually, the simplest correct approach:
    # grad_autoconv[m] = sum_i hn[i]*hn[i+m]
    # d/d(hn[k]) = sum_i [delta_{i,k}hn[i+m] + hn[i]delta_{i+m,k}]
    #            = hn[k+m] (if 0<=k+m<N) + hn[k-m] (if 0<=k-m<N)
    
    # So grad_total[k] = sum_m weights[m] * (hn[k+m] + hn[k-m])
    #                   = sum_m weights[m]*hn[k+m] + sum_m weights[m]*hn[k-m]
    
    # Term 1: sum_m weights[m]*hn[k+m]
    # Term 2: sum_m weights[m]*hn[k-m]
    
    # Term 2 is standard convolution: (weights * hn)[k] = sum_m weights[m]*hn[k-m]
    # Term 1: sum_m weights[m]*hn[k+m]
    # Let's compute Term 1 by reversing weights:
    # sum_m weights[m]*hn[k+m] = sum_m weights[N-1-m]*hn[k+N-1-(N-1-m)]... no
    # sum_m w[m]*h[k+m], let n = m, = sum_n w[n]*h[k+n]
    # This is the cross-correlation of w and h at lag k.
    # crosscorr[k] = sum_n w[n]*h[k+n]
    # = sum_n w[n]*h_flipped[N-1-k-n] where h_flipped[i]=h[N-1-i]
    # = (w * h_flipped)[N-1-k]
    
    # Let me just do it the straightforward way with two FFTs
    # For Term 2: conv_w_hn[k] = sum_m w[m]*hn[k-m]
    # For Term 1: we need sum_m w[m]*hn[k+m]
    #   = sum_m w[m]*hn[k+m]
    #   Define w_rev[m] = w[N-1-m]
    #   Then sum_m w[m]*hn[k+m] = sum_m w_rev[N-1-m]*hn[k+m]
    #   Let j = N-1-m, m = N-1-j
    #   = sum_j w_rev[j]*hn[k+N-1-j]
    #   = (w_rev * hn)[k+N-1]
    
    w_rev = weights[::-1]
    fft_w_rev = np.fft.rfft(w_rev, n=size)
    term1_full = np.fft.irfft(fft_w_rev * fft_hn, n=size)
    term1 = term1_full[N-1:N-1+N]  # indices N-1 to 2N-2
    
    # Hmm, this is getting error-prone. Let me just compute the gradient 
    # using the fact that we can express it as a matrix-vector product.
    # But that's O(N^2). For N=600, that's 360000 operations per gradient eval,
    # which is fine for a few hundred iterations.
    
    # Actually, let me just use the direct formula with vectorized numpy:
    # grad_hn[k] = sum_m weights[m] * (hn[k+m] + hn[k-m])
    
    # Build the gradient directly
    grad_hn = np.zeros(N)
    for m in range(N):
        if weights[m] == 0:
            continue
        # hn[k+m] for k=0..N-1-m
        if m == 0:
            grad_hn += weights[m] * 2.0 * hn
        else:
            # hn[k+m] for k=0..N-1-m
            grad_hn[:N-m] += weights[m] * hn[m:]
            # hn[k-m] for k=m..N-1
            grad_hn[m:] += weights[m] * hn[:N-m]
    
    # Now convert gradient from hn to h:
    # hn = h / h_sum
    # d(hn)/d(h) = (I - hn*hn^T) / h_sum  ... no
    # hn_i = h_i / h_sum
    # d(hn_i)/d(h_j) = delta_{ij}/h_sum - h_i/(h_sum^2)
    # So grad_h = (1/h_sum) * (grad_hn - hn * sum(grad_hn))
    # Wait: d(f)/d(h_j) = sum_i d(f)/d(hn_i) * d(hn_i)/d(h_j)
    # = sum_i grad_hn[i] * (delta_{ij}/h_sum - hn_i/h_sum^2)
    # = grad_hn[j]/h_sum - (1/h_sum^2) * sum_i grad_hn[i]*hn_i
    
    sum_grad_hn_hn = np.dot(grad_hn, hn)
    grad_h = (grad_hn - hn * sum_grad_hn_hn) / h_sum
    
    # Also need to account for the 2N factor in value
    grad_h *= 2.0 * N
    
    return value, grad_h

def construct():
    start_time = time.time()
    
    # Start with uniform
    h0 = np.ones(N)
    best_h = h0.copy()
    
    # Compute initial value
    dx = 1.0 / (2.0 * N)
    mass = dx * np.sum(h0)
    autoconv0 = compute_autoconv(h0)
    best_val = 2.0 * N * np.max(autoconv0)
    
    bounds = [(0.0, None)] * N
    
    # Use multiple betas for smooth max, starting coarse and refining
    betas = [10.0, 30.0, 100.0, 300.0]
    
    for beta in betas:
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        
        def obj_grad(h):
            val, grad = objective_and_grad(h, beta=beta)
            return val, grad
        
        try:
            result = minimize(
                obj_grad,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_h = result.x.copy()
        except Exception as e:
            pass
    
    # Final polish with very high beta (near-exact max)
    if time.time() - start_time < TIME_LIMIT - 10:
        try:
            beta_final = 1000.0
            def obj_grad_final(h):
                val, grad = objective_and_grad(h, beta=beta_final)
                return val, grad
            
            result = minimize(
                obj_grad_final,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 10}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_h = result.x.copy()
        except Exception:
            pass
    
    # Ensure nonnegativity
    best_h = np.clip(best_h, 0, None)
    
    # Compute final score
    dx = 1.0 / (2.0 * N)
    mass = dx * np.sum(best_h)
    if mass > 0:
        autoconv_final = compute_autoconv(best_h)
        final_score = 2.0 * N * np.max(autoconv_final)
    else:
        final_score = 1e10
    
    return list(best_h)
