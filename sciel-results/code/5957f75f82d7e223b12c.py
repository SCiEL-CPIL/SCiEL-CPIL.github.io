"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin before 150s kill
    
    # Baseline: Flat function
    best_f = np.ones(N)
    best_val = np.inf

    def compute_value(f):
        """Compute normalized max autoconvolution."""
        f = np.asarray(f, dtype=np.float64)
        s = np.sum(f)
        if s < 1e-12:
            return np.inf
        # Autoconvolution
        conv = np.convolve(f, f, mode='full')
        max_conv = np.max(conv)
        return max_conv / (s * s) * N

    best_val = compute_value(best_f)

    # Neural Network Parameterization
    # Latent vector z
    D_z = 10
    # Hidden layers
    D_h1 = 32
    D_h2 = 32
    # Output
    D_f = N

    # Initialize Weights (Xavier)
    W1 = np.random.randn(D_z, D_h1) * np.sqrt(2.0 / D_z)
    b1 = np.zeros(D_h1)
    W2 = np.random.randn(D_h1, D_h2) * np.sqrt(2.0 / D_h1)
    b2 = np.zeros(D_h2)
    W3 = np.random.randn(D_h2, D_f) * np.sqrt(2.0 / D_h2)
    b3 = np.zeros(D_f)

    # Adam Optimizer State
    m_z = np.zeros(D_z)
    v_z = np.zeros(D_z)
    t = 0
    lr = 0.005
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    def forward(z):
        h1 = np.maximum(W1.T @ z + b1, 0)  # ReLU
        h2 = np.maximum(W2.T @ h1 + b2, 0) # ReLU
        f = np.log1p(np.exp(W3.T @ h2 + b3)) # Softplus
        return f, h1, h2

    def objective_and_grad(z):
        f, h1, h2 = forward(z)
        s = np.sum(f)
        if s < 1e-12:
            return 1e10, np.zeros(D_z)
        
        # Use log-sum-exp for smooth max approximation
        conv = np.convolve(f, f, mode='full')
        beta_lse = 100.0
        m = np.max(conv)
        if m < 1e-12:
             return 1e10, np.zeros(D_z)
        
        exp_conv = np.exp((conv - m) * beta_lse)
        sum_exp = np.sum(exp_conv)
        weights = exp_conv / sum_exp
        log_sum_exp_val = m + np.log(sum_exp) / beta_lse
        
        val = N * log_sum_exp_val / (s * s)
        
        # Gradient of val wrt f
        # val = N * LSE * s^-2
        # d(val)/df = N * (d(LSE)/df * s^-2 - 2 * LSE * s^-3)
        
        # d(LSE)/df
        # LSE = m + log(sum exp) / beta
        # d(LSE)/dk = weights[k] / beta
        # d(conv[k])/df[i] = f[i+k] + f[i-k]
        
        # Compute gradient of weighted sum of conv
        # grad_lse_conv = sum_k weights[k] * d(conv[k])/df
        # This is equivalent to:
        # grad_f = (weights * f) + (flip(weights) * f) ?
        # Let's use the FFT method for efficiency or direct if N is small enough?
        # N=600, conv length 1199.
        # Direct loop over k is too slow.
        # Use FFT for correlation.
        
        # d(conv[k])/df[i] is the correlation of f with delta at k?
        # Actually, grad_f[i] = sum_k weights[k] (f[i+k] + f[i-k])
        # = sum_k weights[k] f[i+k] + sum_k weights[k] f[i-k]
        
        # Term 1: sum_k weights[k] f[i+k]
        # This is cross-correlation of f and weights.
        # Term 2: sum_k weights[k] f[i-k]
        # This is convolution of weights and f.
        
        nfft = 1
        while nfft < 2 * N:
            nfft *= 2
            
        F = np.fft.rfft(f, n=nfft)
        W = np.fft.rfft(weights, n=nfft)
        
        # Term 2: conv(weights, f)
        # ifft(fft(w) * fft(f))
        conv_wf = np.fft.irfft(W * F, n=nfft)[:N]
        
        # Term 1: corr(f, weights)
        # ifft(fft(f) * conj(fft(w)))
        corr_fw = np.fft.irfft(F * np.conj(W), n=nfft)[:N]
        
        # Note: weights are defined on lags 0 to 2N-2.
        # My weights array is length 2N-1.
        # The FFT padding handles the linear nature.
        
        grad_lse_conv = (corr_fw + conv_wf) / beta_lse
        
        # d(val)/df
        grad_val_f = N * (grad_lse_conv / (s * s) - 2 * log_sum_exp_val / (s ** 3))
        
        # Backprop to z
        pre_act3 = W3.T @ h2 + b3
        df_dpre3 = 1 / (1 + np.exp(-pre_act3)) # Sigmoid
        grad_pre3 = grad_val_f * df_dpre3
        
        grad_h2 = W3 @ grad_pre3
        pre_act2 = W2.T @ h1 + b2
        drelu2 = (pre_act2 > 0).astype(float)
        grad_pre2 = grad_h2 * drelu2
        
        grad_h1 = W2 @ grad_pre2
        pre_act1 = W1.T @ z + b1
        drelu1 = (pre_act1 > 0).astype(float)
        grad_pre1 = grad_h1 * drelu1
        
        grad_z = W1 @ grad_pre1
        
        return val, grad_z

    # Optimization Loop
    z = np.zeros(D_z)
    
    # Warm start with flat?
    # The NN might not produce flat easily.
    # We can try to initialize W3, b3 to produce flat.
    # f = softplus(W3 h2 + b3)
    # If h2 is 0, f = softplus(b3) = log(1+exp(b3)).
    # To get f ~ 1, b3 ~ log(e-1) ~ 0.54.
    b3 = np.full(D_f, 0.54)
    
    for i in range(1000):
        if time.time() > deadline:
            break
            
        val, grad_z = objective_and_grad(z)
        
        if val < best_val:
            best_val = val
            f, _, _ = forward(z)
            best_f = f
        
        # Adam Update
        t += 1
        m_z = beta1 * m_z + (1 - beta1) * grad_z
        v_z = beta2 * v_z + (1 - beta2) * (grad_z ** 2)
        m_hat = m_z / (1 - beta1 ** t)
        v_hat = v_z / (1 - beta2 ** t)
        z -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
    # Final evaluation
    f, _, _ = forward(z)
    val_final = compute_value(f)
    if val_final < best_val:
        best_f = f

    # Ensure non-negative
    best_f = np.maximum(best_f, 0)
    
    return [float(h) for h in best_f]
# EVOLVE-BLOCK-END
