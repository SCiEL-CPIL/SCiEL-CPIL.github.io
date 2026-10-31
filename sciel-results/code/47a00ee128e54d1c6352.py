import numpy as np
import time

N = 600

def compute_metric_and_grad(h):
    """
    Computes the normalized max autoconvolution metric and its gradient with respect to h.
    Uses a soft-max approximation for the max to ensure differentiability.
    """
    h = np.maximum(h, 1e-12) # Ensure positive for log/grad stability if needed, though we optimize log-h
    s = h.sum()
    if s < 1e-15:
        return 1e10, np.zeros_like(h)
    
    # Normalize h to sum to 1 to fix scale, as the metric is scale-invariant
    # Actually, the metric is max(R)/R[0]. R scales with h^2. So it is scale invariant.
    # We can just work with h directly.
    
    # FFT for autoconvolution
    fft_size = 2 * N
    H = np.fft.rfft(h, fft_size)
    P = H * np.conj(H) # Power spectrum
    R = np.fft.irfft(P, fft_size)
    
    # R[0] is the peak
    peak = R[0]
    if peak < 1e-15:
        return 1e10, np.zeros_like(h)
    
    # Sidelobes are R[1:N] and R[N+1:2N-1] (due to symmetry, R[1:N] is enough for max)
    # Actually, linear autoconv of length N has length 2N-1.
    # Indices 0 to 2N-2.
    # R[0] is lag 0.
    # R[1:N] are lags 1 to N-1.
    # R[N:2N-1] are lags -1 to -(N-1).
    # Due to symmetry of real h, R[k] = R[-k].
    # So we only need to check R[1:N].
    
    sidelobes = R[1:N]
    
    # To minimize the max, we use a differentiable proxy: LogSumExp
    # J = log(sum(exp(alpha * sidelobes))) / alpha - log(peak)
    # But we want max(sidelobes) / peak.
    # Let's optimize log(max_sidelobe) - log(peak).
    # Gradient of log(max) is tricky. Let's use the soft-max value.
    
    alpha = 50.0 # Sharpness of the max approximation
    # Shift for numerical stability
    max_side = np.max(sidelobes)
    if max_side < 1e-15:
        max_side = 1e-15
    
    # Soft max
    exp_sides = np.exp(alpha * (sidelobes - max_side))
    sum_exp = np.sum(exp_sides)
    soft_max_val = max_side + np.log(sum_exp) / alpha
    
    metric = soft_max_val / peak
    
    # Gradient
    # d/dh (soft_max_val / peak)
    # = (1/peak) * d(soft_max_val)/dh - (soft_max_val/peak^2) * d(peak)/dh
    
    # d(peak)/dh = 2*h
    # d(soft_max_val)/dh:
    # soft_max_val = max_side + (1/alpha) * log(sum(exp(alpha*(sidelobes - max_side))))
    # Let S = sum(exp(alpha*(sidelobes - max_side)))
    # d(soft_max)/dh = (1/alpha) * (1/S) * d(S)/dh
    # d(S)/dh = sum_k exp(alpha*(sidelobes_k - max_side)) * alpha * d(sidelobes_k)/dh
    # = alpha * sum_k exp(...) * d(sidelobes_k)/dh
    
    # So d(soft_max)/dh = sum_k w_k * d(sidelobes_k)/dh, where w_k = exp(...)/S
    # w_k is the weight of each sidelobe in the soft-max.
    
    weights = exp_sides / sum_exp # Shape (N-1,)
    
    # We need d(sidelobes_k)/dh for k=1..N-1.
    # sidelobes_k = R[k].
    # R = IFFT(|FFT(h)|^2).
    # This is a quadratic form. R = h^T T_k h ? No, R is the convolution.
    # R[k] = sum_i h[i] h[i+k].
    # d R[k] / d h[j] = h[j+k] + h[j-k] (with appropriate bounds/wrapping for linear conv).
    
    # Instead of deriving the full gradient matrix, we can use the fact that
    # R = IFFT(FFT(h)^2).
    # Let's compute the gradient via the chain rule in the frequency domain or use automatic differentiation logic.
    
    # d R / d h is a linear operator.
    # If we perturb h by delta_h, R changes by delta_R.
    # delta_R = IFFT( 2 * FFT(h) * FFT(delta_h) ) (approximately, for real signals).
    # Actually, d/ dh |H|^2 = 2 H dH/dh.
    # dH/dh is the FFT matrix.
    
    # Let's compute the gradient vector grad_h of the soft_max part.
    # grad_softmax = sum_{k=1}^{N-1} w_k * (d R[k] / d h)
    
    # d R[k] / d h[j] = h[j+k] + h[j-k] (if indices valid, else 0 for linear conv).
    # This is equivalent to:
    # Let W be a vector of weights for lags 1..N-1.
    # grad_softmax[j] = sum_{k=1}^{N-1} w_k * (h[j+k] + h[j-k])
    
    # This is a convolution of h with a kernel K defined by weights w.
    # K[k] = w_k for k=1..N-1, K[-k] = w_k, K[0]=0.
    
    # We can compute this via FFT as well.
    # Construct kernel K in time domain (length 2N-1 or N for circular?).
    # Since R is linear autoconv, the gradient is linear convolution.
    
    # Let's build the kernel for the gradient of the soft-max.
    # Kernel length 2N-1.
    # But h has length N.
    
    # Alternative: Compute gradient in frequency domain?
    # J = log(sum exp(alpha R_k)) / alpha - log(R_0)
    # This is getting complex. Let's use a simpler, robust approach:
    # Finite difference gradient is too slow.
    # Let's use the analytic gradient of the max sidelobe if we identify the argmax.
    # The max sidelobe index is fixed (mostly) during optimization.
    # Let k_max be the index of the maximum sidelobe.
    # Then metric = R[k_max] / R[0].
    # grad_metric = (1/R[0]) * dR[k_max]/dh - (R[k_max]/R[0]^2) * dR[0]/dh
    
    # dR[0]/dh = 2h
    # dR[k_max]/dh[j] = h[j+k_max] + h[j-k_max] (linear bounds)
    
    # Let's find the current argmax.
    k_max_idx = np.argmax(sidelobes) + 1 # 1-based lag
    k_max = k_max_idx
    
    # Compute dR[k_max]/dh
    # grad_side = np.zeros(N)
    # for j in range(N):
    #    if j+k_max < N: grad_side[j] += h[j+k_max]
    #    if j-k_max >= 0: grad_side[j] += h[j-k_max]
    
    # Vectorized:
    grad_side = np.zeros(N)
    if k_max > 0:
        # h[j+k]
        grad_side[:N-k_max] += h[k_max:]
        # h[j-k]
        grad_side[k_max:] += h[:N-k_max]
        
    # Total gradient of metric
    grad_metric = (1.0 / peak) * grad_side - (metric * 2.0 / peak) * h
    
    # Note: This gradient assumes k_max is constant. If k_max changes, we have a discontinuity.
    # This is acceptable for a descent method if we use small steps.
    
    return metric, grad_metric

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # Initial guess: Raised Cosine (better than Box)
    x = np.linspace(-1, 1, N)
    h = 0.5 * (1 + np.cos(np.pi * x))
    h = np.maximum(h, 1e-6)
    
    # Normalize
    h = h / h.sum()
    
    current_metric, _ = compute_metric_and_grad(h)
    best_h = h.copy()
    best_metric = current_metric
    
    # Adam Optimizer on log-h parameters?
    # Or just on h with projection h >= 0.
    # Let's use Adam on h directly, with projection.
    
    # Initialize Adam state
    m = np.zeros(N)
    v = np.zeros(N)
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    lr = 0.01
    
    t = 0
    
    # Optimization Loop
    while time.time() < deadline - 5.0:
        t += 1
        
        # Compute gradient
        metric, grad = compute_metric_and_grad(h)
        
        if metric < best_metric:
            best_metric = metric
            best_h = h.copy()
            
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        h -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Project to non-negative
        h = np.maximum(h, 0.0)
        
        # Renormalize to keep scale stable (optional, metric is scale invariant)
        # But preventing underflow/overflow is good.
        s = h.sum()
        if s < 1e-15:
            h = np.ones(N) / N
            break
        h = h / s
        
        # Adaptive learning rate: reduce if oscillating
        if t % 100 == 0:
            # Check if we are stuck
            pass # Simple constant LR is often fine with Adam
            
        # If metric is very low, reduce LR to converge
        if best_metric < 0.1:
            lr = max(0.001, lr * 0.99)
            
    # Final refinement: Coordinate descent on the best result to polish
    # Only if time permits
    if time.time() < deadline - 2.0:
        h_current = best_h.copy()
        val_current = best_metric
        step = 0.05
        for _ in range(20): # Few passes
            if time.time() > deadline - 1.0:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 0.5:
                    break
                old_val = h_current[i]
                # Try small perturbation
                for delta in [step, -step]:
                    if old_val + delta < 0:
                        continue
                    h_trial = h_current.copy()
                    h_trial[i] += delta
                    h_trial = np.maximum(h_trial, 0)
                    s = h_trial.sum()
                    if s < 1e-15: continue
                    h_trial = h_trial / s
                    val_trial, _ = compute_metric_and_grad(h_trial)
                    if val_trial < val_current:
                        h_current = h_trial
                        val_current = val_trial
                        improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break
            
    if val_current < best_metric:
        best_h = h_current
        best_metric = val_current

    return list(best_h)
