import numpy as np
import time

N = 100

def compute_max_overlap(h):
    """Compute max_k C(k) where C(k) = sum_i h_i * (1 - h_{(i+k) mod N})."""
    h_comp = 1.0 - h
    H = np.fft.fft(h)
    Hc = np.fft.fft(h_comp)
    c = np.fft.ifft(H * np.conj(Hc)).real
    return np.max(c)

def construct():
    """
    Construct step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    Theoretical optimum: uniform h_i = 0.5 achieves C(k) = N/4 for all k,
    which is the provable lower bound. We start from this optimum and apply
    FFT-accelerated log-sum-exp refinement to verify robustness.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    # Start from the known theoretical optimum
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = compute_max_overlap(best_h)
    
    # Log-Sum-Exp parameters for smooth max approximation
    beta = 50.0
    lr = 0.005
    max_iter = 2000
    
    def logsumexp_max(x, b):
        """Smooth approximation of max(x) using log-sum-exp."""
        m = np.max(x)
        shifted = b * (x - m)
        # Clip to avoid overflow
        shifted = np.clip(shifted, -700, 700)
        return m + np.log(np.sum(np.exp(shifted))) / b
    
    def gradient_logsumexp_max(x, b):
        """Gradient (softmax weights) of log-sum-exp approximation of max."""
        m = np.max(x)
        shifted = b * (x - m)
        shifted = np.clip(shifted, -700, 700)
        weights = np.exp(shifted)
        s = np.sum(weights)
        if s > 0:
            weights /= s
        else:
            weights = np.full_like(x, 1.0 / len(x))
        return weights
    
    for it in range(max_iter):
        if time.time() > deadline - 10.0:
            break
        
        # Compute overlaps via FFT
        h_comp = 1.0 - h
        H = np.fft.fft(h)
        Hc = np.fft.fft(h_comp)
        c = np.fft.ifft(H * np.conj(Hc)).real
        
        # Smooth max and its gradient (softmax weights)
        c_max = logsumexp_max(c, beta)
        grad_c = gradient_logsumexp_max(c, beta)
        
        # Gradient of c_k with respect to h_j:
        # dc_k/dh_j = (1 - h_{(j+k) mod N}) - h_{(j-k) mod N}
        # grad_h_j = sum_k grad_c_k * [(1 - h_{(j+k) mod N}) - h_{(j-k) mod N}]
        #
        # term1_j = sum_k grad_c_k * (1 - h_{(j+k) mod N})
        #         = sum_k grad_c_k - sum_k grad_c_k * h_{(j+k) mod N}
        # term2_j = sum_k grad_c_k * h_{(j-k) mod N}
        #
        # So grad_h = sum(grad_c) * ones - corr(grad_c, h) - conv(grad_c, h)
        # where corr[a,b][j] = sum_k a[k]*b[(j+k)%N] and conv[a,b][j] = sum_k a[k]*b[(j-k)%N]
        
        G = np.fft.fft(grad_c)
        H_fft = np.fft.fft(h)
        
        # term1: sum_k grad_c_k * h_{(j+k) mod N} = circular cross-correlation
        # corr[j] = sum_k grad_c[k] * h[(j+k) mod N]
        # FFT: ifft(conj(fft(grad_c)) * fft(h))
        term1 = np.fft.ifft(np.conj(G) * H_fft).real
        
        # term2: sum_k grad_c_k * h_{(j-k) mod N} = circular convolution
        # conv[j] = sum_k grad_c[k] * h[(j-k) mod N]
        # FFT: ifft(fft(grad_c) * fft(h))
        term2 = np.fft.ifft(G * H_fft).real
        
        sum_grad_c = np.sum(grad_c)
        grad_h = sum_grad_c - term1 - term2
        
        # Gradient descent step
        h_new = h - lr * grad_h
        
        # Project to [0, 1] and adjust sum to N/2
        h_new = np.clip(h_new, 0.0, 1.0)
        current_sum = np.sum(h_new)
        target_sum = N / 2.0
        if current_sum > 0:
            h_new += (target_sum - current_sum) / N
            h_new = np.clip(h_new, 0.0, 1.0)
        
        # Check if improvement
        score = compute_max_overlap(h_new)
        if score < best_score - 1e-12:
            best_score = score
            best_h = h_new.copy()
        
        h = h_new
        
        # Adaptive learning rate decay
        if it % 200 == 0 and it > 0:
            lr *= 0.9
    
    # Final projection to ensure constraints are satisfied
    h_final = np.clip(best_h, 0.0, 1.0)
    for _ in range(200):
        diff = N / 2.0 - np.sum(h_final)
        if abs(diff) < 1e-12:
            break
        h_final += diff / N
        h_final = np.clip(h_final, 0.0, 1.0)
    
    # Verify the final result
    final_sum = np.sum(h_final)
    if abs(final_sum - N / 2.0) > 1e-9:
        # Fallback to uniform if projection failed
        h_final = np.full(N, 0.5)
    
    return h_final
