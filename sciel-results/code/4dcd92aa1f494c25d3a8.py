"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    def compute_corr_fft(h):
        """Compute circular autocorrelation using FFT."""
        fft_h = np.fft.fft(h, n=2 * N)
        corr = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
        return corr
    
    def max_overlap(h):
        """Compute the actual maximum overlap."""
        corr = compute_corr_fft(h)
        # Overlap at shift k = N/2 - corr[k]
        # We want to minimize max_k overlap
        overlaps = N / 2.0 - corr
        return np.max(overlaps)
    
    def project_to_polytope(h, S):
        """Project h onto { x : sum(x) = S, 0 <= x_i <= 1 } using binary search."""
        h = np.clip(h, 0.0, 1.0)
        current_sum = np.sum(h)
        if abs(current_sum - S) < 1e-12:
            return h
        
        lo = -1.0
        hi = 2.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            projected = np.clip(h - mid, 0.0, 1.0)
            val = np.sum(projected)
            if val > S:
                lo = mid
            else:
                hi = mid
        
        lambda_star = (lo + hi) / 2.0
        return np.clip(h - lambda_star, 0.0, 1.0)
    
    def smooth_max_and_grad(h, beta):
        """
        Returns (smooth_max_value, gradient) for the max overlap.
        Overlap O_k = N/2 - sum h_i h_{i+k}
        We want to minimize max_k O_k.
        """
        corr = compute_corr_fft(h)
        overlaps = N / 2.0 - corr
        
        # Stable log-sum-exp for soft max
        max_o = np.max(overlaps)
        exps = np.exp(beta * (overlaps - max_o))
        sum_exps = np.sum(exps)
        
        # Weights for the soft max
        weights = exps / sum_exps
        
        # Smooth max value
        val = max_o + np.log(sum_exps) / beta
        
        # Gradient of O_k w.r.t h_j:
        # O_k = N/2 - sum_i h_i h_{i+k}
        # dO_k/dh_j = - (h_{j+k} + h_{j-k}) for circular indices
        # Total grad_j = sum_k weights[k] * (- (h_{j+k} + h_{j-k}))
        
        # Efficient computation using circular convolution via FFT
        # term_sum[j] = sum_k weights[k] * (h_{j+k} + h_{j-k})
        # = 2 * sum_k weights[k] * h_{j+k} if weights is symmetric
        # But weights may not be symmetric, so compute directly
        
        # Use FFT for convolution: convolve weights with h
        # c[j] = sum_k weights[k] * h[(j-k) % N]
        fft_w = np.fft.fft(weights, n=2 * N)
        fft_h = np.fft.fft(h, n=2 * N)
        conv1 = np.fft.ifft(fft_w * fft_h).real[:N]  # sum_k w[k] h[j-k]
        
        # For h[j+k], that's sum_k w[k] h[j+k] = sum_k w[N-k] h[j-(N-k)] 
        # Or just compute: d[j] = sum_k weights[k] h[(j+k) % N]
        # This is correlation, not convolution
        # d[j] = sum_k weights[k] h[j+k] = sum_m weights[(j-m) % N] h[m]
        # Let's just use a loop since N=100 is small and weights are sparse-ish
        
        term_sum = np.zeros(N)
        for k in range(N):
            if weights[k] > 1e-10:
                term_sum += weights[k] * (np.roll(h, -k) + np.roll(h, k))
        
        grad = -term_sum
        return val, grad
    
    best_h = np.full(N, 0.5)
    best_score = max_overlap(best_h)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.full(N, 0.5))
    
    # 2. Sinusoidal patterns with various frequencies and amplitudes
    for freq in [1, 2, 3, 4, 5, 7, 11, 13, 17, 23]:
        for amp in [0.1, 0.2, 0.3, 0.4, 0.45]:
            h = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
            h = project_to_polytope(h, N / 2.0)
            inits.append(h)
    
    # 3. Cosine patterns
    for freq in [1, 2, 3, 5, 7]:
        for amp in [0.15, 0.25, 0.35, 0.45]:
            h = 0.5 + amp * np.cos(2 * np.pi * freq * np.arange(N) / N)
            h = project_to_polytope(h, N / 2.0)
            inits.append(h)
    
    # 4. Random perturbations
    for _ in range(15):
        h = 0.5 + 0.15 * rng.standard_normal(N)
        h = project_to_polytope(h, N / 2.0)
        inits.append(h)
    
    # 5. Binary-like sequences (rounded)
    for _ in range(10):
        h = (rng.random(N) > 0.5).astype(float)
        h = project_to_polytope(h, N / 2.0)
        inits.append(h)
    
    # 6. Sawtooth patterns
    for period in [3, 4, 5, 7, 10, 13]:
        for phase in range(period):
            h = np.zeros(N)
            for i in range(N):
                pos = (i - phase) % period
                if pos < period // 2:
                    h[i] = 1.0
                else:
                    h[i] = 0.0
            h = project_to_polytope(h, N / 2.0)
            inits.append(h)
    
    # Optimize from each initialization
    for h0 in inits:
        if time.time() > deadline - 5.0:
            break
        
        h = h0.copy()
        
        # Adaptive parameters
        beta = 5.0  # Sharpness of soft max
        alpha = 0.05  # Step size
        max_steps = 150
        patience = 30
        no_improve = 0
        
        for step in range(max_steps):
            if time.time() > deadline - 3.0:
                break
            
            val, grad = smooth_max_and_grad(h, beta)
            
            # Track actual score
            actual_score = max_overlap(h)
            if actual_score < best_score:
                best_score = actual_score
                best_h = h.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # Gradient step
            h_new = h - alpha * grad
            h_new = project_to_polytope(h_new, N / 2.0)
            
            # Check if step improved
            new_val, _ = smooth_max_and_grad(h_new, beta)
            if new_val < val:
                h = h_new
                # Occasionally increase beta to sharpen
                if step % 20 == 0:
                    beta *= 1.5
                # Decrease step size slowly
                alpha *= 0.98
            else:
                # Backtrack
                alpha *= 0.5
                if alpha < 1e-6:
                    break
            
            # Early stop if no improvement for many steps
            if no_improve > patience:
                break
    
    # Final verification and cleanup
    final_h = best_h.copy()
    final_h = project_to_polytope(final_h, N / 2.0)
    
    # Ensure sum is exactly N/2
    diff = N / 2.0 - np.sum(final_h)
    if abs(diff) > 1e-10:
        mask = (final_h > 1e-6) & (final_h < 1 - 1e-6)
        if np.any(mask):
            final_h[mask] += diff / np.sum(mask)
            final_h = np.clip(final_h, 0.0, 1.0)
    
    return final_h
# EVOLVE-BLOCK-END
