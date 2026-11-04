import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))

    # Initialize with seed
    h_best = np.ones(N)
    val_best = compute_value(h_best)
    
    rng = np.random.default_rng(42)
    
    # We want to minimize value = |2*N*max(c) / S^2|.
    # Scale invariance: let S = sum(h) = N. Then minimize 2*max(c)/N.
    # Equivalent to minimizing max(c) subject to sum(h)=N.
    # Let h = 1 + f, where sum(f)=0.
    # c = conv(h,h). We want to minimize max(c).
    
    # Gradient of max(c) w.r.t h is tricky because max is non-differentiable.
    # However, we can use a soft-max approximation or just use the subgradient 
    # at the argmax. 
    # Let k* = argmax(c). c[k*] = sum_i h[i] h[i+k*].
    # d c[k*] / d h[j] = h[j+k*] + h[j-k*] (with boundary handling).
    # To reduce max(c), we should decrease h[j] where the gradient is positive 
    # and increase where negative? 
    # Actually, we want to move h in the direction that decreases c[k*].
    # grad_h c[k*] has components h[j+k*] + h[j-k*].
    # We update h -= lr * grad, projected to sum=0.
    
    # Better approach: Use FFT to compute c and its gradients efficiently?
    # np.convolve is O(N^2). N=400, so 160k ops. Fast enough.
    # But let's stick to direct computation for simplicity and robustness, 
    # as N is small.
    
    # Strategy: 
    # 1. Start from h=1.
    # 2. Identify the index k* where c is maximum.
    # 3. Compute the "shadow price" gradient g[j] = h[j+k*] + h[j-k*].
    # 4. Update h -= alpha * g, then re-center to maintain sum=0.
    # 5. Repeat.
    
    # To handle the non-smoothness of max, we can use a "top-k" max approximation
    # or just stick to the single argmax which often works for this kind of 
    # "flatten the peak" objective.
    
    # Let's try a few random restarts with this gradient descent.
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Initialize
        if restart == 0:
            h = np.ones(N)
        else:
            # Add some noise to escape local minima
            h = np.ones(N) + rng.normal(0, 0.1, N)
            h -= np.mean(h) # Re-center
        
        # Optimize
        for step in range(2000):
            if time.time() > deadline:
                break
                
            c = np.convolve(h, h)
            k_max = np.argmax(c)
            val_cur = abs(2.0 * N * c[k_max] / (np.sum(h)**2))
            
            if val_cur < val_best:
                val_best = val_cur
                h_best = h.copy()
                
            # Compute gradient of c[k_max] w.r.t h
            # c[k_max] = sum_i h[i] * h[i+k_max]
            # Note: np.convolve is full mode.
            # If k_max >= 0: c[k] = sum_{i} h[i] h[i+k] for i where i+k < N.
            # The derivative w.r.t h[j] is h[j+k] + h[j-k] (if indices valid).
            
            g = np.zeros(N)
            if k_max >= 0:
                # Term 1: h[j] * h[j+k_max] -> d/dh[j] is h[j+k_max]
                # Valid for j such that j+k_max < N => j < N - k_max
                if k_max < N:
                    g[:N-k_max] += h[k_max:N]
                # Term 2: h[j-k_max] * h[j] -> d/dh[j] is h[j-k_max]
                # Valid for j such that j-k_max >= 0 => j >= k_max
                if k_max < N:
                    g[k_max:] += h[:N-k_max]
            else:
                # k_max is negative? In full convolve, indices are 0 to 2N-2.
                # The max is usually near the center (N-1).
                # Let's assume k_max is positive. If not, handle symmetrically.
                # Actually np.argmax returns index in [0, 2N-2].
                # If k_max < N-1, it corresponds to negative lag in correlation?
                # No, conv(h,h) at index k is sum h[i]h[i+k].
                # If k < N-1, it's a positive lag correlation.
                # If k > N-1, let k' = 2(N-1)-k, it's symmetric.
                # Due to symmetry of conv(h,h), c[k] = c[2(N-1)-k].
                # So we can map k_max to the range [0, N-1].
                if k_max > N-1:
                    k_eff = 2*(N-1) - k_max
                else:
                    k_eff = k_max
                
                # Recompute gradient with k_eff
                g = np.zeros(N)
                if k_eff < N:
                    g[:N-k_eff] += h[k_eff:N]
                    g[k_eff:] += h[:N-k_eff]
            
            # Project gradient to sum=0 subspace
            g -= np.mean(g)
            
            # Adaptive step size
            # We want to reduce c[k_max].
            # Try a small step.
            lr = 0.01
            h_new = h - lr * g
            # Re-center to maintain sum = N (or 0 for f)
            # Since we started with sum N, and g has mean 0, sum is preserved.
            # But numerical errors might creep in.
            h_new += (N - np.sum(h_new)) / N
            
            # Check if this step improves the objective
            c_new = np.convolve(h_new, h_new)
            val_new = abs(2.0 * N * np.max(c_new) / (np.sum(h_new)**2))
            
            if val_new < val_cur:
                h = h_new
            else:
                # Reduce learning rate or try different direction?
                # For simplicity, just reduce lr.
                lr *= 0.5
                if lr < 1e-6:
                    break
    
    return h_best
