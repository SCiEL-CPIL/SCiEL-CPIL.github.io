import numpy as np
import time
import sys

N = 600
W = 0.5 / N

def ratio_from_h(h):
    """
    Compute the normalized maximum of the autoconvolution.
    Metric: max(g) / sum(h^2)
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    
    l2_sq = np.sum(h * h)
    if l2_sq < 1e-15:
        return float('inf')
        
    # Exact convolution for step functions
    # The continuous convolution integral is w * sum(h_i h_{k-i})
    conv = np.convolve(h, h, mode='full') * W
    peak = np.max(conv)
    return float(peak / l2_sq)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin

    best_h = None
    best_val = float('inf')

    # Initializations
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    inits = [
        np.ones(N),
        np.exp(-20 * x**2),
        np.exp(-50 * x**2),
        0.5 * (1 + np.cos(2 * np.pi * x)),
        np.clip(1 - np.abs(2*x), 0, None), # Triangle
    ]
    
    rng = np.random.default_rng(42)
    # Add some random smooth inits
    for _ in range(5):
        noise = rng.normal(0, 1, N)
        # Smooth it
        kernel = np.exp(-np.linspace(-5, 5, 21)**2)
        sm = np.convolve(noise, kernel, mode='same')
        inits.append(np.clip(sm, 0, None))

    for h_init in inits:
        if time.time() > deadline - 20.0:
            break
            
        current_h = h_init.copy()
        current_h = np.clip(current_h, 0, None)
        if np.sum(current_h**2) < 1e-10:
            continue
        # Normalize to unit norm to simplify dynamics? 
        # No, ratio is scale invariant, so we can just work with any scale.
        # But normalizing helps with gradient magnitude.
        norm = np.linalg.norm(current_h)
        if norm > 0:
            current_h = current_h / norm
            
        current_val = ratio_from_h(current_h)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()

        # Gradient Descent
        # We minimize R(h) = max(g) / ||h||^2
        # Since we normalized h, ||h||^2 = 1.
        # So we minimize max(g).
        # But if we update h, norm changes.
        # Let's use the full gradient of the ratio.
        
        lr = 0.05
        momentum = 0.9
        v = np.zeros_like(current_h)
        
        # Adam parameters
        m = np.zeros_like(current_h)
        u = np.zeros_like(current_h)
        b1 = 0.9
        b2 = 0.999
        eps = 1e-8
        
        max_iters = 2000
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
                
            # 1. Compute convolution and peak
            conv = np.convolve(current_h, current_h, mode='full') * W
            peak_val = np.max(conv)
            k_star = np.argmax(conv)
            
            # 2. Compute gradient of peak_val w.r.t h
            # grad_g_j = 2 * h[k_star - j] if 0 <= k_star - j < N else 0
            # We need to construct this vector.
            # j goes from 0 to N-1.
            # Indices in h: k_star - j.
            # Range of k_star - j: k_star to k_star - (N-1).
            
            grad_g = np.zeros(N)
            
            # Efficient way:
            # The vector is h reversed and shifted.
            # Let's just loop or use slicing.
            # j=0 -> h[k_star]
            # j=N-1 -> h[k_star - (N-1)]
            
            # Valid indices for h: 0 to N-1.
            # k_star - j must be in [0, N-1].
            # j must be in [k_star - (N-1), k_star].
            # Also j must be in [0, N-1].
            
            start_j = max(0, k_star - (N - 1))
            end_j = min(N, k_star + 1)
            
            if start_j < end_j:
                # j ranges from start_j to end_j - 1
                # h index ranges from k_star - start_j down to k_star - (end_j - 1)
                h_indices = np.arange(k_star - start_j, k_star - (end_j - 1) - 1, -1)
                h_indices = h_indices[(h_indices >= 0) & (h_indices < N)]
                
                # Wait, simpler:
                # grad_g[j] = 2 * h[k_star - j]
                # Let's just compute it directly for the valid range.
                js = np.arange(start_j, end_j)
                h_idx = k_star - js
                valid = (h_idx >= 0) & (h_idx < N)
                if np.any(valid):
                    grad_g[js[valid]] = 2.0 * current_h[h_idx[valid]]
            
            # 3. Gradient of Ratio
            # R = peak_val / ||h||^2
            # h is normalized to 1? 
            # In this loop, we normalize after update.
            # So ||h||^2 is not necessarily 1 during gradient calc if we don't normalize before.
            # Let's normalize h at start of iteration.
            norm_sq = np.sum(current_h**2)
            if norm_sq < 1e-10:
                break
            # If we assume h is normalized, norm_sq = 1.
            # But let's be general.
            # grad_R = (1/norm_sq) * grad_g - (peak_val / norm_sq^2) * 2 * h
            
            # Actually, if we normalize h after every step, we are effectively minimizing peak_val subject to ||h||=1.
            # The gradient of peak_val on the sphere is the projection of grad_g onto the tangent space.
            # proj = grad_g - (grad_g . h / ||h||^2) * h
            # Since we will normalize, we can just use grad_g.
            # Let's use the ratio gradient to be safe.
            
            grad_R = grad_g / norm_sq - (2.0 * peak_val / (norm_sq ** 2)) * current_h
            
            # 4. Adam Update
            t = it + 1
            m = b1 * m + (1 - b1) * grad_R
            u = b2 * u + (1 - b2) * (grad_R ** 2)
            m_hat = m / (1 - b1 ** t)
            u_hat = u / (1 - b2 ** t)
            
            # Adaptive learning rate
            # We need to scale lr appropriately.
            # grad_R magnitude depends on h.
            # Let's use a fixed small lr with Adam.
            step = lr * m_hat / (np.sqrt(u_hat) + eps)
            
            current_h = current_h - step
            current_h = np.clip(current_h, 0, None)
            
            # Normalize
            new_norm = np.linalg.norm(current_h)
            if new_norm < 1e-10:
                # Restart this init
                break
            current_h = current_h / new_norm
            
            new_val = ratio_from_h(current_h)
            
            if new_val < best_val:
                best_val = new_val
                best_h = current_h.copy()
            
            # Decay learning rate
            if it % 100 == 0:
                lr *= 0.95

    return list(best_h)
