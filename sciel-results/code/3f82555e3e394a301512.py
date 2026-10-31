import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    def elapsed():
        return time.time() - start_time
    
    def metric(h):
        """Compute the normalized maximum autoconvolution metric."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        hn = h / s
        # Use FFT for speed
        F = np.fft.fft(hn, n=2*N)
        conv = np.fft.ifft(F * F).real[:2*N-1]
        return np.max(conv)

    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Triangular
    center = N // 2
    tri = np.array([1.0 - abs(i - center) / center if center > 0 else 1.0 for i in range(N)])
    tri = np.clip(tri, 0, None)
    candidates.append(tri)
    
    # Gaussian
    x_grid = np.linspace(-3, 3, N)
    gauss = np.exp(-x_grid**2)
    candidates.append(gauss)
    
    # Inverted Gaussian
    inv_gauss = np.ones(N) - 0.5 * np.exp(-3.0 * np.linspace(-1, 1, N)**2)
    inv_gauss = np.clip(inv_gauss, 0, None)
    candidates.append(inv_gauss)
    
    # Two peaks
    two_peak = np.zeros(N)
    two_peak[N//4] = 1.0
    two_peak[3*N//4] = 1.0
    candidates.append(two_peak)
    
    # Plateaus
    for width_frac in [0.3, 0.5, 0.7]:
        w = int(N * width_frac)
        plateau = np.zeros(N)
        plateau[N//2 - w//2 : N//2 + w//2 + 1] = 1.0
        candidates.append(plateau)
        
    # Random Block Partitions (structural diversity)
    np.random.seed(42)
    for _ in range(5):
        h_rand = np.zeros(N)
        num_blocks = np.random.randint(2, 10)
        block_sizes = np.random.randint(10, N//num_blocks + 1, size=num_blocks)
        total_size = sum(block_sizes)
        if total_size > N:
            block_sizes = block_sizes * (N / total_size)
        start = 0
        for bs in block_sizes:
            bs = int(bs)
            if start + bs > N:
                bs = N - start
            h_rand[start:start+bs] = np.random.uniform(0.5, 1.0)
            start += bs
        candidates.append(h_rand)

    best_h = np.zeros(N)
    best_val = 1e10
    
    # Evaluate candidates
    for h in candidates:
        val = metric(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    # Phase 2: Projected Gradient Descent with Subgradient
    # Objective: minimize max_k conv_k
    # Subgradient of max_k conv_k is the gradient of the term achieving the max.
    # conv = IFFT(FFT(h)^2)
    # conv_k = sum_i h_i h_{i-k} (circular)
    # d(conv_k)/d(h_j) = h_{j-k} + h_{j+k}
    
    def pgd_optimize(h_init, steps=2000, lr_start=0.01, lr_end=0.0001):
        h = h_init.copy()
        s = np.sum(h)
        if s < 1e-15:
            return h, 1e10
        h = h / s
        
        best_local_val = metric(h)
        best_local_h = h.copy()
        
        for step in range(steps):
            if elapsed() > time_limit:
                break
                
            # Normalize
            s = np.sum(h)
            if s < 1e-15:
                break
            h_norm = h / s
            
            # Compute Convolution
            F = np.fft.fft(h_norm, n=2*N)
            conv = np.fft.ifft(F * F).real[:2*N-1]
            
            # Find Peak
            peak_idx = np.argmax(conv)
            
            # Compute Subgradient
            # G_j = h_norm[j - peak_idx] + h_norm[j + peak_idx] (circular)
            G = np.roll(h_norm, peak_idx) + np.roll(h_norm, -peak_idx)
            
            # Learning Rate Schedule
            lr = lr_start * (lr_end / lr_start) ** (step / steps)
            
            # Update
            h = h - lr * G
            
            # Project to Non-negative Simplex
            # 1. Clip negatives
            h = np.maximum(h, 0)
            # 2. Normalize
            s_new = np.sum(h)
            if s_new < 1e-15:
                h = np.ones(N) / N
            else:
                h = h / s_new
                
            # Check Improvement
            val = metric(h)
            if val < best_local_val:
                best_local_val = val
                best_local_h = h.copy()
                
            # Early stopping if no improvement for a long time?
            # For now, just run through steps.
            
        return best_local_h, best_local_val

    # Run PGD from best initial candidate
    if elapsed() < time_limit - 5:
        h_opt, val_opt = pgd_optimize(best_h, steps=3000)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt

    # Phase 3: Try PGD from other promising candidates if time permits
    # Pick top 3 candidates from Phase 1 that weren't the best
    sorted_candidates = []
    for h in candidates:
        val = metric(h)
        sorted_candidates.append((val, h))
    sorted_candidates.sort(key=lambda x: x[0])
    
    for val, h in sorted_candidates[1:4]:
        if elapsed() > time_limit - 10:
            break
        h_opt, val_opt = pgd_optimize(h, steps=1000)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt

    return list(best_h)
