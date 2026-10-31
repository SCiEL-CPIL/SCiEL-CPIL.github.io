import numpy as np
import time

N = 600
DEADLINE = 115

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    i = np.arange(N, dtype=float)
    center = (N - 1) / 2.0
    
    def compute_autoconv(h):
        """Compute full autoconvolution of h (assumed sum=1)."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return conv
    
    def compute_autoconv_max(h):
        """Compute max of autoconvolution of h (assumed sum=1)."""
        return np.max(compute_autoconv(h))
    
    def compute_l1_autoconv(h):
        """Compute L1 norm of autoconvolution of h (assumed sum=1)."""
        return np.sum(np.abs(compute_autoconv(h)))
    
    def compute_smooth_max(h, beta=50.0):
        """Compute smooth max of autoconvolution using log-sum-exp."""
        conv = compute_autoconv(h)
        m = np.max(conv)
        log_sum_exp = m + np.log(np.sum(np.exp((conv - m) * beta))) / beta
        return log_sum_exp
    
    def compute_grad_smooth_max(h, beta=50.0):
        """Compute gradient of smooth max autoconvolution w.r.t. h."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        m = np.max(conv)
        exp_conv = np.exp((conv - m) * beta)
        weights = exp_conv / np.sum(exp_conv)
        
        w_full = np.zeros(size)
        w_full[:N] = weights[:N]
        w_full[N:] = weights[N:]
        
        W = np.fft.fft(w_full, n=size)
        grad_h = 2.0 * np.fft.ifft(H * W).real[:N]
        return grad_h
    
    def compute_grad_l1(h):
        """Compute gradient of L1 norm of autoconvolution w.r.t. h."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        signs = np.sign(conv)
        
        w_full = np.zeros(size)
        w_full[:N] = signs[:N]
        w_full[N:] = signs[N:]
        
        W = np.fft.fft(w_full, n=size)
        grad_h = 2.0 * np.fft.ifft(H * W).real[:N]
        return grad_h
    
    def project_simplex(v):
        """Project v onto the probability simplex (sum=1, v>=0)."""
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1
        ind = np.arange(n) + 1
        cond = u - cssv / ind > 0
        if not np.any(cond):
            return np.ones(n) / n
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / rho
        return np.maximum(v - theta, 0)
    
    # Generate diverse initializations
    inits = []
    
    # Uniform
    inits.append(np.ones(N))
    
    # Bimodal with various parameters
    for sigma in [N/10, N/8, N/6, N/5]:
        for alpha in [0.1, 0.15, 0.2, 0.25, 0.3]:
            pos1 = alpha * N
            pos2 = (1 - alpha) * N
            g1 = np.exp(-((i - pos1) ** 2) / (2 * sigma ** 2))
            g2 = np.exp(-((i - pos2) ** 2) / (2 * sigma ** 2))
            inits.append(g1 + g2)
    
    # Polynomial shapes
    for c, d, e in [(1, 1, 0), (1, 2, 0), (1, 1, 0.5), (1, 0.5, 1), (2, 1, 0), (1, 3, 0)]:
        x = (i - center) / (N / 2)
        val = c + d * x**2 + e * x**4
        inits.append(np.maximum(val, 0.01))
    
    # U-shaped
    u_shape = 1.0 + 2.0 * ((i - center) / (N/2))**2
    inits.append(u_shape)
    
    # W-shape
    w_shape = 1.0 + 1.5 * np.cos(4 * np.pi * i / N)
    inits.append(np.maximum(w_shape, 0.01))
    
    # Random smooth
    np.random.seed(42)
    for _ in range(5):
        freqs = np.random.choice([1, 2, 3, 4, 5], size=3, replace=False)
        phases = np.random.uniform(0, 2*np.pi, size=3)
        amps = np.random.uniform(0.5, 1.5, size=3)
        random_shape = np.ones(N)
        for f, p, a in zip(freqs, phases, amps):
            random_shape += a * np.cos(2 * np.pi * f * i / N + p)
        inits.append(np.maximum(random_shape, 0.01))
    
    best_h = None
    best_val = np.inf
    
    # Stage 1: Evaluate all initializations
    for init in inits:
        if time_left() < 10:
            break
        h = init / np.sum(init)
        val = compute_autoconv_max(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = compute_autoconv_max(best_h)
    
    # Stage 2: Hybrid optimization - minimize weighted combo of smooth max and L1 norm
    # using projected gradient descent onto simplex
    if time_left() > 40:
        current_h = best_h.copy()
        current_val = best_val
        
        # Phase A: Focus on L1 norm to spread mass, then refine peak
        beta = 30.0
        eta = 0.01
        patience = 0
        max_patience = 50
        l1_weight = 0.5  # Weight for L1 term
        
        for k in range(2000):
            if time_left() < 15:
                break
            
            # Compute combined gradient
            grad_smooth = compute_grad_smooth_max(current_h, beta)
            grad_l1 = compute_grad_l1(current_h)
            grad = grad_smooth + l1_weight * grad_l1
            
            # Projected gradient step with simplex projection
            new_h = current_h - eta * grad
            new_h = project_simplex(new_h)
            
            new_val = compute_autoconv_max(new_h)
            
            if new_val < current_val:
                current_h = new_h
                current_val = new_val
                patience = 0
                eta = min(eta * 1.02, 0.05)
                
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
            else:
                patience += 1
                eta *= 0.95
                if patience > max_patience:
                    # Increase beta for sharper max approximation, reduce L1 weight
                    beta *= 1.5
                    l1_weight *= 0.7
                    eta = 0.01
                    patience = 0
        
        # Phase B: Final polish with higher beta, lower L1 weight
        if time_left() > 10:
            beta = 100.0
            eta = 0.005
            l1_weight = 0.1
            for k in range(500):
                if time_left() < 5:
                    break
                
                grad_smooth = compute_grad_smooth_max(current_h, beta)
                grad_l1 = compute_grad_l1(current_h)
                grad = grad_smooth + l1_weight * grad_l1
                
                new_h = current_h - eta * grad
                new_h = project_simplex(new_h)
                
                new_val = compute_autoconv_max(new_h)
                
                if new_val < current_val:
                    current_h = new_h
                    current_val = new_val
                    eta = min(eta * 1.01, 0.02)
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()
                else:
                    eta *= 0.95
    
    # Stage 3: Local coordinate descent refinement
    if time_left() > 15:
        current_h = best_h.copy()
        current_val = best_val
        improved = True
        max_rounds = int(min(10, time_left() // 3))
        
        for round_num in range(max_rounds):
            if time_left() < 5:
                break
            
            improved = False
            order = np.random.permutation(N)
            for idx in order:
                if time_left() < 3:
                    break
                
                for delta in [0.05, 0.02, -0.05, -0.02]:
                    new_h = current_h.copy()
                    new_h[idx] = max(0, current_h[idx] * (1 + delta))
                    s = np.sum(new_h)
                    if s < 1e-10:
                        continue
                    new_h /= s
                    
                    new_val = compute_autoconv_max(new_h)
                    if new_val < current_val - 1e-12:
                        current_h = new_h
                        current_val = new_val
                        improved = True
                        break
            
            if not improved:
                break
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h
    
    return best_h.tolist()
