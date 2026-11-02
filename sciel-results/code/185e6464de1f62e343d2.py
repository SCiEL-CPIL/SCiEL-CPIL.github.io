import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_ratio(h):
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        l_inf = np.max(h)
        if l1 < 1e-15 or l_inf < 1e-15:
            return 0.0
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv < 1e-15:
            return 0.0
        return (l2_sq) / (l1 * l_inf * max_conv)

    def gradient_step(h, lr):
        """Perform one gradient ascent step on log(J) with projection."""
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        if l2_sq < 1e-15 or l1 < 1e-15:
            return h, 0.0
        
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv < 1e-15:
            return h, 0.0
        
        m = np.argmax(h_conv)
        
        # Gradient of log(max_conv) w.r.t. h[i]
        # conv[m] = sum_j h[j]*h[m-j] for valid j
        # d(conv[m])/dh[i] = 2*h[m-i] if 0 <= m-i < N, else 0
        grad_conv = np.zeros(N)
        lo = max(0, m - N + 1)
        hi = min(N - 1, m)
        for i in range(lo, hi + 1):
            grad_conv[i] += 2.0 * h[m - i]
        
        # Gradient of log(J) = 2*log(l2_sq) - log(l1) - log(max_conv)
        # (l_inf handled by normalization)
        grad = 2.0 * h / l2_sq - 1.0 / l1 - grad_conv / max_conv
        
        h_new = h + lr * grad
        h_new = np.maximum(h_new, 0.0)
        max_h = np.max(h_new)
        if max_h > 0:
            h_new = h_new / max_h
        
        return h_new, compute_ratio(h_new)

    def gradient_ascent(h0, max_iters=3000, lr0=0.05):
        h = h0.copy()
        if np.max(h) > 0:
            h = h / np.max(h)
        
        current_ratio = compute_ratio(h)
        best_h = h.copy()
        best_ratio = current_ratio
        lr = lr0
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            h_new, new_ratio = gradient_step(h, lr)
            
            if new_ratio >= current_ratio - 1e-12:
                h = h_new
                current_ratio = new_ratio
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_h = h.copy()
                # Slowly increase lr if making progress
                if it % 50 == 0:
                    lr = min(lr * 1.05, 0.2)
            else:
                # Decrease lr
                lr *= 0.5
                if lr < 1e-6:
                    break
        
        return best_h, best_ratio

    # Generate diverse initializations
    initializations = []
    
    # Sparse random patterns
    for _ in range(6):
        h = np.zeros(N)
        k = np.random.randint(4, 16)
        mask = np.random.choice(N, k, replace=False)
        h[mask] = np.random.uniform(0.2, 1.0)
        initializations.append(h)
    
    # Structured: alternating
    h = np.zeros(N)
    h[::2] = 1.0
    initializations.append(h)
    
    # Every 3rd
    h = np.zeros(N)
    h[::3] = 1.0
    initializations.append(h)
    
    # Every 4th
    h = np.zeros(N)
    h[::4] = 1.0
    initializations.append(h)
    
    # Gaussian
    x = np.linspace(-1, 1, N)
    h = np.exp(-x**2 * 3.0)
    initializations.append(h)
    
    # Cosine bump
    h = np.cos(np.pi * np.linspace(-0.5, 0.5, N))
    h = np.maximum(h, 0)
    initializations.append(h)
    
    # Smooth random
    h = np.random.uniform(0, 1, N)
    kernel = np.ones(3) / 3
    h = np.convolve(h, kernel, mode='same')
    initializations.append(h)
    
    # Sidon-like
    S = [0, 1, 4, 10, 18, 23, 30, 37, 44]
    h = np.zeros(N)
    h[S] = 1.0
    initializations.append(h)
    
    # Random with moderate density
    h = np.random.uniform(0, 1, N)
    h[h < 0.5] = 0
    initializations.append(h)
    
    best_h = None
    best_ratio = 0.0
    
    for init in initializations:
        if time.time() > deadline - 10.0:
            break
        
        h_result, ratio = gradient_ascent(init)
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = h_result
    
    if best_h is None:
        best_h = np.zeros(N)
        best_h[0] = 1.0
    
    return list(best_h)
