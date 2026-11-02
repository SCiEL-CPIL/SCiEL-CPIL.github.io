import numpy as np
import time
import random

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    def score(h):
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 0.0
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 0.0
        return l2_sq / (l1 * max_conv)

    def compute_grad(h, h_target):
        """
        Compute gradient of -log(max_conv) + log(l2_sq) - log(l1)
        w.r.t. h, using the argmax of conv.
        
        We want to maximize: log(l2_sq) - log(l1) - log(max_conv)
        = 2*log(rms) - log(l1) - log(max_conv) where rms^2 = l2_sq/N... 
        Actually let's work with: f = l2_sq / (l1 * max_conv)
        log f = log(l2_sq) - log(l1) - log(max_conv)
        
        d(log l2_sq)/dh_i = 2*h_i / l2_sq
        d(-log l1)/dh_i = -1/l1
        d(-log max_conv)/dh_i: 
          Let m = argmax(conv). conv[m] = sum_j h_j * h[m-j] (with appropriate indexing)
          d(conv[m])/dh_i = h[m-i] + h[i-m] (from both terms in the convolution)
          d(-log max_conv)/dh_i = -d(conv[m])/dh_i / max_conv
                               = -(h[m-i] + h[i-m]) / max_conv
        """
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        
        if max_conv < 1e-10 or l1 < 1e-10 or l2_sq < 1e-10:
            return np.zeros(N)
        
        m = np.argmax(conv)
        
        # Gradient of log(l2_sq) - log(l1) - log(max_conv)
        grad = np.zeros(N)
        
        # d(log l2_sq)/dh_i = 2*h_i / l2_sq
        grad += 2.0 * h / l2_sq
        
        # d(-log l1)/dh_i = -1/l1
        grad -= 1.0 / l1
        
        # d(-log max_conv)/dh_i = -(h[m-i] + h[i-m]) / max_conv
        # Need to handle indexing carefully
        for i in range(N):
            # In full convolution, conv[k] = sum_j h[j] * h[k-j]
            # d(conv[m])/dh[i] = h[m-i] if 0 <= m-i < N else 0
            #                   + h[i-m] if 0 <= i-m < N else 0
            val = 0.0
            if 0 <= m - i < N:
                val += h[m - i]
            if 0 <= i - m < N:
                val += h[i - m]
            grad[i] -= val / max_conv
        
        return grad

    def gradient_flow(h0, max_iters=2000, lr_init=0.1, lr_min=1e-4, cooling=0.999):
        """
        Gradient ascent on log-score using projected gradient.
        """
        h = h0.copy()
        h = np.maximum(h, 0)
        # Normalize to unit L1 for stability
        l1 = np.sum(h)
        if l1 > 1e-10:
            h = h / l1
        
        best_h = h.copy()
        best_s = score(h)
        lr = lr_init
        
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
            
            grad = compute_grad(h, h)
            
            # Projected gradient ascent
            h_new = h + lr * grad
            h_new = np.maximum(h_new, 0)
            
            # Normalize to unit L1
            l1_new = np.sum(h_new)
            if l1_new > 1e-10:
                h_new = h_new / l1_new
            else:
                h_new = h
            
            s_new = score(h_new)
            
            if s_new > best_s:
                best_s = s_new
                best_h = h_new.copy()
            
            # Adaptive learning rate
            if s_new < best_s * 0.99:
                lr *= 0.5
            else:
                lr = max(lr * cooling, lr_min)
            
            h = h_new
        
        return best_h, best_s

    best_h = np.zeros(N)
    best_score = 0.0

    np.random.seed(42)
    random.seed(42)

    # Generate diverse initial conditions
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(N) / N)
    
    # 2. Single spike
    for pos in [0, N//2, N-1]:
        h = np.zeros(N)
        h[pos] = 1.0
        initializations.append(h)
    
    # 3. Random sparse supports
    for k in [5, 10, 15, 20, 25, 30]:
        for _ in range(3):
            S = set(np.random.choice(N, size=k, replace=False))
            h = np.zeros(N)
            for i in S:
                h[i] = 1.0 / k
            initializations.append(h)
    
    # 4. Structured patterns
    # Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0 / (N//2)
    initializations.append(h_alt)
    
    # Cosine-like
    x = np.linspace(0, np.pi, N)
    h_cos = np.abs(np.cos(x))
    h_cos /= np.sum(h_cos)
    initializations.append(h_cos)
    
    # Gaussian-like
    x = np.linspace(-2, 2, N)
    h_gauss = np.exp(-x**2)
    h_gauss /= np.sum(h_gauss)
    initializations.append(h_gauss)
    
    # Run gradient flow from each initialization
    for h0 in initializations:
        if time.time() > deadline - 30:
            break
        
        h_opt, s_opt = gradient_flow(h0, max_iters=1000, lr_init=0.1)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt

    # If nothing better found, return single spike
    if best_score <= 1.0:
        h = [0.0] * N
        h[N // 2] = 1.0
        return h

    return list(best_h)
