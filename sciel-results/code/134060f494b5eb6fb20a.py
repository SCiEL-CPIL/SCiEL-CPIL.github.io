import numpy as np
import time

N = 400

def evaluate(h):
    # h is a numpy array
    conv = np.convolve(h, h)
    max_val = np.max(conv)
    sum_h = np.sum(h)
    if sum_h == 0:
        return float('inf')
    return abs(2 * N * max_val / (sum_h ** 2))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Start with seed
    h = np.ones(N)
    best_val = evaluate(h)
    best_h = h.copy()
    
    # Smooth max surrogate using log-sum-exp
    def smooth_max(conv, beta=50.0):
        # Use log-sum-exp for numerical stability
        m = np.max(conv)
        if m > 0:
            return m + np.log(np.sum(np.exp(beta * (conv - m)))) / beta
        else:
            return m + np.log(np.sum(np.exp(beta * (conv - m)))) / beta
    
    # Compute gradient of smooth max w.r.t. h
    def smooth_max_grad(h, beta=50.0):
        conv = np.convolve(h, h)
        m = np.max(conv)
        weights = np.exp(beta * (conv - m))
        weights /= np.sum(weights)
        # d/dh of sum_j weights[j] * conv[j]
        # conv[j] = sum_k h[k] * h[j-k]
        # Gradient: 2 * sum_j weights[j] * h[j-k] for each k
        # Use FFT for efficiency
        grad = np.zeros(N)
        for j in range(len(conv)):
            w = weights[j]
            if w > 1e-10:
                k_start = max(0, j - (N - 1))
                k_end = min(N, j + 1)
                for k in range(k_start, k_end):
                    idx = j - k
                    if 0 <= idx < N:
                        grad[k] += 2 * w * h[idx]
        return grad
    
    # Faster gradient using FFT
    def smooth_max_grad_fft(h, beta=50.0):
        conv = np.convolve(h, h)
        m = np.max(conv)
        weights = np.exp(beta * (conv - m))
        weights /= np.sum(weights)
        
        # Use the fact that gradient of sum_j w_j * (h*h)[j] w.r.t h[k]
        # = 2 * sum_j w_j * h[j-k]
        # This is 2 * (w * h) reversed and cropped
        # Actually: grad[k] = 2 * sum_j w[j] * h[j-k] for valid j,k
        # Let's use direct computation with vectorization
        grad = np.zeros(N)
        for k in range(N):
            # j ranges such that 0 <= j-k < N, i.e., k <= j < k+N
            j_min = max(0, k)
            j_max = min(len(conv), k + N)
            if j_min < j_max:
                # h[j-k] for j from j_min to j_max-1
                # j-k ranges from j_min-k to j_max-1-k
                h_idx = np.arange(j_min - k, j_max - k)
                valid = (h_idx >= 0) & (h_idx < N)
                if np.any(valid):
                    grad[k] = 2 * np.dot(weights[j_min:j_max][valid], h[h_idx[valid]])
        return grad
    
    # Even faster: use the autocorrelation structure
    def smooth_max_grad_fast(h, beta=50.0):
        conv = np.convolve(h, h)
        m = np.max(conv)
        weights = np.exp(beta * (conv - m))
        weights /= np.sum(weights)
        
        # grad[k] = 2 * sum_{j=k}^{k+N-1} weights[j] * h[j-k]
        # = 2 * (weights * h)[k] where * is cross-correlation
        # Let's compute cross-correlation of weights with h
        # cross_corr[k] = sum_j weights[j] * h[j-k]
        # This is the same as convolve(weights, h[::-1])[N-1-k] or similar
        
        # Use np.correlate
        # np.correlate(a, v, mode='full')[k] = sum_n a[n+k] * v[n]
        # We want sum_j weights[j] * h[j-k] = sum_j weights[j+k] * h[j]
        # = np.correlate(weights, h, mode='full')[N-1-k]
        
        corr = np.correlate(weights, h, mode='full')
        # corr[k] = sum_j weights[j+k] * h[j] for valid j
        # We need for each k in [0,N-1]: sum_j weights[j] * h[j-k]
        # Let i = j-k, then j = i+k, sum_i weights[i+k] * h[i]
        # = corr[N-1-k] if corr is defined as above? Let me check.
        
        # Actually np.correlate(weights, h, 'full')[m] = sum_n weights[n+m] * h[n]
        # So for our k, we want sum_i weights[i+k] * h[i] = corr[k]
        
        grad = 2 * corr[:N]
        return grad
    
    # Objective using smooth max
    def objective(h, beta=50.0):
        conv = np.convolve(h, h)
        sm = smooth_max(conv, beta)
        sum_h = np.sum(h)
        if abs(sum_h) < 1e-10:
            return float('inf')
        return abs(2 * N * sm / (sum_h ** 2))
    
    # Gradient of objective
    def objective_grad(h, beta=50.0):
        conv = np.convolve(h, h)
        m = np.max(conv)
        weights = np.exp(beta * (conv - m))
        weights /= np.sum(weights)
        
        sum_h = np.sum(h)
        if abs(sum_h) < 1e-10:
            return None
        
        # d/dh of 2*N*sm / sum_h^2
        # = 2*N * (grad_sm * sum_h^2 - sm * 2*sum_h * ones) / sum_h^4
        # = 2*N * (grad_sm / sum_h - 2*sm*ones / sum_h^2)
        
        grad_sm = smooth_max_grad_fast(h, beta)
        sm = smooth_max(conv, beta)
        
        grad = 2 * N * (grad_sm / sum_h - 2 * sm / (sum_h ** 2) * np.ones(N))
        return grad
    
    # Phase 1: Gradient descent with smooth objective
    beta = 20.0  # Start with moderate smoothness
    lr = 0.1
    for i in range(5000):
        if time.time() > deadline:
            break
        
        grad = objective_grad(best_h, beta)
        if grad is None:
            # Sum is too small, perturb
            best_h += 0.1 * np.random.randn(N)
            continue
        
        # Adaptive learning rate
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        step = lr * grad / grad_norm
        h_new = best_h - step
        
        # Check improvement
        val_new = evaluate(h_new)
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
            lr = min(lr * 1.05, 1.0)
        else:
            lr *= 0.8
            if lr < 1e-6:
                # Try increasing beta for sharper max
                beta *= 1.2
                lr = 0.05
    
    # Phase 2: Coordinate descent polish
    for coord_pass in range(200):
        if time.time() > deadline:
            break
        
        improved = False
        for k in range(N):
            if time.time() > deadline:
                break
            
            h_try = best_h.copy()
            # Try small perturbations in coordinate k
            for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                h_try[k] = best_h[k] + delta
                val_try = evaluate(h_try)
                if val_try < best_val:
                    best_val = val_try
                    best_h = h_try.copy()
                    improved = True
                    break
        
        if not improved and coord_pass > 50:
            break
    
    # Phase 3: Final random perturbation polish
    for i in range(2000):
        if time.time() > deadline:
            break
        
        pert = np.random.randn(N)
        h_new = best_h + 0.01 * pert
        val_new = evaluate(h_new)
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
    
    return best_h
