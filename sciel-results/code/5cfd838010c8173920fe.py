import numpy as np

N = 600

def _score(h):
    """Compute the normalized maximum autoconvolution value."""
    h = np.asarray(h, dtype=float)
    total = h.sum()
    if total <= 0:
        return float('inf')
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_conv = conv.max()
    return max_conv / (total * total)

def construct():
    best_h = None
    best_score = float('inf')
    
    # We will run Subgradient Descent from multiple initializations
    
    # Initializations
    inits = []
    inits.append(np.ones(N))
    
    np.random.seed(42)
    for _ in range(10):
        h_rand = np.random.rand(N)
        inits.append(h_rand)
        
    # Add some structured inits
    for freq in [1, 2, 5, 10]:
        x = np.arange(N)
        h_sin = 1.0 + 0.8 * np.sin(2 * np.pi * freq * x / N)
        inits.append(np.maximum(h_sin, 0))

    for h0 in inits:
        h = h0.copy()
        # Normalize to sum 1
        if h.sum() == 0:
            continue
        h = h / h.sum()
        
        # Subgradient Descent
        # Objective: f(h) = max(conv) / (sum(h))^2
        # We minimize this.
        
        num_iter = 2000
        # Learning rate schedule
        # t_k = c / sqrt(k)
        c_lr = 0.01
        
        for k in range(num_iter):
            total = h.sum()
            if total == 0:
                break
            conv = np.convolve(h, h, mode='full')
            max_val = conv.max()
            score = max_val / (total * total)
            
            # Update best
            if score < best_score:
                best_score = score
                best_h = h.copy()
                
            # Find indices of max
            # To handle ties, we can pick the first one or average gradients.
            # Picking one is standard subgradient.
            max_idx = np.argmax(conv)
            
            # Gradient of conv[max_idx] w.r.t h
            # conv[k] = sum_i h[i] h[i+k] (with padding)
            # d(conv[k])/d(h[j]) = h[j+k] + h[j-k] (if indices valid, 0 otherwise)
            # Note: np.convolve(h, h, 'full') index mapping:
            # conv[i] corresponds to lag i - (N-1).
            # Let lag = max_idx - (N - 1).
            lag = max_idx - (N - 1)
            
            grad_conv = np.zeros(N)
            if lag >= 0:
                # h[j] * h[j+lag] term
                # d/dh[j] of sum_j h[j] h[j+lag] is h[j+lag] for the first part
                # and h[j-lag] for the second part?
                # Let's derive carefully.
                # c_l = sum_{j=0}^{N-1-lag} h[j] h[j+lag]
                # d c_l / d h[j] = h[j+lag] + h[j-lag] (where defined)
                # For j in 0..N-1-lag: term h[j]h[j+lag] gives h[j+lag].
                # For j in lag..N-1: term h[j-lag]h[j] gives h[j-lag].
                # So grad[j] = h[j+lag] (if j+lag < N) + h[j-lag] (if j-lag >= 0).
                # If lag == 0: c_0 = sum h[j]^2. grad = 2 h.
                if lag == 0:
                    grad_conv = 2 * h
                else:
                    # First part: h[j+lag]
                    grad_conv[:N-lag] += h[lag:]
                    # Second part: h[j-lag]
                    grad_conv[lag:] += h[:N-lag]
            else:
                # lag < 0, let l = -lag
                l = -lag
                # Symmetric case
                if l == 0:
                    grad_conv = 2 * h
                else:
                    grad_conv[:N-l] += h[l:]
                    grad_conv[l:] += h[:N-l]
            
            # Gradient of 1/total^2
            # f = c / total^2
            # df/dh = (grad_c * total^2 - c * 2 * total * 1) / total^4
            # df/dh = grad_c / total^2 - 2 c / total^3
            # df/dh = (grad_c * total - 2 c) / total^3
            
            # Wait, the score is scale invariant. 
            # If we scale h by alpha, score is unchanged.
            # Subgradient descent on a scale-invariant function can be tricky.
            # It's better to minimize c_max subject to sum(h)=1.
            # Lagrangian: L = c_max + lambda (sum(h) - 1) - mu^T h
            # Subgradient of c_max is grad_conv.
            # Subgradient of sum(h) is 1.
            # So we update h <- h - t * (grad_conv - lambda * 1).
            # Then project to sum(h)=1, h>=0.
            # We can estimate lambda or just use the gradient of the ratio directly.
            # Using the ratio gradient:
            # grad_score = (grad_conv * total - 2 * max_val) / total**3
            
            grad_score = (grad_conv * total - 2 * max_val) / (total**3)
            
            # Update
            t = c_lr / np.sqrt(k + 1)
            h_new = h - t * grad_score
            
            # Project to simplex: h >= 0, sum(h) = 1
            # First, project to h >= 0
            h_new = np.maximum(h_new, 0)
            # Then project to sum = 1
            s = h_new.sum()
            if s == 0:
                h_new = np.ones(N) / N
            else:
                h_new = h_new / s
            
            h = h_new
            
    if best_h is None:
        return np.ones(N).tolist()
        
    return best_h.tolist()
