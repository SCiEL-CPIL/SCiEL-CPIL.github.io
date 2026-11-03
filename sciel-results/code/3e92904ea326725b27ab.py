import numpy as np
import time

def construct():
    """
    Minimize the normalized maximum of the autoconvolution for a nonnegative
    step function with 600 equal intervals on [-1/4, 1/4].
    
    Strategy:
    1. The uniform distribution is a known candidate with score 1/N.
    2. We use a structural approach: treat the problem as minimizing the peak
       of the autoconvolution. We start with diverse parametric families and
       refine using a fast coordinate-descent / local search that directly
       minimizes the max lag of the autoconvolution.
    3. We use FFT for fast autoconvolution computation.
    4. We enforce a strict time budget to stay within limits.
    """
    N = 600
    start_time = time.time()
    time_limit = 145.0  # safety margin before 150s kill
    
    def fft_autoconv(f):
        """Compute full autoconvolution via FFT."""
        n = len(f)
        size = 2 * n - 1
        fft_size = 1
        while fft_size < size:
            fft_size *= 2
        F = np.fft.rfft(f, n=fft_size)
        conv = np.fft.irfft(F * F, n=fft_size)
        return conv[:size]

    def compute_score(f):
        """Compute the normalized maximum of the autoconvolution."""
        f = np.asarray(f, dtype=np.float64)
        if np.any(f < -1e-15):
            return np.inf
        f = np.maximum(f, 0.0)
        s = np.sum(f)
        if s < 1e-15:
            return np.inf
        conv = fft_autoconv(f)
        return np.max(conv) / (s * s)

    def normalize(f):
        f = np.maximum(f, 0.0)
        s = np.sum(f)
        if s < 1e-15:
            return np.ones(N) / N
        return f / s

    def local_search(f, max_iter=200000):
        """
        Coordinate descent: iteratively adjust individual bins to reduce
        the max autoconvolution value.
        """
        f = f.copy()
        f = normalize(f)
        best_f = f.copy()
        conv = fft_autoconv(f)
        best_val = np.max(conv)
        
        # Track which lags are near the max to focus optimization
        # We'll use a simple approach: for each coordinate, try small perturbations
        
        # Precompute indices for efficiency
        # For coordinate i, changing f[i] affects lags m where m-i is in [0, N-1]
        # i.e., m in [i, i+N-1], clipped to [0, 2N-2]
        
        step_sizes = [0.01, 0.005, 0.001, 0.0005]
        current_step_idx = 0
        
        for it in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            # Find the current max lag
            max_lag = np.argmax(conv)
            current_max = conv[max_lag]
            
            # Try to reduce the max by adjusting coordinates
            # Pick a random coordinate that contributes to the max lag
            # For lag m, the contributing pairs are (j, m-j) for j in valid range
            m = max_lag
            valid_js = []
            for j in range(N):
                k = m - j
                if 0 <= k < N:
                    valid_js.append(j)
            
            if not valid_js:
                break
            
            # Pick a random valid j
            j = valid_js[np.random.randint(len(valid_js))]
            k = m - j
            
            # Try increasing or decreasing f[j]
            # If j == k, the term is f[j]^2
            # If j != k, the term is 2*f[j]*f[k] (in the sum, but actually f[j]*f[k] + f[k]*f[j])
            
            # We want to reduce conv[m]. 
            # If we decrease f[j], conv[m] decreases (since f is nonnegative)
            # But we need to maintain normalization.
            
            # Simple approach: try small perturbations
            improved = False
            for delta in step_sizes[current_step_idx]:
                # Try decreasing f[j] by delta, and compensating by increasing another bin
                # For simplicity, just decrease f[j] and renormalize
                f_new = f.copy()
                f_new[j] = max(0, f_new[j] - delta)
                f_new = normalize(f_new)
                
                # Recompute conv at lag m (approximate)
                # Actually, let's just recompute full conv for simplicity in this prototype
                # But that's slow. Let's do a partial update.
                
                # For now, let's just accept the step if it looks promising
                # We'll do a more careful update below
                
                break  # Just use the first step size for now
            
            # More careful coordinate descent:
            # For coordinate j, the objective at lag m is linear in f[j] if j != k
            # and quadratic if j == k.
            
            # Let's try a different approach: use the subgradient of the max
            # The subgradient of max_lag conv[lag] w.r.t. f[j] is conv contribution
            # at the max lag.
            
            # For the max lag m, d(conv[m])/d(f[j]) = f[m-j] if 0 <= m-j < N
            subgrad = np.zeros(N)
            m = max_lag
            for j in range(N):
                k = m - j
                if 0 <= k < N:
                    subgrad[j] = f[k]
            
            # Projected gradient step: move opposite to subgradient
            # But we want to minimize, so f -= step * subgrad
            # Then project to f >= 0, sum f = 1
            
            # Use a small step
            step = 0.001
            f_new = f - step * subgrad
            f_new = normalize(f_new)
            
            # Check if this improved the max
            conv_new = fft_autoconv(f_new)
            max_new = np.max(conv_new)
            
            if max_new < best_val - 1e-15:
                best_val = max_new
                best_f = f_new.copy()
                f = f_new.copy()
                conv = conv_new.copy()
                improved = True
            else:
                # Try a smaller step
                step = 0.0001
                f_new = f - step * subgrad
                f_new = normalize(f_new)
                conv_new = fft_autoconv(f_new)
                max_new = np.max(conv_new)
                if max_new < best_val - 1e-15:
                    best_val = max_new
                    best_f = f_new.copy()
                    f = f_new.copy()
                    conv = conv_new.copy()
                    improved = True
            
            if not improved and it % 1000 == 0:
                # Try a different max lag if stuck
                # Find all lags near the max
                threshold = best_val - 1e-6
                near_max = np.where(conv >= threshold)[0]
                if len(near_max) > 1:
                    # Randomly pick one
                    m_alt = near_max[np.random.randint(len(near_max))]
                    subgrad = np.zeros(N)
                    for j in range(N):
                        k = m_alt - j
                        if 0 <= k < N:
                            subgrad[j] = f[k]
                    step = 0.0005
                    f_new = f - step * subgrad
                    f_new = normalize(f_new)
                    conv_new = fft_autoconv(f_new)
                    max_new = np.max(conv_new)
                    if max_new < best_val - 1e-15:
                        best_val = max_new
                        best_f = f_new.copy()
                        f = f_new.copy()
                        conv = conv_new.copy()
                        improved = True
        
        return best_f

    # Generate initial candidates
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + (0.5 / N)
    candidates = []
    
    # Uniform
    f = np.ones(N)
    candidates.append(f / np.sum(f))
    
    # Gaussian with various widths
    for alpha in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        f = np.exp(-alpha * x ** 2)
        candidates.append(f / np.sum(f))
    
    # Raised cosine
    for w in [0.1, 0.15, 0.2, 0.25, 0.4, 0.5]:
        f = np.zeros(N)
        mask = np.abs(x) <= w / 2.0
        f[mask] = np.cos(np.pi * x[mask] / w) ** 2
        if np.sum(f) > 1e-15:
            candidates.append(f / np.sum(f))
    
    # Triangle
    for w in [0.1, 0.2, 0.25, 0.4, 0.5]:
        f = np.zeros(N)
        mask = np.abs(x) <= w / 2.0
        f[mask] = 1.0 - 2.0 * np.abs(x[mask]) / w
        f = np.maximum(f, 0)
        if np.sum(f) > 1e-15:
            candidates.append(f / np.sum(f))
    
    # Best so far is uniform
    best_f = candidates[0].copy()
    best_score = compute_score(best_f)
    
    # Try local search on each candidate
    for i, f_init in enumerate(candidates):
        if time.time() - start_time > time_limit - 10:
            break
        
        f_refined = local_search(f_init)
        score = compute_score(f_refined)
        
        if score < best_score:
            best_score = score
            best_f = f_refined.copy()
    
    # Final check
    if time.time() - start_time > time_limit:
        # Return best found
        pass
    
    return [float(x) for x in best_f]
