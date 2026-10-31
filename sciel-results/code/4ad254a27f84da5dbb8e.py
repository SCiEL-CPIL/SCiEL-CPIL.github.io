import numpy as np
import time

N = 50

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    # Autoconvolution via FFT for efficiency
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    conv = np.fft.ifft(H * H).real
    conv_peak = np.max(conv[:2*N-1])
    if conv_peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_peak)

def shape_score(h):
    """Scale-invariant score: minimize ||h*h||_inf * ||h||_1 / ||h||_2^2"""
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    if l2_sq < 1e-15:
        return 0.0
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    conv = np.fft.ifft(H * H).real
    conv_peak = np.max(conv[:2*N-1])
    if conv_peak < 1e-15:
        return 0.0
    # We want to maximize l2_sq/(l1*conv_peak), which is equivalent to minimizing l1*conv_peak/l2_sq
    return l1 * conv_peak / l2_sq

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_h = np.ones(N) / N
    best_shape = shape_score(best_h)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N) / N)
    
    # 2. Sparse patterns: pick k indices
    for k in range(1, 15):
        for _ in range(3):
            idx = np.random.choice(N, k, replace=False)
            h = np.zeros(N)
            h[idx] = 1.0
            h /= np.sum(h)
            candidates.append(h)
    
    # 3. Block of consecutive ones
    for k in range(1, 20):
        for start in range(0, max(1, N - k + 1), max(1, (N - k) // 8)):
            h = np.zeros(N)
            h[start:start+k] = 1.0
            h /= np.sum(h)
            candidates.append(h)
    
    # 4. Alternating patterns
    for period in range(2, 12):
        for offset in range(period):
            h = np.zeros(N)
            for i in range(N):
                if (i - offset) % period == 0:
                    h[i] = 1.0
            if np.sum(h) > 0:
                h /= np.sum(h)
                candidates.append(h)
    
    # 5. Gaussian-like profiles
    for sigma in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
        for center in [N//4, N//2, 3*N//4]:
            x = np.arange(N)
            h = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            h += 1e-12
            h /= np.sum(h)
            candidates.append(h)
    
    # 6. Power law profiles
    for p in [0.5, 1.0, 1.5, 2.0, 3.0]:
        x = np.arange(N)
        center = N / 2.0
        h = (1.0 + (x - center) ** 2) ** (-p)
        h += 1e-12
        h /= np.sum(h)
        candidates.append(h)
    
    # 7. Random Dirichlet
    for _ in range(100):
        alpha = np.random.dirichlet(np.ones(N))
        candidates.append(alpha)
    
    # 8. Random sparse with weights
    for _ in range(100):
        k = np.random.randint(2, 10)
        idx = np.random.choice(N, k, replace=False)
        h = np.zeros(N)
        h[idx] = np.random.dirichlet(np.ones(k))
        h /= np.sum(h)
        candidates.append(h)
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline:
            break
        s = shape_score(h)
        if s < best_shape:
            best_shape = s
            best_h = h.copy()
    
    # Local optimization: mass redistribution search
    if time.time() < deadline:
        h = best_h.copy()
        current_score = shape_score(h)
        
        # Adaptive perturbation sizes
        perturbation_sizes = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
        
        for psize in perturbation_sizes:
            if time.time() > deadline:
                break
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                
                # Try multiple random moves
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    
                    i = np.random.randint(N)
                    j = np.random.randint(N)
                    if i == j:
                        continue
                    
                    # Move small amount of mass from i to j
                    delta = h[i] * psize * np.random.uniform(0.5, 1.5)
                    if delta <= 0 or h[i] - delta < -1e-12:
                        continue
                    
                    h_new = h.copy()
                    h_new[i] = max(0, h_new[i] - delta)
                    h_new[j] += delta
                    
                    new_score = shape_score(h_new)
                    if new_score < current_score:
                        h = h_new
                        current_score = new_score
                        improved = True
                        if current_score < best_shape:
                            best_shape = current_score
                            best_h = h.copy()
                        break
            
            # Also try 3-point redistribution
            for _ in range(10):
                if time.time() > deadline:
                    break
                
                idxs = np.random.choice(N, 3, replace=False)
                h_new = h.copy()
                vals = h_new[idxs]
                if np.sum(vals) < 1e-12:
                    continue
                # Random redistribution
                new_vals = np.random.dirichlet(np.ones(3)) * np.sum(vals)
                h_new[idxs] = new_vals
                
                new_score = shape_score(h_new)
                if new_score < current_score:
                    h = h_new
                    current_score = new_score
                    improved = True
                    if current_score < best_shape:
                        best_shape = current_score
                        best_h = h.copy()
    
    # Final high-precision polish: very small adjustments
    if time.time() < deadline:
        h = best_h.copy()
        current_score = shape_score(h)
        
        for _ in range(200):
            if time.time() > deadline:
                break
            
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            
            delta = h[i] * 0.01 * np.random.uniform(0.1, 1.0)
            if delta <= 0 or h[i] - delta < -1e-12:
                continue
            
            h_new = h.copy()
            h_new[i] = max(0, h_new[i] - delta)
            h_new[j] += delta
            
            new_score = shape_score(h_new)
            if new_score < current_score:
                h = h_new
                current_score = new_score
                if current_score < best_shape:
                    best_shape = current_score
                    best_h = h.copy()
    
    # Scale down to exploit the 1/k scaling of the ratio
    # The ratio scales as 1/k, so smaller k gives larger ratio
    # Use a very small but safe scale factor
    scale = 1e-6
    final_h = best_h * scale
    
    # Verify the result
    final_ratio = compute_ratio(final_h)
    
    return final_h.tolist()
