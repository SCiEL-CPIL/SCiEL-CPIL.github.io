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
    # Autoconvolution of length N with itself has length 2N-1
    # The FFT gives circular conv, but for zero-padded it's linear
    conv_peak = np.max(conv[:2*N-1])
    if conv_peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_peak)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_score = compute_ratio(best_h)
    
    # Try various structured patterns
    candidates = []
    
    # Delta function
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        candidates.append(h)
    
    # Block of k consecutive 1s
    for k in range(1, min(10, N)):
        for start in range(0, N - k + 1, max(1, (N - k) // 5)):
            h = np.zeros(N)
            h[start:start+k] = 1.0
            candidates.append(h)
    
    # Alternating patterns
    for period in range(2, 8):
        for offset in range(period):
            h = np.zeros(N)
            for i in range(N):
                if (i - offset) % period == 0:
                    h[i] = 1.0
            candidates.append(h)
    
    # Symmetric patterns with varying widths
    for width in range(1, 15):
        center = N // 2
        start = max(0, center - width // 2)
        end = min(N, start + width)
        h = np.zeros(N)
        h[start:end] = 1.0
        candidates.append(h)
        
        # Triangular profile
        h = np.zeros(N)
        for i in range(start, end):
            dist_from_center = abs(i - center)
            h[i] = max(0, 1.0 - dist_from_center / (width / 2 + 1))
        if np.sum(h) > 0:
            candidates.append(h)
    
    # Gaussian-like profiles
    for sigma in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
        x = np.arange(N)
        center = N / 2.0
        h = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        candidates.append(h)
    
    # Power law profiles
    for p in [0.5, 1.0, 2.0, 3.0]:
        x = np.arange(N)
        center = N / 2.0
        h = (1.0 + (x - center) ** 2) ** (-p)
        candidates.append(h)
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline:
            break
        s = compute_ratio(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Local optimization: gradient-free search around best
    if time.time() < deadline:
        # Try perturbing the best solution
        for _ in range(200):
            if time.time() > deadline:
                break
            
            # Random perturbation
            h_new = best_h.copy()
            idx = np.random.randint(0, N)
            h_new[idx] = max(0, h_new[idx] + np.random.normal(0, 0.1))
            
            # Also try scaling
            h_scaled = h_new * np.random.uniform(0.5, 2.0)
            
            for h_try in [h_new, h_scaled]:
                s = compute_ratio(h_try)
                if s > best_score:
                    best_score = s
                    best_h = h_try.copy()
    
    # Final refinement: try small adjustments
    if time.time() < deadline:
        for _ in range(100):
            if time.time() > deadline:
                break
            h_new = best_h.copy()
            for idx in np.random.choice(N, size=3, replace=False):
                h_new[idx] = max(0, h_new[idx] + np.random.normal(0, 0.01))
            s = compute_ratio(h_new)
            if s > best_score:
                best_score = s
                best_h = h_new.copy()
    
    return best_h.tolist()
