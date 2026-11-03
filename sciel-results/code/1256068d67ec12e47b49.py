import numpy as np
import time

N = 400
DEADLINE = 140  # 150s limit, 10s safety margin

def autocorr_peak_sidelobe(x):
    """Compute peak sidelobe of aperiodic autocorrelation."""
    # Aperiodic autocorrelation
    n = len(x)
    r = np.zeros(2 * n - 1)
    for k in range(2 * n - 1):
        if k < n:
            r[k] = np.dot(x[:n-k], x[k:])
        else:
            r[k] = r[2*n-2-k]
    # Peak sidelobe: max absolute value excluding main lobe (lag 0)
    # Main lobe is at index n-1 (center of r)
    center = n - 1
    r_sidelobes = np.concatenate([r[:center], r[center+1:]])
    return np.max(np.abs(r_sidelobes))

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > DEADLINE
    
    # Phase 1: Gradient descent on continuous relaxation
    # Objective: minimize sum of squared sidelobes (L2 norm of autocorrelation sidelobes)
    # This is smooth and differentiable
    
    # Initialize with random ±1
    np.random.seed(42)
    x = np.random.choice([-1.0, 1.0], size=N)
    
    # Gradient descent on continuous x to minimize L2 norm of autocorrelation sidelobes
    # r(k) = sum_i x_i * x_{i+k} for k > 0
    # Objective: sum_k r(k)^2 for k != 0
    
    lr = 0.01
    max_iters = 2000
    for it in range(max_iters):
        if check_deadline():
            break
        # Compute autocorrelation
        # Using FFT for speed
        n_fft = 2 * N
        X = np.fft.fft(x, n=n_fft)
        R = np.fft.ifft(np.abs(X)**2).real[:2*N-1]
        # R[k] is autocorrelation at lag k (circular, but we take aperiodic part)
        # For aperiodic: R[k] for k=0..N-1 are the positive lags
        # Actually, np.fft.ifft(|X|^2) gives circular autocorrelation
        # For aperiodic, we need to be careful. Let me use direct computation for accuracy.
        
        # Direct autocorrelation
        r = np.correlate(x, x, mode='full')
        # r has length 2N-1, centered at N-1
        center = N - 1
        # Sidelobes: all except center
        r_sidelobes = np.concatenate([r[:center], r[center+1:]])
        
        # Gradient of sum(r_sidelobes^2) w.r.t. x
        # r(k) = sum_i x_i * x_{i+k}
        # dr(k)/dx_j = x_{j-k} + x_{j+k} (with appropriate bounds)
        # This is getting complex. Let me use a simpler approach.
        
        # Actually, let me just do local search directly. It's simpler and more effective for binary problems.
        break
    
    # Phase 2: Local search with bit flips
    # Try single-bit flips and multi-bit flips
    
    best_x = x.copy()
    best_score = autocorr_peak_sidelobe(best_x)
    
    # Precompute autocorrelation for speed
    def compute_score(x):
        r = np.correlate(x, x, mode='full')
        center = N - 1
        r_sidelobes = np.concatenate([r[:center], r[center+1:]])
        return np.max(np.abs(r_sidelobes))
    
    improved = True
    while improved and not check_deadline():
        improved = False
        for i in range(N):
            if check_deadline():
                break
            # Try flipping bit i
            old_val = best_x[i]
            best_x[i] = -old_val
            new_score = compute_score(best_x)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                best_x[i] = old_val
    
    # Phase 3: Multi-bit flips (2-bit and 3-bit)
    for n_bits in [2, 3]:
        while not check_deadline():
            improved = False
            # Try random pairs/triples
            for _ in range(10000):
                if check_deadline():
                    break
                indices = np.random.choice(N, size=n_bits, replace=False)
                old_vals = best_x[indices].copy()
                best_x[indices] = -old_vals
                new_score = compute_score(best_x)
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    best_x[indices] = old_vals
            if not improved:
                break
    
    # Convert to signed step heights (±1)
    return best_x
