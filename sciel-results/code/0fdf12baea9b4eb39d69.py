import time
import numpy as np

N = 600

def compute_metric(h):
    """Compute normalized maximum autoconvolution."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    s = h.sum()
    if s < 1e-15:
        return 1e10
    hn = h / s
    # Linear autoconvolution via FFT
    fft_size = 2 * N
    F = np.fft.rfft(hn, fft_size)
    conv = np.fft.irfft(F * F, fft_size)[:2 * N - 1]
    return conv.max()

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    def elapsed():
        return time.time() - start_time
    
    # Initialize with uniform
    h_best = np.ones(N)
    best_val = compute_metric(h_best)
    
    # Phase 1: Try various structured candidates
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Triangular
    center = N // 2
    tri = np.array([max(0, 1.0 - abs(i - center) / center) for i in range(N)])
    candidates.append(tri)
    
    # Gaussian
    x = np.linspace(-3, 3, N)
    gauss = np.exp(-x**2)
    candidates.append(gauss)
    
    # Raised cosine
    rc = 0.5 * (1 + np.cos(np.pi * np.linspace(-1, 1, N)))
    candidates.append(rc)
    
    # Inverse Gaussian (peaks at edges)
    inv_gauss = 1.0 - 0.3 * np.exp(-3.0 * np.linspace(-1, 1, N)**2)
    inv_gauss = np.maximum(inv_gauss, 0)
    candidates.append(inv_gauss)
    
    # Two-peak
    two = np.zeros(N)
    two[N//4] = 1.0
    two[3*N//4] = 1.0
    candidates.append(two)
    
    # Plateau
    for w_frac in [0.3, 0.5, 0.7, 0.9]:
        w = int(N * w_frac)
        plat = np.zeros(N)
        plat[N//2 - w//2 : N//2 + w//2 + 1] = 1.0
        candidates.append(plat)
    
    # Evaluate candidates
    for c in candidates:
        if elapsed() > deadline:
            break
        c = np.maximum(c, 0)
        if c.sum() < 1e-15:
            continue
        val = compute_metric(c)
        if val < best_val:
            best_val = val
            h_best = c.copy()
    
    # Phase 2: Water-filling inspired iterative refinement
    # The idea: iteratively raise the minimum height where the local autoconvolution peak is highest
    # to flatten the convolution profile
    
    h_current = h_best.copy()
    h_current = np.maximum(h_current, 1e-10)
    
    # Compute the autoconvolution to identify where peaks are
    def get_conv_profile(h):
        h = np.maximum(h, 0)
        s = h.sum()
        if s < 1e-15:
            return np.zeros(2*N-1)
        hn = h / s
        fft_size = 2 * N
        F = np.fft.rfft(hn, fft_size)
        conv = np.fft.irfft(F * F, fft_size)[:2 * N - 1]
        return conv
    
    # Iterative water-filling: raise low parts where conv is high
    for iteration in range(200):
        if elapsed() > deadline - 5:
            break
        
        # Compute current convolution profile
        conv = get_conv_profile(h_current)
        max_conv = conv.max()
        
        # Identify regions where convolution is near maximum
        # These are the "hot spots" we want to flatten
        threshold = max_conv * 0.95
        hot_mask = conv >= threshold
        
        # For each hot region, find which input indices contribute most
        # and slightly reduce them, while raising the minimum elsewhere
        if np.any(hot_mask):
            # Get the positions of hot spots
            hot_positions = np.where(hot_mask)[0]
            
            # For each hot position, the contribution comes from pairs (i,j) where i+j = pos
            # We want to redistribute: reduce high values, increase low values
            # Simple approach: slightly flatten the distribution
            
            # Compute local variance of h
            h_mean = h_current.mean()
            h_std = h_current.std()
            
            if h_std > 1e-10:
                # Apply gentle smoothing: mix with uniform
                alpha = 0.01  # Small mixing parameter
                h_new = (1 - alpha) * h_current + alpha * h_mean
                h_new = np.maximum(h_new, 0)
                
                # Also try to boost the minimum
                min_h = h_new.min()
                if min_h < h_mean * 0.5:
                    # Raise the bottom 10% towards the mean
                    sorted_idx = np.argsort(h_new)
                    bottom_10 = sorted_idx[:N//10]
                    h_new[bottom_10] += 0.005 * (h_mean - h_new[bottom_10])
                    h_new = np.maximum(h_new, 0)
                
                val_new = compute_metric(h_new)
                if val_new < best_val:
                    best_val = val_new
                    h_best = h_new.copy()
                    h_current = h_new.copy()
    
    # Phase 3: Cosine series refinement
    i = np.arange(N)
    M_max = 8
    
    # Start from current best
    h_cos = h_best.copy()
    
    for M in range(1, M_max + 1):
        if elapsed() > deadline - 5:
            break
        
        # Build cosine basis
        cos_basis = np.column_stack([np.cos(2 * np.pi * k * i / N) for k in range(1, M + 1)])
        
        # Try small perturbations
        for k in range(M):
            if elapsed() > deadline - 3:
                break
            for amp in [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
                for sign in [1, -1]:
                    h_test = h_cos + sign * amp * cos_basis[:, k]
                    h_test = np.maximum(h_test, 0)
                    if h_test.sum() < 1e-15:
                        continue
                    val = compute_metric(h_test)
                    if val < best_val:
                        best_val = val
                        h_best = h_test.copy()
                        h_cos = h_test.copy()
    
    # Phase 4: Final local polish with small random perturbations
    if elapsed() < deadline - 10:
        rng = np.random.RandomState(42)
        for _ in range(50):
            if elapsed() > deadline - 5:
                break
            h_test = h_best + rng.normal(0, 0.01, N)
            h_test = np.maximum(h_test, 0)
            if h_test.sum() < 1e-15:
                continue
            val = compute_metric(h_test)
            if val < best_val:
                best_val = val
                h_best = h_test.copy()
    
    # Final: ensure nonnegative and return
    h_best = np.maximum(h_best, 0.0)
    if h_best.sum() < 1e-15:
        h_best = np.ones(N)
    
    return list(h_best)
