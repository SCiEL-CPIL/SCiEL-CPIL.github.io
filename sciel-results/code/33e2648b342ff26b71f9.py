import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 1.0 / N

def _autoconv_sup(h):
    """Compute sup norm of autoconvolution of step function with heights h."""
    n = len(h)
    num_points = 2 * n - 1
    
    # Use FFT for efficient convolution
    # Pad to length 2n for linear convolution
    h_padded = np.zeros(2 * n)
    h_padded[:n] = h
    
    # FFT-based convolution
    H = np.fft.fft(h_padded, n=4*n)
    conv = np.fft.ifft(H * H).real
    
    # Extract the relevant part: indices 0 to 2n-2
    conv_relevant = conv[:2*n-1]
    
    # The convolution gives sum h[i]*h[j] for each lag
    # But we need to account for the overlap structure
    # For step functions, the autoconvolution at k*DELTA is:
    # DELTA * sum_{i,j} h[i]*h[j] * overlap_length(i,j,k)
    
    # The FFT gives us the discrete convolution, but we need the continuous
    # overlap integral. Let me compute it directly but vectorized.
    
    ff_vals = np.zeros(num_points)
    
    for k in range(num_points):
        total = 0.0
        i_min = max(0, k - (n - 1))
        i_max = min(n - 1, k)
        for i in range(i_min, i_max + 1):
            j_min = max(0, k - i - 1)
            j_max = min(n - 1, k - i + 1)
            if j_min > j_max:
                continue
            js = np.arange(j_min, j_max + 1)
            lo = np.maximum(i, k - js)
            hi = np.minimum(i + 1, k - js + 1)
            overlaps = np.maximum(0, hi - lo)
            total += np.sum(h[i] * h[js] * overlaps)
        ff_vals[k] = total * DELTA
    
    return np.max(ff_vals)

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = _autoconv_sup(h)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def _smooth_autoconv_sup(h, beta=100.0):
    """Compute smooth approximation of sup norm using log-sum-exp."""
    n = len(h)
    num_points = 2 * n - 1
    ff_vals = np.zeros(num_points)
    
    for k in range(num_points):
        total = 0.0
        i_min = max(0, k - (n - 1))
        i_max = min(n - 1, k)
        for i in range(i_min, i_max + 1):
            j_min = max(0, k - i - 1)
            j_max = min(n - 1, k - i + 1)
            if j_min > j_max:
                continue
            js = np.arange(j_min, j_max + 1)
            lo = np.maximum(i, k - js)
            hi = np.minimum(i + 1, k - js + 1)
            overlaps = np.maximum(0, hi - lo)
            total += np.sum(h[i] * h[js] * overlaps)
        ff_vals[k] = total * DELTA
    
    # Log-sum-exp for numerical stability
    max_val = np.max(ff_vals)
    if max_val == 0:
        return 0.0
    return max_val + np.log(np.sum(np.exp((ff_vals - max_val) * beta))) / beta

def _smooth_ratio(h, beta=100.0):
    """Compute smooth ratio."""
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = _smooth_autoconv_sup(h, beta)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def generate_mian_chowla(n, max_size):
    """Generate a Mian-Chowla (Sidon) set."""
    sidon = [0]
    used_sums = {0}
    for candidate in range(1, n):
        valid = True
        for i in sidon:
            if i + candidate in used_sums:
                valid = False
                break
        if valid:
            sidon.append(candidate)
            used_sums.add(candidate + candidate)
            for i in sidon[:-1]:
                used_sums.add(i + candidate)
        if len(sidon) >= max_size:
            break
    return sidon

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    np.random.seed(42)
    
    initializations = []
    
    # 1. Mian-Chowla sets of various sizes
    for size in [8, 10, 12, 14, 16]:
        s = generate_mian_chowla(N, size)
        h_mc = np.zeros(N)
        for idx in s:
            h_mc[idx] = 1.0
        initializations.append(h_mc)
        
    # 2. Random sparse vectors
    for _ in range(20):
        h_sp = np.zeros(N)
        num_ones = np.random.randint(5, 15)
        indices = np.random.choice(N, num_ones, replace=False)
        h_sp[indices] = np.random.uniform(0.5, 1.5, num_ones)
        initializations.append(h_sp)
        
    # 3. Random dense vectors
    for _ in range(10):
        h_rd = np.random.uniform(0, 1, N)
        initializations.append(h_rd)
        
    # 4. Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    initializations.append(h_alt)
    
    # 5. Every 3
    h_sp3 = np.zeros(N)
    h_sp3[::3] = 1.0
    initializations.append(h_sp3)
    
    # 6. Gaussian
    t = np.linspace(-1, 1, N)
    initializations.append(np.exp(-t**2/0.3))
    
    for h0 in initializations:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        h0 = np.maximum(h0, 0)
        s0 = np.sum(h0)
        if s0 < 1e-15:
            continue
        h0 = h0 / s0
        
        # Optimize using L-BFGS-B on the smooth surrogate
        def neg_smooth_ratio(x):
            x = np.maximum(x, 0)
            s = np.sum(x)
            if s < 1e-15:
                return 0.0
            return -_smooth_ratio(x / s, beta=50.0)
        
        try:
            result = minimize(neg_smooth_ratio, h0.copy(), method='L-BFGS-B',
                             bounds=[(0, None)] * N,
                             options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            
            h_candidate = np.maximum(result.x, 0)
            s = np.sum(h_candidate)
            if s > 1e-15:
                h_candidate = h_candidate / s
                sc = _ratio(h_candidate)
                if sc > best_score:
                    best_score = sc
                    best_h = h_candidate.copy()
        except Exception:
            continue
    
    # Final polish with Nelder-Mead on the exact ratio
    if time.time() - start_time < DEADLINE - 10:
        try:
            def neg_ratio(x):
                x = np.maximum(x, 0)
                s = np.sum(x)
                if s < 1e-15:
                    return 0.0
                return -_ratio(x / s)
            
            result = minimize(neg_ratio, best_h.copy(), method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            h_candidate = np.maximum(result.x, 0)
            s = np.sum(h_candidate)
            if s > 1e-15:
                h_candidate = h_candidate / s
                sc = _ratio(h_candidate)
                if sc > best_score:
                    best_score = sc
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    return best_h.tolist()
