import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _true_autoconv_max(h):
    """Compute max of autoconvolution over all lags (including lag 0)."""
    h = np.asarray(h, dtype=float)
    # Full autoconvolution: c[k] = sum_i h[i]*h[i+k] for k = -(N-1)..(N-1)
    c = np.correlate(h, h, mode='full')
    return float(np.max(c))

def _score(h):
    h = np.asarray(h, dtype=float)
    l2_sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 == 0 or l2_sq == 0:
        return 0.0
    conv_max = _true_autoconv_max(h)
    if conv_max == 0:
        return 0.0
    return l2_sq / (l1 * conv_max)

def _neg_score(h):
    return -_score(h)

def _sidon_set(n, m):
    """Find a Sidon-like set of m positions in [0, n-1] minimizing autoconvolution peaks."""
    best_set = None
    best_score = 0.0
    rng = np.random.RandomState(12345)
    for _ in range(200):
        s = set(rng.choice(n, size=m, replace=False))
        # Compute max autoconvolution off-diagonal
        h = np.zeros(n)
        for pos in s:
            h[pos] = 1.0
        c = np.correlate(h, h, mode='full')
        # Exclude lag 0 (center)
        center = n - 1
        c_no_center = np.delete(c, center)
        peak = np.max(c_no_center)
        # We want to minimize peak, so maximize negative
        if peak < best_score or best_set is None:
            best_score = peak
            best_set = sorted(s)
    return best_set if best_set else list(range(m))

def _water_fill_init(n, m, rng):
    """Spectral water-filling: start with sparse Sidon-like set, then refine heights."""
    # Start with a sparse set
    if m <= 10:
        positions = _sidon_set(n, m)
    else:
        positions = sorted(rng.choice(n, size=m, replace=False).tolist())
    
    h = np.zeros(n)
    for pos in positions:
        h[pos] = 1.0
    
    # Iterative water-filling: adjust heights based on local spectral density
    for it in range(50):
        # Compute FFT-based spectral density
        H = np.fft.fft(h, n=2*n)
        spectral_density = np.abs(H)**2
        
        # For each position, estimate local interference
        # Use the fact that autoconvolution peak at lag k relates to spectral structure
        # Simple heuristic: reduce height where it contributes to high autoconv peaks
        
        # Compute current autoconvolution
        c = np.correlate(h, h, mode='full')
        center = n - 1
        
        # For each nonzero position, compute its contribution to off-center peaks
        # and adjust height inversely
        for i in range(n):
            if h[i] > 0:
                # Contribution of h[i] to autoconv at each lag
                # c[k] includes h[i]*h[i+k] and h[i+k]*h[i]
                # Approximate: reduce h[i] if it's part of many high peaks
                contrib = 0.0
                for k in range(1, n):
                    if i + k < n:
                        contrib += c[center + k] * h[i+k]
                    if i - k >= 0:
                        contrib += c[center - k] * h[i-k]
                # Inverse proportional to local interference
                h[i] = 1.0 / (1.0 + 0.1 * contrib)
        
        # Normalize to keep sum reasonable
        if np.sum(h) > 0:
            h = h / np.sum(h) * m  # keep average height around 1
    
    return np.maximum(h, 0)

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # 20s safety margin
    
    best_val = 0.0
    best_h = np.ones(N)
    
    # Update with seed
    v = _score(best_h)
    if v > best_val:
        best_val = v
        best_h = best_h.copy()
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initial points
    starts = []
    
    # Uniform
    starts.append(np.ones(N))
    
    # Sparse Sidon-like sets
    for m in [5, 8, 10, 12, 15]:
        h_sparse = np.zeros(N)
        positions = _sidon_set(N, m)
        for pos in positions:
            h_sparse[pos] = 1.0
        starts.append(h_sparse)
    
    # Water-filling initializations
    for m in [8, 10, 12, 15, 20]:
        h_wf = _water_fill_init(N, m, rng)
        if np.sum(h_wf) > 0:
            starts.append(h_wf)
    
    # Gaussian-like
    x = np.linspace(-1, 1, N)
    starts.append(np.exp(-x**2 / 0.1))
    starts.append(np.exp(-x**2 / 0.5))
    
    # Bimodal
    x = np.linspace(-1, 1, N)
    starts.append(np.exp(-(x-0.5)**2/0.05) + np.exp(-(x+0.5)**2/0.05))
    
    # Random
    for _ in range(15):
        s = rng.random(N)
        starts.append(s)
    
    # Random with sparsity
    for _ in range(10):
        s = rng.random(N)
        mask = rng.random(N) > 0.7
        s[mask] = 0.0
        starts.append(s)
    
    for i, x0 in enumerate(starts):
        if time.time() > deadline:
            break
        
        x0 = np.maximum(x0, 0.0)
        if np.sum(x0) == 0:
            continue
        
        # Try BFGS
        try:
            res = minimize(_neg_score, x0, method='BFGS', 
                          bounds=[(0, None)] * N,
                          options={'maxiter': 100, 'gtol': 1e-7})
            h_opt = np.maximum(res.x, 0.0)
            v = _score(h_opt)
            if v > best_val:
                best_val = v
                best_h = h_opt.copy()
        except Exception:
            pass
        
        # Try Nelder-Mead on best result from this start
        if time.time() > deadline:
            break
        try:
            res = minimize(_neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8})
            h_opt = np.maximum(res.x, 0.0)
            v = _score(h_opt)
            if v > best_val:
                best_val = v
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final refinement
    if time.time() < deadline - 10:
        try:
            res = minimize(_neg_score, best_h, method='BFGS',
                          bounds=[(0, None)] * N,
                          options={'maxiter': 500, 'gtol': 1e-9})
            h_opt = np.maximum(res.x, 0.0)
            v = _score(h_opt)
            if v > best_val:
                best_val = v
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Additional refinement with Nelder-Mead
    if time.time() < deadline - 5:
        try:
            res = minimize(_neg_score, best_h, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            h_opt = np.maximum(res.x, 0.0)
            v = _score(h_opt)
            if v > best_val:
                best_val = v
                best_h = h_opt.copy()
        except Exception:
            pass
    
    return list(best_h)
