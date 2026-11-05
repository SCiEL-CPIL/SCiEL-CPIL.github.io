import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import fft, ifft

N = 400
DEADLINE = 145.0

def _value(h):
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))

def _smooth_objective(h, beta=50.0):
    """Smooth objective using log-sum-exp for max."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max = m + log_sum / beta
    return abs(2.0 * N * soft_max / (s * s))

def _smooth_grad(h, beta=50.0):
    """Analytic gradient of smooth objective using FFT."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18, np.zeros_like(h)
    
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    weights = np.exp(shifted)
    wsum = np.sum(weights)
    weights /= wsum
    soft_max = m + np.log(wsum) / beta
    
    # Gradient of soft_max w.r.t. h: d(soft_max)/d(h_k) = sum_j weights[j] * d(conv_j)/d(h_k)
    # conv[j] = sum_i h[i]*h[j-i] (with valid indices for full convolution)
    # d(conv_j)/d(h_k) = sum_i [delta_{i,k}*h[j-i] + h[i]*delta_{j-i,k}]
    #                  = h[j-k] + h[k-(j-N+1)] (with bounds checking)
    # For full convolution of length 2N-1: conv[j] = sum_{i=max(0,j-N+1)}^{min(N-1,j)} h[i]*h[j-i]
    # d(conv_j)/d(h_k) = h[j-k] (if 0<=j-k<N) + h[k-(j-N+1)] (if 0<=k-(j-N+1)<N and j-k != k-(j-N+1))
    # Simplified: for full conv, d(conv_j)/d(h_k) = h[j-k] + h[2N-1-j-k] if both in range
    
    # More carefully: conv[j] = sum_{i=0}^{N-1} h[i]*h[j-i] where h is zero-padded outside [0,N-1]
    # d(conv_j)/d(h_k) = h[j-k] + h[k-j+N-1]... let me just use the fact that:
    # If we define H as zero-padded to length 2N-1, then conv = H * H (circular if padded to 2N-1)
    # Actually, let's use the direct formula: d(conv_j)/d(h_k) = h[j-k] if 0<=j-k<N, plus h[k-(j-N+1)] if 0<=k-(j-N+1)<N
    
    # For efficiency, note that sum_j weights[j]*h[j-k] = (weights * h)[k] (cross-correlation)
    # And sum_j weights[j]*h[k-(j-N+1)] = sum_j weights[j]*h[N-1+j-k] = (weights * h)[N-1-k] (with appropriate indexing)
    
    # Let's compute: g[k] = sum_j weights[j] * (h[j-k] + h[N-1-(j-k)] if in range)
    # This is essentially the convolution of weights with h, but weights has length 2N-1 and h has length N
    
    # Let's pad h to length 2N-1 and compute the cross-correlation properly
    h_padded = np.zeros(2*N-1)
    h_padded[:N] = h
    
    # d(conv_j)/d(h_k) = h[j-k] + h[k-j+N-1] for valid indices
    # Let's compute sum_j weights[j]*h[j-k] for each k: this is the cross-correlation of weights and h
    # corr[k] = sum_j weights[j]*h[j-k] = sum_j weights[j]*h_padded[j-k]
    
    # Using FFT for cross-correlation:
    # corr = ifft(fft(weights, 2*N-1) * conj(fft(h_padded, 2*N-1)))
    
    size = 2*N-1
    W = fft(weights, size)
    H = fft(h_padded, size)
    corr = ifft(W * np.conj(H)).real
    
    # corr[k] = sum_j weights[j]*h_padded[j-k] = sum_j weights[j]*h[j-k] (with h zero outside [0,N-1])
    
    # Similarly, sum_j weights[j]*h[k-j+N-1]: let i = k-j+N-1, so j = k+N-1-i
    # = sum_i weights[k+N-1-i]*h[i] = sum_i h[i]*weights[N-1+k-i]
    # This is the cross-correlation of h and weights at lag N-1+k
    corr2 = ifft(np.conj(H) * W).real
    # corr2[k] = sum_j h[j]*weights[j-k]... let me think more carefully
    
    # Actually, let's just compute the gradient numerically but efficiently using the structure
    # d(soft_max)/d(h_k) = sum_j weights[j] * d(conv_j)/d(h_k)
    # d(conv_j)/d(h_k) = h[j-k] + h[k-j+N-1] if both indices valid
    
    # Let's compute this directly but vectorized:
    # For each k, sum_j weights[j]*h[j-k] where 0<=j-k<N
    # This is the cross-correlation (weights * h)[k] in the sense of:
    # (w*h)[k] = sum_j w[j]*h[j-k]
    
    # And sum_j weights[j]*h[k-j+N-1] where 0<=k-j+N-1<N, i.e., j in [k+1, k+N]
    # Let i = k-j+N-1, so j = k+N-1-i, i in [0, N-1]
    # = sum_i h[i]*weights[k+N-1-i] = sum_i h[i]*weights[N-1+k-i]
    # This is the cross-correlation (h * weights)[N-1+k]
    
    # So d(soft_max)/d(h_k) = (w*h)[k] + (h*w)[N-1+k]
    # where (a*b)[k] = sum_j a[j]*b[j-k]
    
    # Using FFT: (w*h)[k] = ifft(fft(w) * conj(fft(h)))
    # (h*w)[k] = ifft(conj(fft(h)) * fft(w)) = same as above (real part)
    
    # So d(soft_max)/d(h_k) = 2 * real(ifft(fft(weights) * conj(fft(h_padded))))[k]
    # But we need to be careful about the second term's index
    
    # Let me just compute it numerically but in a vectorized way for safety:
    grad_soft_max = np.zeros(N)
    for k in range(N):
        # d(conv_j)/d(h_k) = h[j-k] if 0<=j-k<N, plus h[k-j+N-1] if 0<=k-j+N-1<N
        # j ranges from 0 to 2N-2
        # First term: j-k in [0,N-1] => j in [k, k+N-1]
        j1 = np.arange(k, min(k+N, 2*N-1))
        if len(j1) > 0:
            idx1 = j1 - k
            term1 = np.sum(weights[j1] * h[idx1])
        else:
            term1 = 0.0
        
        # Second term: k-j+N-1 in [0,N-1] => j in [k+1, k+N]
        j2 = np.arange(k+1, min(k+N+1, 2*N-1))
        if len(j2) > 0:
            idx2 = k - j2 + N - 1
            term2 = np.sum(weights[j2] * h[idx2])
        else:
            term2 = 0.0
        
        grad_soft_max[k] = term1 + term2
    
    # Now the full objective: f = abs(2*N*soft_max/s^2)
    # Assuming soft_max > 0 (which it usually is for reasonable h), f = 2*N*soft_max/s^2
    # df/d(h_k) = 2*N * (grad_soft_max[k]/s^2 - 2*soft_max*s_k/s^3)
    # where s_k = 1 for all k (since s = sum h, ds/dh_k = 1)
    
    if soft_max >= 0:
        grad = 2.0 * N * (grad_soft_max / (s * s) - 2.0 * soft_max / (s * s * s))
    else:
        grad = -2.0 * N * (grad_soft_max / (s * s) - 2.0 * soft_max / (s * s * s))
    
    val = abs(2.0 * N * soft_max / (s * s))
    return val, grad

def _diverse_init(seed=42):
    """Generate diverse initial points."""
    rng = np.random.RandomState(seed)
    inits = []
    
    # All ones (baseline)
    inits.append(np.ones(N))
    
    # Random smooth functions
    for _ in range(8):
        coeffs = rng.randn(10) * 0.5
        x = np.linspace(-1, 1, N)
        h = 1.0
        for i, c in enumerate(coeffs):
            h += c * np.cos(i * np.pi * x)
        h = np.maximum(h, -2.0)  # Keep finite
        inits.append(h)
    
    # Block patterns
    for block_size in [10, 20, 50]:
        h = np.ones(N)
        n_blocks = N // block_size
        for b in range(n_blocks):
            h[b*block_size:(b+1)*block_size] = rng.randn() * 0.5 + 1.0
        inits.append(h)
    
    # Sine waves with random phases
    for freq in [1, 2, 3, 5]:
        phase = rng.uniform(0, 2*np.pi)
        amp = rng.uniform(0.5, 1.5)
        x = np.arange(N)
        h = 1.0 + amp * np.sin(2*np.pi*freq*x/N + phase)
        inits.append(h)
    
    return inits

def _local_refine(h0, time_budget=10.0):
    """Local refinement using L-BFGS-B with smooth objective."""
    t_start = time.time()
    
    def objective(h):
        if time.time() - t_start > time_budget:
            raise TimeoutError()
        return _smooth_objective(h, beta=50.0)
    
    def gradient(h):
        if time.time() - t_start > time_budget:
            raise TimeoutError()
        _, g = _smooth_grad(h, beta=50.0)
        return g
    
    try:
        result = minimize(
            objective,
            h0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 100, 'maxfun': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        return result.x
    except (TimeoutError, Exception):
        return h0

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    best_h = np.ones(N)
    best_val = _value(best_h)  # 2.0
    
    # Phase 1: Multi-start local optimization from diverse initializations
    inits = _diverse_init(seed=42)
    
    for i, h_init in enumerate(inits):
        if time_left() < 15:
            break
        
        # Ensure valid initialization
        if not np.all(np.isfinite(h_init)) or abs(np.sum(h_init)) < 1e-12:
            continue
        
        # Quick evaluation
        v_init = _value(h_init)
        
        # Local refinement
        h_refined = _local_refine(h_init, time_budget=min(10.0, time_left() - 5.0))
        
        if not np.all(np.isfinite(h_refined)):
            h_refined = h_init
        
        v = _value(h_refined)
        if v < best_val:
            best_val = v
            best_h = h_refined.copy()
    
    # Phase 2: Additional random restarts if time permits
    if time_left() > 20:
        rng = np.random.RandomState(123)
        for _ in range(10):
            if time_left() < 10:
                break
            
            # Generate random smooth function
            coeffs = rng.randn(15) * 0.3
            x = np.linspace(-1, 1, N)
            h = 1.0
            for k, c in enumerate(coeffs):
                h += c * np.cos(k * np.pi * x + rng.uniform(0, np.pi))
            
            if not np.all(np.isfinite(h)) or abs(np.sum(h)) < 1e-12:
                continue
            
            h_refined = _local_refine(h, time_budget=min(8.0, time_left() - 5.0))
            
            if not np.all(np.isfinite(h_refined)):
                h_refined = h
            
            v = _value(h_refined)
            if v < best_val:
                best_val = v
                best_h = h_refined.copy()
    
    # Phase 3: Final polish with higher beta for sharper max approximation
    if time_left() > 10:
        def objective_sharp(h):
            return _smooth_objective(h, beta=200.0)
        
        def gradient_sharp(h):
            _, g = _smooth_grad(h, beta=200.0)
            return g
        
        try:
            result = minimize(
                objective_sharp,
                best_h,
                jac=gradient_sharp,
                method='L-BFGS-B',
                options={'maxiter': 50, 'maxfun': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if np.all(np.isfinite(result.x)):
                v = _value(result.x)
                if v < best_val:
                    best_val = v
                    best_h = result.x.copy()
        except Exception:
            pass
    
    # Safety: ensure valid output
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h
