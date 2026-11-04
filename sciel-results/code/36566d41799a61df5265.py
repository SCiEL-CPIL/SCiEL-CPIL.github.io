import numpy as np
from scipy import optimize
import time

N = 600
DELTA = (1.0 / 2.0) / N  # interval width = 1/1200

def compute_autoconv(h):
    """Compute autoconvolution of step function with heights h on N equal intervals."""
    h = np.asarray(h, dtype=np.float64)
    size = 2 * N - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    conv = np.fft.irfft(H * H, n=fft_size)[:size]
    autoconv = DELTA * conv
    return autoconv

def compute_score(h):
    """Compute the normalized max autoconvolution score."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12):
        return np.inf
    h = np.maximum(h, 0.0)
    autoconv = compute_autoconv(h)
    max_ac = np.max(autoconv)
    integral = DELTA * np.sum(h)
    if integral <= 0:
        return np.inf
    score = max_ac * 0.5 / (integral ** 2)
    return score

def smooth_objective_and_grad(h, beta=50.0):
    """
    Compute smooth approximation of the score using log-sum-exp, plus gradient.
    Score = 0.5 * max(autoconv) / integral^2
    Smooth: 0.5 * (1/beta) * log(sum(exp(beta * autoconv))) / integral^2
    """
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return 1e10, np.zeros_like(h)
    
    autoconv = compute_autoconv(h)
    integral = DELTA * np.sum(h)
    if integral <= 0:
        return 1e10, np.zeros_like(h)
    
    # Log-sum-exp for stability
    max_ac = np.max(autoconv)
    exp_ac = np.exp(beta * (autoconv - max_ac))
    log_sum_exp = max_ac + np.log(np.sum(exp_ac)) / beta
    
    score = 0.5 * log_sum_exp / (integral ** 2)
    
    # Gradient with respect to h
    # d(autoconv_k)/d(h_j) = DELTA * (h_{k-j} + h_{j-k}) for valid indices
    # autoconv has length 2N-1, h has length N
    # For k in [0, 2N-2], j in [0, N-1]:
    #   d(autoconv_k)/d(h_j) = DELTA * (h[k-j] if 0<=k-j<N else 0 + h[j-k] if 0<=j-k<N else 0)
    
    # Build gradient efficiently
    # grad_h[j] = sum_k w_k * d(autoconv_k)/d(h_j)
    #           = sum_k w_k * DELTA * (h[k-j] + h[j-k]) (with bounds)
    
    # This is a convolution-like operation
    # term1[j] = sum_k w_k * h[k-j] (cross-correlation)
    # term2[j] = sum_k w_k * h[j-k] (same, by symmetry if w is symmetric)
    
    # w is not symmetric in general. Let's compute directly.
    # For efficiency, note that:
    # sum_k w_k * h[k-j] for k in [0, 2N-2], j in [0, N-1]
    # Let m = k - j, then k = m + j, m ranges from -j to 2N-2-j
    # Valid m: 0 <= m < N, so 0 <= m+j < N => m < N-j
    # Also need m >= 0, so m in [0, min(N-1, 2N-2-j)]
    
    # This is getting complex. Let's use a different approach:
    # Since N=600, we can compute the gradient by direct summation for each j,
    # but that's O(N^2) = 360000 ops, which is fine.
    
    # Better: use the FFT-based approach
    # autoconv = DELTA * ifft(FFT(h)^2)
    # The gradient of autoconv with respect to h can be computed as:
    # grad_autoconv = DELTA * ifft(2 * FFT(h) * FFT(dh)) for each dh
    # But we need the full Jacobian transposed.
    
    # Alternative: since autoconv_k = DELTA * sum_j h_j * h_{k-j},
    # d(autoconv)/d(h) is a (2N-1) x N matrix.
    # We need w^T * d(autoconv)/d(h) where w is (2N-1) vector.
    
    # result[j] = DELTA * sum_k w_k * (h[k-j] + h[j-k])
    #            = DELTA * (sum_k w_k * h[k-j] + sum_k w_k * h[j-k])
    
    # First term: sum_k w_k * h[k-j], k from 0 to 2N-2, j from 0 to N-1
    # Let's define w_ext of length 2N-1
    # For fixed j: sum_{k=0}^{2N-2} w_k * h[k-j] where h[m] = 0 if m<0 or m>=N
    # = sum_{m=-j}^{2N-2-j} w_{m+j} * h[m] where h[m] nonzero for m in [0,N-1]
    # = sum_{m=max(0,-j)}^{min(N-1, 2N-2-j)} w_{m+j} * h[m]
    # = sum_{m=0}^{min(N-1, 2N-2-j)} w_{m+j} * h[m]  (since j>=0, -j<=0)
    
    # Similarly second term: sum_k w_k * h[j-k]
    # = sum_{k=0}^{2N-2} w_k * h[j-k] where h[m] nonzero for m in [0,N-1]
    # Let m = j-k, k = j-m
    # = sum_{m=j-(2N-2)}^{j} w_{j-m} * h[m] where m in [0,N-1]
    # = sum_{m=max(0, j-(2N-2))}^{min(j, N-1)} w_{j-m} * h[m]
    # = sum_{m=0}^{min(j, N-1)} w_{j-m} * h[m]  (since j-(2N-2) <= 0 for j<N)
    
    # So result[j] = DELTA * (sum_{m=0}^{min(N-1, 2N-2-j)} w_{m+j} * h[m] + sum_{m=0}^{min(j, N-1)} w_{j-m} * h[m])
    
    # This is still O(N^2) total. With N=600, that's 360K multiplications, very fast.
    
    grad_h = np.zeros(N)
    for j in range(N):
        # First sum: m from 0 to min(N-1, 2N-2-j)
        upper1 = min(N-1, 2*N-2-j)
        if upper1 >= 0:
            idx1 = np.arange(upper1+1)
            grad_h[j] += DELTA * np.dot(w[idx1+j], h[idx1])
        
        # Second sum: m from 0 to min(j, N-1)
        upper2 = min(j, N-1)
        if upper2 >= 0:
            idx2 = np.arange(upper2+1)
            grad_h[j] += DELTA * np.dot(w[j-idx2], h[idx2])
    
    d_lse_dh = grad_h  # This is d(LSE)/d(h) since LSE = max_ac + log(sum(exp(beta*(ac-max_ac)))/beta)
    # Wait, I need to be more careful.
    # LSE_smooth = (1/beta) * log(sum(exp(beta * autoconv)))
    # d(LSE_smooth)/d(autoconv_k) = exp(beta*autoconv_k) / sum(exp(beta*autoconv)) = w_k
    # So d(LSE_smooth)/d(h_j) = sum_k w_k * d(autoconv_k)/d(h_j) = grad_h[j] as computed
    
    d_int_dh = DELTA * np.ones(N)
    
    d_score_dh = 0.5 * (d_lse_dh / (integral ** 2) - 2 * log_sum_exp / (integral ** 3) * d_int_dh)
    
    return score, d_score_dh

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Start with uniform
    h = np.ones(N)
    best_score = compute_score(h)
    best_h = h.copy()
    
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5/N  # midpoints
    
    # 1. Parametric Fourier Optimization with smooth objective
    n_modes = 15
    def param_to_h(params):
        h = 1.0
        for k in range(1, n_modes + 1):
            h += params[2*k-2] * np.cos(2*np.pi*k*x)
            h += params[2*k-1] * np.sin(2*np.pi*k*x)
        return np.maximum(h, 0)
        
    def objective_and_grad_fourier(params):
        h = param_to_h(params)
        score, grad_h = smooth_objective_and_grad(h, beta=50.0)
        
        # Chain rule: d(score)/d(params) = d(score)/d(h) * d(h)/d(params)
        grad_params = np.zeros(2 * n_modes)
        for k in range(1, n_modes + 1):
            grad_params[2*k-2] = np.dot(grad_h, -2*np.pi*k*np.sin(2*np.pi*k*x))
            grad_params[2*k-1] = np.dot(grad_h, 2*np.pi*k*np.cos(2*np.pi*k*x))
        
        return score, grad_params
        
    # Multi-start L-BFGS-B with analytical gradient
    np.random.seed(42)
    best_fourier_score = np.inf
    best_fourier_params = None
    
    for restart in range(20):
        if time.time() > deadline - 25:
            break
        if restart == 0:
            x0 = np.zeros(2 * n_modes)
        else:
            x0 = np.random.randn(2 * n_modes) * 0.3
            
        try:
            res = optimize.minimize(objective_and_grad_fourier, x0, method='L-BFGS-B', 
                                    jac=True,
                                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6})
            if res.fun < best_fourier_score:
                best_fourier_score = res.fun
                best_fourier_params = res.x.copy()
                h_candidate = param_to_h(res.x)
                actual_score = compute_score(h_candidate)
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = h_candidate.copy()
        except Exception:
            pass
            
    # 2. Direct Optimization on 600 heights with smooth objective and analytical gradient
    if time.time() < deadline - 15:
        h_start = best_h.copy()
        
        def objective_and_grad_direct(h_vec):
            if np.any(h_vec < 0):
                return 1e10, np.zeros_like(h_vec)
            return smooth_objective_and_grad(h_vec, beta=50.0)
            
        try:
            res_direct = optimize.minimize(objective_and_grad_direct, h_start, method='L-BFGS-B',
                                           jac=True,
                                           bounds=[(0, None)] * N,
                                           options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6})
            if res_direct.fun < best_score:
                h_candidate = res_direct.x
                actual_score = compute_score(h_candidate)
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = h_candidate.copy()
        except Exception:
            pass
        
    # 3. Block structure optimization with Differential Evolution
    # Represent the step function as a convex combination of a small set of disjoint, symmetric indicator functions for sub-intervals
    if time.time() < deadline - 10:
        n_blocks_options = [2, 3, 4]
        
        for n_blocks in n_blocks_options:
            if time.time() > deadline - 10:
                break
            
            n_params = 3 * n_blocks
            bounds = []
            for _ in range(n_blocks):
                bounds.append((0.0, float(N - 1)))   # start bin
                bounds.append((0.0, float(N - 1)))   # end bin
                bounds.append((-5.0, 5.0))           # weight raw
            
            def objective_block(params):
                h = np.zeros(N)
                for i in range(n_blocks):
                    s_raw = params[3 * i]
                    e_raw = params[3 * i + 1]
                    w_raw = params[3 * i + 2]
                    
                    # Convert raw params to bin indices [0, N-1]
                    s_idx = int(np.clip(np.round(s_raw), 0, N - 1))
                    e_idx = int(np.clip(np.round(e_raw), 0, N - 1))
                    
                    if e_idx >= s_idx:
                        # Weight is non-negative via softplus
                        w = np.log1p(np.exp(np.clip(w_raw, -50, 50)))
                        h[s_idx:e_idx + 1] += w
                
                val = compute_score(h)
                if not np.isfinite(val):
                    return 1e10
                return val
            
            # Initial population with symmetric block structures
            init_pop = []
            rng = np.random.default_rng(42)
            for _ in range(20):
                p = np.zeros(n_params)
                # Create symmetric pairs of blocks
                for i in range(n_blocks):
                    # Center of domain is N//2
                    center = N // 2
                    # Random offset
                    offset = int(rng.uniform(50, N // 2 - 50))
                    width = int(rng.uniform(20, 100))
                    
                    if i % 2 == 0:
                        # Left block
                        s = max(0, center - offset - width // 2)
                        e = min(N - 1, center - offset + width // 2)
                    else:
                        # Right block
                        s = max(0, center + offset - width // 2)
                        e = min(N - 1, center + offset + width // 2)
                    
                    p[3 * i] = float(s)
                    p[3 * i + 1] = float(e)
                    p[3 * i + 2] = float(rng.uniform(-1, 1))
                
                init_pop.append(p)
            
            try:
                res_de = optimize.differential_evolution(
                    objective_block,
                    bounds,
                    maxiter=100,
                    tol=1e-6,
                    popsize=20,
                    init=np.array(init_pop),
                    seed=42
                )
                
                h_candidate = np.zeros(N)
                for i in range(n_blocks):
                    s_raw = res_de.x[3 * i]
                    e_raw = res_de.x[3 * i + 1]
                    w_raw = res_de.x[3 * i + 2]
                    
                    s_idx = int(np.clip(np.round(s_raw), 0, N - 1))
                    e_idx = int(np.clip(np.round(e_raw), 0, N - 1))
                    
                    if e_idx >= s_idx:
                        w = np.log1p(np.exp(np.clip(w_raw, -50, 50)))
                        h_candidate[s_idx:e_idx + 1] += w
                
                actual_score = compute_score(h_candidate)
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = h_candidate.copy()
            except Exception:
                pass
        
        # Final local refinement using the best block structure as initial guess
        if time.time() < deadline - 5:
            h_start = best_h.copy()
            
            def objective_and_grad_direct_final(h_vec):
                if np.any(h_vec < 0):
                    return 1e10, np.zeros_like(h_vec)
                return smooth_objective_and_grad(h_vec, beta=100.0)
            
            try:
                res_final = optimize.minimize(objective_and_grad_direct_final, h_start, method='L-BFGS-B',
                                              jac=True,
                                              bounds=[(0, None)] * N,
                                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                if res_final.fun < best_score:
                    h_candidate = res_final.x
                    actual_score = compute_score(h_candidate)
                    if actual_score < best_score:
                        best_score = actual_score
                        best_h = h_candidate.copy()
            except Exception:
                pass
    
    return best_h.tolist()
