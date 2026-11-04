import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 600
W = 1.0 / 1200.0

# Precompute fine grid for fast evaluation (for final score)
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT_FINE = 1 << (2 * FINE_N - 1).bit_length()

# For optimization, we work directly on the 600 bins
# Convolution of two length 600 arrays has length 1199
N_OPT_FFT = 1 << (2 * N - 1).bit_length()

def ratio_from_h(h):
    """Compute the normalized maximum of the autoconvolution."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    
    # Use fine grid for accurate max? Or coarse?
    # The task says "normalized maximum of its autoconvolution".
    # The seed program uses fine grid. I should stick to fine grid for the final score to be consistent with "seed/value".
    # However, for optimization, coarse is faster.
    
    f_fine = h[CELL]
    
    F = np.fft.rfft(f_fine, N_FFT_FINE)
    conv = np.fft.irfft(F * F, N_FFT_FINE)[:2 * FINE_N - 1]
    conv *= DX
    max_conv = np.max(conv)
    l2_sq = np.sum(f_fine ** 2) * DX
    if l2_sq < 1e-15:
        return float('inf')
    return max_conv / l2_sq

def params_to_h(params):
    """
    Convert parameters to step function heights.
    params: [a1, x1, s1, a2, x2, s2, ..., aK, xK, sK]
    """
    K = len(params) // 3
    h = np.zeros(FINE_N)
    for i in range(K):
        a_raw = params[3*i]
        x_c = params[3*i + 1]
        s_raw = params[3*i + 2]
        
        a = np.log1p(np.exp(a_raw))
        s = np.exp(s_raw) if s_raw < 20 else 1e10
        s = max(s, 1e-4)
        x_c = np.clip(x_c, -0.25, 0.25)
        
        h += a * np.exp(-0.5 * ((X_FINE - x_c) / s) ** 2)
    
    sums = np.zeros(N)
    counts = np.zeros(N)
    np.add.at(sums, CELL, h)
    np.add.at(counts, CELL, 1)
    counts[counts == 0] = 1
    h_bins = sums / counts
    
    return h_bins

def objective_gaussian(params):
    h = params_to_h(params)
    val = ratio_from_h(h)
    if not np.isfinite(val):
        return 1e10
    return val

def smooth_objective(h, tau=50.0):
    """
    Smooth surrogate for the ratio max(h*h) / ||h||_2^2
    """
    h = np.clip(h, 0.0, None)
    norm_sq = np.sum(h**2)
    if norm_sq < 1e-15:
        return 1e10
    
    # Compute convolution
    H = np.fft.rfft(h, N_OPT_FFT)
    conv = np.fft.irfft(H * H, N_OPT_FFT)[:2*N - 1]
    
    # Smooth max
    max_smooth = logsumexp(tau * conv) / tau
    
    return max_smooth / norm_sq

def smooth_gradient(h, tau=50.0):
    """
    Gradient of the smooth surrogate
    """
    h = np.clip(h, 0.0, None)
    norm_sq = np.sum(h**2)
    if norm_sq < 1e-15:
        return np.zeros(N)
    
    H = np.fft.rfft(h, N_OPT_FFT)
    conv = np.fft.irfft(H * H, N_OPT_FFT)[:2*N - 1]
    
    # Softmax weights
    # p_k = exp(tau * conv_k) / sum(exp(tau * conv))
    # Using logsumexp for stability
    max_conv_val = np.max(conv)
    exp_conv = np.exp(tau * (conv - max_conv_val))
    sum_exp = np.sum(exp_conv)
    p = exp_conv / sum_exp
    
    # q = tau * p
    q = tau * p
    
    # max_smooth = logsumexp(tau * conv) / tau
    max_smooth = (np.log(sum_exp) + tau * max_conv_val) / tau
    
    # Gradient of max_smooth w.r.t h
    # dM/dh_m = 2 * (q * h)_m
    # q is length 1199, h is length 600
    # We need (q * h) for indices corresponding to h
    # q[k] corresponds to conv[k].
    # conv[k] = sum_j h[j] h[k-j]
    # d conv[k] / d h[m] = 2 h[k-m] (if k-m in range, else 0? No, convolution definition)
    # Let's use FFT for the gradient of M
    # G_M = 2 * ifft(fft(q) * fft(h))
    # Note: q is defined on the convolution support.
    # We need to embed q into an array of size N_OPT_FFT
    q_full = np.zeros(N_OPT_FFT, dtype=np.complex128)
    q_full[:len(q)] = q
    Q = np.fft.rfft(q_full, N_OPT_FFT)
    # Wait, q is real.
    # The gradient of M = sum_k tau p_k conv_k
    # dM/dh = 2 * conv(q, h) ?
    # Let's verify.
    # M = sum_k w_k (h*h)_k
    # dM/dh_m = sum_k w_k (2 h_{k-m})
    # This is 2 * (w * h)_m where * is convolution.
    # So we compute convolution of q and h.
    
    Q_fft = np.fft.rfft(q, N_OPT_FFT) # q is length 1199, padded to N_OPT_FFT
    # Actually, np.fft.rfft pads with zeros.
    # H is already computed.
    grad_M = np.fft.irfft(Q_fft * H, N_OPT_FFT)[:N]
    grad_M *= 2.0
    
    # Gradient of J = M / norm_sq
    # dJ = (dM * norm_sq - M * d(norm_sq)) / norm_sq^2
    # d(norm_sq) = 2h
    grad_J = (grad_M * norm_sq - 2.0 * max_smooth * h) / (norm_sq * norm_sq)
    
    return grad_J

def construct():
    deadline = time.time() + 140.0 # Safety margin
    
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h(best_h)
    
    # Phase 1: Gaussian Initialization
    # Keep this part similar to the original to ensure a good start
    K = 5
    np.random.seed(42)
    
    inits = []
    # 1. Uniform-ish
    p0 = np.zeros(K * 3)
    for i in range(K):
        p0[3*i] = np.log(1.0 / K)
        p0[3*i+1] = np.random.uniform(-0.2, 0.2)
        p0[3*i+2] = np.log(0.1 + np.random.rand()*0.2)
    inits.append(p0)
    
    # 2. Single dominant bump
    for shift in [-0.15, 0, 0.15]:
        p0 = np.zeros(K * 3)
        p0[0] = np.log(1.0)
        p0[1] = shift
        p0[2] = np.log(0.15)
        for i in range(1, K):
            p0[3*i] = np.log(0.01)
            p0[3*i+1] = np.random.uniform(-0.2, 0.2)
            p0[3*i+2] = np.log(0.05)
        inits.append(p0)
        
    # 3. Random smooth shapes
    for _ in range(5):
        p0 = np.zeros(K * 3)
        for i in range(K):
            p0[3*i] = np.random.uniform(-2, 1)
            p0[3*i+1] = np.random.uniform(-0.25, 0.25)
            p0[3*i+2] = np.random.uniform(np.log(0.05), np.log(0.3))
        inits.append(p0)

    # Optimize Gaussians quickly
    for p0 in inits:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(
                objective_gaussian, 
                p0, 
                method='L-BFGS-B', 
                options={'maxiter': 20, 'ftol': 1e-6}
            )
            if res.fun < best_val:
                h_candidate = params_to_h(res.x)
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Phase 2: Direct Bin Optimization with Smooth Surrogate
    # Use the best Gaussian result as initial guess
    h_init = best_h.copy()
    # Normalize to avoid scale issues? The objective is scale invariant.
    # L-BFGS-B might drift in scale. We can normalize h in the callback?
    # Or just let it run. The gradient is zero along the scale direction?
    # No, the gradient of the ratio is not zero along scale, but the function is constant.
    # Actually, dJ/dh has a component.
    # To be safe, we can constrain sum(h^2)=1?
    # L-BFGS-B doesn't do equality constraints easily.
    # We can just run it.
    
    tau = 20.0
    
    def obj_wrap(h):
        return smooth_objective(h, tau)
    
    def grad_wrap(h):
        return smooth_gradient(h, tau)
    
    try:
        res_opt = minimize(
            obj_wrap, 
            h_init, 
            jac=grad_wrap, 
            method='L-BFGS-B', 
            bounds=[(0, None)] * N,
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5}
        )
        
        if res_opt.fun < best_val:
            h_candidate = res_opt.x
            # Ensure non-negative
            h_candidate = np.clip(h_candidate, 0, None)
            val = ratio_from_h(h_candidate)
            if val < best_val and np.isfinite(val):
                best_val = val
                best_h = h_candidate.copy()
    except Exception as e:
        pass

    # Phase 3: Local Refinement with Exact Objective (Coordinate Descent / Random Steps)
    # The smooth optimum might be slightly suboptimal for the exact max.
    # Do a few steps of projected gradient or random search on bins.
    
    if time.time() < deadline - 5.0:
        h_curr = best_h.copy()
        val_curr = best_val
        
        # Simple coordinate descent on the exact objective
        # This is slow, so do limited iterations
        for _ in range(200):
            if time.time() > deadline - 2.0:
                break
            
            # Pick a random subset of bins to perturb
            idx = np.random.choice(N, 20, replace=False)
            h_try = h_curr.copy()
            # Small random steps
            h_try[idx] += np.random.randn(20) * 0.005
            h_try = np.clip(h_try, 0, None)
            
            val_try = ratio_from_h(h_try)
            if val_try < val_curr:
                val_curr = val_try
                h_curr = h_try
                best_val = val_curr
                best_h = h_curr

    return list(best_h)
