import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 140.0  # 10s safety margin

def _check_deadline():
    return time.time() < DEADLINE

def _autoconv_fft(s):
    """Compute linear autocorrelation via FFT. Returns ac[0..N-1] where ac[k]=sum_i s[i]*s[i+k]."""
    n = len(s)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    f = np.fft.rfft(s, size)
    power = f * np.conj(f)
    ac = np.fft.irfft(power, size)[:n]
    return ac

def _objective_only(s):
    """Compute max|ac| / (sum(s))^2"""
    ac = _autoconv_fft(s)
    sm = np.sum(s)
    if abs(sm) < 1e-10:
        return np.inf
    return np.max(np.abs(ac)) / (sm * sm)

def _soft_max_and_grad(ac, beta):
    """Soft max of |ac| and its gradient w.r.t. ac."""
    abs_ac = np.abs(ac)
    # log-sum-exp trick
    m = np.max(beta * abs_ac)
    exps = np.exp(beta * abs_ac - m)
    logsumexp = m + np.log(np.sum(exps))
    val = logsumexp / beta
    weights = exps / np.sum(exps)
    grad_ac = beta * weights * np.sign(ac)
    return val, grad_ac

def _objective_and_grad(s, beta=10.0):
    """Objective and gradient for soft-max surrogate."""
    ac = _autoconv_fft(s)
    sm = np.sum(s)
    if abs(sm) < 1e-10:
        return np.inf, np.zeros_like(s)
    
    val, grad_ac = _soft_max_and_grad(ac, beta)
    
    # Gradient of ac[k] w.r.t. s[j]:
    # ac[k] = sum_{i=0}^{N-1-k} s[i]*s[i+k]
    # dac[k]/ds[j] = s[j+k] (if j+k<N) + s[j-k] (if j-k>=0)
    # Vectorized: grad_s = ac_grad convolved with s (circular, but we need linear)
    # Use FFT: grad_s = ifft(fft(grad_ac) * conj(fft(s))) gives circular correlation
    # For linear: zero-pad to 2N-1
    size = 1
    while size < 2 * N - 1:
        size *= 2
    g_ac_padded = np.zeros(size)
    g_ac_padded[:N] = grad_ac
    s_padded = np.zeros(size)
    s_padded[:N] = s
    
    G = np.fft.rfft(g_ac_padded, size)
    S = np.fft.rfft(s_padded, size)
    
    # Circular correlation in padded space gives linear correlation
    # corr[k] = sum_j g_ac[j] * s[j+k] for k>=0, and g_ac[j+k]*s[j] for k<0
    # Actually: ifft(fft(g) * conj(fft(s))) gives circular cross-correlation
    # We want: grad_s[j] = sum_k grad_ac[k] * (s[j+k] + s[j-k])
    # This is: grad_s = (grad_ac * s) + (grad_ac * s)[::-1] type operation
    # Let's use: grad_s = ifft(fft(grad_ac) * conj(fft(s))) + ifft(fft(grad_ac)[::-1] * conj(fft(s)))
    # Simpler: just do it via FFT correlation
    
    # corr_forward[k] = sum_j grad_ac[j] * s[j+k]  (for k = 0, 1, ..., N-1)
    # corr_backward[k] = sum_j grad_ac[j] * s[j-k]  (for k = 0, 1, ..., N-1)
    # grad_s[k] = corr_forward[k] + corr_backward[k]
    
    # Using FFT: circular cross-correlation of grad_ac and s
    # corr[m] = sum_j grad_ac[j] * conj(s[j-m]) for complex, but for real:
    # ifft(fft(grad_ac) * conj(fft(s))) gives sum_j grad_ac[j] * s[j+m] (circular)
    
    corr_fwd = np.fft.irfft(G * np.conj(S), size)[:N]
    
    # For backward: sum_j grad_ac[j] * s[j-k] = sum_j grad_ac[j+k] * s[j]
    # This is ifft(fft(grad_ac[::-1]) * conj(fft(s)))... 
    # Actually: grad_s[j] = sum_k grad_ac[k]*s[j+k] + sum_k grad_ac[k]*s[j-k]
    # = sum_k grad_ac[k]*s[j+k] + sum_k grad_ac[k+k-j]*s[j]... let me think differently
    # 
    # Let's just compute directly with vectorized numpy:
    # For each k, grad_ac[k] contributes:
    #   to grad_s[j] for j=0..N-1-k: grad_ac[k]*s[j+k]
    #   to grad_s[j] for j=k..N-1: grad_ac[k]*s[j-k]
    # 
    # Vectorized approach using FFT:
    # Define a = grad_ac (length N), b = s (length N)
    # We want c[j] = sum_k a[k]*b[j+k] + sum_k a[k]*b[j-k]
    # = (a * b)[j] + (a * b_rev)[j] where * is cross-correlation
    
    # Cross-correlation: (a ⋆ b)[j] = sum_k a[k]*b[j+k]
    # This equals ifft(fft(a) * conj(fft(b))) at index j (for circular, padded for linear)
    
    # Forward: sum_k a[k]*b[j+k]
    # Pad both to size, compute
    a_pad = np.zeros(size)
    a_pad[:N] = grad_ac
    b_pad = np.zeros(size)
    b_pad[:N] = s
    
    A = np.fft.rfft(a_pad, size)
    B = np.fft.rfft(b_pad, size)
    
    # Circular cross-correlation: ifft(A * conj(B))
    # This gives sum_k a[k] * b[j+k] (circularly)
    # With zero-padding to 2N-1, this gives linear cross-correlation for j=0..N-1
    cross_fwd = np.fft.irfft(A * np.conj(B), size)[:N]
    
    # Backward: sum_k a[k]*b[j-k] = sum_k a[j-k]*b[j] ... 
    # = sum_m a[m]*b[j-m] where m=k
    # This is the reverse cross-correlation
    # ifft(conj(A) * B) gives sum_k b[k]*a[j-k] = sum_m a[m]*b[j-m] (by symmetry for real)
    # Actually for real signals: cross_corr_rev[j] = sum_k a[k]*b[j-k]
    # = ifft(conj(A) * B) ... let me verify
    # A = fft(a), B = fft(b)
    # ifft(A * conj(B))[j] = sum_k a[k] b[j+k] (circular)
    # ifft(conj(A) * B)[j] = sum_k a[j-k] b[k] = sum_m a[m] b[j-m] (with m=j-k)
    
    cross_bwd = np.fft.irfft(np.conj(A) * B, size)[:N]
    
    grad_s = cross_fwd + cross_bwd
    
    # Normalize: we're minimizing max|ac|/(sum(s))^2
    # Let f = max|ac|, g = (sum(s))^2
    # d/ds [f/g] = (g*df - f*2*sum(s)*ds_sum) / g^2
    # df/ds = grad_s (from soft max)
    # ds_sum/ds = 1
    obj_val = val / (sm * sm)
    grad_obj = (sm * sm * grad_s - 2 * sm * val * np.ones(N)) / (sm * sm * sm * sm)
    
    return obj_val, grad_obj

def _epigraph_objective_and_grad(x, beta=10.0):
    """Epigraph formulation: minimize t subject to |ac[k]| <= t*(sum(s))^2 for all k.
    x = [s_0, ..., s_{N-1}, t]
    """
    s = x[:N]
    t = x[N]
    
    ac = _autoconv_fft(s)
    sm = np.sum(s)
    if abs(sm) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    # Penalty for violations: sum of max(0, |ac[k]| - t*sm^2)^2
    denom = sm * sm
    violations = np.maximum(0, np.abs(ac) - t * denom)
    penalty = np.sum(violations ** 2)
    
    # Gradient
    val, grad_ac = _soft_max_and_grad(ac, beta)
    
    # Use soft max as surrogate for max|ac|
    # Objective: t + penalty_weight * penalty
    # Let's use: minimize t + 0.5 * sum(max(0, |ac|/denom - t)^2)
    
    normed_ac = np.abs(ac) / denom
    v = np.maximum(0, normed_ac - t)
    obj = t + 0.5 * np.sum(v ** 2)
    
    # Gradient
    grad_t = 1.0 - np.sum(v)
    grad_s_ac = beta * (np.exp(beta * normed_ac - np.max(beta * normed_ac)) / 
                        np.sum(np.exp(beta * normed_ac - np.max(beta * normed_ac)))) * np.sign(ac) / denom
    
    # d/ds [normed_ac] = grad_s_ac / denom - 2*sm*|ac|*grad_s_ac / denom^2
    # This is getting complex. Let's use a simpler approach:
    # Just minimize the soft-max objective directly with L-BFGS
    
    return obj, np.zeros_like(x)  # Fallback: just return t

def _spectral_init(seed=0):
    """Generate initial signal using spectral method."""
    rng = np.random.RandomState(seed)
    # Random phases
    phases = rng.uniform(0, 2 * np.pi, N)
    # Flat magnitude spectrum
    magnitudes = np.ones(N)
    # Construct signal
    f = magnitudes * np.exp(1j * phases)
    # Make it real by symmetrizing
    f_sym = np.zeros(2 * N, dtype=complex)
    f_sym[:N] = f
    f_sym[N:] = np.conj(f[::-1])
    s = np.fft.ifft(f_sym).real[:N]
    # Normalize to ±1
    s = np.sign(s)
    s[s == 0] = 1
    return s

def _lbfgs_optimize(s_init, beta=10.0, maxiter=200):
    """Run L-BFGS on the soft-max objective."""
    def fun_and_grad(s):
        if not _check_deadline():
            return np.inf, np.zeros_like(s)
        val, grad = _objective_and_grad(s, beta)
        if not np.isfinite(val):
            return 1e10, np.zeros_like(s)
        return val, grad
    
    res = minimize(fun_and_grad, s_init, method='L-BFGS-B', 
                   jac=True, options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
    return res.x, res.fun

def _coordinate_descent(s, max_iters=10000):
    """Coordinate descent: flip individual bits to improve."""
    s = s.copy()
    current_val = _objective_only(s)
    
    for it in range(max_iters):
        if not _check_deadline():
            break
        
        improved = False
        # Try flipping each position
        for j in range(N):
            s[j] *= -1
            new_val = _objective_only(s)
            if new_val < current_val - 1e-12:
                current_val = new_val
                improved = True
            else:
                s[j] *= -1  # revert
        
        if not improved:
            break
    
    return s, current_val

def construct():
    start = time.time()
    
    best_s = None
    best_val = np.inf
    
    # Multi-restart: spectral init + L-BFGS + coordinate descent
    n_restarts = 5
    
    for r in range(n_restarts):
        if not _check_deadline():
            break
        
        # Generate initial signal
        s_init = _spectral_init(seed=r * 12345 + 67890)
        
        # Also try all-ones as one restart
        if r == 0:
            s_init = np.ones(N)
        
        # L-BFGS optimization
        try:
            s_opt, val = _lbfgs_optimize(s_init, beta=10.0, maxiter=100)
            if _check_deadline() and val < best_val:
                best_val = val
                best_s = s_opt.copy()
        except Exception:
            pass
        
        # Round to ±1
        if best_s is not None:
            s_rounded = np.sign(best_s)
            s_rounded[s_rounded == 0] = 1
            rv = _objective_only(s_rounded)
            if rv < best_val:
                best_val = rv
                best_s = s_rounded.copy()
        
        # Coordinate descent polish
        if best_s is not None and _check_deadline():
            s_cd, cd_val = _coordinate_descent(best_s, max_iters=5000)
            if cd_val < best_val:
                best_val = cd_val
                best_s = s_cd.copy()
    
    # Final polish with more coordinate descent iterations
    if best_s is not None and _check_deadline():
        s_final, final_val = _coordinate_descent(best_s, max_iters=20000)
        if final_val < best_val:
            best_val = final_val
            best_s = s_final.copy()
    
    # Ensure we return something valid
    if best_s is None:
        best_s = np.ones(N)
    
    return best_s
