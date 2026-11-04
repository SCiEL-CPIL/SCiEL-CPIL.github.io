"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_deadline = None
_best_h = None
_best_val = None

def _init_deadline():
    global _deadline
    _deadline = time.time() + 145.0

def _check_time():
    return time.time() < _deadline - 2.0

def _autoconv(h):
    """Full autoconvolution via FFT: O(N log N)."""
    n = len(h)
    size = 2 * n - 1
    f = np.fft.rfft(h, size)
    return np.fft.irfft(f * f, size)

def _objective_and_grad(h):
    """Compute value and gradient analytically."""
    n = len(h)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(h)
    
    conv = _autoconv(h)
    max_val = np.max(conv)
    idx = np.argmax(conv)
    
    # value = |2 * N * max(conv) / s^2|
    raw = 2.0 * n * max_val / (s * s)
    val = abs(raw)
    
    # Gradient of f = 2*N*conv[idx] / s^2 with respect to h
    # d(conv[idx])/dh[j] = sum over k: h[k] where idx = j + k (with offset for full conv)
    # Full conv: conv[m] = sum_{j+k=m} h[j]*h[k] for m=0..2n-2
    # So d(conv[idx])/dh[j] = h[idx-j] if 0<=idx-j<n else 0 (for the product h[j]*h[idx-j])
    # But also h[idx-j]*h[j] contributes, so it's 2*h[idx-j] when j != idx-j, and h[j] when j==idx-j
    # More carefully: conv[idx] = sum_j h[j]*h[idx-j] (where both indices valid)
    # d/dh[j] = h[idx-j] + (if j != idx-j) h[idx-j] ... wait
    # conv[idx] = sum_{j=0}^{n-1} h[j] * h[idx-j] (with h[k]=0 for k<0 or k>=n)
    # d(conv[idx])/dh[j] = h[idx-j] + sum_{j'} [j'=j term from other side]
    # Actually: conv[idx] = sum_j h[j] * h[idx-j]
    # d/dh[j0] = h[idx-j0] + (contribution from j' such that idx-j' = j0, i.e., j' = idx-j0)
    # If idx-j0 != j0: d/dh[j0] = h[idx-j0] + h[idx-j0] = 2*h[idx-j0]... no wait
    # Let me be more careful. conv[idx] = sum_j h[j] * h[idx-j]
    # d(conv[idx])/dh[j0] = sum_j [delta(j,j0)*h[idx-j] + h[j]*delta(idx-j, j0)]
    # = h[idx-j0] + h[idx-j0] if j0 != idx-j0, or h[j0] if j0 == idx-j0
    # Wait: second term: sum_j h[j]*delta(idx-j, j0) = h[idx-j0] (if idx-j0 is valid index)
    # First term: sum_j delta(j,j0)*h[idx-j] = h[idx-j0] (if idx-j0 valid)
    # So d(conv[idx])/dh[j0] = h[idx-j0] + h[idx-j0] = 2*h[idx-j0] when j0 != idx-j0
    # And when j0 == idx-j0 (i.e., 2*j0 == idx): d = h[j0] + h[j0] = 2*h[j0]... 
    # Actually both terms give h[idx-j0] = h[j0], so it's 2*h[j0] still? No:
    # When j0 = idx - j0, the two terms are the same term h[j0]*h[j0], derivative is 2*h[j0].
    # So in all cases: d(conv[idx])/dh[j0] = 2*h[idx-j0] if 0 <= idx-j0 < n, else 0? 
    # Hmm, let me just compute it numerically for the gradient to be safe, but use FFT for the conv value.
    
    # Actually, let's use the formula: d(conv[m])/dh[j] = h[m-j] + (1 if m-2j==0 else 0)*0 ... 
    # Simplest: d(conv[m])/dh[j] = h[m-j] if 0<=m-j<n, plus we need to account for symmetry.
    # For full linear convolution: (h*h)[m] = sum_k h[k]*h[m-k]
    # d/dh[j] (h*h)[m] = h[m-j] + h[m-j] = 2*h[m-j] when m-j != j
    # = h[m-j] when m-j == j (i.e., m = 2j)
    # Wait no: d/dh[j] sum_k h[k]h[m-k] = h[m-j] (from k=j term) + h[m-j] (from m-k=j term, i.e., k=m-j)
    # If j != m-j: two distinct terms, each contributes h[m-j], total 2*h[m-j]
    # If j == m-j (m=2j): only one term h[j]*h[j], derivative is 2*h[j]
    # So in all cases: d/dh[j] (h*h)[m] = 2*h[m-j] if 0<=m-j<n, else 0
    # But when m-j == j, it's still 2*h[j], which is consistent.
    
    m = idx
    dh_conv = np.zeros(n)
    valid_j = np.arange(n)
    km = m - valid_j  # m - j for each j
    mask = (km >= 0) & (km < n)
    dh_conv[mask] = 2.0 * h[km[mask]]
    
    # Now f = 2*N*conv[m] / s^2
    # df/dh[j] = 2*N * (dh_conv[j] * s^2 - conv[m] * 2*s * 1) / s^4
    # = 2*N * (dh_conv[j] * s - 2*conv[m]) / s^3
    
    grad = 2.0 * n * (dh_conv * s - 2.0 * max_val) / (s ** 3)
    
    # If raw is negative, val = -raw, so gradient sign flips
    if raw < 0:
        grad = -grad
    
    return val, grad

def _make_initial_conditions():
    """Generate diverse initial conditions."""
    inits = []
    
    # 1. All ones (baseline)
    inits.append(np.ones(N))
    
    # 2. Ramp
    inits.append(np.linspace(1, -1, N))
    
    # 3. Cosine bump centered
    x = np.linspace(-1, 1, N)
    inits.append(1.0 + 0.5 * np.cos(2 * np.pi * x))
    
    # 4. Sharp bump
    inits.append(np.exp(-100 * x**2))
    
    # 5. Wide bump
    inits.append(np.exp(-10 * x**2))
    
    # 6. Two bumps
    inits.append(np.exp(-50 * (x - 0.3)**2) + np.exp(-50 * (x + 0.3)**2))
    
    # 7. Random with small noise
    rng = np.random.default_rng(42)
    inits.append(1.0 + 0.1 * rng.standard_normal(N))
    
    # 8. Alternating
    inits.append(np.where(np.arange(N) % 2 == 0, 1.0, -0.5))
    
    # 9. Triangle wave
    inits.append(1.0 - 2.0 * np.abs(x))
    
    # 10. Sine
    inits.append(1.0 + 0.5 * np.sin(4 * np.pi * x))
    
    return inits

def construct():
    global _best_h, _best_val
    _init_deadline()
    
    best_h = np.ones(N)
    best_val = 2.0
    
    inits = _make_initial_conditions()
    
    for init_h in inits:
        if not _check_time():
            break
        
        h0 = init_h.astype(np.float64)
        
        try:
            result = minimize(
                _objective_and_grad,
                h0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxfun': 2000,
                    'maxls': 50,
                }
            )
            
            if result.success or result.fun < best_val:
                val = result.fun
                if val < best_val:
                    best_val = val
                    best_h = result.x.copy()
        except Exception:
            continue
        
        if not _check_time():
            break
    
    # Verify result
    s = np.sum(best_h)
    if abs(s) < 1e-12:
        return np.ones(N)
    
    conv = _autoconv(best_h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    
    if val >= 2.0:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
