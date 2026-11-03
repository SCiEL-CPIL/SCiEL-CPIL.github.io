"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Strategy: Spectral parameterization.
We represent h via its discrete Fourier coefficients.
h = IDFT(c).
Constraints:
1. sum(h) = N/2  => c[0] = 0.5 (fixed)
2. h in [0, 1]   => enforced via projection in real space after each update.
Objective:
Minimize max_k C_k, where C_k is derived from autocorrelation.
Autocorrelation in frequency domain is |C|^2.
We optimize the complex coefficients c[1..N/2] (and conjugates) to minimize
the smooth maximum of the overlap values.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill

def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h

def _spectral_to_h(c):
    """
    Convert spectral coefficients to real space h.
    c is a vector of length N//2 + 1 (for rfft).
    c[0] is the DC term (fixed at 0.5 for sum=N/2).
    The optimizer varies c[1:].
    """
    # c is the full rfft spectrum
    # We need to ensure c[0] is exactly 0.5
    c_full = c.copy()
    c_full[0] = 0.5
    # Inverse FFT
    h = np.fft.irfft(c_full, n=N)
    # Project to [0, 1] and fix sum
    h = _fix_sum(h)
    return h

def _spectral_objective(c, beta):
    """
    Objective function in spectral domain.
    c: array of length N//2 + 1
    Returns: (value, gradient)
    """
    h = _spectral_to_h(c)
    
    # Compute C_k
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for soft max
    
    val = m + np.log(sum_exp) / beta
    
    # Gradient with respect to h
    # dC_k/dh_j = -(2/N) * ( h_{j-k} (if j>=k) + h_{j+k} (if j+k<N) )
    grad_h = np.zeros(N)
    for k in range(N):
        if w[k] == 0.0:
            continue
        factor = -(2.0 / N) * w[k]
        if k > 0:
            grad_h[k:] += factor * h[:N-k]
        if k < N:
            grad_h[:N-k] += factor * h[k:]
            
    # Chain rule: grad_c = DFT(grad_h)
    # Since h = irfft(c), the gradient of the objective w.r.t c is fft(grad_h)
    # Note: scipy irfft uses normalization 1/N for forward, 1 for inverse?
    # np.fft.irfft(x, n) = ifft(x) * 2 (for real part)
    # The gradient of real-valued function f(h) w.r.t complex c is simply fft(grad_h)
    # adjusted for the real FFT structure.
    grad_c = np.fft.rfft(grad_h, n=N)
    
    # We only optimize c[1:]
    # c[0] is fixed, so grad_c[0] is ignored.
    # Also, due to symmetry, the imaginary parts of the spectrum correspond to sine terms.
    # The optimizer treats c as a real vector of length N//2+1.
    # Wait, rfft returns complex numbers.
    # We need to map the complex gradient to the real parameter vector used by SLSQP.
    # SLSQP operates on real vectors.
    # Let's represent the spectral coefficients as a real vector:
    # [c0, c1_real, c1_imag, c2_real, c2_imag, ...]
    # This requires a custom mapping.
    
    # Alternative: Use a real-valued spectral representation.
    # h_i = 0.5 + sum_{k=1}^{N/2} ( a_k cos(2pi k i / N) + b_k sin(2pi k i / N) )
    # This is equivalent to Fourier series.
    # Let's switch to this representation to avoid complex number handling in SLSQP.
    
    # We will re-implement the objective to work with coefficients [a_1..a_M, b_1..b_M]
    # where M = N//2 (if N even) or (N-1)//2.
    # For N=100, M=50.
    # h = 0.5 + sum_{k=1}^{50} ( a_k cos(2pi k i / 100) + b_k sin(2pi k i / 100) )
    
    # This function _spectral_objective is currently receiving c which is rfft output.
    # We need to change the interface to accept the real coefficient vector.
    
    raise NotImplementedError("Refactoring to real coefficients")

def _real_spectral_to_h(coeffs):
    """
    coeffs: [a_1..a_M, b_1..b_M]
    Returns h.
    """
    i = np.arange(N)
    h = np.full(N, 0.5)
    M = N // 2
    # a is coeffs[:M], b is coeffs[M:]
    a = coeffs[:M]
    b = coeffs[M:]
    
    # Vectorized construction
    # h_i = 0.5 + sum_{k=1}^M ( a_k cos(2pi k i / N) + b_k sin(2pi k i / N) )
    # This is essentially an inverse DFT (real part) of the coefficients.
    # Let's build the complex spectrum and use irfft.
    # Spectrum C_k:
    # C_0 = 0.5
    # C_k = (a_k - i b_k) / 2  for k=1..M-1 ?
    # Standard DFT: h_n = 1/N sum_{k=0}^{N-1} C_k e^{i 2pi k n / N}
    # For real h: C_{N-k} = conj(C_k).
    # h_n = C_0 + sum_{k=1}^{N/2-1} ( C_k e^{i 2pi k n / N} + C_{N-k} e^{i 2pi (N-k) n / N} ) + C_{N/2} e^{i pi n}
    # C_{N-k} = conj(C_k) = a_k/2 + i b_k/2
    # C_k e^{i...} + conj(C_k) e^{-i...} = 2 Re(C_k e^{i...}) = 2 Re( (a_k/2 - i b_k/2) (cos + i sin) )
    # = 2 ( a_k/2 cos + b_k/2 sin ) = a_k cos + b_k sin.
    # So yes, if we define C_k = (a_k - i b_k)/2, then the term is a_k cos + b_k sin.
    # And for k=N/2, term is a_{N/2} cos(pi n) = a_{N/2} (-1)^n.
    # In this case b_{N/2} is 0 (sin(pi n)=0).
    
    C = np.zeros(N//2 + 1, dtype=np.complex128)
    C[0] = 0.5
    for k in range(1, M + 1):
        C[k] = (a[k-1] - 1j * b[k-1]) / 2.0
        
    h = np.fft.irfft(C, n=N)
    h = _fix_sum(h)
    return h

def _real_spectral_objective(coeffs, beta):
    """
    Objective and gradient for real spectral coefficients.
    """
    h = _real_spectral_to_h(coeffs)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    
    val = m + np.log(sum_exp) / beta
    
    # Gradient w.r.t h
    grad_h = np.zeros(N)
    for k in range(N):
        if w[k] == 0.0:
            continue
        factor = -(2.0 / N) * w[k]
        if k > 0:
            grad_h[k:] += factor * h[:N-k]
        if k < N:
            grad_h[:N-k] += factor * h[k:]
            
    # Gradient w.r.t coeffs
    # h = 0.5 + sum ( a_k cos(...) + b_k sin(...) )
    # dh/d a_k = cos(2pi k i / N)
    # dh/d b_k = sin(2pi k i / N)
    # grad_coeff = grad_h . d h / d coeff
    # grad_a_k = sum_i grad_h[i] cos(2pi k i / N)
    # grad_b_k = sum_i grad_h[i] sin(2pi k i / N)
    
    # This is essentially the DFT of grad_h.
    # DFT_k(grad_h) = sum_i grad_h[i] e^{-i 2pi k i / N}
    # Real part = sum grad_h cos
    # Imag part = - sum grad_h sin
    # So grad_a = Re(DFT(grad_h)), grad_b = -Im(DFT(grad_h))
    
    G = np.fft.rfft(grad_h, n=N)
    M = N // 2
    grad_a = np.real(G[1:M+1])
    grad_b = -np.imag(G[1:M+1])
    
    # Scale factor?
    # h = irfft(C). C_k = (a_k - i b_k)/2.
    # The relationship is linear.
    # Let's verify with a small example or just trust the chain rule via finite diff if needed, 
    # but analytical is better.
    # d h_n / d a_k = cos(2pi k n / N).
    # d h_n / d b_k = sin(2pi k n / N).
    # grad_a_k = sum_n grad_h_n cos(2pi k n / N).
    # This matches Re(DFT(grad_h)_k).
    
    grad_coeffs = np.concatenate([grad_a, grad_b])
    
    return val, grad_coeffs

def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    M = N // 2
    # Initial coefficients: All zero (h=0.5)
    # Or start from a cosine wave to break symmetry?
    # h_i = 0.5 + 0.5 cos(2pi i / N)
    # This corresponds to a_1 = 0.5, others 0.
    coeffs0 = np.zeros(2 * M)
    coeffs0[0] = 0.5 # a_1
    
    h0 = _real_spectral_to_h(coeffs0)
    _update_best(h0)
    
    # Try a few different starts
    starts = [
        coeffs0,
        np.random.normal(0, 0.1, 2*M),
        np.zeros(2*M)
    ]
    
    def _run_opt(coeffs_init, beta, maxiter, ftol):
        nonlocal best_h, best_score
        
        # No bounds on coeffs directly, but h is projected.
        # However, large coeffs lead to large h values which get clipped, causing non-smoothness.
        # We can add loose bounds on coeffs to keep h in a reasonable range.
        # Max contribution of one term is |a_k|. Sum of M terms.
        # If sum |a_k| < 0.5, h stays in [0,1] roughly.
        # Let's use bounds [-0.5, 0.5] for each coeff.
        bounds = [(-0.5, 0.5)] * (2 * M)
        
        def obj_func(x):
            val, _ = _real_spectral_objective(x, beta)
            return val
            
        def jac_func(x):
            _, grad = _real_spectral_objective(x, beta)
            return grad
            
        try:
            res = minimize(obj_func, coeffs_init.copy(), jac=jac_func, method='SLSQP',
                           bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': ftol})
            if res.success or res.fun < best_score:
                h_res = _real_spectral_to_h(res.x)
                _update_best(h_res)
        except Exception:
            pass

    # Phase 1: Optimize from cosine start
    if time.time() - t0 < _DEADLINE - 30:
        _run_opt(coeffs0, beta=50.0, maxiter=200, ftol=1e-9)
        
    # Phase 2: Random starts
    if time.time() - t0 < _DEADLINE - 20:
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 10:
                break
            c_rand = np.random.normal(0, 0.1, 2*M)
            _run_opt(c_rand, beta=50.0, maxiter=100, ftol=1e-8)
            
    # Phase 3: Polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        # Recover coefficients from best_h?
        # Or just re-optimize from the best_h's spectral representation.
        C_best = np.fft.rfft(best_h, n=N)
        # Convert C_best to coeffs
        M = N // 2
        a = 2 * np.real(C_best[1:M+1])
        b = -2 * np.imag(C_best[1:M+1])
        coeffs_best = np.concatenate([a, b])
        _run_opt(coeffs_best, beta=100.0, maxiter=300, ftol=1e-12)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _fix_sum(best_h)
    return best_h
