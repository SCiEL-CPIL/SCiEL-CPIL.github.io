"""
Optimized program for the third autocorrelation inequality.
Combines Chebyshev polynomial initialization with direct time-domain 
soft-max optimization.
"""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft
from scipy.optimize import minimize

def _eval_value(h):
    """Compute the objective value using FFT-based convolution."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _convolve_fft(h):
    """Compute full autoconvolution using FFT."""
    n_conv = 2 * len(h) - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _soft_max(x, beta):
    """Numerically stable soft maximum."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    exp_vals = np.exp(beta * (x - m))
    sm = m + np.log(np.sum(exp_vals)) / beta
    return sm

def _soft_max_and_grad(x, beta):
    """Soft maximum and its gradient w.r.t. x."""
    m = np.max(x)
    if not np.isfinite(m):
        return m, np.zeros_like(x)
    exp_vals = np.exp(beta * (x - m))
    sum_exp = np.sum(exp_vals)
    sm = m + np.log(sum_exp) / beta
    grad = beta * exp_vals / sum_exp
    return sm, grad

def _objective_soft_direct(h, beta):
    """
    Soft-max surrogate objective in time domain.
    """
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    h_norm = h * (N / s)
    conv = _convolve_fft(h_norm)
    sm = _soft_max(conv, beta)
    return abs(2.0 * N * sm / (N * N))

def _objective_soft_direct_grad(h, beta):
    """
    Soft-max surrogate objective and its gradient.
    Returns (value, gradient) where gradient is w.r.t. h.
    """
    if not np.all(np.isfinite(h)):
        return 1e10, np.zeros_like(h)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(h)
    
    # Scale factor for normalization
    scale = N / s
    h_norm = h * scale
    
    conv = _convolve_fft(h_norm)
    sm, sm_grad = _soft_max_and_grad(conv, beta)
    
    # Objective: V = 2 * N * sm / (N*N) = 2*sm/N
    # dV/d(conv_k) = 2/N * sm_grad[k]
    
    # d(conv_k)/d(h_i) = 2 * h_norm[k-i] if 0<=k-i<N else 0
    # So dV/d(h_i) = sum_k (2/N * sm_grad[k]) * 2 * h_norm[k-i] for 0<=k-i<N
    # = (4/N) * sum_k sm_grad[k] * h_norm[k-i]
    # This is a cross-correlation of sm_grad with h_norm
    
    # Compute: grad_h[i] = (4/N) * sum_k sm_grad[k] * h_norm[k-i] for 0<=k-i<N
    # Let j = k-i, so k = i+j, and 0<=j<N
    # grad_h[i] = (4/N) * sum_{j=0}^{N-1} sm_grad[i+j] * h_norm[j]
    # But sm_grad has length 2N-1, and i+j can be up to (N-1)+(N-1)=2N-2, which is fine.
    
    # This is a convolution/correlation. Let's compute it directly using FFT or direct sum.
    # For efficiency, use FFT-based correlation.
    # grad_h[i] = (4/N) * sum_{j=0}^{N-1} sm_grad[i+j] * h_norm[j]
    # This is like a cross-correlation. Let's use the fact that:
    # sum_j a[i+j] * b[j] = (a * b)[i] where * is cross-correlation
    
    # Actually, let's just compute it via matrix-free approach using FFT.
    # Define a vector g of length 2N-1 as sm_grad.
    # We want for each i in [0,N-1]: sum_{j=0}^{N-1} g[i+j] * h_norm[j]
    # This is a sliding dot product, which can be computed via convolution.
    # Let's flip h_norm: h_flip[j] = h_norm[N-1-j]
    # Then sum_j g[i+j] * h_norm[j] = sum_j g[i+j] * h_flip[N-1-j]
    # = (g * h_flip)[i + N - 1] where * is convolution
    
    h_flip = h_norm[::-1]
    # convolve g (length 2N-1) with h_flip (length N)
    n_corr = len(sm_grad) + len(h_flip) - 1
    n_fft_corr = 1
    while n_fft_corr < n_corr:
        n_fft_corr *= 2
    G = fft(sm_grad, n=n_fft_corr)
    HF = fft(h_flip, n=n_fft_corr)
    corr = ifft(G * HF, n=n_fft_corr).real
    
    # corr[m] = sum_k sm_grad[k] * h_flip[m-k] = sum_k sm_grad[k] * h_norm[N-1-(m-k)]
    # We want sum_j sm_grad[i+j] * h_norm[j]
    # Let m = i + N - 1, then corr[m] = sum_k sm_grad[k] * h_norm[N-1-m+k]
    # = sum_k sm_grad[k] * h_norm[N-1-i-N+1+k] = sum_k sm_grad[k] * h_norm[k-i]
    # Hmm, that's not quite right. Let me reconsider.
    
    # Actually, let's just do it more carefully with a direct approach for clarity,
    # or use a different formulation.
    
    # Alternative: dV/dh[i] = (4/N) * sum_{k=0}^{2N-2} sm_grad[k] * (2*h_norm[k-i] if 0<=k-i<N else 0) / 2
    # Wait, I already accounted for the factor of 2 in d(conv_k)/d(h_i).
    # dV/dh[i] = sum_k (dV/dconv[k]) * (dconv[k]/dh[i])
    # = sum_k (2/N * sm_grad[k]) * (2 * h_norm[k-i] if 0<=k-i<N else 0)
    # = (4/N) * sum_{k: 0<=k-i<N} sm_grad[k] * h_norm[k-i]
    
    # Let j = k-i, so k = i+j, j in [0, N-1]
    # = (4/N) * sum_{j=0}^{N-1} sm_grad[i+j] * h_norm[j]
    
    # This is a valid computation. For N=400, this is 400*400=160K operations per gradient,
    # which is fine. But let's use FFT for efficiency.
    
    # We can compute this as a cross-correlation.
    # Define A[i+j] for i in [0,N-1], j in [0,N-1].
    # result[i] = sum_{j=0}^{N-1} A[i+j] * h_norm[j]
    
    # This is equivalent to: for each i, take sm_grad[i:i+N] and dot with h_norm.
    # We can compute all of these at once using a Toeplitz-like structure, but FFT is cleaner.
    
    # Use the relationship: result = ifft(fft(sm_grad, n) * conj(fft(h_norm_padded, n)))
    # where h_norm_padded is h_norm placed at the end of a zero array.
    
    # Actually, the simplest correct approach: 
    # result[i] = <sm_grad[i:i+N], h_norm>
    # This is a cross-correlation of sm_grad with h_norm, evaluated at lags 0 to N-1.
    
    # Cross-correlation: (sm_grad ⋆ h_norm)[i] = sum_j sm_grad[i+j] * conj(h_norm[j])
    # = sum_j sm_grad[i+j] * h_norm[j] (since real)
    
    # We can compute this via: ifft(fft(sm_grad) * conj(fft(h_norm)))
    # But we need to be careful with lengths.
    
    n_total = 2 * N - 1  # length of sm_grad
    # Pad h_norm to length n_total
    h_pad = np.zeros(n_total)
    h_pad[:N] = h_norm
    
    # Cross-correlation: result[i] = sum_j sm_grad[i+j] * h_pad[j] for j where i+j < n_total
    # Using FFT: result = ifft(fft(sm_grad) * conj(fft(h_pad)))
    # But this gives circular cross-correlation. We need linear.
    
    # For linear cross-correlation of lengths n_total and N:
    # result has length n_total + N - 1
    # result[i] = sum_j sm_grad[i-j] * h_pad[j] ... this is getting complicated.
    
    # Let's just do the direct computation since N=400 is small enough.
    # 400 * 400 = 160,000 multiply-adds, which is fast in NumPy.
    
    # Create a matrix where row i is sm_grad[i:i+N]
    # Then result = matrix @ h_norm
    # Using stride tricks or just a loop-free approach:
    
    # Actually, let's use scipy.signal.correlate or just do it manually.
    # For N=400, a simple approach:
    idx = np.arange(N)
    # sm_grad[i + j] for i in [0,N-1], j in [0,N-1]
    # This is a matrix M[i,j] = sm_grad[i+j]
    # result[i] = sum_j M[i,j] * h_norm[j]
    
    # We can build M using broadcasting:
    # M = sm_grad[np.arange(N)[:, None] + np.arange(N)[None, :]]
    # result = M @ h_norm
    
    M = sm_grad[np.arange(N)[:, None] + np.arange(N)[None, :]]
    result = M @ h_norm
    
    grad_h = (4.0 / N) * result * scale  # Chain rule: h_norm = h * scale, so dh_norm/dh = scale
    
    # Also need to account for the fact that V uses abs(). 
    # If sm >= 0, dV/dh = grad_h. If sm < 0, dV/dh = -grad_h.
    # But typically sm > 0 for reasonable inputs.
    if sm < 0:
        grad_h = -grad_h
    
    V = abs(2.0 * sm / N)
    
    return V, grad_h

def _chebyshev_basis(k, x):
    """
    Compute Chebyshev polynomials of the second kind U_0..U_{k-1} at points x.
    U_0(x) = 1
    U_1(x) = 2x
    U_{n+1}(x) = 2x U_n(x) - U_{n-1}(x)
    """
    U = np.zeros((k, len(x)))
    U[0, :] = 1.0
    if k > 1:
        U[1, :] = 2.0 * x
    for i in range(2, k):
        U[i, :] = 2.0 * x * U[i-1, :] - U[i-2, :]
    return U

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses Chebyshev polynomial initialization followed by direct 
    time-domain optimization with a soft-max surrogate.
    """
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    # Start with all-ones (value 2.0)
    best_h = np.ones(N)
    best_val = 2.0

    # Phase 1: Chebyshev Polynomial Initialization
    # We parameterize h as a linear combination of the first M Chebyshev polynomials.
    # h[i] = sum_{k=0}^{M-1} c_k U_k(x_i)
    # We want to minimize the autocorrelation peak.
    # The grid x_i is on [-1, 1].
    x_grid = np.linspace(-1.0, 1.0, N)
    
    # Try a few different numbers of basis functions and random coefficients
    rng = np.random.RandomState(42)
    
    for M in [5, 10, 20]:
        if time.time() > deadline - 100:
            break
        U = _chebyshev_basis(M, x_grid)
        
        # Try random coefficient vectors
        for trial in range(50):
            if time.time() > deadline - 100:
                break
            coeffs = rng.randn(M) * 0.5
            h = U.T @ coeffs
            # Ensure sum is not zero and finite
            s = np.sum(h)
            if abs(s) < 1e-6 or not np.all(np.isfinite(h)):
                continue
            # Normalize to sum N for consistent scale during evaluation
            h_norm = h * (N / s)
            v = _eval_value(h_norm)
            if v < best_val:
                best_val = v
                best_h = h_norm.copy()

    # Also try the simple cosine family as a fallback if Chebyshev didn't improve much
    if best_val > 1.5:
        for m in range(1, 21):
            if time.time() > deadline - 90:
                break
            for a in np.linspace(-0.8, 0.8, 17):
                h = 1.0 + a * np.cos(2.0 * np.pi * m * np.arange(N) / N)
                v = _eval_value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()

    # Phase 2: Direct Time-Domain Optimization with Soft-Max (with analytical gradients)
    # We optimize the raw vector h directly.
    # We use a multi-stage approach with increasing beta for the soft-max.
    
    # Initial point
    h_current = best_h.copy()
    
    # Beta schedule: start low for smooth landscape, increase for precision
    betas = [5.0, 20.0, 100.0, 500.0, 2000.0]
    
    for beta in betas:
        if time.time() > deadline - 20:
            break
            
        def obj_func(h_vec, beta=beta):
            return _objective_soft_direct(h_vec, beta)
        
        def obj_grad_func(h_vec, beta=beta):
            val, grad = _objective_soft_direct_grad(h_vec, beta)
            return val, grad
        
        # We perform a few random restarts around the current best
        # but for high beta, we stick closer to the previous best to avoid getting stuck in local minima of the sharper landscape.
        restarts = 5 if beta < 50 else 2
        
        for r in range(restarts):
            if time.time() > deadline - 10:
                break
                
            if r == 0:
                h_start = h_current.copy()
            else:
                # Add small noise scaled by the previous beta's "temperature"
                noise_scale = 0.1 / (1 + r)
                h_start = h_current + noise_scale * rng.randn(N)
            
            try:
                result = minimize(
                    obj_grad_func,
                    h_start,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                h_opt = result.x
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    # Evaluate exact value
                    v_opt = _eval_value(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        h_current = h_opt.copy()
            except Exception:
                continue

    # Phase 3: Final Polish with very high beta (with gradient)
    if time.time() < deadline - 5:
        beta_final = 5000.0
        
        def obj_final(h_vec):
            return _objective_soft_direct(h_vec, beta_final)
        
        def obj_final_grad(h_vec):
            val, grad = _objective_soft_direct_grad(h_vec, beta_final)
            return val, grad
        
        try:
            result = minimize(
                obj_final_grad,
                best_h,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            h_opt = result.x
            if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                v_opt = _eval_value(h_opt)
                if v_opt < best_val:
                    best_val = v_opt
                    best_h = h_opt.copy()
        except Exception:
            pass

    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h
# EVOLVE-BLOCK-END
