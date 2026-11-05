"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_chebyt

_deadline = None

def _init_deadline():
    global _deadline
    _deadline = time.time() + 145.0

def _check_time():
    return time.time() < _deadline - 5.0

def _autoconv(h):
    """Full autoconvolution via FFT: O(N log N)."""
    n = len(h)
    size = 2 * n - 1
    f = np.fft.rfft(h, size)
    return np.fft.irfft(f * f, size)

def _compute_value(h):
    """Compute the raw value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _objective_and_grad(h):
    """Compute value and gradient analytically for full-dim optimization."""
    n = len(h)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(h)
    
    conv = _autoconv(h)
    max_val = np.max(conv)
    idx = np.argmax(conv)
    
    raw = 2.0 * n * max_val / (s * s)
    val = abs(raw)
    
    m = idx
    # d(conv_m)/d(h_i) = h[m-i] + h[i-m] for valid indices
    dh_conv = np.zeros(n)
    for i in range(n):
        j1 = m - i
        j2 = i - m
        if 0 <= j1 < n:
            dh_conv[i] += h[j1]
        if 0 <= j2 < n:
            dh_conv[i] += h[j2]
    
    # d(raw)/d(h_i) = 2*n*(dh_conv_i * s - 2*max_val) / s^3
    grad = 2.0 * n * (dh_conv * s - 2.0 * max_val) / (s ** 3)
    
    if raw < 0:
        grad = -grad
    
    return val, grad

def _soft_max(x, beta=100.0):
    """Smooth approximation of max using log-sum-exp."""
    m = np.max(x)
    shifted = (x - m) * beta
    shifted = np.clip(shifted, -50, 50)
    return m + np.log(np.sum(np.exp(shifted)) / len(x)) / beta

def _soft_max_grad(x, beta=100.0):
    """Gradient of soft-max w.r.t. x."""
    m = np.max(x)
    shifted = (x - m) * beta
    shifted = np.clip(shifted, -50, 50)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    return weights

def _objective_soft_and_grad(coeffs, basis, beta=50.0):
    """
    Objective in coefficient space using soft-max for smoothness.
    coeffs[0] is offset (dc term), coeffs[1:] are Chebyshev coefficients.
    """
    h = coeffs @ basis.T
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(coeffs)
    
    conv = _autoconv(h)
    
    # Use soft max for differentiability
    m = _soft_max(conv, beta)
    
    # Gradient of soft max: softmax weights
    shifted = (conv - m) * beta
    shifted = np.clip(shifted, -50, 50)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    
    # d/dconv = weights
    # We need d h / d coeffs = basis.T
    # d conv / d h: conv is autoconvolution.
    # Let c_k = sum_j h_j h_{k-j}.
    # dc_k / dh_i = h_{k-i} + h_{i-k} (for valid indices)
    
    # d value / d h_i = d value / d m * d m / d conv_k * d conv_k / dh_i
    # value = 2*N*m/s^2
    # d value / d m = 2*N/s^2
    
    dval_dm = 2.0 * N / (s * s)
    
    # d m / d conv_k = weights[k]
    
    # d conv_k / d h_i:
    # conv_k = sum_j h_j h_{k-j}
    # d conv_k / dh_i = h_{k-i} + h_{i-k}
    
    # grad_h_i = sum_k weights[k] * (h_{k-i} + h_{i-k}) * dval_dm
    # This is a convolution of weights with h, plus h convolved with weights.
    # Since weights are symmetric? No, weights come from conv which is symmetric.
    # conv is symmetric around N-1.
    # weights are symmetric around N-1.
    
    # Let's compute grad_h via FFT for efficiency?
    # d conv / dh is a Toeplitz-like operation.
    # Actually, let's just compute it directly if N is small enough, or use FFT.
    # N=400, conv size 799.
    
    # grad_h = dval_dm * (weights * h + weights[::-1] * h) ?
    # Let's check.
    # conv_k = sum_j h_j h_{k-j}
    # d conv_k / dh_i = h_{k-i} + h_{i-k}
    # grad_h_i = sum_k weights[k] (h_{k-i} + h_{i-k})
    # Let j = k-i => k = i+j. Sum_j weights[i+j] h_j. This is correlation of weights and h.
    # Let l = i-k => k = i-l. Sum_l weights[i-l] h_l. This is correlation of h and weights.
    
    # Since weights are symmetric (conv is symmetric, soft max preserves symmetry if input is symmetric),
    # weights[k] = weights[2(N-1)-k].
    # The two terms are identical.
    # So grad_h_i = 2 * dval_dm * sum_j weights[i+j] h_j
    
    # This is a convolution of h and weights, but shifted.
    # Let's use np.correlate or FFT.
    # weights has length 2N-1. h has length N.
    
    # Using FFT:
    # corr = ifft(fft(weights) * conj(fft(h)))
    # Need to align indices correctly.
    
    # Simpler: just use np.convolve if not too slow? N=400, 799*400 ~ 300k ops. Very fast.
    # But we are in a loop.
    
    # Let's use the property that grad_h is proportional to the "autocorrelation" of weights and h.
    # Actually, just computing the gradient numerically might be safer and fast enough for N=400?
    # No, analytic is better.
    
    # Let's compute grad_h using FFT.
    # weights is length 2N-1.
    # We want g_i = sum_k w_k h_{k-i} + w_k h_{i-k}
    
    # Let's just do it with np.convolve for simplicity and correctness, N=400 is small.
    # conv_wh = np.convolve(h, weights, mode='full')
    # The term sum_j w_{i+j} h_j is related to cross-correlation.
    
    # Let's stick to the direct summation for gradient in coeff space?
    # grad_coeff = basis.T @ grad_h
    
    # Computing grad_h:
    # g_i = sum_{k} w_k (h_{k-i} + h_{i-k})
    # Note h_j is 0 outside [0, N-1].
    
    # This is equivalent to:
    # g = 2 * (w * h) where * is correlation?
    
    # Let's use np.correlate(weights, h, mode='full')?
    # np.correlate(a, v) computes sum_k a[k+i] conj(v[k]).
    
    # Let's just compute grad_h via a loop if N is small, or FFT.
    # FFT approach:
    # Pad h to length 2N-1.
    # H = fft(h, 2N-1)
    # W = fft(weights, 2N-1)
    # Product P = W * conj(H) ?
    
    # Actually, the gradient of the sum of squares is 2*conv.
    # Here we have weighted sum.
    
    # Let's use a simpler, robust method for gradient:
    # Finite differences on h? No, we need grad w.r.t coeffs.
    
    # Let's compute grad_h analytically using the symmetry of weights.
    # w is symmetric.
    # g_i = 2 * sum_j w_{i+j} h_j
    # This is the cross-correlation of w and h, evaluated at lag i?
    # corr(w, h)[i] = sum_j w[j+i] h[j]
    # Yes.
    
    # So grad_h = 2 * dval_dm * corr(w, h)
    # We need corr(w, h) for i in 0..N-1.
    
    # np.correlate(w, h, mode='full') gives lags from -(N-1) to (2N-2).
    # We want lags 0 to N-1.
    # The 'full' output index k corresponds to lag k - (len(w)-1).
    # len(w) = 2N-1.
    # So index i corresponds to lag i - (2N-2).
    # We want lag i. So index = i + 2N - 2.
    
    # Let's verify.
    # np.correlate([1,2], [3,4], 'full')
    # lag -1: 0
    # lag 0: 1*3 + 2*4 = 11
    # lag 1: 1*4 = 4
    # Output: [0, 11, 4]
    # Indices: 0, 1, 2.
    # len(w)=2. offset = 2-1 = 1.
    # lag = index - 1.
    # index 1 -> lag 0.
    
    # So we need indices from (2N-2) to (2N-2 + N-1) = 3N-3.
    
    try:
        corr = np.correlate(weights, h, mode='full')
        start_idx = (len(weights) - 1)
        grad_h = 2.0 * dval_dm * corr[start_idx : start_idx + N]
    except Exception:
        return 1e10, np.zeros_like(coeffs)
    
    grad_coeffs = basis.T @ grad_h
    
    # Chain rule for abs?
    # value = |2*N*m/s^2|
    # If 2*N*m/s^2 > 0, grad is as above.
    # If < 0, grad is negated.
    raw_sign = 1.0 if (2.0 * N * m / (s*s)) > 0 else -1.0
    grad_coeffs *= raw_sign
    
    val = abs(2.0 * N * m / (s * s))
    
    return val, grad_coeffs

def _make_initial_conditions():
    """Generate diverse initial conditions."""
    inits = []
    x = np.linspace(-1, 1, N)
    rng = np.random.default_rng(42)
    
    inits.append(np.ones(N))
    inits.append(np.linspace(1, -1, N))
    inits.append(1.0 + 0.5 * np.cos(2 * np.pi * x))
    inits.append(np.exp(-100 * x**2))
    inits.append(np.exp(-10 * x**2))
    inits.append(np.exp(-50 * (x - 0.3)**2) + np.exp(-50 * (x + 0.3)**2))
    inits.append(1.0 + 0.1 * rng.standard_normal(N))
    inits.append(np.where(np.arange(N) % 2 == 0, 1.0, -0.5))
    inits.append(1.0 - 2.0 * np.abs(x))
    inits.append(1.0 + 0.5 * np.sin(4 * np.pi * x))
    inits.append(np.exp(-30*(x-0.4)**2) + np.exp(-30*(x)**2) + np.exp(-30*(x+0.4)**2))
    inits.append(np.sign(np.sin(3*np.pi*x)))
    inits.append(2.0 * (x - np.floor(x + 0.5)))
    inits.append(1.0 + 0.3 * np.sin(8 * np.pi * x))
    inits.append(1.0 + 0.4 * np.sin(2*np.pi*x) + 0.2*np.sin(6*np.pi*x))
    inits.append(1.0 + 0.3 * rng.standard_normal(N))
    
    from scipy.ndimage import gaussian_filter1d
    noise = rng.standard_normal(N)
    inits.append(1.0 + 0.5 * gaussian_filter1d(noise, sigma=5))
    inits.append(np.linspace(-1, 2, N))
    inits.append(np.concatenate([np.linspace(-1, 1, N//2), np.linspace(1, -1, N - N//2)]))
    inits.append(1.0 - 0.5 * x**2)
    
    return inits

def construct():
    _init_deadline()
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # Phase 1: Low-dimensional Chebyshev basis search using differential evolution
    # This reduces the search space from 400 to ~8-10 dimensions
    num_coeffs_list = [8, 12, 16]
    basis_cache = {}
    
    for num_coeffs in num_coeffs_list:
        if not _check_time():
            break
        
        if num_coeffs not in basis_cache:
            x = np.linspace(-1, 1, N)
            basis = np.zeros((N, num_coeffs))
            for k in range(num_coeffs):
                basis[:, k] = eval_chebyt(k, x)
            basis_cache[num_coeffs] = basis
        basis = basis_cache[num_coeffs]
        
        # Define bounds for differential evolution
        # c0 should be positive to keep sum(h) > 0
        bounds = [(0.5, 5.0)] + [(-3.0, 3.0)] * (num_coeffs - 1)
        
        try:
            def de_objective(coeffs):
                h = coeffs @ basis.T
                s = np.sum(h)
                if abs(s) < 1e-12:
                    return 1e10
                conv = _autoconv(h)
                return abs(2.0 * N * np.max(conv) / (s * s))
            
            result = differential_evolution(
                de_objective,
                bounds,
                maxiter=80,
                popsize=20,
                tol=1e-10,
                seed=42,
                polish=True,
                init='sobol',
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_h = (result.x @ basis.T).copy()
                
        except Exception:
            continue
    
    # Phase 2: Try each initial condition with moderate gradient-based optimization
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
                    'maxiter': 200,
                    'ftol': 1e-13,
                    'gtol': 1e-11,
                    'maxfun': 600,
                    'maxls': 30,
                }
            )
            
            val = result.fun
            if val < best_val:
                best_val = val
                best_h = result.x.copy()
        except Exception:
            continue
    
    # Phase 3: Polish the best result with high precision
    if best_val < 2.0 and _check_time():
        try:
            result = minimize(
                _objective_and_grad,
                best_h,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 500,
                    'ftol': 1e-16,
                    'gtol': 1e-14,
                    'maxfun': 1500,
                    'maxls': 50,
                }
            )
            val = result.fun
            if val < best_val:
                best_val = val
                best_h = result.x.copy()
        except Exception:
            pass
        
        # Additional polish with even tighter tolerance
        if best_val < 2.0 and _check_time():
            try:
                result = minimize(
                    _objective_and_grad,
                    best_h,
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 200,
                        'ftol': 1e-18,
                        'gtol': 1e-16,
                        'maxfun': 600,
                        'maxls': 40,
                    }
                )
                val = result.fun
                if val < best_val:
                    best_val = val
                    best_h = result.x.copy()
            except Exception:
                pass
    
    # Phase 4: If time remains, try random perturbations around best
    if best_val < 2.0 and _check_time():
        rng = np.random.default_rng(123)
        for _ in range(8):
            if not _check_time():
                break
            perturbation = 0.03 * rng.standard_normal(N)
            h_try = best_h + perturbation
            try:
                result = minimize(
                    _objective_and_grad,
                    h_try,
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 200,
                        'ftol': 1e-15,
                        'gtol': 1e-13,
                        'maxfun': 800,
                        'maxls': 40,
                    }
                )
                val = result.fun
                if val < best_val:
                    best_val = val
                    best_h = result.x.copy()
            except Exception:
                continue
    
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
