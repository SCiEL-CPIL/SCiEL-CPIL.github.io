"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fftpack import dct, idct
from scipy.special import eval_gegenbauer

def _objective_hard(h):
    """Compute the hard objective value."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_obj(h, beta):
    """
    Soft-max approximation of the objective.
    Minimizes 2*N*soft_max(conv) / (sum(h))^2
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    # Numerically stable log-sum-exp
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum_exp = m + np.log(np.sum(np.exp(shifted))) / beta
    val = 2.0 * N * log_sum_exp / (s * s)
    return val

def _soft_max_grad(h, beta):
    """
    Gradient of the soft-max objective: d/dh [2*N*soft_max(conv)/(sum(h))^2]
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros(N)
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    w = exp_shifted / sum_exp  # weights for each conv entry
    
    # d(soft_max)/dh[i] = sum_j w_j * d(conv[j])/dh[i]
    # conv[j] = sum_{i=0}^{N-1} h[i] * h[i+j-N+1] (with valid indices)
    # d(conv[j])/dh[i] = h[i+j-N+1] (if 0<=i+j-N+1<N) + h[i-j+N-1] (if 0<=i-j+N-1<N)
    # Let k = j - N + 1, ranges from -(N-1) to N-1
    # For each i, sum over k: w[k+N-1] * (h[i+k] + h[i-k]) with valid bounds
    
    # Efficient computation using convolution
    # grad_softmax[i] = sum_k w[k+N-1] * (h[i+k] + h[i-k])
    # This is: convolve(w, h)[N-1+i] + convolve(w, h)[N-1-i] (approximately)
    # More precisely, use direct computation with FFT for speed
    
    # Build the kernel: for lag k, weight is w[k+N-1]
    # The gradient at position i is: sum_{k} w[k+N-1] * (h[i+k] + h[i-k])
    # = (w * h)[i + N-1] + (w * h)[N-1 - i] where * denotes cross-correlation
    
    # Use FFT for efficient computation
    n_fft = 2 * N
    w_full = np.zeros(n_fft)
    # w has length 2N-1, indexed 0..2N-2
    # For lag k from -(N-1) to N-1, weight is w[k+N-1]
    # In the convolution, we need to place weights appropriately
    
    # Actually, let's compute directly using the structure
    # grad[i] = sum_{k=-(N-1)}^{N-1} w[k+N-1] * (h[i+k] if 0<=i+k<N else 0)
    #           + sum_{k=-(N-1)}^{N-1} w[k+N-1] * (h[i-k] if 0<=i-k<N else 0)
    
    # The second sum is the same as the first with k -> -k, and w is symmetric (since conv is symmetric)
    # So grad[i] = 2 * sum_{k=-(N-1)}^{N-1} w[k+N-1] * h[i+k] (with valid bounds)
    # But w is NOT necessarily symmetric in general... wait, conv IS symmetric, so w IS symmetric.
    # w[j] = exp(beta*(conv[j]-m)) / sum_exp, and conv[j] = conv[2N-2-j], so w[j] = w[2N-2-j]
    
    # So we can compute: grad[i] = 2 * sum_{k} w[k+N-1] * h[i+k] with valid bounds
    # This is a convolution of h with w (centered)
    
    # Let's use np.convolve for this
    # w_centered: shift w so that index N-1 is at position 0
    # w has length 2N-1. We want to convolve h with w such that:
    # result[i] = sum_j h[j] * w[i-j+N-1] (if valid)
    
    # Actually, let me just compute it directly with a loop-free approach
    # Create a matrix or use broadcasting
    
    # Simpler: use the fact that this is a Toeplitz-like system
    # Let's build the weight vector for positive lags and negative lags
    # w_pos[k] = w[k+N-1] for k = 0, 1, ..., N-1 (lag +k)
    # w_neg[k] = w[N-1-k] for k = 1, 2, ..., N-1 (lag -k) = w_pos[N-1-k] (by symmetry)
    
    # grad[i] = sum_{k=0}^{N-1} w[N-1+k] * h[i+k] (if i+k < N)
    #           + sum_{k=1}^{N-1} w[N-1-k] * h[i-k] (if i-k >= 0)
    #           + w[N-1] * h[i]  (the k=0 term, counted once in each sum but we need to be careful)
    
    # Actually for k=0: h[i+0] and h[i-0] = h[i], so it's 2*w[N-1]*h[i]
    # For k>0: w[N-1+k]*h[i+k] + w[N-1-k]*h[i-k]
    # By symmetry w[N-1+k] = w[N-1-k] for k>0? No! w[j] = w[2N-2-j]
    # w[N-1+k] and w[N-1-k]: 2N-2-(N-1+k) = N-1-k, so w[N-1+k] = w[N-1-k]. Yes!
    
    # So grad[i] = 2 * sum_{k=0}^{N-1} w[N-1+k] * h[i+k] (with valid bounds, but k=0 gives 2*w[N-1]*h[i])
    # Wait, that's not right either. Let me be more careful.
    
    # d(conv[j])/dh[i] for j = i+k+N-1: contributes h[i+k] if 0<=i+k<N
    # d(conv[j])/dh[i] for j = i-k+N-1: contributes h[i-k] if 0<=i-k<N
    # For j = N-1 (k=0): both give h[i], so total 2*h[i] * w[N-1]
    # For j = N-1+k (k>0): gives h[i+k] if valid, weight w[N-1+k]
    # For j = N-1-k (k>0): gives h[i-k] if valid, weight w[N-1-k]
    # And w[N-1+k] = w[N-1-k] by symmetry
    
    # So grad_softmax[i] = 2*w[N-1]*h[i] + 2*sum_{k=1}^{N-1} w[N-1+k]*h[i+k]*(i+k<N) + 2*sum_{k=1}^{N-1} w[N-1-k]*h[i-k]*(i-k>=0)
    # = 2*sum_{k=-(N-1)}^{N-1} w[N-1+max(0,k)]*... this is getting messy
    
    # Let me just use a direct approach with np.roll or explicit loops for clarity
    # Since N=400, a simple loop over k is O(N^2) = 160000 operations, which is fine
    
    grad_sm = np.zeros(N)
    for k in range(-(N-1), N):
        w_val = w[k + N - 1]
        if k >= 0:
            # h[i+k] valid for i = 0 to N-1-k
            if N-1-k >= 0:
                grad_sm[:N-k] += w_val * h[k:]
        else:
            # h[i-k] = h[i+|k|] valid for i = |k| to N-1
            abs_k = -k
            if N-1-abs_k >= 0:
                grad_sm[abs_k:] += w_val * h[:N-abs_k]
    
    # Now the full gradient of the objective:
    # V = 2*N*soft_max / s^2
    # dV/dh[i] = 2*N/s^2 * grad_sm[i] - 2*N*soft_max*2/s^4 * 1
    # = 2*N/s^2 * (grad_sm[i] - 2*soft_max/s)
    
    soft_max_val = m + np.log(sum_exp) / beta
    grad_obj = 2.0 * N / (s * s) * (grad_sm - 2.0 * soft_max_val / s)
    
    return grad_obj

def _chebyshev_init(rng, n_coeffs=20):
    """
    Generate initialization using Chebyshev polynomials of the second kind.
    h[i] = sum_{m=0}^{M-1} a_m * U_m(x_i) + c
    where x_i are uniform on [-1, 1]
    """
    x = np.linspace(-1, 1, N, endpoint=False)
    h = np.zeros(N)
    # Add DC component
    h += rng.uniform(0.5, 1.5)
    # Add Chebyshev components
    for m in range(1, n_coeffs):
        # U_m(x) = sin((m+1)*arcsin(x)) / sqrt(1-x^2) approximately
        # Use scipy's eval_gegenbauer for stability
        # U_m(x) = Gegenbauer(m+1, 1/2, x) / (m+1) * something... 
        # Actually, U_n(x) = P_{n+1}^{(0,1)}(x) * (n+1)
        # Let's just use the recurrence
        if m == 1:
            U = 2 * x
        elif m == 2:
            U = 4 * x**2 - 1
        else:
            # Recurrence: U_m(x) = 2x*U_{m-1}(x) - U_{m-2}(x)
            U_prev2 = 2 * x  # U_1
            U_prev1 = 4 * x**2 - 1  # U_2
            for mm in range(3, m + 1):
                U_curr = 2 * x * U_prev1 - U_prev2
                U_prev2 = U_prev1
                U_prev1 = U_curr
            U = U_prev1
        
        coeff = rng.standard_normal() * 0.3 / (m + 1)
        h += coeff * U
    
    # Ensure positive sum
    if abs(np.sum(h)) < 0.1:
        h += 0.5
    
    return h

def _coordinate_descent_polish(h, max_iters=5000, step_scale=0.01):
    """
    Coordinate descent polish: iterate over coordinates, try small perturbations.
    """
    best_val = _objective_hard(h)
    h_curr = h.copy()
    rng = np.random.default_rng(123)
    
    for it in range(max_iters):
        if time.time() > deadline_global - 5.0:
            break
        
        i = rng.integers(0, N)
        # Try a few step sizes
        for scale in [step_scale, step_scale * 0.5, step_scale * 2.0]:
            delta = rng.standard_normal() * scale * abs(h_curr[i] + 0.1)
            h_test = h_curr.copy()
            h_test[i] += delta
            
            val_test = _objective_hard(h_test)
            if val_test < best_val:
                best_val = val_test
                h_curr = h_test
                break
    
    return h_curr, best_val

def construct():
    global deadline_global
    start_time = time.time()
    deadline_global = start_time + 145.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_val = _objective_hard(best_h)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Chebyshev polynomial initializations (structured)
    for n_coeffs in [10, 15, 20, 30]:
        for trial in range(3):
            h0 = _chebyshev_init(rng, n_coeffs)
            initializations.append(h0)
    
    # 2. Random normal with positive mean
    for _ in range(5):
        h0 = rng.standard_normal(N)
        h0 += 0.5
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)
        
    # 3. Sign sequences with DC offset
    for _ in range(5):
        h0 = rng.choice([-1, 1], size=N).astype(float)
        h0 += 0.3 * rng.standard_normal()
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)
        
    # 4. Sine waves
    for freq in [1, 2, 3, 5]:
        t = np.linspace(0, 2*np.pi*freq, N, endpoint=False)
        h0 = np.sin(t + rng.random()*2*np.pi)
        h0 += 0.8
        initializations.append(h0)

    # Phase 1: Time-domain optimization with analytic gradients
    betas = [50.0, 100.0, 200.0]
    
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline_global - 40.0:
            break
        
        h_curr = h0.copy()
        
        for beta in betas:
            if time.time() > deadline_global - 40.0:
                break
            
            def obj_func(x):
                return _soft_max_obj(x, beta)
            
            def grad_func(x):
                return _soft_max_grad(x, beta)
            
            try:
                res = minimize(obj_func, h_curr, jac=grad_func,
                              method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30})
                h_new = res.x
            except Exception:
                h_new = h_curr
            
            val_new = _objective_hard(h_new)
            
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
                h_curr = h_new.copy()
            else:
                val_curr = _objective_hard(h_curr)
                if val_new < val_curr:
                    h_curr = h_new.copy()

    # Phase 2: Coordinate descent polish on best result
    if time.time() < deadline_global - 20.0:
        h_cd, val_cd = _coordinate_descent_polish(best_h, max_iters=3000)
        if val_cd < best_val:
            best_val = val_cd
            best_h = h_cd.copy()

    # Phase 3: Final high-precision polish with L-BFGS-B
    if time.time() < deadline_global - 10.0:
        for beta in [200.0, 500.0]:
            if time.time() > deadline_global - 5.0:
                break
            
            def obj_func(x):
                return _soft_max_obj(x, beta)
            
            def grad_func(x):
                return _soft_max_grad(x, beta)
            
            try:
                res = minimize(obj_func, best_h, jac=grad_func,
                              method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
                val = _objective_hard(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass

    # Phase 4: Another coordinate descent pass
    if time.time() < deadline_global - 5.0:
        h_cd, val_cd = _coordinate_descent_polish(best_h, max_iters=2000)
        if val_cd < best_val:
            best_val = val_cd
            best_h = h_cd.copy()

    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _objective_hard(best_h)
        
    return best_h
# EVOLVE-BLOCK-END
