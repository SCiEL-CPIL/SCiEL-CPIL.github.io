"""Signed step functions minimizing max |autoconv| / (integral)^2 via spectral basis optimization."""
import numpy as np
import time
import warnings
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

N = 400
START_TIME = time.time()
TIME_LIMIT = 110  # Safety margin before 120s evaluator limit

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT. Returns array of length 2N-1."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _score(x):
    """Compute the C3' score: max |autoconv| / (sum(x))^2."""
    integral = np.sum(x)
    if abs(integral) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    max_abs = np.max(np.abs(ac))
    return max_abs / (integral * integral)

def _log_sum_exp(a, axis=None):
    """Numerically stable log-sum-exp."""
    a_max = np.max(a, axis=axis, keepdims=True)
    if np.any(np.isinf(a_max)):
        return a_max
    return np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max

def _soft_max_abs_ac(x, beta=10.0):
    """Differentiable soft approximation of max |autoconv| using log-sum-exp."""
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    if beta < 1e-6:
        return np.max(abs_ac)
    scaled = beta * abs_ac
    # Log-sum-exp gives (1/beta) * log(sum(exp(beta * |ac|)))
    lse = _log_sum_exp(scaled)
    return lse / beta

def _objective_and_grad(params, beta=10.0):
    """
    Objective: soft_max_abs_ac / (sum(x))^2 where x = tanh(params).
    Returns (value, gradient).
    """
    x = np.tanh(params)
    integral = np.sum(x)
    if abs(integral) < 1e-10:
        return 1e10, np.zeros_like(params)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    # Soft max via log-sum-exp
    if beta < 1e-6:
        max_val = np.max(abs_ac)
        # Gradient of max |ac| is tricky, use subgradient
        idx = np.argmax(abs_ac)
        d_dac = np.zeros_like(ac)
        d_dac[idx] = np.sign(ac[idx])
    else:
        scaled = beta * abs_ac
        m = np.max(scaled)
        exp_scaled = np.exp(scaled - m)
        sum_exp = np.sum(exp_scaled)
        soft_max = (m + np.log(sum_exp)) / beta
        
        # Gradient: d(soft_max)/d(ac_k) = beta * sign(ac_k) * exp(beta*|ac_k|) / sum(exp(beta*|ac|))
        d_dac = np.sign(ac) * exp_scaled / sum_exp
    
    # d(soft_max)/d(x_j) = sum_k d_dac[k] * d(ac_k)/d(x_j)
    # ac_k = sum_i x_i * x_{k-i} (for valid indices)
    # d(ac_k)/d(x_j) = x_{k-j} + x_{j-k} (with bounds checking)
    # More efficiently: d(ac)/dx = 2 * convolution(x, x) but shifted... 
    # Actually ac = x * x (linear conv), so d(ac_k)/d(x_j) = x_{k-j} + x_{j-k} if in range
    # Let's compute via FFT: d(ac)/dx_j = 2 * (x convolved with delta_j) but that's just 2*x shifted
    # Better: use the fact that ac = ifft(|fft(x)|^2)
    # d(ac_k)/d(x_j) = 2 * Re{ ifft( fft(x) * conj(fft(e_j)) ) } where e_j is impulse at j
    # fft(e_j) = exp(-2*pi*i*j*k/n)
    # So d(ac)/dx_j = 2 * Re{ ifft( fft(x) * exp(2*pi*i*j*k/n) ) } = 2 * circular_shift(x, j)
    # But we need linear convolution, not circular. For n_fft >= 2N-1, circular = linear.
    
    # Efficient approach: d(ac)/dx = 2 * (x padded and convolved) 
    # Let's just compute it directly using the structure
    # ac[k] = sum_{i=0}^{N-1} x[i] * x[k-i] where x[m]=0 if m<0 or m>=N
    # d(ac[k])/d(x[j]) = x[k-j] + x[j-k] if both indices valid
    
    # For efficiency, use FFT-based gradient
    L_fft = 1
    L = 2 * N - 1
    while L_fft < L:
        L_fft *= 2
    
    X = np.fft.rfft(x, n=L_fft)
    # d(ac)/dx_j: ac = ifft(X^2), d(ac_k)/d(x_j) = 2*Re{ifft(X * conj(fft(e_j)))}
    # fft(e_j) for j in [0, N-1] is exp(-2*pi*i*j*freq)
    # So we need ifft(X * exp(2*pi*i*j*freq)) for each j
    # This is equivalent to circular cross-correlation
    
    # Alternative: since ac = x * x (linear), d(ac)/dx = 2 * (x * delta) = 2*x but that's not right
    # Let me think again. ac[k] = sum_i x[i]x[k-i]. 
    # d(ac[k])/d(x[j]) = x[k-j] + x[j-k] (when indices valid, 0 otherwise)
    # This is 2 * (x convolved with x) at position... no.
    # It's: for fixed j, d(ac[k])/d(x[j]) = x[k-j] (if 0<=k-j<N) + x[j-k] (if 0<=j-k<N)
    # = x[k-j] + x[-(k-j)] = 2 * x[k-j] if x is symmetric... but x is not necessarily symmetric.
    # Wait, x is just a vector. x[k-j] is defined for 0<=k-j<N, and x[j-k] is defined for 0<=j-k<N.
    
    # Let's compute d_ac_dx as a matrix implicitly via FFT
    # d(ac)/dx_j = 2 * Re{ ifft( X * conj(FFT(e_j)) ) } where FFT(e_j) = exp(-i*2*pi*j*freq)
    # = 2 * Re{ ifft( X * exp(i*2*pi*j*freq) ) }
    # = 2 * circular_correlation(x, e_j) = 2 * x shifted by j (circularly)
    
    # For linear convolution with zero-padding, this works directly.
    # d_ac_dx[k, j] = 2 * x[(k - j) mod L_fft] ... but we need linear, so zero outside [0, N-1]
    
    # Let's just compute the gradient w.r.t. x directly
    d_ac_dx = np.zeros((L, N))
    for j in range(N):
        # d(ac[k])/d(x[j]) = x[k-j] if 0 <= k-j < N, plus x[j-k] if 0 <= j-k < N
        k1 = np.arange(N) + j  # k = j + i, i in [0, N-1], so k-j = i
        # x[k-j] = x[i] for i in [0, N-1]
        d_ac_dx[k1, j] += x
        k2 = np.arange(N)  # k = j - i, so j-k = i, k = j - i
        valid = (k2 >= 0) & (k2 < L)
        if np.any(valid):
            d_ac_dx[k2[valid], j] += x[~valid][::-1]  # Hmm, this is getting complicated
    
    # Let me use a cleaner FFT approach
    # d(ac)/dx_j = 2 * ifft(X * exp(2*pi*i*j*k/L_fft)) real part, restricted to [0, L-1]
    freqs = np.fft.rfftfreq(L_fft) * L_fft
    exp_matrix = np.exp(2j * np.pi * np.outer(freqs, np.arange(N)) / L_fft)  # shape (L_fft//2+1, N)
    X_exp = X[:, np.newaxis] * exp_matrix  # (L_fft//2+1, N)
    d_ac_dx_fft = np.fft.irfft(X_exp, n=L_fft, axis=0)  # (L_fft, N)
    d_ac_dx = 2.0 * d_ac_dx_fft[:L, :]
    
    # Now d(soft_max)/d(x_j) = sum_k d_dac[k] * d_ac_dx[k, j]
    d_dx = d_ac_dx.T @ d_dac  # shape (N,)
    
    # d(soft_max)/d(params_j) = d(soft_max)/d(x_j) * (1 - x_j^2)
    d_x_params = 1.0 - x**2
    
    # Objective: f = soft_max / integral^2
    # df/d(params_j) = [d(soft_max)/d(params_j) * integral^2 - soft_max * 2 * integral * d(integral)/d(params_j)] / integral^4
    # d(integral)/d(params_j) = 1 - x_j^2
    soft_max_val = _soft_max_abs_ac(x, beta)
    d_softmax_dparams = d_dx * d_x_params
    
    d_integral_dparams = d_x_params
    denom = integral**4
    grad = (d_softmax_dparams * integral**2 - 2 * soft_max_val * integral * d_integral_dparams) / denom
    
    value = soft_max_val / (integral * integral)
    return value, grad

def _local_search_flip(x, max_iter=300, time_limit=30.0):
    """Local search by flipping individual signs."""
    start_time = time.time()
    x = x.copy().astype(float)
    current_score = _score(x)
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        improved = False
        # Try flipping each element, track best flip
        best_i = -1
        best_new_score = current_score
        for i in range(N):
            x[i] = -x[i]
            new_score = _score(x)
            if new_score < best_new_score:
                best_new_score = new_score
                best_i = i
            x[i] = -x[i]  # revert
        
        if best_i >= 0 and best_new_score < current_score:
            x[best_i] = -x[best_i]
            current_score = best_new_score
            improved = True
        else:
            break
    return x, current_score

def _pairwise_swap_search(x, max_iter=100, time_limit=20.0):
    """Local search by swapping pairs of signs."""
    start_time = time.time()
    x = x.copy().astype(float)
    current_score = _score(x)
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        improved = False
        # Try swapping a pair of opposite-sign elements
        pos_indices = np.where(x > 0)[0]
        neg_indices = np.where(x < 0)[0]
        
        # Sample some pairs to avoid O(N^2)
        n_samples = min(200, len(pos_indices) * len(neg_indices))
        for _ in range(n_samples):
            if time.time() - start_time > time_limit:
                break
            i = pos_indices[np.random.randint(len(pos_indices))]
            j = neg_indices[np.random.randint(len(neg_indices))]
            x[i], x[j] = x[j], x[i]
            new_score = _score(x)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                x[i], x[j] = x[j], x[i]
        
        if not improved:
            break
    return x, current_score

def construct():
    """Main entry point: construct signed step function minimizing C3'."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    np.random.seed(12345)
    
    best_x = None
    best_score = np.inf
    
    def _check_time():
        return time.time() > deadline - 5
    
    # Strategy 1: Start with all-ones and optimize via spectral basis
    if not _check_time():
        # All ones
        x0 = np.ones(N)
        params0 = np.zeros(N)  # arctanh(1) is inf, so use large values
        params0 = np.full(N, 3.0)  # tanh(3) ≈ 0.995
        
        # Optimize with L-BFGS
        try:
            res = minimize(
                lambda p: _objective_and_grad(p, beta=5.0)[0],
                params0,
                jac=lambda p: _objective_and_grad(p, beta=5.0)[1],
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x_new = np.tanh(res.x)
            x_new = np.sign(x_new)
            x_new[x_new == 0] = 1
            score_new = _score(x_new)
            if score_new < best_score:
                best_score = score_new
                best_x = x_new.copy()
        except Exception:
            pass
    
    # Strategy 2: Random initializations with gradient optimization
    n_restarts = 4
    for restart in range(n_restarts):
        if _check_time():
            break
        
        if restart == 0:
            # Alternating
            x0 = np.sign(np.cos(np.arange(N) * np.pi / 2))
            x0[x0 == 0] = 1
        elif restart == 1:
            # Random
            x0 = np.random.choice([-1, 1], size=N)
        elif restart == 2:
            # Block pattern
            block_len = 16
            n_blocks = N // block_len
            blocks = np.random.choice([-1, 1], size=n_blocks)
            x0 = np.repeat(blocks, block_len)
            if len(x0) < N:
                x0 = np.concatenate([x0, np.random.choice([-1, 1], size=N - len(x0))])
        else:
            # Low-frequency pattern
            freq = np.random.uniform(0.01, 0.1)
            x0 = np.sign(np.cos(2 * np.pi * freq * np.arange(N) + np.random.uniform(0, 2*np.pi)))
            x0[x0 == 0] = 1
        
        # First do local search
        if not _check_time():
            x_local, score_local = _local_search_flip(x0, max_iter=100, time_limit=10.0)
            if score_local < best_score:
                best_score = score_local
                best_x = x_local.copy()
        
        # Then gradient optimization
        if not _check_time():
            params0 = np.arctanh(np.clip(x_local, -0.99, 0.99))
            try:
                # Annealing beta
                for beta in [3.0, 8.0, 20.0]:
                    if _check_time():
                        break
                    res = minimize(
                        lambda p: _objective_and_grad(p, beta=beta)[0],
                        params0,
                        jac=lambda p: _objective_and_grad(p, beta=beta)[1],
                        method='L-BFGS-B',
                        options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    x_new = np.tanh(res.x)
                    x_new = np.sign(x_new)
                    x_new[x_new == 0] = 1
                    score_new = _score(x_new)
                    if score_new < best_score:
                        best_score = score_new
                        best_x = x_new.copy()
                        params0 = res.x.copy()
            except Exception:
                pass
    
    # Strategy 3: Prolate spheroidal wave basis approach
    # Represent x as a linear combination of prolate spheroidal wave functions
    # and optimize the coefficients
    if not _check_time() and best_x is not None:
        # Use a small number of basis functions
        n_basis = min(20, N)
        # Generate prolate spheroidal wave functions (approximate with bandlimited sinc)
        # For simplicity, use DCT basis which is related
        t = np.arange(N)
        basis = np.zeros((n_basis, N))
        for k in range(n_basis):
            basis[k] = np.cos(np.pi * (k + 0.5) * (t + 0.5) / N)
        
        # Optimize coefficients
        def _basis_objective(coeffs):
            x = np.tanh(coeffs @ basis)
            integral = np.sum(x)
            if abs(integral) < 1e-10:
                return 1e10
            ac = _autoconv_fft(x)
            max_abs = np.max(np.abs(ac))
            return max_abs / (integral * integral)
        
        # Initialize coefficients from current best
        x_curr = best_x
        # Least squares fit to get initial coefficients
        coeffs0, _, _, _ = np.linalg.lstsq(basis, x_curr, rcond=None)
        coeffs0 = np.arctanh(np.clip(coeffs0 @ basis, -0.99, 0.99)) if False else coeffs0
        
        try:
            res = minimize(
                _basis_objective,
                coeffs0,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-10}
            )
            x_new = np.tanh(res.x @ basis)
            x_new = np.sign(x_new)
            x_new[x_new == 0] = 1
            score_new = _score(x_new)
            if score_new < best_score:
                best_score = score_new
                best_x = x_new.copy()
        except Exception:
            pass
    
    # Final polish with local search
    if best_x is not None and not _check_time():
        x_polished, score_polished = _local_search_flip(best_x, max_iter=200, time_limit=15.0)
        if score_polished < best_score:
            best_score = score_polished
            best_x = x_polished
        
        # Pairwise swap polish
        if not _check_time():
            x_swap, score_swap = _pairwise_swap_search(best_x, max_iter=50, time_limit=10.0)
            if score_swap < best_score:
                best_score = score_swap
                best_x = x_swap
    
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x.astype(float)
