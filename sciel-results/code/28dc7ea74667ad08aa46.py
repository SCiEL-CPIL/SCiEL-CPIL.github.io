"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_value(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv)) / (s ** 2)

def _lse_and_grad(h, temperature=1.0):
    """Compute log-sum-exp of conv(h,h) and its gradient w.r.t. h.
    
    LSE(conv) = max(conv) + log(sum(exp((conv - max)/T))) * T
    This is a smooth upper bound on max(conv), converging as T -> 0.
    
    Returns (lse_value, grad_h) where grad_h is dLSE/dh.
    """
    conv = np.convolve(h, h)
    max_c = np.max(conv)
    shifted = (conv - max_c) / temperature
    # Clip for numerical stability
    shifted = np.clip(shifted, -100, 100)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum = np.log(sum_exp)
    lse_val = max_c + temperature * log_sum
    
    # Gradient: dLSE/dconv_k = exp((conv_k - max)/T) / sum_exp = softmax weights
    weights = exp_shifted / sum_exp
    
    # dLSE/dh_i = sum_k weights[k] * dconv_k/dh_i
    # conv_k = sum_j h_j * h_{k-j}
    # dconv_k/dh_i = h_{k-i} + h_{k-i} (when k-i in [0,N-1])... 
    # Actually: conv_k = sum_{j=0}^{N-1} h_j * h_{k-j} where h is zero-padded outside [0,N-1]
    # dconv_k/dh_i: for each j where j=i, term is h_{k-i}; for each j where k-j=i i.e. j=k-i, term is h_j = h_{k-i}
    # But we need to be careful: conv_k = sum_j h_j h_{k-j}
    # d/dh_i [h_j h_{k-j}] = delta(j,i)*h_{k-j} + delta(k-j,i)*h_j
    # = delta(j,i)*h_{k-i} + delta(j,k-i)*h_j
    # Sum over j: h_{k-i} (if k-i in range) + h_{k-i} (if k-i in range) ... wait
    # Sum_j delta(j,i) h_{k-j} = h_{k-i} (if 0 <= k-i <= N-1)
    # Sum_j delta(j, k-i) h_j = h_{k-i} (if 0 <= k-i <= N-1)
    # So dconv_k/dh_i = 2 * h_{k-i} if 0 <= k-i <= N-1, else 0
    # Wait, that's not right. Let me re-derive.
    # conv_k = sum_{j=0}^{N-1} h_j * h_{k-j} where h_m = 0 if m < 0 or m >= N
    # dconv_k/dh_i = sum_j [delta(j,i) * h_{k-j} + h_j * delta(k-j,i)]
    # First term: sum_j delta(j,i) h_{k-j} = h_{k-i} (valid if 0<=k-i<N)
    # Second term: sum_j h_j delta(k-j,i) = sum_j h_j delta(j, k-i) = h_{k-i} (valid if 0<=k-i<N)
    # So dconv_k/dh_i = 2*h_{k-i} if 0 <= k-i < N, else 0
    # Hmm, but this double counts when i = k-i, i.e., 2i = k.
    # Actually no, the formula is correct. Each occurrence of h_i in the sum contributes.
    
    # More efficient: dconv_k/dh_i = h_{k-i} + h_{k-i} = 2*h_{k-i} for valid k-i
    # But actually we can compute grad_h[i] = sum_k weights[k] * (h_{k-i} + h_{k-i}) for valid indices
    # = 2 * sum_k weights[k] * h_{k-i}
    # Let's define: for each i, grad_h[i] = 2 * sum_{k: 0<=k-i<N} weights[k] * h_{k-i}
    # Let m = k-i, so k = m+i, sum over m where 0<=m<N and 0<=m+i<2N-1
    # grad_h[i] = 2 * sum_{m=max(0,-i)}^{min(N-1, 2N-2-i)} weights[m+i] * h[m]
    # This is a correlation of weights with h!
    
    # grad_h[i] = 2 * (weights * h)_{correlation}[i]
    # Specifically, grad_h = 2 * np.convolve(weights, h)[N-1:N-1+N]
    # Wait: np.convolve(weights, h) gives full convolution of length 2N-1
    # (weights * h)[i] = sum_m weights[m] * h[i-m] ... that's cross-correlation reversed
    # Let me think again.
    # We want: for each i, sum over m of weights[m+i] * h[m]
    # This is the cross-correlation of weights and h at lag i: <weights, h shifted by i>
    # = sum_m weights[m+i] * h[m]
    # np.correlate(weights, h, mode='full')[N-1+i] = sum_m weights[m] * h[m+i] ... not quite
    # Let's just use: grad_h[i] = 2 * sum_m weights[m+i]*h[m]
    # This is equivalent to: take weights, shift it by i, multiply element-wise with h, sum
    # Or equivalently: it's the cross-correlation.
    # np.correlate(weights, h, mode='full')[2*N-2-i] ... let me just compute it directly with FFT or loop for clarity
    
    # Efficient approach: grad_h = 2 * np.correlate(weights, h, mode='same') might not be exactly right
    # Let me verify: np.correlate(a, v, mode='full')[k] = sum_n a[n+k] * conj(v[n])
    # So np.correlate(weights, h, 'full')[k] = sum_n weights[n+k] * h[n]
    # We want for each i: sum_m weights[m+i] * h[m] = np.correlate(weights, h, 'full')[i]
    # The full correlation has length 2N-1, indexed 0 to 2N-2
    # For i in [0, N-1], we use index i
    # For i in [N, 2N-2], that would be negative lags
    
    # So: grad_h[i] = 2 * np.correlate(weights, h, 'full')[i] for i=0..N-1
    corr_full = np.correlate(weights, h, mode='full')
    grad_h = 2.0 * corr_full[:N]
    
    return lse_val, grad_h

def _smooth_obj_grad(h, temperature=1.0):
    """Compute smooth objective and its gradient.
    
    Objective: 2*N*LSE(conv) / s^2 where s = sum(h)
    We minimize this (it's an upper bound on the true value).
    """
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10, np.zeros(N)
    
    lse_val, dLSE_dh = _lse_and_grad(h, temperature)
    
    # f = 2*N*lse / s^2
    f = 2.0 * N * lse_val / (s ** 2)
    
    # df/dh_i = 2*N * [dLSE_dh_i * s^2 - lse * 2*s * 1] / s^4
    #         = 2*N * [dLSE_dh_i * s - 2*lse] / s^3
    grad_f = 2.0 * N * (dLSE_dh * s - 2.0 * lse_val) / (s ** 3)
    
    return f, grad_f

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_h = np.ones(N)
    best_val = _compute_value(best_h)  # should be 2.0

    # Strategy 1: Try various structured initializations
    candidates = []

    # Constant (seed)
    candidates.append(np.ones(N))

    # Triangular concentration in center
    idx = np.arange(N)
    center = (N - 1) / 2.0
    tri = np.maximum(0, 1 - np.abs(idx - center) / center)
    if np.sum(tri) > 0:
        candidates.append(tri / np.sum(tri))

    # Gaussian centered
    gauss = np.exp(-((idx - center) ** 2) / (2 * (N / 4) ** 2))
    if np.sum(gauss) > 0:
        candidates.append(gauss / np.sum(gauss))

    # Broad Gaussian
    gauss2 = np.exp(-((idx - center) ** 2) / (2 * (N / 2) ** 2))
    if np.sum(gauss2) > 0:
        candidates.append(gauss2 / np.sum(gauss2))

    # Random perturbations of the constant
    rng = np.random.default_rng(42)
    for trial in range(15):
        h = np.ones(N) + rng.standard_normal(N) * 0.3
        if abs(np.sum(h)) > 1e-9:
            candidates.append(h / np.sum(h))

    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline:
            break
        val = _compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    # Strategy 2: Local optimization using smooth surrogate with analytical gradient
    # Use decreasing temperature for sharper approximation
    temperatures = [5.0, 2.0, 1.0, 0.5, 0.2]
    
    for temp in temperatures:
        if time.time() > deadline - 15.0:
            break
        try:
            # Define objective and gradient for L-BFGS-B
            def obj_and_grad(x):
                f, g = _smooth_obj_grad(x, temperature=temp)
                # Clip gradient for stability
                g = np.clip(g, -1e6, 1e6)
                return f, g
            
            res = minimize(
                obj_and_grad,
                best_h,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            if time.time() < deadline - 5.0:
                val = _compute_value(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
        except Exception:
            pass

    # Strategy 3: Try from other good starting points
    if time.time() < deadline - 15.0:
        for h0 in candidates[:6]:
            if time.time() > deadline - 10.0:
                break
            try:
                def obj_and_grad2(x):
                    f, g = _smooth_obj_grad(x, temperature=1.0)
                    g = np.clip(g, -1e6, 1e6)
                    return f, g
                
                res = minimize(
                    obj_and_grad2,
                    h0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
                )
                if time.time() < deadline - 5.0:
                    val = _compute_value(res.x)
                    if val < best_val:
                        best_val = val
                        best_h = res.x.copy()
            except Exception:
                pass

    # Strategy 4: Final high-precision polish with very low temperature
    if time.time() < deadline - 5.0:
        try:
            def obj_and_grad_final(x):
                f, g = _smooth_obj_grad(x, temperature=0.1)
                g = np.clip(g, -1e6, 1e6)
                return f, g
            
            res = minimize(
                obj_and_grad_final,
                best_h,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
            val = _compute_value(res.x)
            if val < best_val:
                best_val = val
                best_h = res.x.copy()
        except Exception:
            pass

    # Strategy 5: Quick random local search
    if time.time() < deadline - 2.0:
        rng2 = np.random.default_rng(123)
        for _ in range(50):
            if time.time() > deadline - 1.0:
                break
            direction = rng2.standard_normal(N)
            for scale in [0.001, 0.005, 0.01, 0.05]:
                h_try = best_h + scale * direction
                val = _compute_value(h_try)
                if val < best_val:
                    best_val = val
                    best_h = h_try.copy()
                    break

    return best_h
# EVOLVE-BLOCK-END
