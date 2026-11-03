import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400

def _autoconv(h):
    """Compute full autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv(h)
    return np.max(np.abs(ac)) / (s * s)

def _surrogate_and_grad(h, beta=100.0):
    """
    Compute the smooth surrogate objective and its gradient.
    Surrogate: (1/beta) * log(sum(exp(beta * |c_k|))) / (sum h)^2
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(h)
    
    ac = _autoconv(h)
    abs_ac = np.abs(ac)
    
    # Stable log-sum-exp
    max_ac = np.max(abs_ac)
    if max_ac < 1e-12:
        log_sum = 0.0
        weights = np.zeros_like(ac)
    else:
        shifted = abs_ac - max_ac
        exp_shifted = np.exp(beta * shifted)
        log_sum = max_ac + np.log(np.sum(exp_shifted)) / beta
        
        # Weights for the gradient
        # w_k = (exp(beta * |c_k|) / sum(exp(...))) * beta * sign(c_k)
        # Note: sum(exp_shifted) is the denominator without max_ac factor
        # sum(exp(beta * |c_k|)) = exp(beta * max_ac) * sum(exp_shifted)
        # So w_k = exp_shifted_k / sum(exp_shifted) * beta * sign(c_k)
        weights = exp_shifted / np.sum(exp_shifted) * beta * np.sign(ac)
    
    obj = log_sum / (s * s)
    
    # Gradient of log_sum w.r.t h
    # d(log_sum)/dh = (1/beta) * (1/Z) * dZ/dh
    # dZ/dh = sum_k exp(beta * |c_k|) * beta * sign(c_k) * dc_k/dh
    # dZ/dh = beta * sum_k (exp(beta * |c_k|)/Z) * sign(c_k) * dc_k/dh
    # Let p_k = exp(beta * |c_k|)/Z. Then dZ/dh = beta * sum_k p_k sign(c_k) dc_k/dh
    # d(log_sum)/dh = (1/beta) * (1/Z) * beta * sum_k p_k sign(c_k) dc_k/dh
    #                = (1/Z) * sum_k p_k sign(c_k) dc_k/dh
    # Wait, Z = sum(exp(beta * |c_k|)).
    # p_k = exp(beta * |c_k|)/Z.
    # So d(log_sum)/dh = sum_k p_k sign(c_k) dc_k/dh.
    
    # dc_k/dh_j = h_{k-j} + h_{j-k}
    # sum_k p_k sign(c_k) (h_{k-j} + h_{j-k})
    # Let w_k = p_k sign(c_k).
    # Term 1: sum_k w_k h_{k-j} = (w * h)_j
    # Term 2: sum_k w_k h_{j-k} = (w * h_{-1})_j
    
    w = weights # This is p_k * beta * sign(c_k). We need p_k * sign(c_k).
    # My weights calculation above included beta.
    # Let's re-calculate weights for gradient correctly.
    # p_k = exp_shifted_k / sum(exp_shifted)
    # grad_log_sum_j = sum_k p_k sign(ac_k) (h_{k-j} + h_{j-k})
    
    p = exp_shifted / np.sum(exp_shifted) if max_ac >= 1e-12 else np.zeros_like(ac)
    w_grad = p * np.sign(ac)
    
    # Compute convolution w_grad * h and w_grad * h_rev
    # w_grad has length 2N-1. h has length N.
    # We need linear convolution.
    # (w_grad * h)_j = sum_k w_grad[k] h[k-j]
    # This is equivalent to cross-correlation or convolution depending on definition.
    # Let's use fftconvolve.
    
    # h is length N. w_grad is length 2N-1.
    # We want y_j = sum_k w_grad[k] h[k-j].
    # This is the cross-correlation of w_grad and h?
    # Correlation R_xy[l] = sum_n x[n] y[n+l].
    # Convolution (x*y)[n] = sum_k x[k] y[n-k].
    # So sum_k w_grad[k] h[k-j] is not standard convolution.
    # Let h_rev[m] = h[N-1-m].
    # Or simply use the property that sum_k w[k] h[k-j] = (w * h_{-1})_j?
    # Let's just compute the gradient term by term using FFT of the "kernel".
    
    # Actually, it's easier to compute the gradient of the sum of exponentials directly.
    # d/dh_j sum_k exp(beta |c_k|) = sum_k exp(beta |c_k|) beta sign(c_k) (h_{k-j} + h_{j-k})
    # Let A_k = exp(beta |c_k|) beta sign(c_k).
    # Then grad_j = sum_k A_k h_{k-j} + sum_k A_k h_{j-k}
    # Term 1: sum_k A_k h_{k-j}. This is the cross-correlation of A and h, evaluated at j?
    # (A * h)[j] = sum_k A[k] h[j-k].
    # We have sum_k A[k] h[k-j]. Let m = k-j => k = m+j.
    # sum_m A[m+j] h[m]. This is the cross-correlation of A and h at lag -j?
    # Actually, sum_k A[k] h[k-j] is the convolution of A and h_reversed?
    # Let's use numpy.correlate or fftconvolve carefully.
    
    # A_k is defined for k in 0..2N-2.
    # h_j is defined for j in 0..N-1.
    # We want G_j = sum_k A_k h_{k-j} + A_k h_{j-k}.
    
    # Let's compute C1_j = sum_k A_k h_{k-j}.
    # This is equivalent to the convolution of A and h, but with indices shifted?
    # If we define A_padded and h_padded, we can use fftconvolve.
    # Note that h_{k-j} is 0 if k-j is outside [0, N-1].
    
    # Let's compute the full convolution A * h.
    # (A * h)[n] = sum_k A[k] h[n-k].
    # We want sum_k A[k] h[k-j].
    # Let n = k-j => k = n+j.
    # sum_n A[n+j] h[n].
    # This is the cross-correlation of A and h, specifically (h * A)[j]?
    # Correlation R[j] = sum_n h[n] A[n+j].
    # So C1_j = R_A_h[j] where R is cross-correlation of A and h.
    
    # Similarly, C2_j = sum_k A_k h_{j-k}.
    # This is exactly the convolution (A * h)[j].
    
    # So grad_j = (h corr A)[j] + (A * h)[j].
    
    # We can compute both using FFT.
    # A has length 2N-1. h has length N.
    
    fsize = 1
    while fsize < (2*N - 1) + N:
        fsize *= 2
        
    H_A = np.fft.rfft(A, fsize)
    H_h = np.fft.rfft(h, fsize)
    
    # Convolution A * h
    conv_A_h = np.fft.irfft(H_A * H_h, fsize)[:2*N-1]
    
    # Cross-correlation h corr A
    # (h corr A)[j] = sum_n h[n] A[n+j]
    # This is equivalent to conv(h, A_rev) where A_rev is reversed A?
    # Or simply use the property that corr(x, y) = conv(x, y_rev).
    # A_rev[k] = A[2N-2-k].
    A_rev = A[::-1]
    H_A_rev = np.fft.rfft(A_rev, fsize)
    corr_h_A = np.fft.irfft(H_h * H_A_rev, fsize)[:2*N-1]
    
    # We need indices 0 to N-1 for the gradient.
    grad_log_sum = (corr_h_A[:N] + conv_A_h[:N])
    
    # Total gradient
    # d(obj)/dh = (1/s^2) grad_log_sum - (2/s^3) s * 1 = (1/s^2) grad_log_sum - (2/s^2) 1
    # Wait, obj = log_sum / s^2.
    # d(obj)/dh = (1/s^2) d(log_sum)/dh - log_sum * 2/s^3 * ds/dh
    # ds/dh = 1.
    # d(obj)/dh = (1/s^2) grad_log_sum - 2 * log_sum / s^3 * 1
    #           = (1/s^2) (grad_log_sum - 2 * log_sum / s)
    
    grad_obj = (grad_log_sum - 2 * log_sum / s) / (s * s)
    
    return obj, grad_obj

def _local_search_flip(h, max_passes=50, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    """
    h = h.copy()
    ac = _autoconv(h)
    cur_sum = float(np.sum(h))
    best_val = float(np.max(np.abs(ac)) / (cur_sum * cur_sum))
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        best_j = -1
        best_js = best_val
        best_new_ac = None
        best_new_sum = cur_sum
        
        # Evaluate all possible single flips
        for j in range(N):
            if deadline is not None and time.time() > deadline:
                break
            val = h[j]
            delta = -2.0 * val
            new_ac = ac.copy()
            start = j
            end = j + N
            new_ac[start:end] += -4.0 * val * h
            new_ac[j] += 4.0 * val * val
            new_sum = cur_sum + delta
            nsq = new_sum ** 2
            if nsq < 1e-10:
                new_score = np.inf
            else:
                new_score = float(np.max(np.abs(new_ac)) / nsq)
            
            if new_score < best_js:
                best_js = new_score
                best_j = j
                best_new_ac = new_ac
                best_new_sum = new_sum
        
        if best_j == -1 or best_js >= best_val - 1e-12:
            break
            
        h[best_j] = -h[best_j]
        ac = best_new_ac
        cur_sum = best_new_sum
        best_val = best_js
        improved = True
        
        if not improved:
            break
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 140.0
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()
        
    rng = np.random.default_rng(42)
    
    # Gradient Descent with Smooth Surrogate
    for restart in range(10):
        if time.time() > deadline - 10:
            break
            
        # Initialize h
        if restart == 0:
            h = np.ones(N) / N
        else:
            # Random initialization with zero mean (to be normalized)
            h = rng.standard_normal(N)
            h -= np.mean(h)
            if np.sum(h) == 0:
                h[0] += 1e-6
        
        s = np.sum(h)
        if abs(s) < 1e-9:
            continue
        h = h / s
        
        curr_score = _score(h)
        if curr_score < best_score:
            best_score = curr_score
            best_h = h.copy()
            
        # Gradient Descent Loop
        beta = 50.0
        lr = 0.01
        for step in range(200):
            if time.time() > deadline - 5:
                break
                
            obj, grad = _surrogate_and_grad(h, beta)
            
            # Normalize gradient to have unit norm to stabilize step size
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
                
            # Update
            h_new = h - lr * grad
            
            # Project to sum=1
            s_new = np.sum(h_new)
            if abs(s_new) < 1e-9:
                break
            h_new = h_new / s_new
            
            new_score = _score(h_new)
            
            if new_score < curr_score:
                h = h_new
                curr_score = new_score
                if curr_score < best_score:
                    best_score = curr_score
                    best_h = h.copy()
                # Reduce learning rate if improving
                lr *= 0.95
            else:
                # If not improving, reduce learning rate and try again
                lr *= 0.5
                if lr < 1e-6:
                    break
                    
        # Local Search Polish
        if time.time() < deadline - 5:
            h_ls, val_ls = _local_search_flip(h, max_passes=20, deadline=deadline-3)
            if val_ls < best_score:
                best_score = val_ls
                best_h = h_ls.copy()
                
    return best_h

# EVOLVE-BLOCK-END
