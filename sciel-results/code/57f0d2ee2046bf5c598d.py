"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score_normalized(h):
    """
    Compute the objective: max|autoconv| / (sum h)^2.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    max_ac = np.max(np.abs(ac[:size]))
    return max_ac / (s * s)

def _objective_and_grad(h):
    """
    Objective: max |A(k)|.
    We use a soft-max approximation to make it differentiable everywhere,
    or we can just use the max index.
    For robustness, we minimize the sum of exp(lambda * |A(k)|) / N.
    """
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    ac = ac[:size]
    
    # Use a log-sum-exp trick for the max
    # max |x| approx (1/L) log sum exp(L |x|)
    L = 50.0
    # Clip to avoid overflow
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    if max_val < 1e-9:
        return 0.0, np.zeros_like(h)
    
    # Normalize for stability
    shifted = L * (abs_ac - max_val)
    sum_exp = np.sum(np.exp(shifted))
    obj = max_val + np.log(sum_exp) / L
    
    # Gradient
    # d/dh (max |A|) is dominated by the terms with large |A|
    # grad_obj = (1/L) * sum exp(L|A_k|) * sign(A_k) * dA_k/dh
    # We approximate the gradient by weighting the gradients of individual A_k by their softmax weights
    weights = np.exp(shifted) / sum_exp
    
    # dA_k/dh = h convolved with delta at k?
    # A(k) = sum h_i h_{i+k}
    # dA(k)/dh_j = h_{j+k} + h_{j-k} (with boundary handling)
    # This is equivalent to the convolution of h with a kernel that is 1 at -k and +k.
    # Or simpler: A = ifft(|H|^2).
    # Let's compute gradient via FFT.
    # A = ifft(H * conj(H)) ? No, A is real, A = ifft(H * H) for real h?
    # H = fft(h). A = ifft(H * conj(H)) is power spectrum? No.
    # Autocorrelation A(k) = sum h_i h_{i+k}.
    # FFT(A) = H * conj(H) = |H|^2.
    # So A = ifft(|H|^2).
    # dA/dh = ?
    # Let's use the property that A = conv(h, h_flipped).
    # grad_h = conv(grad_A, h_flipped) + conv(grad_A, h_flipped) ?
    # Actually, d/dh (sum w_k A(k)) = sum w_k (h_{shifted} + h_{shifted})
    
    # Efficient gradient calculation:
    # We want G = sum_k weights_k * (dA_k/dh)
    # dA_k/dh = h_{k} * delta?
    # Let's use the FFT domain.
    # A = ifft(|H|^2)
    # This is tricky to differentiate directly in time domain for the gradient of max.
    # Let's stick to time domain convolution for gradient, it's O(N^2) but N=400 is small.
    # Or use the fact that dA/dh is a Toeplitz matrix.
    
    # Simpler: Finite difference? No, too slow.
    # Let's use the analytic gradient of A(k).
    # A(k) = <h, h_shifted_k>
    # dA(k)/dh = h_shifted_k + h_shifted_-k
    
    # We can compute the gradient vector g such that g = sum_k weights_k * (h_shifted_k + h_shifted_-k)
    # This is equivalent to:
    # g = conv(weights, h) + conv(weights_flipped, h)?
    # Let's verify.
    # sum_k w_k h_{i+k} = (w * h)_i
    # sum_k w_k h_{i-k} = (w_flipped * h)_i ? No.
    # sum_k w_k h_{i-k} = sum_j w_{i-j} h_j = (w * h)_i if w is symmetric?
    # w_k = weights[|k|]? No, weights are indexed by k in 0..2N-2.
    # A(k) for k in 0..N-1 is positive lags. A(-k) = A(k).
    # So we only need to sum over k >= 0.
    # g_i = sum_{k=0}^{N-1} w_k (h_{i+k} + h_{i-k})
    # This can be done with FFTs.
    
    # Let's construct the gradient using FFTs.
    # g = ifft( fft(w_full) * H ) ?
    # Let W = fft(w_full) where w_full is the weights placed in the full convolution vector.
    # Then g = ifft(W * H) + ifft(W * H) ?
    
    # Let's just do it with np.convolve for simplicity and correctness, N=400 is fast enough.
    # w_full length 2N-1.
    # w_full[k] = weights[k] for k in 0..N-1 (positive lags)
    # w_full[2N-2-k] = weights[k] for k in 0..N-1 (negative lags, symmetric)
    # Actually A(k) is symmetric.
    
    w_full = np.zeros(size)
    w_full[:N] = weights[:N]
    w_full[N:] = weights[:N-1][::-1] # weights[N-1] is at N-1, so N..2N-2 correspond to N-1..1
    
    # dA/dh = conv(w_full, h) + conv(w_full_flipped, h)?
    # Let's check d/dh_i sum_k w_k (h_{i+k} h_{i+k} ?? No.
    # A(k) = sum_j h_j h_{j+k}
    # dA(k)/dh_i = h_{i+k} + h_{i-k}
    # g_i = sum_k w_k (h_{i+k} + h_{i-k})
    # g = conv(w, h) + conv(w_rev, h) ?
    # conv(w, h)_i = sum_k w_k h_{i-k}
    # conv(w_rev, h)_i = sum_k w_rev_k h_{i-k} = sum_k w_{-k} h_{i-k} = sum_j w_j h_{i+j}
    # So g = conv(w_full, h) + conv(w_full, h) = 2 * conv(w_full, h) ?
    # Wait, w_full contains both positive and negative indices effectively if we define it properly.
    # If w_full is defined on 0..2N-2, and we convolve with h, we get the sum.
    # Let's just use np.convolve.
    
    g = np.convolve(w_full, h, mode='same') # This might be off by shift.
    # Better:
    # g = np.zeros(N)
    # for k in range(N):
    #    g += w_full[k] * (np.roll(h, k) + np.roll(h, -k))
    # This is O(N^2). 400^2 = 160,000. Very fast.
    
    g = np.zeros(N)
    for k in range(N):
        if w_full[k] > 1e-12:
            g += w_full[k] * (np.roll(h, k) + np.roll(h, -k))
            
    # Scale gradient
    grad = g
    
    return obj, grad

def construct():
    t_start = time.time()
    deadline = t_start + 115.0
    
    # Start with a random initialization or a chirp
    rng = np.random.default_rng(42)
    
    best_val = np.inf
    best_h = None
    
    # Try a few initializations
    inits = []
    # 1. Random
    inits.append(rng.normal(size=N))
    # 2. Chirp
    t = np.linspace(0, 1, N, endpoint=False)
    inits.append(np.cos(2 * np.pi * 50 * t**2))
    # 3. Constant (to check baseline)
    inits.append(np.ones(N))
    
    for h0 in inits:
        if time.time() > deadline - 10:
            break
        
        # Normalize sum to 1
        h0 = h0 / np.sum(h0)
        
        # Optimize
        try:
            res = minimize(
                _objective_and_grad,
                h0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_opt = res.x
            # Project to sum=1
            h_opt = h_opt / np.sum(h_opt)
            
            val = _score_normalized(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt
                
        except Exception as e:
            pass

    # Fallback if optimization failed
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = _score_normalized(best_h)

    return best_h

# EVOLVE-BLOCK-END
