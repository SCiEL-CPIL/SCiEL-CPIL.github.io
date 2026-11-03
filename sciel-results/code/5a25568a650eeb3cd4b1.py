"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from math import log
import time

N = 400

# EVOLVE-BLOCK-START
def _autoconv(x):
    """Compute full autoconvolution of x (length N) via FFT."""
    n = 2 * N - 1
    # zero-pad to next power of 2 for FFT efficiency
    m = 1
    while m < n:
        m *= 2
    X = np.fft.rfft(x, m)
    Y = np.fft.irfft(X * X, m)
    return Y[:n]

def _score(x):
    """Score = max |autoconv| / (sum x)^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv(x)
    max_abs = np.max(np.abs(ac))
    return max_abs / (s * s)

def _gradient_softmax(x, lam=1000.0):
    """
    Compute gradient of soft-max surrogate: (1/lam) * log(sum exp(lam*|ac_i|))
    approximated via softmax weighting.
    Returns (grad, surrogate_value)
    """
    ac = _autoconv(x)
    abs_ac = np.abs(ac)
    # Soft-max: f = (1/lam) * log(sum exp(lam * abs_ac))
    # grad of f w.r.t. x: (1/lam) * sum_j (lam * sign(ac_j) * exp(lam*abs_ac_j) / sum exp(lam*abs_ac)) * d|ac_j|/dx
    # d|ac_j|/dx = sign(ac_j) * dac_j/dx
    # So grad_j = sign(ac_j) * w_j * dac_j/dx where w_j = exp(lam*abs_ac_j)/sum
    # dac_j/dx = ac reversed appropriately for autoconvolution
    
    # For autoconvolution: ac[j] = sum_i x[i]*x[j-i]
    # dac[j]/dx[k] = x[j-k] + x[k-j]  (for valid indices)
    # This is equivalent to: dac/dx[k] = 2 * (x * x)[j-k] but only for k where indices valid
    
    # More precisely, ac = x * x (full convolution)
    # d(ac[j])/dx[k] = x[j-k] + x[k-j] if both in [0,N-1]
    
    # Compute weights
    max_val = np.max(abs_ac)
    shifted = lam * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp  # shape (n,)
    
    # surrogate value
    surrogate = (max_val + log(sum_exp)) / lam
    
    # Compute gradient: grad[k] = sum_j sign(ac[j]) * weights[j] * (x[j-k] + x[k-j])
    # This is a convolution of weights*sign(ac) with x, plus its reverse
    
    w_sign = weights * np.sign(ac)  # shape (n,)
    
    # grad[k] = sum_j w_sign[j] * (x[j-k] + x[k-j])
    # = sum_j w_sign[j] * x[j-k] + sum_j w_sign[j] * x[k-j]
    
    # First term: for each k, sum over j where 0<=j-k<N: w_sign[j]*x[j-k]
    # Let m = j-k, so j = k+m, sum over m where 0<=k+m<N: w_sign[k+m]*x[m]
    # This is a convolution of w_sign with x
    
    # Second term: sum_j w_sign[j]*x[k-j]
    # Let m = k-j, so j = k-m, sum over m where 0<=k-m<N: w_sign[k-m]*x[m]
    # This is cross-correlation of w_sign with x
    
    # Both can be computed via FFT
    n_full = 2 * N - 1
    m_fft = 1
    while m_fft < n_full:
        m_fft *= 2
    
    W = np.fft.rfft(w_sign, m_fft)
    X = np.fft.rfft(x, m_fft)
    
    # First term: conv(w_sign, x) -> index j-k, we want sum_j w_sign[j] x[j-k]
    # In convolution terms: (w_sign * x)[k] = sum_j w_sign[j] x[k-j] -- that's the second term!
    # Wait, let me be careful.
    
    # Standard convolution: (a*b)[n] = sum_k a[k] b[n-k]
    # Term 2: sum_j w_sign[j] x[k-j] = (w_sign * x)[k] -- this is standard conv at index k
    
    # Term 1: sum_j w_sign[j] x[j-k]
    # Let m = -j, then sum_m w_sign[-m] x[m-k]... not clean.
    # Alternatively: sum_j w_sign[j] x[j-k] = sum_j w_sign[j] x_ref[k-j] where x_ref[m] = x[-m]?
    # Actually, define x_rev[j] = x[N-1-j]. Then x[j-k] relates to x_rev.
    # Simpler: Term 1 = sum_j w_sign[j] x[j-k]. Let i = j-k, j = i+k:
    # sum_i w_sign[i+k] x[i] -- this is cross-correlation of w_sign with x at lag k.
    # corr[k] = sum_i w_sign[i+k] x[i]
    
    # Cross-correlation can be done as conv(x, w_sign_reversed) or via FFT.
    # corr_wx[k] = sum_i w_sign[i+k] x[i]
    # = (x * w_sign_rev)[k + (N-1)] approximately... let me just use direct FFT approach.
    
    # Actually, let's use a different approach. Note that:
    # grad[k] = sum_j w_sign[j] * (x[j-k] + x[k-j])
    
    # Let's compute this via two convolutions using FFT.
    
    # For the full autoconvolution ac[j] = sum_{i} x[i] x[j-i] (with x defined as 0 outside [0,N-1])
    # d ac[j]/d x[k] = x[j-k] + x[k-j]
    
    # grad[k] = sum_j w_sign[j] (x[j-k] + x[k-j])
    
    # Define h[j] = w_sign[j]
    # Term A: sum_j h[j] x[j-k] 
    # Term B: sum_j h[j] x[k-j] = sum_j h[j] x[(k-j)] = (h * x)[k] if we define conv properly
    
    # For Term B: (h * x)[k] = sum_j h[j] x[k-j] -- standard convolution, valid for k in [0, 2N-2]
    # For Term A: sum_j h[j] x[j-k]. Let l = j-k, j = l+k: sum_l h[l+k] x[l]
    # This is the cross-correlation of h and x at lag k: (h \star x)[k] = sum_l h[l+k] x[l]
    # Cross-correlation = convolution with reversed: (h \star x)[k] = (x * h_rev)[k + shift]
    
    # h_rev[j] = h[N-1-j] for j in [0,N-1], but h has length n_full = 2N-1
    # Let's just compute both terms via explicit FFT of padded signals.
    
    # Term B: conv(h, x) at indices 0..N-1
    # Pad h to length m_fft, pad x to length m_fft
    H = W  # already computed
    # conv result
    conv_B = np.fft.irfft(H * X, m_fft)[:n_full]
    
    # Term A: cross-correlation sum_l h[l+k] x[l] for k=0..N-1
    # = sum_l h[l+k] x[l]
    # Reverse h: h_rev[j] = h[n_full-1-j]
    h_rev = h[::-1].copy()
    H_rev = np.fft.rfft(h_rev, m_fft)
    # conv(h_rev, x)[k + (n_full-1)] should give cross-correlation... 
    # (h_rev * x)[n] = sum_j h_rev[j] x[n-j] = sum_j h[n_full-1-j] x[n-j]
    # Let i = n_full-1-j, j = n_full-1-i: = sum_i h[i] x[n-(n_full-1-i)] = sum_i h[i] x[n-n_full+1+i]
    # We want sum_i h[i+k] x[i]. Set i' = i+k: sum_i' h[i'] x[i'-k]... 
    # This is getting complicated. Let me just do a direct approach with matrix-free but O(N^2) for the gradient since N=400 is small.
    
    # Direct computation for N=400: O(N^2) per gradient step
    grad = np.zeros(N)
    for k in range(N):
        s = 0.0
        for j in range(n_full):
            # x[j-k] is valid if 0 <= j-k < N
            # x[k-j] is valid if 0 <= k-j < N
            val = 0.0
            if 0 <= j - k < N:
                val += x[j - k]
            if 0 <= k - j < N:
                val += x[k - j]
            s += w_sign[j] * val
        grad[k] = s
    
    return grad / N, surrogate

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_x = None
    best_score = np.inf
    
    # Start with random signs
    np.random.seed(42)
    
    # Try a few initializations
    best_init_score = np.inf
    best_init = None
    
    for trial in range(5):
        x = np.random.choice([-1.0, 1.0], size=N)
        # Center: subtract mean to make sum ~ 0? No, we want sum != 0 for the ratio.
        # Actually, for signed steps, sum can be anything. Let's just use random signs.
        sc = _score(x)
        if sc < best_init_score:
            best_init_score = sc
            best_init = x.copy()
    
    if best_init is None:
        return np.ones(N)
    
    best_x = best_init.copy()
    best_score = best_init_score
    
    # Gradient descent on soft-max surrogate
    lam = 500.0
    lr = 0.1
    x = best_x.copy()
    
    for it in range(200):
        if time.time() > deadline - 30.0:
            break
        
        grad, _ = _gradient_softmax(x, lam)
        
        # Project to nearest sign pattern (or keep continuous for now, project at end)
        x_new = x - lr * grad
        # Soft projection: keep values in [-2, 2] to avoid wild excursions
        x_new = np.clip(x_new, -2.0, 2.0)
        
        sc_new = _score(np.sign(x_new))
        if sc_new < best_score:
            best_score = sc_new
            best_x = np.sign(x_new).copy()
        
        x = x_new
        lr *= 0.99
    
    # Local search: flip individual signs to improve
    x = best_x.copy()
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for i in range(N):
            x[i] *= -1.0
            sc = _score(x)
            if sc < best_score:
                best_score = sc
                best_x = x.copy()
                improved = True
            else:
                x[i] *= -1.0  # revert
    
    return best_x
# EVOLVE-BLOCK-END
