import numpy as np
import time

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _smooth_max_acf(A, tau=0.1):
    """Soft-max of |A[1:]| via log-sum-exp for gradient computation."""
    vals = np.abs(A[1:])
    # Shift for numerical stability
    m = np.max(vals)
    if m < 1e-12:
        return 0.0
    shifted = (vals - m) / tau
    log_sum = m + tau * np.log(np.sum(np.exp(shifted)))
    return float(log_sum)

def _grad_smooth_max(A, s, tau=0.1):
    """
    Gradient of smooth max autocorrelation w.r.t. s.
    We use the fact that A = circular_conv(s, s) for the full N-point circular ACF.
    d(A[k])/ds[i] = s[i-k] + s[i+k] (circular)
    The smooth max is sum_k w_k * |A[k]| where w_k = exp((|A[k]|-m)/tau) / sum(...)
    grad_s[i] = sum_k w_k * sign(A[k]) * (s[i-k] + s[i+k])
    """
    vals = np.abs(A[1:])
    m = np.max(vals)
    if m < 1e-12:
        return np.zeros(N)
    shifted = (vals - m) / tau
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    # weights has length N-1, corresponding to lags 1..N-1
    signs = np.sign(A[1:])
    # grad_s[i] = sum_{k=1}^{N-1} weights[k-1] * signs[k-1] * (s[i-k] + s[i+k])
    # This is a convolution of (weights*signs) with s, plus its flip
    kernel = np.zeros(N)
    kernel[1:] = weights * signs
    # For lag k, contribution is s[i-k] + s[i+k]
    # sum_k w_k s[i-k] = (kernel * s)[i] where kernel[k] = w_k
    # sum_k w_k s[i+k] = (kernel * s)[i] with kernel flipped? 
    # Actually: sum_k w_k s[i-k] = (w * s)[i] where (w*s)[i] = sum_k w[k] s[i-k]
    # and sum_k w_k s[i+k] = sum_k w[k] s[i+k] = sum_j w[i-j] s[j] ... 
    # Let's just do it via FFT for efficiency
    F_w = np.fft.rfft(kernel)
    F_s = np.fft.rfft(s.astype(np.float64))
    # sum_k w_k s[i-k] is circular convolution: ifft(fft(w)*fft(s))
    conv1 = np.fft.irfft(F_w * F_s, n=N)
    # sum_k w_k s[i+k] = sum_k w_k s[i+k]. Let j=i+k, k=j-i. sum_j w[j-i] s[j].
    # This is the circular cross-correlation or convolution with flipped kernel.
    # flip kernel: kernel_flip[k] = w[-k] = w[N-k] for k!=0, w[0]=w[0]
    # Actually simpler: sum_k w_k s[i+k] = (w_rev * s)[i] where w_rev is reversed
    # w_rev[k] = w[N-k] (with w_rev[0]=w[0])
    kernel_rev = np.zeros(N)
    kernel_rev[0] = kernel[0]
    kernel_rev[1:] = kernel[:0:-1]  # kernel[N-1], kernel[N-2], ..., kernel[1]
    F_wr = np.fft.rfft(kernel_rev)
    conv2 = np.fft.irfft(F_wr * F_s, n=N)
    
    grad = conv1 + conv2
    return grad

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin

    best_val = float('inf')
    best_seq = np.ones(N)

    def _try_sequence(s):
        nonlocal best_val, best_seq
        A = _calc_acf(s)
        v = _max_abs_acf_from_array(A)
        if v < best_val:
            best_val = v
            best_seq = s.copy()
        return v, A

    def _smooth_gradient_descent(s, n_iters=2000, lr=0.01, tau=0.1):
        """
        Smooth gradient descent on the continuous relaxation.
        We minimize the smooth max of |ACF[1:]| over real-valued s.
        Then project to binary.
        """
        s = s.astype(np.float64).copy()
        for it in range(n_iters):
            if time.time() > deadline:
                break
            A = _calc_acf(s)
            g = _grad_smooth_max(A, s, tau)
            # Gradient step
            s -= lr * g
            # Project to bounded interval [-1, 1] to keep scale reasonable
            s = np.clip(s, -1.0, 1.0)
            # Occasionally project to binary to check
            if it % 100 == 0:
                s_bin = np.sign(s)
                s_bin[s_bin == 0] = 1
                _try_sequence(s_bin)
        # Final projection
        s_bin = np.sign(s)
        s_bin[s_bin == 0] = 1
        _try_sequence(s_bin)
        return s

    def _stochastic_rounding(s_continuous, n_samples=100):
        """
        Stochastic rounding: for each element, round to +1 or -1 with probability
        proportional to its value. Preserve mean.
        """
        best_s = None
        best_v = float('inf')
        p = (s_continuous + 1) / 2  # probability of +1
        p = np.clip(p, 0.0, 1.0)
        for _ in range(n_samples):
            if time.time() > deadline:
                break
            u = np.random.random(N)
            s_sample = np.where(u < p, 1.0, -1.0)
            A = _calc_acf(s_sample)
            v = _max_abs_acf_from_array(A)
            if v < best_v:
                best_v = v
                best_s = s_sample.copy()
        if best_s is not None:
            _try_sequence(best_s)
        return best_s

    def _greedy_polish(s, max_passes=100):
        """Single-bit greedy descent."""
        s = s.copy()
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            # Randomize order to avoid bias
            order = np.random.permutation(N)
            for i in order:
                if time.time() > deadline:
                    break
                s[i] *= -1
                A = _calc_acf(s)
                new_val = _max_abs_acf_from_array(A)
                if new_val < current_val:
                    current_val = new_val
                    improved = True
                else:
                    s[i] *= -1  # revert
            if not improved:
                break
        _try_sequence(s)
        return s

    def _block_flip_anneal(s, n_iters=500, block_size=4):
        """Simulated annealing with block flips."""
        s = s.copy()
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        best_s = s.copy()
        best_v = current_val
        T = 1.0
        cooling = 0.99
        for it in range(n_iters):
            if time.time() > deadline:
                break
            # Pick a random block to flip
            start = np.random.randint(0, N - block_size)
            old_block = s[start:start+block_size].copy()
            s[start:start+block_size] *= -1
            A_new = _calc_acf(s)
            new_val = _max_abs_acf_from_array(A_new)
            delta = new_val - current_val
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-6)):
                current_val = new_val
                A = A_new
                if new_val < best_v:
                    best_v = new_val
                    best_s = s.copy()
            else:
                s[start:start+block_size] = old_block
            T *= cooling
        _try_sequence(best_s)
        return best_s

    # Main search loop
    n_inits = 20

    for init in range(n_inits):
        if time.time() > deadline:
            break

        # Random ±1 initialization
        s = np.random.randint(0, 2, size=N) * 2 - 1
        _try_sequence(s)

        # Smooth gradient descent on continuous relaxation
        s_cont = _smooth_gradient_descent(s.astype(np.float64), n_iters=500, lr=0.05, tau=0.5)
        
        # Stochastic rounding of continuous solution
        if time.time() < deadline - 30:
            s_rounded = _stochastic_rounding(s_cont, n_samples=50)
            if s_rounded is not None:
                s_polished = _greedy_polish(s_rounded, max_passes=30)
            else:
                s_polished = s

        # Block flip annealing from current best
        if time.time() < deadline - 20:
            s_annealed = _block_flip_anneal(best_seq, n_iters=300, block_size=4)

        # Greedy polish from best
        if time.time() < deadline - 10:
            s_polished = _greedy_polish(best_seq, max_passes=50)

    # Final intensive polish if time permits
    if time.time() < deadline - 15:
        s_final = _greedy_polish(best_seq, max_passes=200)
        # Try block annealing on the polished result
        if time.time() < deadline - 5:
            s_final2 = _block_flip_anneal(s_final, n_iters=200, block_size=8)
            _greedy_polish(s_final2, max_passes=100)

    return best_seq
