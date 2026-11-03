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

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin, evaluator kills at 150s

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

    def _mean_field_iterate(s, n_iters=200, beta=0.5, decay=0.999):
        """
        Deterministic mean-field / iterative thresholding.
        Interpret the desired flat spectrum as an Ising Hamiltonian with
        couplings derived from the inverse FFT of a target power spectrum.
        Each site's effective field is the sum of neighbor contributions;
        we threshold to get the new sign.
        """
        s = s.copy()
        # Target: flat power spectrum -> delta-like autocorrelation at lag 0 only.
        # Effective field for site i: sum_j J[i-j] * s[j] where J is the
        # "desired" kernel. For flat spectrum, J[k] ~ delta[k] (identity),
        # but we add a small long-range component to break degeneracy.
        #
        # Practical approach: use the current ACF to guide sign updates.
        # If ACF at lag k is too large, we want to reduce it by flipping
        # signs that contribute to that lag.
        #
        # Simpler mean-field: compute local field h_i = sum_j K[i-j]*s[j]
        # where K is a fixed kernel designed to produce flat spectrum.
        # For a flat spectrum, K should be approximately delta, but we add
        # a small negative feedback from high lags to suppress peaks.

        # Build a kernel that penalizes large autocorrelation at all lags
        # K[k] = 1 if k=0, -alpha/k^2 for k != 0 (power-law decay)
        alpha = beta
        kernel = np.zeros(N)
        kernel[0] = 1.0
        for k in range(1, N):
            kernel[k] = -alpha / (k ** 2)
            kernel[N - k] = kernel[k]  # symmetric

        for it in range(n_iters):
            if time.time() > deadline:
                break
            # Compute local field: h = s * kernel (circular convolution)
            F_s = np.fft.rfft(s.astype(np.float64))
            F_k = np.fft.rfft(kernel)
            h = np.fft.irfft(F_s * F_k, n=N)
            # Threshold: new sign = sign(h)
            new_s = np.sign(h)
            # Handle zeros: keep old sign
            new_s[new_s == 0] = s[new_s == 0]
            # Check for convergence
            if np.all(new_s == s):
                break
            s = new_s

        return s

    def _greedy_polish(s, max_passes=100):
        """Single-bit greedy descent."""
        s = s.copy()
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
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

    # Multiple random initializations, each followed by mean-field + polish
    n_inits = 50

    for init in range(n_inits):
        if time.time() > deadline:
            break

        # Random ±1 initialization
        s = np.random.randint(0, 2, size=N) * 2 - 1
        _try_sequence(s)

        # Mean-field iteration with varying beta
        beta = 0.3 + 0.7 * (init / max(1, n_inits - 1))
        s_mf = _mean_field_iterate(s, n_iters=300, beta=beta)
        _try_sequence(s_mf)

        # Greedy polish
        s_pol = _greedy_polish(s_mf, max_passes=50)

        # If we're doing well, we can do more polish
        if best_val < 10.0 and time.time() < deadline - 30:
            _greedy_polish(s_pol, max_passes=100)

    # Final: if we have time, do one more round of mean-field from best
    if time.time() < deadline - 10:
        s_final = _mean_field_iterate(best_seq, n_iters=500, beta=0.5)
        _try_sequence(s_final)
        _greedy_polish(s_final, max_passes=200)

    return best_seq
