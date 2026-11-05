import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.fft import fft, ifft

N = 400

def construct():
    start_time = time.time()
    DEADLINE = 142

    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        return val

    best_value = np.inf
    best_h = np.ones(N)
    best_value = evaluate(best_h)

    def smooth_objective(x, beta=150.0):
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = np.convolve(h, h)
        m = np.max(conv)
        shifted = conv - m
        log_sum_exp = m + np.log(np.sum(np.exp(beta * shifted))) / beta
        val = abs(2.0 * N * log_sum_exp / (s * s))
        if not np.isfinite(val):
            return 1e10
        return val

    def smooth_grad(x, beta=150.0):
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return np.zeros_like(h)

        conv = np.convolve(h, h)
        m = np.max(conv)
        shifted = conv - m
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        M = m + np.log(sum_exp) / beta

        # Gradient of M w.r.t h using FFT-based correlation
        # grad_M = 2 * correlate(weights, h) cropped to N
        # Use FFT for efficiency
        n_conv = len(weights) + N - 1
        n_fft = 1 << (n_conv - 1).bit_length()
        H = fft(h, n=n_fft)
        W = fft(weights, n=n_fft)
        # correlate(weights, h) = ifft(fft(weights) * conj(fft(h)))
        corr_full = ifft(W * np.conj(H)).real
        # The full correlation has length n_conv, we need the part corresponding to indices 0..N-1
        # Actually, dC_k/dh_j: C_k = sum_i h_i h_{k-i}
        # grad_M_j = sum_k weights[k] * (h_{k-j} + h_{j-(k-j)})... 
        # Actually the standard result: grad_M = 2 * np.correlate(weights, h, 'full')[N-1:N-1+N]
        # Let's verify: np.correlate(a, v, 'full') at lag k gives sum_i a_{i+k} v_i
        # We want sum_k weights[k] * (dC_k/dh_j)
        # C_k = sum_i h_i h_{k-i}
        # dC_k/dh_j = h_{k-j} (if 0<=k-j<N) + h_{k-j} (if k-j==j? no)
        # Actually dC_k/dh_j = h_{k-j} + h_{j} if j == k-j? No.
        # C_k = h_j h_{k-j} + ... 
        # The term h_j appears when i=j: h_j h_{k-j}.
        # The term h_{k-j} appears when i=k-j: h_{k-j} h_j.
        # So dC_k/dh_j = h_{k-j} (from i=j) + h_j (from i=k-j) if k-j != j.
        # If k-j == j, then C_k has h_j^2, so dC_k/dh_j = 2 h_j.
        # In general, dC_k/dh_j = h_{k-j} + (h_j if k != 2j else 0)? No.
        # dC_k/dh_j = h_{k-j} + h_{k-j} if k-j==j? No.
        # Let's just use the numerical gradient or a simple loop since N=400.
        # But FFT is faster. Let's use the property:
        # grad_M_j = 2 * (weights * h)_j (cross correlation)
        # np.correlate(weights, h, 'full') gives r[k] = sum_i weights[i+k] h[i]
        # We want sum_k weights[k] h_{k-j}.
        # Let i = k-j, so k = i+j.
        # sum_i weights[i+j] h[i] = r[j] from correlate(weights, h, 'full')?
        # np.correlate(a, v, 'full')[k] = sum_n a[n+k] v[n]
        # So correlate(weights, h, 'full')[j] = sum_n weights[n+j] h[n]
        # This is exactly what we want for one part.
        # But we also need the case where h_j is the varying term.
        # Actually, the gradient of sum_k w_k C_k is 2 * correlate(w, h) cropped appropriately.
        
        # For safety and speed, let's use the loop since N=400 is small.
        # 400*400 = 160k operations per gradient call. With 1000 iterations, 160M ops. Might be slow in Python.
        # Let's use FFT properly.
        
        # The autocorrelation R = convolve(h, h). 
        # R = IFT( |FFT(h)|^2 )? No, that's circular.
        # For linear convolution, we can use FFT with zero-padding.
        
        # Let's just do the loop. It's safer.
        grad_M = np.zeros(N)
        for k in range(len(weights)):
            wk = weights[k]
            if abs(wk) < 1e-12:
                continue
            j_start = max(0, k - N + 1)
            j_end = min(N - 1, k)
            if j_start > j_end:
                continue
            idx = np.arange(j_start, j_end + 1)
            h_vals = h[k - idx]
            grad_M[idx] += 2.0 * wk * h_vals

        Q = 2.0 * N * M / (s * s)
        if abs(Q) < 1e-12:
            return np.zeros_like(h)

        grad_Q = (2.0 * N / (s * s)) * grad_M - (4.0 * N * M / (s * s * s)) * np.ones(N)
        return np.sign(Q) * grad_Q

    # Strategy: Chirp-like sinusoidal initialization + multi-start optimization
    # h(n) = A * cos(2*pi*f*n + 0.5*c*n^2) + DC
    # This should have flat spectrum and low autocorrelation peaks.

    rng = np.random.RandomState(123)

    inits = [np.ones(N)]

    # Generate chirp-based inits
    for _ in range(20):
        f = rng.uniform(0.5, 5.0)
        c = rng.uniform(-2.0, 2.0)
        dc = rng.uniform(-0.5, 0.5)
        n = np.arange(N)
        phase = 2 * np.pi * f * n + 0.5 * c * n * n
        h = np.cos(phase) + dc
        # Ensure non-zero sum
        if abs(np.sum(h)) < 0.1:
            h += 0.1
        inits.append(h)

    # Random inits
    for _ in range(10):
        h = rng.normal(0, 1, N)
        if abs(np.sum(h)) < 0.1:
            h += 0.1
        inits.append(h)

    for h0 in inits:
        if time.time() - start_time > DEADLINE:
            break
        try:
            res = minimize(
                smooth_objective,
                h0,
                jac=smooth_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_opt = res.x
            val_opt = evaluate(h_opt)
            if val_opt < best_value:
                best_value = val_opt
                best_h = h_opt
        except Exception:
            pass

    return best_h
