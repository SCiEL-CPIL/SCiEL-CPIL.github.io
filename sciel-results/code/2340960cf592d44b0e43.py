"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 600

# EVOLVE-BLOCK-START

def _autoconv_peak(h):
    """Return the normalized maximum autoconvolution of h (lower is better)."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    s = h.sum()
    if s < 1e-15:
        return 1e10
    hn = h / s
    fft_size = 2 * N
    F = np.fft.rfft(hn, fft_size)
    conv = np.fft.irfft(F * F, fft_size)[:2 * N - 1]
    return float(conv.max())

def _autoconv_l1(h):
    """Return the L1 norm of the normalized autoconvolution of h."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    s = h.sum()
    if s < 1e-15:
        return 1e10
    hn = h / s
    fft_size = 2 * N
    F = np.fft.rfft(hn, fft_size)
    conv = np.fft.irfft(F * F, fft_size)[:2 * N - 1]
    return float(np.sum(np.abs(conv)))

def _autoconv(h):
    """Return the full normalized autoconvolution array."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    s = h.sum()
    if s < 1e-15:
        return np.zeros(2 * N - 1)
    hn = h / s
    fft_size = 2 * N
    F = np.fft.rfft(hn, fft_size)
    conv = np.fft.irfft(F * F, fft_size)[:2 * N - 1]
    return conv

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def time_left():
        return deadline - time.time()

    # Phase 1: Generate diverse structured candidates
    candidates = []

    # Uniform
    candidates.append(np.ones(N))

    # Triangular
    center = N // 2
    tri = np.array([max(0.0, 1.0 - abs(i - center) / center) for i in range(N)])
    candidates.append(tri)

    # Gaussian
    x = np.linspace(-3, 3, N)
    candidates.append(np.exp(-x**2))

    # Raised cosine
    rc = 0.5 * (1 + np.cos(np.pi * np.linspace(-1, 1, N)))
    candidates.append(rc)

    # Inverse Gaussian (peaks at edges)
    inv_gauss = 1.0 - 0.3 * np.exp(-3.0 * np.linspace(-1, 1, N)**2)
    candidates.append(np.maximum(inv_gauss, 0))

    # Two-peak
    two = np.zeros(N)
    two[N//4] = 1.0
    two[3*N//4] = 1.0
    candidates.append(two)

    # Plateaus
    for w_frac in [0.3, 0.5, 0.7, 0.9]:
        w = int(N * w_frac)
        plat = np.zeros(N)
        plat[N//2 - w//2 : N//2 + w//2 + 1] = 1.0
        candidates.append(plat)

    # Ramp
    ramp = np.linspace(1, 0, N)
    candidates.append(np.maximum(ramp, 0))

    # Evaluate all candidates
    h_best = np.ones(N)
    best_val = _autoconv_peak(h_best)

    for c in candidates:
        if time_left() < 10:
            break
        c = np.maximum(c, 0)
        if c.sum() < 1e-15:
            continue
        val = _autoconv_peak(c)
        if val < best_val:
            best_val = val
            h_best = c.copy()

    # Phase 2: Iterative Proportional Fitting (IPF) to minimize L1 of autoconvolution
    # Model h as a probability distribution; IPF adjusts h to reduce overall spread
    # of the convolution mass, which indirectly lowers the peak.
    #
    # Strategy: Use the autoconvolution itself as a target to fit.
    # We want h such that its autoconvolution is as flat as possible.
    # IPF step: h_new[i] = h[i] * (target[i] / current_conv[i])^{eta}
    # where target is a flat distribution and eta is a relaxation parameter.
    
    h_ipf = h_best.copy()
    h_ipf = np.maximum(h_ipf, 1e-10)
    h_ipf = h_ipf / h_ipf.sum()
    
    best_ipf_val = _autoconv_peak(h_ipf)
    h_ipf_best = h_ipf.copy()
    
    eta_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    
    for eta in eta_values:
        if time_left() < 15:
            break
        h_curr = h_ipf.copy()
        for iteration in range(200):
            if time_left() < 10:
                break
            conv = _autoconv(h_curr)
            # Avoid division by zero
            conv_safe = np.maximum(conv, 1e-15)
            # Target: uniform distribution over the convolution support
            target = np.ones_like(conv) / len(conv)
            # IPF update
            ratio = target / conv_safe
            h_new = h_curr * np.power(ratio, eta)
            h_new = np.maximum(h_new, 0)
            if h_new.sum() < 1e-15:
                break
            h_new = h_new / h_new.sum()
            
            val = _autoconv_peak(h_new)
            if val < best_ipf_val:
                best_ipf_val = val
                h_ipf_best = h_new.copy()
            
            # Check convergence
            if np.max(np.abs(h_new - h_curr)) < 1e-10:
                break
            h_curr = h_new
        
        h_ipf = h_curr
    
    if best_ipf_val < best_val:
        best_val = best_ipf_val
        h_best = h_ipf_best.copy()

    # Phase 3: Log-parameterized coordinate descent for peak refinement
    # h_i = exp(x_i) / sum(exp(x_j))
    def softmax(x):
        x_shifted = x - np.max(x)
        e = np.exp(x_shifted)
        return e / e.sum()

    h_init = h_best.copy()
    h_init = np.maximum(h_init, 1e-10)
    x0 = np.log(h_init / h_init.sum())
    
    x = x0.copy()
    current_val = _autoconv_peak(softmax(x))
    
    step_sizes = [0.01, 0.05, 0.1, 0.2]
    
    for epoch in range(3):
        if time_left() < 20:
            break
        for _ in range(3):  # Passes
            if time_left() < 10:
                break
            for i in range(N):
                if time_left() < 5:
                    break
                improved = False
                for step in step_sizes:
                    for sign in [1, -1]:
                        x_trial = x.copy()
                        x_trial[i] += sign * step
                        h_trial = softmax(x_trial)
                        val_trial = _autoconv_peak(h_trial)
                        if val_trial < current_val:
                            x = x_trial
                            current_val = val_trial
                            improved = True
                            if current_val < best_val:
                                best_val = current_val
                                h_best = h_trial.copy()
                if time_left() < 5:
                    break

    # Phase 4: Cosine series refinement
    i = np.arange(N)
    M_max = 8
    h_cos = h_best.copy()
    
    for M in range(1, M_max + 1):
        if time_left() < 5:
            break
        cos_basis = np.column_stack([np.cos(2 * np.pi * k * i / N) for k in range(1, M + 1)])
        for k in range(M):
            if time_left() < 3:
                break
            for amp in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
                for sign in [1, -1]:
                    h_test = h_cos + sign * amp * cos_basis[:, k]
                    h_test = np.maximum(h_test, 0)
                    if h_test.sum() < 1e-15:
                        continue
                    val = _autoconv_peak(h_test)
                    if val < best_val:
                        best_val = val
                        h_best = h_test.copy()
                        h_cos = h_test.copy()

    # Final: ensure nonnegative and return
    h_best = np.maximum(h_best, 0.0)
    if h_best.sum() < 1e-15:
        h_best = np.ones(N)
    
    return list(h_best)

# EVOLVE-BLOCK-END
