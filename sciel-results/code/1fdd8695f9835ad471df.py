"""
Autocorrelation inequality optimizer.
Strategy: Spectral parameterization with optimized power spectrum,
combined with direct optimization in time domain.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    best_value = np.inf
    best_h = np.ones(N).astype(float)

    # FFT size for fast convolution
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
    fft_size = max(fft_size, 1024)

    def fast_convolve(h):
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        return conv[:2 * N - 1]

    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = fast_convolve(h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid

    val_seed, _ = evaluate(best_h)
    best_value = val_seed

    # --- Spectral approach: parameterize h via its DFT ---
    # The autoconvolution in time domain corresponds to |H|^2 in frequency domain.
    # Specifically, if H = DFT(h), then DFT(h*h) = H^2 (up to scaling).
    # We want to minimize max(h*h) = max of inverse DFT of H^2.
    # 
    # Key insight: if we make H have a specific structure, we can control the autoconvolution.
    # For instance, if H is constant (h is a delta), the autoconvolution is a delta.
    # If H has a null at DC (sum h = 0), we can't use it directly since sum must be nonzero.
    # But we can have H[0] = c (small) and other components designed to minimize the peak.

    # Approach: parameterize H[k] for k=1,...,N-1 (and H[0] separately)
    # and optimize to minimize the peak of the autoconvolution.

    # Use a smooth surrogate: minimize log-sum-exp of |autoconv| entries
    def spectral_objective(params):
        """
        params: array of length N
        params[0] = log(|H[0]|)
        params[1:N] = log(|H[k]|) for k=1,...,N-1
        We fix the phases to be zero for simplicity (real-valued h requires H[N-k]=conj(H[k])).
        """
        if time.time() - start_time > DEADLINE:
            return 1e10

        # Reconstruct H
        log_mags = params
        mags = np.exp(log_mags)
        
        # H is the real FFT: H[0], H[1], ..., H[N//2]
        # For real h: H[N-k] = conj(H[k])
        # rfft gives N//2+1 values
        n_rfft = N // 2 + 1
        
        # Map params to rfft magnitudes
        # params[0] -> H[0]
        # params[1] -> H[1], params[2] -> H[2], ..., params[N//2] -> H[N//2]
        # For k > N//2, H[k] = H[N-k] (for real signal)
        
        H_mags = np.zeros(n_rfft)
        H_mags[0] = mags[0]
        for k in range(1, n_rfft):
            if k < N:
                H_mags[k] = mags[k]
            else:
                H_mags[k] = 0
        
        # Set phases to zero (all positive) for now
        H = H_mags.copy()
        
        # Reconstruct h
        h = np.fft.irfft(H, n=N)
        
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        
        conv = fast_convolve(h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        
        if not np.isfinite(val):
            return 1e10
        return val

    # --- Direct optimization with better initialization ---
    rng = np.random.RandomState(42)

    def smooth_objective(x, beta=50.0):
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = fast_convolve(h)
        abs_conv = np.abs(conv)
        max_c = np.max(abs_conv)
        shifted = abs_conv - max_c
        log_sum_exp = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
        val = abs(2.0 * N * log_sum_exp / (s * s))
        if not np.isfinite(val):
            return 1e10
        return val

    def smooth_grad(x, beta=50.0):
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return np.zeros_like(h)
        
        conv = fast_convolve(h)
        abs_conv = np.abs(conv)
        max_c = np.max(abs_conv)
        shifted = abs_conv - max_c
        exp_vals = np.exp(beta * shifted)
        weights = exp_vals / np.sum(exp_vals)
        
        w_signed = weights * np.sign(conv)
        
        fft_size_corr = 1
        while fft_size_corr < (2 * N - 1) + N - 1:
            fft_size_corr *= 2
        
        h_rev = h[::-1]
        W = np.fft.rfft(w_signed, n=fft_size_corr)
        Hrev = np.fft.rfft(h_rev, n=fft_size_corr)
        conv_full = np.fft.irfft(W * Hrev, n=fft_size_corr)
        
        g = conv_full[N - 1: 2 * N - 1]
        grad_lse = 2.0 * g
        
        lse = max_c + np.log(np.sum(exp_vals)) / beta
        f = 2.0 * N * lse / (s * s)
        
        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
        
        if f >= 0:
            grad = df_dh
        else:
            grad = -df_dh
        
        return grad

    # Generate diverse initial points
    initial_points = []
    
    # Seed
    initial_points.append(np.ones(N).astype(float))
    
    # Random uniform
    for _ in range(30):
        initial_points.append(rng.uniform(-1, 1, N))
    
    # Bimodal
    for _ in range(30):
        h = np.zeros(N)
        k = rng.randint(50, 300)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        initial_points.append(h)
    
    # Alternating signs
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    initial_points.append(h_alt)
    
    # Low-frequency cosines
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        initial_points.append(np.sin(2 * np.pi * freq * t))
    
    # Parameterized bump profiles
    i_arr = np.arange(N)
    for center in [N // 4, N // 2, 3 * N // 4]:
        for width in [15, 25, 40, 60, 80, 120, 160, 200]:
            h_bump = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2))
            initial_points.append(h_bump)
            initial_points.append(-h_bump)
            h_double = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2)) - \
                       np.exp(-((i_arr - (N - 1 - center)) ** 2) / (2 * width ** 2))
            initial_points.append(h_double)
    
    # Random Fourier synthesis
    for n_comp in [3, 5, 7, 10, 15, 20, 30]:
        for _ in range(10):
            freqs = rng.randint(1, 60, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.3, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.sin(2 * np.pi * f * t + p)
            if abs(np.sum(h_fourier)) < 0.1:
                h_fourier += 0.05
            initial_points.append(h_fourier)
    
    # Mixed sine/cosine spectral constructions
    for n_comp in [5, 10, 15, 20, 30]:
        for _ in range(8):
            freqs = rng.randint(1, 80, n_comp)
            h_spec = np.zeros(N)
            for f in freqs:
                a_s = rng.uniform(0.2, 1.5)
                a_c = rng.uniform(0.2, 1.5)
                p_s = rng.uniform(0, 2 * np.pi)
                p_c = rng.uniform(0, 2 * np.pi)
                h_spec += a_s * np.sin(2 * np.pi * f * t + p_s)
                h_spec += a_c * np.cos(2 * np.pi * f * t + p_c)
            if abs(np.sum(h_spec)) < 0.1:
                h_spec += 0.03
            initial_points.append(h_spec)
    
    # Anti-symmetric patterns
    from scipy.ndimage import gaussian_filter1d
    for width in [20, 40, 60, 80, 100]:
        h_anti = np.zeros(N)
        h_anti[:N // 2] = 1.0
        h_anti[N // 2:] = -1.0
        h_smooth = gaussian_filter1d(h_anti, sigma=width / 4.0)
        initial_points.append(h_smooth)
        initial_points.append(-h_smooth)
    
    # Spectral designs: optimize the power spectrum
    # We want H[k] such that the autoconvolution peak is minimized.
    # Try: H[0] = c, H[k] = a * exp(-b*k^2) for k>=1
    for c in [0.1, 0.5, 1.0, 2.0]:
        for a in [0.1, 0.5, 1.0]:
            for b in [0.01, 0.05, 0.1, 0.2, 0.5]:
                H_spec = np.zeros(N // 2 + 1)
                H_spec[0] = c
                for k in range(1, N // 2 + 1):
                    H_spec[k] = a * np.exp(-b * k ** 2)
                h_spec = np.fft.irfft(H_spec, n=N)
                s = np.sum(h_spec)
                if abs(s) > 1e-10:
                    initial_points.append(h_spec)
    
    # Try flat spectrum with random phases
    for _ in range(50):
        H_flat = np.ones(N // 2 + 1) * rng.uniform(0.5, 2.0)
        # Random phases
        phases = rng.uniform(0, 2 * np.pi, N // 2 + 1)
        H_complex = H_flat * np.exp(1j * phases)
        H_complex[0] = abs(H_complex[0])  # DC is real
        # Make conjugate symmetric
        h_flat = np.fft.irfft(H_complex, n=N)
        s = np.sum(h_flat)
        if abs(s) > 1e-10:
            initial_points.append(h_flat)
    
    # --- Optimize from each initial point ---
    for init_h in initial_points:
        if time.time() - start_time > DEADLINE:
            break
        
        s = np.sum(init_h)
        if abs(s) < 1e-10:
            init_h = init_h + 0.1
            s = np.sum(init_h)
        
        # Evaluate initial point
        val_init, valid = evaluate(init_h)
        if valid and val_init < best_value:
            best_value = val_init
            best_h = init_h.copy()
        
        # Try L-BFGS-B with different betas
        for beta in [20.0, 40.0, 80.0, 120.0]:
            if time.time() - start_time > DEADLINE:
                break
            
            try:
                result = minimize(
                    lambda x, b=beta: smooth_objective(x, b),
                    init_h,
                    jac=lambda x, b=beta: smooth_grad(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'maxfun': 600, 'gtol': 1e-7}
                )
                
                h_opt = result.x
                val_opt, valid = evaluate(h_opt)
                if valid and val_opt < best_value:
                    best_value = val_opt
                    best_h = h_opt.copy()
                    if best_value < 1.0:
                        break
            except Exception:
                pass
    
    # --- Local polish ---
    if time.time() - start_time < DEADLINE - 5:
        h_curr = best_h.copy()
        lr = 0.1
        improved = True
        polish_iter = 0
        while improved and time.time() - start_time < DEADLINE - 2 and polish_iter < 20:
            improved = False
            polish_iter += 1
            for coord in rng.permutation(N):
                if time.time() - start_time > DEADLINE - 2:
                    break
                for delta in [lr, -lr, lr * 0.5, -lr * 0.5, lr * 0.25, -lr * 0.25]:
                    h_try = h_curr.copy()
                    h_try[coord] += delta
                    val_try, valid = evaluate(h_try)
                    if valid and val_try < best_value:
                        best_value = val_try
                        best_h = h_try
                        h_curr = h_try
                        improved = True
                        break
            if not improved:
                lr *= 0.6
                if lr < 1e-5:
                    break
    
    return best_h
