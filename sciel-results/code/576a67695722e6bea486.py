"""Optimize step heights to minimize autocorrelation peak via multi-start gradient descent."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    
    Strategy:
    1. Use FFT-based fast convolution and gradient computation.
    2. Multi-start optimization with diverse initializations.
    3. Smooth surrogate (softmax) with analytic gradient for L-BFGS-B.
    4. Final coordinate descent polish.
    """
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin before 150s kill
    
    best_value = np.inf
    best_h = np.ones(N).astype(float)
    
    # FFT size for fast convolution/correlation
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
    fft_size = max(fft_size, 1024)
    
    def fast_convolve(h):
        """Compute full autoconvolution using FFT."""
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        return conv[:2*N-1]
    
    def fast_grad_lse(weights, h):
        """
        Compute grad_lse[i] = 2 * sum_{m=0}^{N-1} weights[m+i] * h[m]
        using FFT-based cross-correlation for O(N log N) speed.
        
        We want: g[i] = sum_{m=0}^{N-1} weights[i+m] * h[m] for i=0..N-1
        This is a valid-mode correlation. Using FFT:
        Pad weights and h to size >= len(weights)+len(h)-1 = 3N-2
        corr_full = ifft(fft(weights_padded) * conj(fft(h_padded)))
        The valid correlation is at indices [N-1 : 2N-2] of corr_full
        """
        # Pad to sufficient size
        pad_size = 1
        while pad_size < 3 * N - 2:
            pad_size *= 2
        pad_size = max(pad_size, 1024)
        
        w_padded = np.zeros(pad_size)
        h_padded = np.zeros(pad_size)
        w_padded[:len(weights)] = weights
        h_padded[:len(h)] = h
        
        W = np.fft.rfft(w_padded)
        H = np.fft.rfft(h_padded)
        
        # Cross-correlation: ifft(FFT(w) * conj(FFT(h)))
        corr_full = np.fft.irfft(W * np.conj(H), n=pad_size)
        
        # Valid correlation starts at index N-1
        corr_valid = corr_full[N-1:2*N-2]
        
        return 2.0 * corr_valid
    
    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = fast_convolve(h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid
    
    # First evaluate the seed
    val_seed, _ = evaluate(best_h)
    best_value = val_seed
    
    def smooth_objective(x, beta=80.0):
        """Smooth surrogate using log-sum-exp of conv entries."""
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = fast_convolve(h)
        max_c = np.max(conv)
        shifted = conv - max_c
        # Numerically stable log-sum-exp
        log_sum_exp = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
        val = abs(2.0 * N * log_sum_exp / (s * s))
        if not np.isfinite(val):
            return 1e10
        return val
    
    def smooth_grad(x, beta=80.0):
        """Analytic gradient of the smooth surrogate, FFT-accelerated."""
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return np.zeros_like(h)
        
        conv = fast_convolve(h)
        max_c = np.max(conv)
        shifted = conv - max_c
        exp_vals = np.exp(beta * shifted)
        weights = exp_vals / np.sum(exp_vals)
        
        # grad_lse[i] = 2 * sum_{m=0}^{N-1} weights[m+i] * h[m]
        grad_lse = fast_grad_lse(weights, h)
        
        # lse value
        lse = max_c + np.log(np.sum(exp_vals)) / beta
        f = 2.0 * N * lse / (s * s)
        
        # df_dh[i] = 2*N*grad_lse[i]/s^2 - 4*N*lse/s^3
        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
        
        if f >= 0:
            grad = df_dh
        else:
            grad = -df_dh
        
        return grad
    
    rng = np.random.RandomState(42)
    
    # --- Generate diverse initial points ---
    initial_points = []
    
    # Seed
    initial_points.append(np.ones(N).astype(float))
    
    # Random uniform
    for _ in range(20):
        initial_points.append(rng.uniform(-1, 1, N))
    
    # Bimodal: mostly zero with random spikes
    for _ in range(20):
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
    for freq in [1, 2, 3, 5, 7, 10]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        initial_points.append(np.cos(2 * np.pi * freq * t) + 0.1 * rng.normal(0, 1, N))
    
    # Multi-mode cosine combinations (inspired by frequency-domain search)
    for n_comp in [2, 3, 4, 5]:
        for _ in range(10):
            freqs = rng.randint(1, 20, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.3, 1.5, n_comp)
            h_multi = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_multi += a * np.cos(2 * np.pi * f * t + p)
            initial_points.append(h_multi)
    
    # Parameterized bump profiles
    i_arr = np.arange(N)
    for center in [N//4, N//2, 3*N//4]:
        for width in [15, 25, 40, 60, 80, 120, 160, 200]:
            h_bump = np.exp(-((i_arr - center)**2) / (2 * width**2))
            initial_points.append(h_bump)
            initial_points.append(-h_bump)
            # Double bump
            h_double = np.exp(-((i_arr - center)**2) / (2 * width**2)) - np.exp(-((i_arr - (N-1-center))**2) / (2 * width**2))
            initial_points.append(h_double)
            # Bump with offset
            h_off = np.exp(-((i_arr - center)**2) / (2 * width**2)) + 0.3 * np.ones(N)
            initial_points.append(h_off)
    
    # Random Fourier synthesis
    for n_comp in [3, 5, 7]:
        for _ in range(8):
            freqs = rng.randint(1, 30, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.5, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.cos(2 * np.pi * f * t + p)
            initial_points.append(h_fourier)
    
    # --- Optimize from each initial point using L-BFGS-B ---
    for init_h in initial_points:
        if time.time() - start_time > DEADLINE:
            break
        
        # Normalize to have non-zero sum
        s = np.sum(init_h)
        if abs(s) < 1e-10:
            init_h = init_h + 0.1
            s = np.sum(init_h)
        
        # Try multiple beta values for the smooth surrogate
        for beta in [50.0, 80.0, 120.0]:
            if time.time() - start_time > DEADLINE:
                break
            
            try:
                result = minimize(
                    lambda x: smooth_objective(x, beta=beta),
                    init_h,
                    jac=lambda x: smooth_grad(x, beta=beta),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 400, 'gtol': 1e-7}
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
        
        if best_value < 1.0:
            break
        
        # Also try exact evaluation of the initial point
        val_init, valid = evaluate(init_h)
        if valid and val_init < best_value:
            best_value = val_init
            best_h = init_h.copy()
    
    # --- Local polish using coordinate descent on the best solution ---
    if time.time() - start_time < DEADLINE - 5:
        h_curr = best_h.copy()
        lr = 0.05
        improved = True
        while improved and time.time() - start_time < DEADLINE - 2:
            improved = False
            for coord in rng.permutation(N):
                if time.time() - start_time > DEADLINE - 2:
                    break
                for delta in [lr, -lr, lr*0.5, -lr*0.5, lr*0.25, -lr*0.25, lr*0.125, -lr*0.125]:
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
                lr *= 0.7
                if lr < 1e-4:
                    break
    
    return best_h
# EVOLVE-BLOCK-END
