"""Model-and-optimize bootstrap: parameterize step heights and optimize the objective."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    DEADLINE = 148  # 2s safety margin before 150s kill
    
    best_value = np.inf
    best_h = np.ones(N).astype(float)
    
    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid
    
    val_seed, _ = evaluate(best_h)
    best_value = val_seed
    
    def smooth_objective(x, beta=80.0):
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = np.convolve(h, h)
        max_c = np.max(conv)
        shifted = conv - max_c
        # Numerically stable log-sum-exp
        log_sum_exp = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
        val = abs(2.0 * N * log_sum_exp / (s * s))
        if not np.isfinite(val):
            return 1e10
        return val
    
    # Precompute FFT size for fast convolution/correlation
    fft_size = 2 * N - 1
    
    def smooth_grad(x, beta=80.0):
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return np.zeros_like(h)
        
        conv = np.convolve(h, h)
        max_c = np.max(conv)
        shifted = conv - max_c
        weights = np.exp(beta * shifted)
        weights /= np.sum(weights)
        
        # grad_lse[i] = 2 * sum_{m=0}^{N-1} weights[m+i] * h[m] for i=0..N-1
        # This is cross-correlation: for each i, dot(weights[i:i+N], h)
        # Use FFT: correlate(weights, h) where correlate(a,b)[k] = sum_n a[n+k]*b[n]
        # We need sum_{m=0}^{N-1} weights[m+i] * h[m] = correlate(weights, h) at lag i
        # np.correlate(weights, h, mode='full') gives lags from -(N-1) to (2N-2-N+1)
        # Actually let's use FFT directly for clarity and speed
        # convolve(weights, flip(h)) gives sum_k weights[k] * h[flip_len-1-k]
        # Let me just use np.correlate which is clear
        # np.correlate(a, v, 'full')[k] = sum_n a[n+k] * v[n] for valid k
        # We want sum_m weights[m+i] * h[m] which is correlate(weights, h, 'full')
        # The full output has length len(weights)+len(h)-1 = 3N-2
        # correlate(weights, h, 'full')[i] = sum_n weights[n+i] * h[n] for n where both defined
        # For i in 0..N-1, n ranges 0..N-1-i... no that's not right
        # Let me think: correlate(a, v, 'full') has length la+lv-1
        # Index k corresponds to lag k-(lv-1) in some conventions
        # numpy docs: c_k = sum_n a_{n+k} * conj(v_n)
        # For real: c_k = sum_n a[n+k] * v[n]
        # k ranges from -(lv-1) to la-1 in the full output
        # The full output index j corresponds to k = j - (lv-1)
        # So for k=i (i>=0), j = i + lv - 1 = i + N - 1
        # We need i from 0 to N-1, so j from N-1 to 2N-2
        # len(weights) = 2N-1, len(h) = N, so full len = 3N-2
        # j goes from 0 to 3N-3
        # For i=0: j = N-1
        # For i=N-1: j = 2N-2
        # That's within range since 3N-3 >= 2N-2 for N>=1
        corr_full = np.correlate(weights, h, mode='full')
        # Extract the N values we need: indices N-1 through 2N-2
        grad_lse = 2.0 * corr_full[N-1:2*N-1]
        
        lse = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
        f = 2.0 * N * lse / (s * s)
        
        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
        
        if f >= 0:
            grad = df_dh
        else:
            grad = -df_dh
        
        return grad
    
    rng = np.random.RandomState(123)
    
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
        k = rng.randint(50, 350)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        initial_points.append(h)
    
    # Alternating signs
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    initial_points.append(h_alt)
    
    # Low-frequency cosines
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 4, 5, 7, 10, 15, 20]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        initial_points.append(np.sin(2 * np.pi * freq * t))
        initial_points.append(np.cos(2 * np.pi * freq * t) + 0.2 * rng.normal(0, 1, N))
    
    # Parameterized bump profiles
    i_arr = np.arange(N)
    for center in [N//8, N//4, N//2, 3*N//4, 7*N//8]:
        for width in [10, 20, 30, 50, 75, 100, 150, 200]:
            h_bump = np.exp(-((i_arr - center)**2) / (2 * width**2))
            initial_points.append(h_bump)
            initial_points.append(-h_bump)
            # Double bump with opposite signs
            h_double = np.exp(-((i_arr - center)**2) / (2 * width**2)) - np.exp(-((i_arr - (N-1-center))**2) / (2 * width**2))
            initial_points.append(h_double)
            # Bump with offset
            h_off = np.exp(-((i_arr - center)**2) / (2 * width**2)) + 0.5 * np.ones(N)
            initial_points.append(h_off)
            # Two bumps same sign
            h_two = np.exp(-((i_arr - center)**2) / (2 * width**2)) + np.exp(-((i_arr - (N-1-center))**2) / (2 * width**2))
            initial_points.append(h_two)
    
    # Random Fourier synthesis
    for n_comp in [3, 5, 8, 12]:
        for _ in range(8):
            freqs = rng.randint(1, 40, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.5, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.cos(2 * np.pi * f * t + p)
            initial_points.append(h_fourier)
    
    # Structured patterns: blocks of +1 and -1
    for block_size in [10, 20, 50, 100]:
        n_blocks = N // block_size
        for _ in range(5):
            h_blocks = np.zeros(N)
            for b in range(n_blocks):
                sign = rng.choice([-1, 1])
                h_blocks[b*block_size:(b+1)*block_size] = sign
            initial_points.append(h_blocks)
    
    # Smooth random via Gaussian filter
    from scipy.ndimage import gaussian_filter1d
    for _ in range(15):
        h_smooth = rng.uniform(-1, 1, N)
        h_smooth = gaussian_filter1d(h_smooth, sigma=rng.uniform(2, 20))
        initial_points.append(h_smooth)
    
    # Multi-mode cosine combinations (key insight from knowledge)
    t = np.linspace(0, 1, N, endpoint=False)
    for n_modes in [2, 3, 4, 5, 6, 8]:
        for _ in range(12):
            freqs = sorted(rng.randint(1, 50, n_modes))
            # Ensure distinct frequencies
            while len(set(freqs)) < n_modes:
                freqs = sorted(rng.randint(1, 50, n_modes))
            phases = rng.uniform(0, 2 * np.pi, n_modes)
            amps = rng.uniform(0.5, 2.0, n_modes)
            h_multi = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_multi += a * np.cos(2 * np.pi * f * t + p)
            initial_points.append(h_multi)
            # Also try with sine components
            h_multi_sin = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_multi_sin += a * np.sin(2 * np.pi * f * t + p)
            initial_points.append(h_multi_sin)
            # Mixed sin/cos
            h_mixed = np.zeros(N)
            for i, (f, p, a) in enumerate(zip(freqs, phases, amps)):
                if i % 2 == 0:
                    h_mixed += a * np.cos(2 * np.pi * f * t + p)
                else:
                    h_mixed += a * np.sin(2 * np.pi * f * t + p)
            initial_points.append(h_mixed)
    
    # Low-pass filtered random with various cutoffs
    for cutoff in [3, 5, 8, 12, 20, 30]:
        for _ in range(8):
            h_rand = rng.uniform(-1, 1, N)
            # Simple low-pass via FFT
            h_fft = np.fft.rfft(h_rand)
            h_fft[cutoff:] = 0
            h_lowpass = np.fft.irfft(h_fft, N)
            initial_points.append(h_lowpass)
    
    # Anti-aliasing patterns: designed to minimize autocorrelation peak
    # Try patterns with zero mean and specific spectral properties
    for _ in range(20):
        # Random phases at selected frequencies
        n_freqs = rng.randint(3, 15)
        freqs = rng.choice(np.arange(1, 100), size=n_freqs, replace=False)
        phases = rng.uniform(0, 2 * np.pi, n_freqs)
        h_anti = np.zeros(N)
        for f, p in zip(freqs, phases):
            h_anti += np.cos(2 * np.pi * f * t + p)
        # Normalize to have sum ~1
        s_anti = np.sum(h_anti)
        if abs(s_anti) > 1e-6:
            h_anti /= s_anti
        initial_points.append(h_anti)
    
    # --- Optimize from each initial point using L-BFGS-B ---
    # Sort initial points by their initial value to prioritize good starts
    init_values = []
    for init_h in initial_points:
        val_init, valid = evaluate(init_h)
        init_values.append((val_init, init_h))
    init_values.sort(key=lambda x: x[0])
    
    # Take top 120 initial points for optimization (faster gradient allows more)
    n_optimize = min(120, len(init_values))
    top_inits = [h for _, h in init_values[:n_optimize]]
    
    # Also include some diverse ones
    for h in init_values[n_optimize:n_optimize+20]:
        top_inits.append(h[1] if isinstance(h, tuple) else h)
    
    for init_h in top_inits:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Normalize to have non-zero sum
        s = np.sum(init_h)
        if abs(s) < 1e-10:
            init_h = init_h + 0.1
            s = np.sum(init_h)
        
        # Try to optimize with different betas
        for beta in [50.0, 80.0, 120.0, 160.0]:
            if time.time() - start_time > DEADLINE - 3:
                break
            
            try:
                result = minimize(
                    lambda x, b=beta: smooth_objective(x, b),
                    init_h,
                    jac=lambda x, b=beta: smooth_grad(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'maxfun': 800, 'gtol': 1e-8}
                )
                
                h_opt = result.x
                val_opt, valid = evaluate(h_opt)
                if valid and val_opt < best_value:
                    best_value = val_opt
                    best_h = h_opt.copy()
                    if best_value < 0.82:
                        break
            except Exception:
                pass
        
        # Also try exact evaluation of the initial point
        val_init, valid = evaluate(init_h)
        if valid and val_init < best_value:
            best_value = val_init
            best_h = init_h.copy()
    
    # --- Local polish using gradient-based optimization on exact objective ---
    if time.time() - start_time < DEADLINE - 5:
        # Use a higher beta for sharper max approximation and more iterations
        for beta in [150.0, 200.0, 300.0, 400.0]:
            if time.time() - start_time > DEADLINE - 3:
                break
            try:
                result = minimize(
                    lambda x, b=beta: smooth_objective(x, b),
                    best_h,
                    jac=lambda x, b=beta: smooth_grad(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 600, 'maxfun': 1200, 'gtol': 1e-9}
                )
                h_opt = result.x
                val_opt, valid = evaluate(h_opt)
                if valid and val_opt < best_value:
                    best_value = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # --- Second polish: try from top-5 initial solutions that weren't the best ---
    if time.time() - start_time < DEADLINE - 4:
        for init_h in init_values[1:6]:
            if time.time() - start_time > DEADLINE - 3:
                break
            h_start = init_h[1] if isinstance(init_h, tuple) else init_h
            s = np.sum(h_start)
            if abs(s) < 1e-10:
                h_start = h_start + 0.1
            for beta in [200.0, 300.0]:
                if time.time() - start_time > DEADLINE - 3:
                    break
                try:
                    result = minimize(
                        lambda x, b=beta: smooth_objective(x, b),
                        h_start,
                        jac=lambda x, b=beta: smooth_grad(x, b),
                        method='L-BFGS-B',
                        options={'maxiter': 400, 'maxfun': 800, 'gtol': 1e-9}
                    )
                    h_opt = result.x
                    val_opt, valid = evaluate(h_opt)
                    if valid and val_opt < best_value:
                        best_value = val_opt
                        best_h = h_opt.copy()
                except Exception:
                    pass
    
    # --- Final: try random perturbations of the best solution ---
    if time.time() - start_time < DEADLINE - 2:
        for _ in range(80):
            if time.time() - start_time > DEADLINE - 2:
                break
            # Use smaller perturbations as we get closer to deadline
            scale = 0.02 if time.time() - start_time > DEADLINE - 5 else 0.01
            h_perturb = best_h + rng.normal(0, scale, N)
            val_p, valid_p = evaluate(h_perturb)
            if valid_p and val_p < best_value:
                best_value = val_p
                best_h = h_perturb.copy()
                # Quick local refinement from improved point
                if time.time() - start_time < DEADLINE - 2:
                    try:
                        result = minimize(
                            lambda x: smooth_objective(x, 300.0),
                            best_h,
                            jac=lambda x: smooth_grad(x, 300.0),
                            method='L-BFGS-B',
                            options={'maxiter': 300, 'maxfun': 600, 'gtol': 1e-9}
                        )
                        h_opt = result.x
                        val_opt, valid = evaluate(h_opt)
                        if valid and val_opt < best_value:
                            best_value = val_opt
                            best_h = h_opt.copy()
                    except Exception:
                        pass
    
    return best_h
# EVOLVE-BLOCK-END
