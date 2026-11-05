import numpy as np
import time
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

N = 400

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
    
    # Precompute FFT size
    fft_size = 2 * N - 1
    
    def smooth_objective(x, beta=80.0):
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = np.convolve(h, h)
        max_c = np.max(conv)
        shifted = conv - max_c
        log_sum_exp = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
        val = abs(2.0 * N * log_sum_exp / (s * s))
        if not np.isfinite(val):
            return 1e10
        return val
    
    def smooth_grad(x, beta=80.0):
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return np.zeros_like(h)
        
        conv = np.convolve(h, h)
        max_c = np.max(conv)
        shifted = conv - max_c
        weights = np.exp(beta * shifted)
        sum_weights = np.sum(weights)
        weights /= sum_weights
        
        # Compute gradient of log-sum-exp term
        # grad_lse[i] = beta * (sum_{m>=0} w[m+i] h[m] + sum_{n>=0} w[i-n] h[n])
        # This is related to cross-correlation.
        # Let's use FFT.
        # w is length 2N-1. h is length N.
        # We need to compute for each i in 0..N-1:
        # term1[i] = sum_{m=0}^{N-1} w[m+i] h[m]
        # term2[i] = sum_{n=0}^{N-1} w[i-n] h[n]
        
        # Pad w and h to size fft_size
        w_pad = np.zeros(fft_size)
        w_pad[:len(w)] = w
        h_pad = np.zeros(fft_size)
        h_pad[:N] = h
        
        # term1 is convolution of w and h?
        # c1[i] = sum_m w[m+i] h[m].
        # This is the cross-correlation of w and h at lag i?
        # np.correlate(w, h, 'valid')? No, lengths differ.
        # Let's use FFT.
        # FFT(w_pad) * FFT(h_pad) gives linear convolution.
        # conv_wh = ifft(fft(w_pad) * fft(h_pad))
        # conv_wh[k] = sum_j w[j] h[k-j].
        # We want sum_m w[m+i] h[m]. Let j=m+i. sum_j w[j] h[j-i].
        # This is conv_wh[i]?
        # conv_wh[i] = sum_j w[j] h[i-j].
        # If we set k=i-j, then j=i-k. sum_k w[i-k] h[k].
        # This is exactly term2[i] if w and h were symmetric?
        
        # Let's just compute term1 and term2 using the convolution result.
        # conv_wh = np.convolve(w, h, 'full')
        # conv_wh[k] = sum_j w[j] h[k-j].
        # term1[i] = sum_m w[m+i] h[m]. Let j=m+i. sum_j w[j] h[j-i].
        # This is conv_wh[i] if we define conv_wh[i] = sum_j w[j] h[i-j]?
        # No, np.convolve(a, b)[k] = sum_j a[j] b[k-j].
        # So conv_wh[i] = sum_j w[j] h[i-j].
        # We want sum_m w[m+i] h[m].
        # Let j = m+i. Then m = j-i.
        # sum_j w[j] h[j-i].
        # This is the same form as conv_wh[i] but with h[j-i] instead of h[i-j].
        # If h is symmetric, it's the same.
        
        # Let's use the property that term1[i] + term2[i] is the gradient.
        # Actually, grad_lse[i] = beta * (term1[i] + term2[i]).
        # term2[i] = sum_n w[i-n] h[n]. Let k=i-n. sum_k w[k] h[i-k].
        # This IS conv_wh[i].
        # So term2[i] = conv_wh[i].
        # What is term1[i]?
        # term1[i] = sum_m w[m+i] h[m].
        # Let's check if term1[i] = conv_wh[2N-2-i]?
        # conv_wh[2N-2-i] = sum_j w[j] h[2N-2-i-j].
        # Let m = 2N-2-i-j. Then j = 2N-2-i-m.
        # sum_m w[2N-2-i-m] h[m].
        # This is not term1.
        
        # Let's just compute term1 and term2 separately using FFT or direct sum if N is small.
        # N=400. Direct sum is 400*400 = 160k ops. Very fast.
        # But we can do better.
        
        # Let's use np.correlate.
        # term2[i] = np.correlate(w, h, 'valid')?
        # np.correlate(a, v)[k] = sum_n a[n+k] v[n].
        # So np.correlate(w, h, 'valid')[i] = sum_n w[n+i] h[n].
        # This is exactly term1[i]!
        # So term1 = np.correlate(w, h, 'valid')
        
        # And term2[i] = sum_n w[i-n] h[n].
        # This is np.correlate(h, w, 'valid')?
        # np.correlate(h, w)[i] = sum_n h[n+i] w[n].
        # No.
        # term2[i] = sum_k w[k] h[i-k]. This is convolution.
        # conv_wh = np.convolve(w, h, 'full')
        # term2 = conv_wh[:N]?
        # conv_wh[i] = sum_j w[j] h[i-j].
        # For i < N, h[i-j] is non-zero for j in [i-N+1, i].
        # w[j] is non-zero for j in [0, 2N-2].
        # So yes, term2[i] = conv_wh[i].
        
        term1 = np.correlate(w, h, 'valid')
        conv_wh = np.convolve(w, h, 'full')
        term2 = conv_wh[:N]
        
        grad_lse = beta * (term1 + term2)
        
        lse = max_c + np.log(sum_weights) / beta
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
    for _ in range(15):
        h_smooth = rng.uniform(-1, 1, N)
        h_smooth = gaussian_filter1d(h_smooth, sigma=rng.uniform(2, 20))
        initial_points.append(h_smooth)
    
    # --- Optimize from each initial point using L-BFGS-B ---
    # Sort initial points by their initial value to prioritize good starts
    init_values = []
    for init_h in initial_points:
        val_init, valid = evaluate(init_h)
        init_values.append((val_init, init_h))
    init_values.sort(key=lambda x: x[0])
    
    # Take top 80 initial points for optimization
    top_inits = [h for _, h in init_values[:80]]
    
    # Also include some random ones for diversity
    for h in initial_points[80:100]:
        top_inits.append(h)
    
    for init_h in top_inits:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Normalize to have non-zero sum
        s = np.sum(init_h)
        if abs(s) < 1e-10:
            init_h = init_h + 0.1
            s = np.sum(init_h)
        
        # Try to optimize with different betas
        for beta in [50.0, 80.0, 120.0]:
            if time.time() - start_time > DEADLINE - 3:
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
                    if best_value < 1.05:
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
        for beta in [150.0, 200.0, 300.0]:
            if time.time() - start_time > DEADLINE - 3:
                break
            try:
                result = minimize(
                    lambda x, b=beta: smooth_objective(x, b),
                    best_h,
                    jac=lambda x, b=beta: smooth_grad(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'maxfun': 1000, 'gtol': 1e-8}
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
        for _ in range(50):
            if time.time() - start_time > DEADLINE - 2:
                break
            h_perturb = best_h + rng.normal(0, 0.03, N)
            val_p, valid_p = evaluate(h_perturb)
            if valid_p and val_p < best_value:
                best_value = val_p
                best_h = h_perturb.copy()
                # Quick local refinement from improved point
                if time.time() - start_time < DEADLINE - 2:
                    try:
                        result = minimize(
                            lambda x: smooth_objective(x, 200.0),
                            best_h,
                            jac=lambda x: smooth_grad(x, 200.0),
                            method='L-BFGS-B',
                            options={'maxiter': 200, 'maxfun': 400, 'gtol': 1e-7}
                        )
                        h_opt = result.x
                        val_opt, valid = evaluate(h_opt)
                        if valid and val_opt < best_value:
                            best_value = val_opt
                            best_h = h_opt.copy()
                    except Exception:
                        pass
    
    return best_h
