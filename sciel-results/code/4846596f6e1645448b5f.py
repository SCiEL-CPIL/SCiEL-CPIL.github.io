"""
Autocorrelation inequality optimizer.
Minimizes |2*N*max(conv(h,h))/sum(h)^2| over N=400 step heights.
Strategy:
1. Global search in Fourier coefficient space via Differential Evolution.
2. Local refinement via L-BFGS-B with log-sum-exp smoothed surrogate (multi-beta annealing).
3. Final polish via exact subgradient on the true max.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.ndimage import gaussian_filter1d

N = 400

def construct():
    start_time = time.time()
    DEADLINE = 143  # 7s safety margin before 150s kill

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

    # Precompute FFT size for autoconvolution
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    def fast_convolve(h):
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        return conv[:2 * N - 1]

    # For gradient: we need sum over m of w[m] * h[m + N - 1 - lag]
    # This is a cross-correlation between w (length 2N-1) and h (length N)
    # Precompute the FFT size for this
    fft_size_cc = 1
    while fft_size_cc < (2 * N - 1) + N - 1:
        fft_size_cc *= 2

    def fast_grad_lse(weights, h):
        """
        Compute grad_lse[i] = 2 * sum_{m} weights[m] * h[m + N - 1 - i] for i=0..N-1.
        This is 2 * (weights cross-correlated with h)[N-1 : 2N-1]
        """
        # cross-correlation: c[k] = sum_j a[j] * b[j+k]
        # We want: for each i, sum_m weights[m] * h[m + (N-1-i)]
        # Let d = N-1-i, so we need sum_m weights[m] * h[m+d] for d = 0, 1, ..., N-1
        # This is the cross-correlation of weights with h at lags 0..N-1
        
        # Using FFT: correlate(weights, h) via convolution with reversed h
        h_rev = h[::-1]
        H_rev = np.fft.rfft(h_rev, n=fft_size_cc)
        W = np.fft.rfft(weights, n=fft_size_cc)
        conv_result = np.fft.irfft(H_rev * W, n=fft_size_cc)
        
        # conv_result[k] = sum_j h_rev[j] * weights[k-j] = sum_j h[N-1-j] * weights[k-j]
        # We want sum_m weights[m] * h[m+d] = sum_m weights[m] * h[m+d]
        # Let's think differently.
        # sum_m weights[m] * h[m+d] for d = N-1-i
        # = sum_m weights[m] * h[m + N - 1 - i]
        
        # Alternative: use np.correlate-style via FFT
        # conv of weights and h: (w * h)[k] = sum_j w[j] * h[k-j]
        # We want sum_m w[m] * h[m+d] = sum_m w[m] * h[m+d]
        # Let j = m, then = sum_j w[j] * h[j+d]
        # This is the cross-correlation at lag -d (or d depending on convention)
        
        # Simplest: just use the convolution result properly
        # (w * h_rev)[k] = sum_j w[j] * h_rev[k-j] = sum_j w[j] * h[N-1-(k-j)] = sum_j w[j] * h[N-1-k+j]
        # We want sum_j w[j] * h[j+d] where d = N-1-i
        # So N-1-k+j = j+d => N-1-k = d => k = N-1-d = N-1-(N-1-i) = i
        # So (w * h_rev)[i] = sum_j w[j] * h[j + N-1-i] for the range where indices are valid
        
        # Actually let me just verify: 
        # conv_result[k] = sum_j h_rev[j] * W[k-j] = sum_j h[N-1-j] * weights[k-j]
        # Let m = k-j, j = k-m: sum_m h[N-1-(k-m)] * weights[m] = sum_m weights[m] * h[N-1-k+m]
        # We want sum_m weights[m] * h[m+d] where d = N-1-i
        # So N-1-k+m = m+d => N-1-k = d => k = N-1-d = N-1-(N-1-i) = i
        # So conv_result[i] = sum_m weights[m] * h[m + N-1-i] for i in valid range
        
        # The valid range: we need 0 <= m < 2N-1 and 0 <= N-1-k+m < N
        # For k=i (0<=i<N): N-1-i+m ranges from N-1-i to N-1-i+2N-2
        # We need N-1-i+m >= 0 (always true for m>=0, i<N) and N-1-i+m <= N-1
        # So m <= i. Also m >= 0.
        # Hmm, this is getting complicated. Let me just use a direct approach for correctness.
        
        # Actually, the cleanest way: 
        # grad_lse[i] = 2 * sum_m weights[m] * h[m + N - 1 - i]
        # For this to be nonzero, we need 0 <= m < 2N-1 and 0 <= m+N-1-i < N
        # => 0 <= m and m < 2N-1 and i-N+1 <= m <= i
        
        # Let me just use the FFT approach correctly:
        # We want c[i] = sum_m w[m] * h[m + d_i] where d_i = N-1-i
        # This is the cross-correlation (w ⋆ h)[d_i] but with specific indexing
        
        # Let's use: if we define v[m] = w[m] for m in 0..2N-2, and extend h to be 0 outside 0..N-1
        # Then c[i] = sum_m v[m] * h[m + N - 1 - i]
        # = sum_m v[m] * h_ext[m + N - 1 - i] where h_ext is h zero-padded
        
        # This is (v * h_rev_padded)[N-1-i] where h_rev_padded is h reversed and zero-padded
        # Let me just compute it via the convolution I already have
        
        # conv_result[k] = sum_j h_rev[j] * weights[k-j] where h_rev[j] = h[N-1-j]
        # = sum_j weights[k-j] * h[N-1-j]
        # Let m = k-j: = sum_m weights[m] * h[N-1-(k-m)] = sum_m weights[m] * h[N-1-k+m]
        # We want: sum_m weights[m] * h[m+N-1-i]
        # So N-1-k+m = m+N-1-i => k = i
        # Therefore conv_result[i] = sum_m weights[m] * h[m+N-1-i] ✓
        
        # But we need to be careful about the valid range of m
        # The sum is over all m where both weights[m] and h[N-1-k+m] are defined
        # weights has length 2N-1 (indices 0..2N-2)
        # h has length N (indices 0..N-1), so we need 0 <= N-1-i+m < N => i-N+1 <= m < i+1
        # Combined with 0 <= m < 2N-1
        
        # The FFT convolution handles this automatically via zero-padding
        # So conv_result[i] for i = 0..N-1 should give us what we want
        
        grad_lse = 2.0 * conv_result[:N]
        return grad_lse

    def smooth_objective_and_grad(x, beta):
        """Compute LSE-smoothed objective and gradient."""
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(h)
        
        conv = fast_convolve(h)
        max_c = np.max(conv)
        shifted = conv - max_c
        max_shifted = np.max(shifted)
        if max_shifted > 500:
            return 1e10, np.zeros_like(h)
        
        exp_terms = np.exp(beta * (shifted - max_shifted))
        sum_weights = np.sum(exp_terms)
        if sum_weights < 1e-300:
            return 1e10, np.zeros_like(h)
        
        lse = max_c + max_shifted + np.log(sum_weights) / beta
        raw = 2.0 * N * lse / (s * s)
        
        if not np.isfinite(raw):
            return 1e10, np.zeros_like(h)
        
        f = abs(raw)
        
        # Gradient of lse w.r.t. h
        weights = exp_terms / sum_weights
        grad_lse = fast_grad_lse(weights, h)
        
        # d/dh [2N * lse / s^2] = 2N * (grad_lse * s - 2*lse) / s^3
        df_dh = 2.0 * N * (grad_lse * s - 2.0 * lse) / (s ** 3)
        
        if raw < 0:
            grad = -df_dh
        else:
            grad = df_dh
        
        if not np.all(np.isfinite(grad)):
            grad = np.nan_to_num(grad, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return f, grad

    # --- Phase 1: Global Search in Fourier Coefficient Space ---
    t = np.arange(N) / N
    K = 25  # Number of Fourier components
    
    # Use both cosine and sine for more flexibility
    # Actually, let's use complex exponentials for generality
    # h[i] = sum_k (a_k * cos(2*pi*k*i/N) + b_k * sin(2*pi*k*i/N))
    # That's 2K parameters. Let's use K=15 for 30 params total.
    K_cos = 15
    K_sin = 15
    total_params = K_cos + K_sin
    
    basis = np.zeros((N, total_params))
    for k in range(1, K_cos + 1):
        basis[:, k-1] = np.cos(2 * np.pi * k * t)
    for k in range(1, K_sin + 1):
        basis[:, K_cos + k - 1] = np.sin(2 * np.pi * k * t)
    
    def fourier_to_h(coeffs):
        return basis @ coeffs

    def fourier_objective(coeffs):
        h = fourier_to_h(coeffs)
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = fast_convolve(h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        if not np.isfinite(val):
            return 1e10
        return val

    bounds = [(-3.0, 3.0) for _ in range(total_params)]
    
    try:
        if time.time() - start_time < DEADLINE - 80:
            de_result = differential_evolution(
                fourier_objective,
                bounds,
                seed=42,
                maxiter=80,
                polish=False,
                tol=1e-6,
                init='sobol',
                popsize=15,
                update_interval=10,
                mutation=(0.3, 0.9),
                recombination=0.7,
                seed=42
            )
            
            if de_result.fun < best_value:
                best_coeffs = de_result.x
                h_candidate = fourier_to_h(best_coeffs)
                val, valid = evaluate(h_candidate)
                if valid and val < best_value:
                    best_value = val
                    best_h = h_candidate.copy()
    except Exception:
        pass

    # --- Phase 2: Local Refinement with Multi-Beta Annealing ---
    candidates_to_refine = []
    
    # Best from DE
    if best_value < 2.0:
        candidates_to_refine.append(best_h.copy())
    
    # Seed
    candidates_to_refine.append(np.ones(N).astype(float))
    
    # Random smooth perturbations of best
    rng = np.random.RandomState(123)
    for _ in range(4):
        noise = rng.normal(0, 0.15, N)
        noise = gaussian_filter1d(noise, sigma=8)
        h_perturbed = best_h + noise
        s = np.sum(h_perturbed)
        if abs(s) > 1e-10:
            candidates_to_refine.append(h_perturbed)
    
    # A few random smooth signals
    for _ in range(3):
        freqs = rng.randint(1, 10, size=5)
        amplitudes = rng.uniform(-1, 1, size=5)
        phases = rng.uniform(0, 2*np.pi, size=5)
        h_rand = np.zeros(N)
        for f, a, p in zip(freqs, amplitudes, phases):
            h_rand += a * np.cos(2*np.pi*f*t + p)
        s = np.sum(h_rand)
        if abs(s) > 1e-10:
            candidates_to_refine.append(h_rand)

    # Beta schedule: start low (smooth), increase (sharper)
    beta_schedule = [20.0, 50.0, 100.0, 200.0, 400.0]
    
    for init_h in candidates_to_refine:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        s = np.sum(init_h)
        if abs(s) < 1e-10:
            init_h = init_h + 0.1
        
        current_h = init_h.copy()
        
        for beta in beta_schedule:
            if time.time() - start_time > DEADLINE - 10:
                break
            
            try:
                result = minimize(
                    lambda x, b=beta: smooth_objective_and_grad(x, b)[0],
                    current_h,
                    jac=lambda x, b=beta: smooth_objective_and_grad(x, b)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40}
                )
                
                val, valid = evaluate(result.x)
                if valid and val < best_value:
                    best_value = val
                    best_h = result.x.copy()
                    current_h = result.x.copy()
                elif np.all(np.isfinite(result.x)):
                    current_h = result.x.copy()
                    
            except Exception:
                continue

    # --- Phase 3: Exact Subgradient Polish ---
    if time.time() - start_time < DEADLINE - 3:
        def exact_obj_grad(h):
            s = np.sum(h)
            if abs(s) < 1e-12:
                return 1e10, np.zeros_like(h)
            
            conv = fast_convolve(h)
            max_val = np.max(conv)
            raw = 2.0 * N * max_val / (s * s)
            val = abs(raw)
            
            # Find all indices achieving the max (within tolerance)
            tol = 1e-10
            max_indices = np.where(np.abs(conv - max_val) < tol)[0]
            
            # Subgradient: average of gradients at all max indices
            dh_conv = np.zeros(N)
            for idx in max_indices:
                km = idx - np.arange(N)
                mask = (km >= 0) & (km < N)
                dh_conv[mask] += h[km[mask]] + h[mask]
            
            # Average over max indices
            n_max = len(max_indices)
            if n_max > 0:
                dh_conv /= n_max
            
            grad = 2.0 * N * (dh_conv * s - 2.0 * max_val) / (s ** 3)
            
            if raw < 0:
                grad = -grad
            
            if not np.all(np.isfinite(grad)):
                grad = np.nan_to_num(grad, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return val, grad

        try:
            result = minimize(
                exact_obj_grad,
                best_h,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            
            val, valid = evaluate(result.x)
            if valid and val < best_value:
                best_value = val
                best_h = result.x.copy()
        except Exception:
            pass

    # --- Phase 4: Additional refinement with different beta if time permits ---
    if time.time() - start_time < DEADLINE - 5:
        for beta in [500.0, 800.0]:
            if time.time() - start_time > DEADLINE - 3:
                break
            try:
                result = minimize(
                    lambda x, b=beta: smooth_objective_and_grad(x, b)[0],
                    best_h,
                    jac=lambda x, b=beta: smooth_objective_and_grad(x, b)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 50}
                )
                
                val, valid = evaluate(result.x)
                if valid and val < best_value:
                    best_value = val
                    best_h = result.x.copy()
            except Exception:
                continue

    # Final verification
    s = np.sum(best_h)
    if abs(s) < 1e-12:
        return np.ones(N)
    
    conv = fast_convolve(best_h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    
    if not np.isfinite(val) or val >= 2.0:
        return np.ones(N)
    
    return best_h
