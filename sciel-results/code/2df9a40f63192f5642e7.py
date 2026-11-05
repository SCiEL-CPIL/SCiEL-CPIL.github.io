import numpy as np
import time
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

N = 400

def construct():
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin before 150s kill

    best_value = np.inf
    best_h = np.ones(N).astype(float)

    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val if valid else np.inf

    val_seed = evaluate(best_h)
    best_value = val_seed

    # Precompute FFT sizes for fast convolution
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    fft_size_cc = 1
    while fft_size_cc < N + 2 * N - 1:
        fft_size_cc *= 2

    def fast_convolve(h):
        """Fast convolution via FFT."""
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        return conv[:2 * N - 1]

    def fast_cross_corr(weights, h):
        """
        Compute grad_lse[i] = 2 * sum_{m=0}^{N-1} weights[i+m] * h[m] for i=0..N-1
        using FFT-based cross-correlation.
        """
        h_rev = h[::-1]
        H_rev = np.fft.rfft(h_rev, n=fft_size_cc)
        W = np.fft.rfft(weights, n=fft_size_cc)
        conv_result = np.fft.irfft(H_rev * W, n=fft_size_cc)
        
        # Extract the relevant part
        grad_lse = 2.0 * conv_result[N-1:2*N-1]
        
        return grad_lse

    def smooth_objective_and_grad(x, beta=150.0):
        """Compute both objective and gradient in one pass to avoid redundant FFTs."""
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
        f_val = 2.0 * N * lse / (s * s)
        
        if not np.isfinite(f_val):
            return 1e10, np.zeros_like(h)
        
        # Gradient of lse w.r.t. h
        weights = exp_terms / sum_weights
        grad_lse = fast_cross_corr(weights, h)
        
        # Gradient of f = 2*N*lse/s^2
        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
        
        # Gradient of |f|
        if f_val >= 0:
            grad = df_dh
        else:
            grad = -df_dh
        
        return f_val, grad

    def smooth_objective(x, beta=150.0):
        """Smooth surrogate using log-sum-exp of conv entries."""
        f, _ = smooth_objective_and_grad(x, beta)
        return f

    def smooth_grad(x, beta=150.0):
        """Analytic gradient of the smooth surrogate."""
        _, grad = smooth_objective_and_grad(x, beta)
        return grad

    rng = np.random.RandomState(42)

    # --- Generate diverse initial points ---
    initial_points = []

    # Seed
    initial_points.append(np.ones(N).astype(float))

    # Random uniform
    for _ in range(10):
        initial_points.append(rng.uniform(-1, 1, N))

    # Bimodal: mostly zero with random spikes
    for _ in range(10):
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
    for freq in [1, 2, 3, 5, 7, 10, 15, 20]:
        initial_points.append(np.cos(2 * np.pi * freq * t))

    # Parameterized bump profiles
    i_arr = np.arange(N)
    for center in [N // 4, N // 2, 3 * N // 4]:
        for width in [25, 60, 120, 200]:
            h_bump = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2))
            initial_points.append(h_bump)
            h_double = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2)) - np.exp(-((i_arr - (N - 1 - center)) ** 2) / (2 * width ** 2))
            initial_points.append(h_double)

    # Random Fourier synthesis
    for n_comp in [3, 5, 7]:
        for _ in range(3):
            freqs = rng.randint(1, 40, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.5, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.cos(2 * np.pi * f * t + p)
            initial_points.append(h_fourier)

    # Structured patterns: blocks of +1 and -1
    for block_size in [20, 50, 100]:
        n_blocks = N // block_size
        for _ in range(3):
            h_blocks = np.zeros(N)
            for b in range(n_blocks):
                sign = rng.choice([-1, 1])
                h_blocks[b*block_size:(b+1)*block_size] = sign
            initial_points.append(h_blocks)

    # Smooth random via Gaussian filter
    for _ in range(10):
        h_smooth = rng.uniform(-1, 1, N)
        h_smooth = gaussian_filter1d(h_smooth, sigma=rng.uniform(2, 20))
        initial_points.append(h_smooth)

    # --- Spectral null initial points: design h such that H[0] (DC) is small ---
    # The autoconvolution peak is dominated by the DC term |H[0]|^2 / N.
    # If we design h to have a spectral null at DC (i.e., sum(h) is small relative to energy),
    # the peak autoconvolution can be reduced.
    # We generate random vectors in the subspace orthogonal to the all-ones vector,
    # then add a small DC component to keep sum(h) non-zero.
    
    # Projection matrix onto subspace orthogonal to 1
    ones = np.ones(N)
    ones_norm_sq = np.dot(ones, ones)
    
    for _ in range(15):
        # Random vector
        v = rng.normal(0, 1, N)
        # Project out DC component
        dc_component = np.dot(v, ones) / ones_norm_sq * ones
        v_zero_dc = v - dc_component
        # Add small DC offset to ensure sum(h) != 0
        dc_offset = rng.uniform(0.1, 0.5) * np.linalg.norm(v_zero_dc) / N
        h_spectral = v_zero_dc + dc_offset * ones
        initial_points.append(h_spectral)
    
    # More structured spectral nulls: use random phases in Fourier domain
    for _ in range(10):
        # Generate random spectrum with null at DC
        spectrum = rng.normal(0, 1, N // 2 + 1) + 1j * rng.normal(0, 1, N // 2 + 1)
        spectrum[0] = 0.0  # Null at DC
        # Ensure the spectrum is conjugate symmetric for real h
        if N % 2 == 0:
            spectrum_full = np.zeros(N, dtype=complex)
            spectrum_full[:N // 2 + 1] = spectrum
            spectrum_full[N // 2 + 1:] = np.conj(spectrum[1:N // 2][::-1])
        else:
            spectrum_full = np.zeros(N, dtype=complex)
            spectrum_full[:N // 2 + 1] = spectrum
            spectrum_full[N // 2 + 1:] = np.conj(spectrum[1:N // 2][::-1])
        h_fourier_null = np.real(np.fft.ifft(spectrum_full))
        # Add small DC offset
        dc_offset = rng.uniform(0.1, 0.5) * np.linalg.norm(h_fourier_null) / N
        h_fourier_null += dc_offset
        initial_points.append(h_fourier_null)

    # Sort by initial value (best first)
    def sort_key(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = fast_convolve(h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    initial_points.sort(key=sort_key)

    # --- Optimize from best initial points ---
    for h0 in initial_points:
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

    # --- Polish pass: high-precision refinement on the best result ---
    if time.time() - start_time < DEADLINE - 10:
        try:
            # Use a higher beta for sharper approximation of max
            def polish_objective(x):
                f, _ = smooth_objective_and_grad(x, beta=300.0)
                return f

            def polish_grad(x):
                _, grad = smooth_objective_and_grad(x, beta=300.0)
                return grad

            res_polish = minimize(
                polish_objective,
                best_h,
                jac=polish_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_polished = res_polish.x
            val_polished = evaluate(h_polished)
            if val_polished < best_value:
                best_value = val_polished
                best_h = h_polished
        except Exception:
            pass

    # --- Final local search: small perturbations around the best solution ---
    if time.time() - start_time < DEADLINE - 5:
        for _ in range(20):
            if time.time() - start_time > DEADLINE:
                break
            # Small random perturbation
            noise = rng.normal(0, 0.01, N)
            h_perturbed = best_h + noise
            try:
                val_perturbed = evaluate(h_perturbed)
                if val_perturbed < best_value:
                    best_value = val_perturbed
                    best_h = h_perturbed
            except Exception:
                pass

    return best_h
