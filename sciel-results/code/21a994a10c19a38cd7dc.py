import random
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def ratio(h):
        """Compute R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) for non-negative h."""
        h = np.asarray(h, dtype=float)
        if len(h) == 0:
            return 0.0
        h1 = np.sum(h)
        if h1 <= 0:
            return 0.0
        h2 = np.sum(h * h)
        c = np.convolve(h, h)
        cmax = np.max(c)
        if cmax <= 0:
            return 0.0
        return h2 / (h1 * cmax)

    def soft_max(c, tau=0.1):
        """Smooth approximation of max(c) using log-sum-exp."""
        c = np.asarray(c, dtype=float)
        cmax = np.max(c)
        if cmax <= 0:
            return 0.0
        shifted = c / tau
        shifted = shifted - np.max(shifted)
        lse = cmax + tau * np.log(np.sum(np.exp(shifted)))
        return lse

    def soft_max_grad(c, tau=0.1):
        """Gradient of soft_max with respect to c."""
        c = np.asarray(c, dtype=float)
        cmax = np.max(c)
        if cmax <= 0:
            return np.zeros_like(c)
        shifted = c / tau
        shifted = shifted - np.max(shifted)
        weights = np.exp(shifted)
        weights /= np.sum(weights)
        return weights

    def optimize_smooth(h_init, max_iter=500, ftol=1e-10, tau=0.05):
        """
        Optimize h using SLSQP with smooth log-sum-exp surrogate for autocorrelation peak.
        Variables: x = [h_0, ..., h_{N-1}]
        Maximize h2/(h1 * softmax(c)) subject to sum(h)=1, h >= 0
        Equivalent to minimizing -h2/(h1 * softmax(c))
        """
        h_init = np.asarray(h_init, dtype=float)
        s = np.sum(h_init)
        if s <= 0:
            h_init = np.ones(N) / N
            s = 1.0
        h_init = h_init / s

        def obj(x):
            h = np.clip(x, 0, None)
            s = np.sum(h)
            if s <= 0:
                return 0.0
            h = h / s
            h2 = np.sum(h * h)
            c = np.convolve(h, h)
            sm = soft_max(c, tau)
            if sm <= 0:
                return 0.0
            return -h2 / (s * sm)

        def obj_grad(x):
            h = np.clip(x, 0, None)
            s = np.sum(h)
            if s <= 0:
                return np.zeros(N)
            h = h / s
            h2 = np.sum(h * h)
            c = np.convolve(h, h)
            sm = soft_max(c, tau)
            if sm <= 0:
                return np.zeros(N)
            # d/dh of -h2/(s*sm) where s=sum(h)
            # Let f = h2/(s*sm), we want -grad(f)
            # df/dh_i = (2*h_i*s*sm - h2*sm - h2*s*dsmdh_i) / (s^2 * sm^2)
            # dsmdh_i = sum_k (dc_k/dh_i) * w_k where w_k = softmax weights
            c_grad = soft_max_grad(c, tau)
            # dc_k/dh_i = sum_j (delta_{k,i+j} * h_j + delta_{k,j+i} * h_j) for j valid
            # More precisely: c_k = sum_i h_i * h_{k-i}
            # dc_k/dh_i = h_{k-i} + h_{i-k} (with bounds check)
            dsmdh = np.zeros(N)
            for k in range(2 * N - 1):
                wk = c_grad[k]
                if abs(wk) < 1e-15:
                    continue
                for i in range(N):
                    j1 = k - i
                    j2 = i - k
                    if 0 <= j1 < N:
                        dsmdh[i] += wk * h[j1]
                    if 0 <= j2 < N:
                        dsmdh[i] += wk * h[j2]
            
            # f = h2 / (s * sm)
            # df/dh_i = [2*h_i * s * sm - h2 * sm - h2 * s * dsmdh_i] / (s^2 * sm^2)
            #          + correction for the 1/s factor: -h2/(s^2*sm) for the s dependence
            # Actually let's use quotient rule more carefully
            # f = h2 / (s * sm)
            # df/dh_i = (2*h_i * (s*sm) - h2 * (ds/dh_i * sm + s * dsmdh_i)) / (s*sm)^2
            # ds/dh_i = 1
            df_dh = (2 * h * (s * sm) - h2 * (sm + s * dsmdh)) / (s * sm) ** 2
            # But we also need to account for the normalization h = x/s
            # For simplicity, use finite-difference-free analytical form
            # The gradient of obj = -f is -df_dh
            grad = -df_dh
            # Project to non-negative: zero out gradient for components where x < 0
            mask = x > 0
            grad[~mask] = 0.0
            return grad

        def constraint_sum(x):
            return np.sum(x) - 1.0

        def constraint_sum_grad(x):
            return np.ones(N)

        x0 = h_init.copy()
        bounds = [(0.0, None)] * N

        cons = [
            {'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad}
        ]

        try:
            res = minimize(
                obj,
                x0,
                method='SLSQP',
                jac=obj_grad,
                bounds=bounds,
                constraints=cons,
                options={'maxiter': max_iter, 'ftol': ftol}
            )

            h_res = res.x
            h_res = np.clip(h_res, 0, None)
            s = np.sum(h_res)
            if s > 0:
                h_res = h_res / s
            else:
                return None, 0.0
            ratio_res = ratio(h_res)
            return h_res, ratio_res
        except Exception:
            pass
        return None, 0.0

    def optimize_epigraph(h_init, max_iter=300, ftol=1e-10):
        """
        Optimize h using SLSQP with epigraph formulation.
        Variables: x = [h_0, ..., h_{N-1}, M]
        Minimize M subject to sum(h)=1, M >= c_k for all k, h >= 0
        """
        def obj(x):
            return x[N]

        def obj_grad(x):
            grad = np.zeros(N + 1)
            grad[N] = 1.0
            return grad

        def constraint_sum(x):
            return np.sum(x[:N]) - 1.0

        def constraint_sum_grad(x):
            grad = np.zeros(N + 1)
            grad[:N] = 1.0
            return grad

        def constraints_autoconv(x):
            h = x[:N]
            M = x[N]
            c = np.convolve(h, h)
            return M - c

        def constraints_autoconv_grad(x):
            h = x[:N]
            grad = np.zeros((2 * N - 1, N + 1))
            for k in range(2 * N - 1):
                derivs = np.zeros(N)
                for i in range(N):
                    j = k - i
                    if 0 <= j < N:
                        derivs[i] = 2.0 * h[j]
                grad[k, :N] = -derivs
                grad[k, N] = 1.0
            return grad

        h_init = np.asarray(h_init, dtype=float)
        s = np.sum(h_init)
        if s <= 0:
            h_init = np.ones(N) / N
            s = 1.0
        h_init = h_init / s
        c_init = np.convolve(h_init, h_init)
        M_init = np.max(c_init)
        x0 = np.concatenate([h_init, [M_init]])

        bounds = [(0.0, None)] * N + [(0.0, None)]

        cons = [
            {'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad},
            {'type': 'ineq', 'fun': constraints_autoconv, 'jac': constraints_autoconv_grad}
        ]

        try:
            res = minimize(
                obj,
                x0,
                method='SLSQP',
                jac=obj_grad,
                bounds=bounds,
                constraints=cons,
                options={'maxiter': max_iter, 'ftol': ftol}
            )

            if res.success or res.fun < M_init:
                h_res = res.x[:N]
                h_res = np.clip(h_res, 0, None)
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                else:
                    return None, 0.0
                ratio_res = ratio(h_res)
                return h_res, ratio_res
        except Exception:
            pass
        return None, 0.0

    def spectral_basis_search(num_trials=200):
        """
        Generate candidate height vectors as |DFT of sparse random complex signals|.
        The magnitude of a sparse DFT creates structured non-negative vectors whose
        autocorrelation has controlled peak structure.
        """
        candidates = []
        n_freq = 2 * N  # Use zero-padded FFT size for better frequency resolution

        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break

            # Choose sparsity level - favor higher sparsity for better peak separation
            K = random.randint(3, min(18, N // 2))

            # Random sparse complex signal in frequency domain
            # Bias toward well-separated frequencies for Sidon-like behavior
            if random.random() < 0.5:
                # Well-separated: choose from non-adjacent bins
                all_bins = list(range(n_freq))
                random.shuffle(all_bins)
                selected = []
                for b in all_bins:
                    if all(abs(b - s) > 2 for s in selected):
                        selected.append(b)
                        if len(selected) == K:
                            break
                freq_indices = np.array(selected[:K])
            else:
                freq_indices = np.random.choice(n_freq, K, replace=False)

            phases = np.random.uniform(0, 2 * np.pi, K)
            # Vary amplitudes more broadly to explore different interference patterns
            amplitudes = np.random.uniform(0.3, 1.0, K)

            # Build sparse frequency vector
            F = np.zeros(n_freq, dtype=complex)
            for k_idx, amp, ph in zip(freq_indices, amplitudes, phases):
                F[k_idx] = amp * np.exp(1j * ph)

            # Inverse FFT to get time-domain signal
            h_complex = np.fft.ifft(F)
            h = np.abs(h_complex[:N])  # Take first N samples, magnitude

            # Normalize
            s = np.sum(h)
            if s > 0:
                h = h / s
                r = ratio(h)
                if r > 0.1:  # Only keep reasonable candidates
                    candidates.append((r, h.copy()))

        # Sort by ratio
        candidates.sort(key=lambda x: -x[0])
        return candidates

    def gaussian_basis_search(num_trials=100):
        """
        Generate candidates as mixtures of Gaussian bumps at random positions.
        This creates smooth, localized structures.
        """
        candidates = []
        x = np.linspace(0, N - 1, N)

        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break

            # Choose number of bumps
            n_bumps = random.randint(1, 5)
            centers = np.random.uniform(0, N - 1, n_bumps)
            widths = np.random.uniform(1.0, 5.0, n_bumps)
            amplitudes = np.random.uniform(0.5, 1.0, n_bumps)

            h = np.zeros(N)
            for c, w, a in zip(centers, widths, amplitudes):
                h += a * np.exp(-0.5 * ((x - c) / w) ** 2)

            s = np.sum(h)
            if s > 0:
                h = h / s
                r = ratio(h)
                if r > 0.1:
                    candidates.append((r, h.copy()))

        candidates.sort(key=lambda x: -x[0])
        return candidates

    def random_sparse_search(num_trials=200):
        """
        Generate random sparse vectors with varying sparsity and weight distributions.
        """
        candidates = []

        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break

            # Random sparsity
            K = random.randint(1, N)
            support = np.random.choice(N, K, replace=False)

            # Random weight distribution
            if random.random() < 0.3:
                # Uniform weights
                weights = np.ones(K) / K
            elif random.random() < 0.5:
                # Geometric weights
                base = random.uniform(0.7, 0.95)
                weights = base ** np.arange(K)
                weights /= np.sum(weights)
            else:
                # Random Dirichlet
                alpha = np.random.uniform(0.5, 3.0, K)
                weights = np.random.dirichlet(alpha)

            h = np.zeros(N)
            h[support] = weights
            r = ratio(h)
            if r > 0.1:
                candidates.append((r, h.copy()))

        candidates.sort(key=lambda x: -x[0])
        return candidates

    def structured_supports():
        """
        Generate structured support patterns (Sidon-like, arithmetic, etc.)
        """
        candidates = []

        # Arithmetic progressions
        for start in range(0, N, 3):
            for step in [2, 3, 4, 5, 7]:
                support = list(range(start, N, step))
                if len(support) >= 2:
                    h = np.zeros(N)
                    h[support] = 1.0 / len(support)
                    r = ratio(h)
                    if r > 0.1:
                        candidates.append((r, h.copy()))

        # Power-of-2 supports
        for base in [2, 3]:
            support = []
            val = 0
            while val < N:
                support.append(val)
                val = val * base + 1
            if len(support) >= 2:
                h = np.zeros(N)
                h[support] = 1.0 / len(support)
                r = ratio(h)
                if r > 0.1:
                    candidates.append((r, h.copy()))

        # Alternating patterns
        for period in [2, 3, 4, 5, 7]:
            for offset in range(period):
                h = np.zeros(N)
                for i in range(N):
                    if i % period == offset:
                        h[i] = 1.0
                s = np.sum(h)
                if s > 0:
                    h /= s
                    r = ratio(h)
                    if r > 0.1:
                        candidates.append((r, h.copy()))

        # Uniform
        h_uni = np.ones(N) / N
        candidates.append((ratio(h_uni), h_uni.copy()))

        candidates.sort(key=lambda x: -x[0])
        return candidates

    def coordinate_refine(h, max_iters=50):
        """
        Simple coordinate ascent: optimize one coordinate at a time.
        """
        h = np.array(h, dtype=float)
        s = np.sum(h)
        if s <= 0:
            return h

        current_ratio = ratio(h)

        for iteration in range(max_iters):
            if time.time() > deadline - 3:
                break

            improved = False
            for i in range(N):
                if time.time() > deadline - 3:
                    break

                # Try varying h[i] while keeping sum fixed
                # h[i] = t, others scaled to maintain sum=1
                for frac in [0.5, 0.7, 0.8, 0.9, 1.1, 1.3, 1.5, 2.0]:
                    h_test = h.copy()
                    h_test[i] = h[i] * frac
                    s_test = np.sum(h_test)
                    if s_test <= 0:
                        continue
                    h_test = h_test / s_test
                    r_test = ratio(h_test)
                    if r_test > current_ratio:
                        current_ratio = r_test
                        h = h_test
                        improved = True

            if not improved:
                break

        return h

    # Generate all candidate initializations
    all_candidates = []

    # Structured supports
    structured = structured_supports()
    all_candidates.extend(structured)

    # Spectral basis
    spectral = spectral_basis_search(180)
    all_candidates.extend(spectral)

    # Gaussian basis
    gaussian = gaussian_basis_search(100)
    all_candidates.extend(gaussian)

    # Random sparse
    rand_sparse = random_sparse_search(180)
    all_candidates.extend(rand_sparse)

    # Sort all by ratio
    all_candidates.sort(key=lambda x: -x[0])

    # Deduplicate approximately
    unique_candidates = []
    for r, h in all_candidates:
        # Check if similar to existing
        is_dup = False
        for _, h_existing in unique_candidates:
            if np.linalg.norm(h - h_existing) < 0.1:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append((r, h))

    unique_candidates.sort(key=lambda x: -x[0])

    # Optimize top candidates using smooth optimizer
    best_h = None
    best_score = 0.0

    max_optimizations = 50
    opt_count = 0

    for init_ratio, h_init in unique_candidates:
        if time.time() > deadline - 35:
            break
        if opt_count >= max_optimizations:
            break

        # Try smooth optimization with moderate tau
        h_opt, s_opt = optimize_smooth(h_init, max_iter=600, ftol=1e-10, tau=0.05)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
        opt_count += 1

    # Additional random restarts if time allows
    while time.time() < deadline - 30:
        K = random.randint(2, 20)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K

        h_opt, s_opt = optimize_smooth(h_rand, max_iter=300, ftol=1e-10, tau=0.05)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    # Multi-tau refinement: progressively tighten the soft-max
    if best_h is not None and time.time() < deadline - 20:
        for tau in [0.02, 0.01, 0.005, 0.002, 0.001]:
            if time.time() > deadline - 10:
                break
            h_ref = coordinate_refine(best_h, max_iters=20)
            s_ref = ratio(h_ref)
            if s_ref > best_score:
                best_score = s_ref
                best_h = h_ref.copy()
            
            h_opt, s_opt = optimize_smooth(best_h, max_iter=800, ftol=1e-12, tau=tau)
            if h_opt is not None and s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # Coordinate refinement on best solution
    if best_h is not None and time.time() < deadline - 10:
        h_ref = coordinate_refine(best_h, max_iters=40)
        s_ref = ratio(h_ref)
        if s_ref > best_score:
            best_score = s_ref
            best_h = h_ref.copy()

    # Final high-precision polish with very small tau
    if best_h is not None and time.time() < deadline - 5:
        h_polished, s_polished = optimize_smooth(best_h, max_iter=2000, ftol=1e-13, tau=0.0005)
        if h_polished is not None and s_polished > best_score:
            best_score = s_polished
            best_h = h_polished.copy()

    # One more coordinate refinement
    if best_h is not None and time.time() < deadline - 2:
        h_ref2 = coordinate_refine(best_h, max_iters=30)
        s_ref2 = ratio(h_ref2)
        if s_ref2 > best_score:
            best_score = s_ref2
            best_h = h_ref2.copy()

    # Fallback
    if best_h is None:
        best_h = np.ones(N) / N
        best_score = ratio(best_h)

    # Ensure non-negative and normalized
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
# EVOLVE-BLOCK-END
