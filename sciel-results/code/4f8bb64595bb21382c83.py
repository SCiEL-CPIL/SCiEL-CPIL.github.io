import numpy as np
from scipy import optimize
import time

N = 600

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    conv = np.convolve(h, h, 'full')
    total = np.sum(h)
    if total == 0:
        return 0.0
    return np.max(conv) / (total * total)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for the 150s wall-clock limit

    # Initialize with all ones
    best_h = np.ones(N)
    best_score = compute_autoconv_max(best_h)

    # Precompute FFT-related quantities for fast evaluation
    # We'll work in the frequency domain where the objective is smoother

    def soft_objective_freq(x, T=0.01):
        """
        Soft objective in frequency domain.
        x: candidate h (length N)
        T: temperature for soft-max
        Returns: soft approximation of max |H(k)|^2 / (sum h)^2
        """
        h = np.maximum(x, 0)
        total = np.sum(h)
        if total < 1e-10:
            return 1e10
        h_norm = h / total
        H = np.fft.fft(h_norm)
        power = np.abs(H)**2
        pmax = np.max(power)
        if pmax < 1e-15:
            return 1e10
        shifted = (power - pmax) / T
        shifted = np.clip(shifted, -50, 50)
        soft_max = pmax + T * np.log(np.sum(np.exp(shifted)))
        return soft_max

    def exact_objective_freq(x):
        """Exact objective: max |H(k)|^2 / (sum h)^2"""
        h = np.maximum(x, 0)
        total = np.sum(h)
        if total < 1e-10:
            return 1e10
        h_norm = h / total
        H = np.fft.fft(h_norm)
        power = np.abs(H)**2
        return np.max(power)

    # Strategy: The problem is to minimize the peak of the power spectrum |H(k)|^2
    # subject to h >= 0 and sum(h) = 1 (after normalization).
    # 
    # Key insight: A flat power spectrum (all |H(k)|^2 equal) would be ideal,
    # but with h >= 0, we can't achieve perfect flatness. The uniform h gives
    # H(0) = 1, H(k) = 0 for k != 0, which has peak = 1.
    # 
    # We want to spread the energy more evenly across frequency bins.
    # This is similar to designing a low-peak-sidelobe sequence.
    #
    # Approach: Use projected gradient descent on the soft-max objective,
    # with multiple restarts and adaptive temperature annealing.

    np.random.seed(42)

    # Generate diverse initializations
    initializations = []

    # 1. Uniform
    initializations.append(np.ones(N))

    # 2. Raised cosine
    t = np.linspace(0, 1, N, endpoint=False)
    rc = 0.5 + 0.5 * np.cos(2 * np.pi * t)
    initializations.append(np.maximum(rc, 0.01))

    # 3. Triangular
    tri = 1 - np.abs(2 * t - 1)
    initializations.append(np.maximum(tri, 0.01))

    # 4. Gaussian
    gauss = np.exp(-4 * (t - 0.5)**2)
    initializations.append(np.maximum(gauss, 0.01))

    # 5. Inverse raised cosine (peaked at edges)
    irc = 0.5 - 0.5 * np.cos(2 * np.pi * t)
    initializations.append(np.maximum(irc, 0.01))

    # 6. Two-lobe cosine
    two_lobe = 0.5 + 0.4 * np.cos(4 * np.pi * t)
    initializations.append(np.maximum(two_lobe, 0.01))

    # 7. Three-lobe cosine
    three_lobe = 0.5 + 0.3 * np.cos(6 * np.pi * t)
    initializations.append(np.maximum(three_lobe, 0.01))

    # 8. Random smooth
    freq_rand = np.random.randn(N)
    # Low-pass filter to make it smooth
    H_rand = np.fft.fft(freq_rand)
    H_rand[np.abs(np.arange(N) - N//2) > N//8] = 0
    rand_smooth = np.real(np.fft.ifft(H_rand))
    initializations.append(np.maximum(rand_smooth - np.min(rand_smooth) + 0.1, 0.01))

    # 9. Comb pattern
    comb = np.ones(N)
    comb[::3] = 0.3
    initializations.append(np.maximum(comb, 0.01))

    # 10. Alternating high/low
    alt = np.ones(N)
    alt[::2] = 0.5
    initializations.append(np.maximum(alt, 0.01))

    def projected_gradient_descent(h0, n_iters=500, lr=0.05, T=0.01, cooldown_iters=100):
        """
        Projected gradient descent on the soft-max frequency objective.
        """
        h = np.maximum(h0, 0).copy()
        best_local = h.copy()
        best_local_score = compute_autoconv_max(h)

        for it in range(n_iters):
            if time.time() > deadline - 20:
                break

            # Adaptive temperature
            if it < cooldown_iters:
                T_curr = T
            else:
                T_curr = T * (1 - (it - cooldown_iters) / max(1, n_iters - cooldown_iters))
                T_curr = max(T_curr, 0.001)

            # Compute gradient of soft-max objective
            h_pos = np.maximum(h, 0)
            total = np.sum(h_pos)
            if total < 1e-10:
                break
            h_norm = h_pos / total
            H = np.fft.fft(h_norm)
            power = np.abs(H)**2
            pmax = np.max(power)

            if pmax < 1e-15:
                break

            shifted = (power - pmax) / T_curr
            shifted = np.clip(shifted, -50, 50)
            weights = np.exp(shifted)
            weights /= np.sum(weights)

            # Gradient of soft-max with respect to h_norm
            # d(soft_max)/d(h_norm_j) = sum_k weights_k * d|H(k)|^2/d(h_norm_j)
            # d|H(k)|^2/d(h_norm_j) = 2 * Re{conj(H(k)) * exp(-2*pi*i*j*k/N)}
            # = 2 * Re{conj(H(k)) * W[j,k]} where W is the DFT matrix

            # Efficient computation:
            # grad = 2 * Re{IFFT(weights * conj(H))} ... let me verify
            # |H(k)|^2 = H(k) * conj(H(k))
            # d|H(k)|^2/dh_j = conj(H(k)) * dH(k)/dh_j + H(k) * dconj(H(k))/dh_j
            # dH(k)/dh_j = exp(-2*pi*i*j*k/N)
            # So d|H(k)|^2/dh_j = conj(H(k)) * exp(-2*pi*i*j*k/N) + H(k) * exp(2*pi*i*j*k/N)
            # = 2 * Re{conj(H(k)) * exp(-2*pi*i*j*k/N)}

            # For the weighted sum:
            # grad_j = sum_k weights_k * 2 * Re{conj(H(k)) * exp(-2*pi*i*j*k/N)}
            # = 2 * Re{sum_k weights_k * conj(H(k)) * exp(-2*pi*i*j*k/N)}
            # = 2 * Re{IFFT(weights * conj(H))} * N ... need to check normalization

            # numpy's ifft: ifft(x)_j = (1/N) sum_k x_k * exp(2*pi*i*j*k/N)
            # We want: sum_k weights_k * conj(H(k)) * exp(-2*pi*i*j*k/N)
            # = conj( sum_k conj(weights_k) * H(k) * exp(2*pi*i*j*k/N) )  [if weights are real]
            # weights are real, so:
            # = conj( N * ifft(weights * H)_j ) ... let me just compute directly

            # Actually, let's use the direct formula:
            # grad_j = 2 * Re{sum_k weights_k * conj(H(k)) * exp(-2*pi*i*j*k/N)}

            # Using FFT: let g = ifft(weights * conj(H))
            # g_j = (1/N) sum_k weights_k * conj(H(k)) * exp(2*pi*i*j*k/N)
            # We need exp(-2*pi*i*j*k/N), which is conj(exp(2*pi*i*j*k/N))
            # So sum_k weights_k * conj(H(k)) * exp(-2*pi*i*j*k/N) = conj(N * g_j) if weights real?
            # Let's just compute it directly for correctness, N=600 is small enough

            # More efficient: use the fact that
            # sum_k weights_k * conj(H(k)) * exp(-2*pi*i*j*k/N)
            # Let a_k = weights_k * conj(H(k))
            # Then we want sum_k a_k * exp(-2*pi*i*j*k/N) for each j
            # This is N * ifft(a)_j conjugated... 
            # ifft(a)_j = (1/N) sum_k a_k * exp(2*pi*i*j*k/N)
            # conj(ifft(a)_j) = (1/N) sum_k conj(a_k) * exp(-2*pi*i*j*k/N)
            # Since weights are real: conj(a_k) = weights_k * H(k)
            # So conj(N * ifft(weights * conj(H)))_j = sum_k weights_k * H(k) * exp(-2*pi*i*j*k/N)
            # That's not quite what we want.

            # Let me just do it the straightforward way with matrix-vector product
            # For N=600, this is 600x600 which is fine
            j_idx = np.arange(N)
            k_idx = np.arange(N)
            # W[j,k] = exp(-2*pi*i*j*k/N)
            # We need: grad_j = 2 * Re{sum_k weights_k * conj(H(k)) * W[j,k]}
            # = 2 * Re{W @ (weights * conj(H))}

            # To avoid forming the full matrix, use FFT:
            # sum_k a_k * exp(-2*pi*i*j*k/N) where a_k = weights_k * conj(H(k))
            # This is the DFT of a, evaluated at j
            # numpy's fft computes: fft(a)_j = sum_k a_k * exp(-2*pi*i*j*k/N)
            # So grad_j = 2 * Re{fft(weights * conj(H))_j}

            a = weights * np.conj(H)
            grad_h_norm = 2 * np.real(np.fft.fft(a))

            # Now account for normalization: h_norm = h / total
            # d(h_norm)/d(h_j) = 1/total - h_j/total^2
            # d(objective)/d(h_j) = sum_i (d objective/d h_norm_i) * d(h_norm_i)/d(h_j)
            # = grad_h_norm_j / total - (sum_i grad_h_norm_i * h_norm_i) / total^2
            # But we also need to account for the fact that objective depends on h_norm
            # and h_norm = h/total, so:
            # d(objective)/d(h_j) = grad_h_norm_j / total - (dot(grad_h_norm, h_norm)) / total

            dot_grad_hnorm = np.dot(grad_h_norm, h_norm)
            grad_h = (grad_h_norm - dot_grad_hnorm * h_norm) / total

            # Projected gradient step
            h_new = h - lr * grad_h
            h_new = np.maximum(h_new, 0)

            # Check improvement
            score_new = compute_autoconv_max(h_new)
            if score_new < best_local_score:
                best_local_score = score_new
                best_local = h_new.copy()

            h = h_new

        return best_local, best_local_score

    # Run projected gradient descent from each initialization
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 30:
            break

        # Try different learning rates and temperatures
        for lr in [0.02, 0.05, 0.1]:
            for T in [0.005, 0.01, 0.02]:
                if time.time() > deadline - 25:
                    break

                h_opt, score_opt = projected_gradient_descent(
                    h0, n_iters=300, lr=lr, T=T, cooldown_iters=100
                )

                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()

    # Final polish with L-BFGS-B on the soft objective
    if time.time() < deadline - 15:
        try:
            result = optimize.minimize(
                lambda x: soft_objective_freq(x, T=0.001),
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_polished = np.maximum(result.x, 0)
            score_polished = compute_autoconv_max(h_polished)
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished.copy()
        except Exception:
            pass

    # Another polish with even lower temperature
    if time.time() < deadline - 10:
        try:
            result = optimize.minimize(
                lambda x: soft_objective_freq(x, T=0.0001),
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            h_final = np.maximum(result.x, 0)
            score_final = compute_autoconv_max(h_final)
            if score_final < best_score:
                best_score = score_final
                best_h = h_final.copy()
        except Exception:
            pass

    return list(best_h)
