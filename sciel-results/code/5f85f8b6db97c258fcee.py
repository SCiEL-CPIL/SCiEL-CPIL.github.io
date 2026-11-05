import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill

    # Precompute Chebyshev basis
    x = np.linspace(-1, 1, N)
    n_basis = 16
    basis = np.zeros((n_basis, N))
    basis[0] = 1.0
    if n_basis > 1:
        basis[1] = x
    for k in range(2, n_basis):
        basis[k] = 2 * x * basis[k-1] - basis[k-2]

    # Precompute FFT size
    fft_size = 1 << (2 * N - 1).bit_length()

    def compute_value(h):
        """Compute the exact value using FFT for speed, with fallback to direct."""
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        H = np.fft.rfft(h, fft_size)
        Z = np.fft.irfft(H * H, fft_size)[:2 * N - 1]
        return abs(2.0 * N * np.max(Z) / (s * s))

    def compute_h(c):
        return basis.dot(c)

    # Strategy: Use structured frequency-domain search with Chebyshev coefficients
    # 1. Start with seed (c = [1, 0, ..., 0])
    # 2. Use multi-start local optimization with different random initializations
    # 3. Use log-sum-exp surrogate for smooth gradient computation
    # 4. Use projected gradient descent with momentum

    def smooth_max(Z, beta=50.0):
        """Log-sum-exp approximation of max(Z)."""
        Z_max = np.max(Z)
        if Z_max < 0:
            return Z_max
        return Z_max + np.log(np.sum(np.exp(beta * (Z - Z_max)))) / beta

    def smooth_value(c, beta=50.0):
        """Smooth surrogate of the objective."""
        h = compute_h(c)
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(c)
        H = np.fft.rfft(h, fft_size)
        Z = np.fft.irfft(H * H, fft_size)[:2 * N - 1]
        # Smooth max
        smax = smooth_max(Z, beta)
        val = abs(2.0 * N * smax / (s * s))
        return val, h

    # Multi-start optimization
    rng = np.random.RandomState(42)
    best_c = np.zeros(n_basis)
    best_c[0] = 1.0
    best_val = compute_value(compute_h(best_c))  # Should be 2.0

    # Try several different random initializations
    n_starts = 8
    for start in range(n_starts):
        if time.time() > deadline - 20:
            break

        if start == 0:
            c = best_c.copy()
        else:
            c = rng.randn(n_basis) * 0.5
            c[0] = 1.0  # Keep sum(h) away from zero

        # Local optimization using projected gradient descent
        lr = 0.05
        momentum = 0.9
        v = np.zeros(n_basis)
        local_best_c = c.copy()
        local_best_val = compute_value(compute_h(c))
        beta = 30.0

        n_steps = 800
        for step in range(n_steps):
            if time.time() > deadline - 10:
                break

            # Decay learning rate
            lr = 0.05 * (0.98 ** (step // 50))
            # Increase beta for sharper max approximation
            beta = 30.0 + step * 0.1

            h = compute_h(c)
            s = np.sum(h)
            if abs(s) < 1e-10:
                c[0] += 0.1
                continue

            H = np.fft.rfft(h, fft_size)
            Z = np.fft.irfft(H * H, fft_size)[:2 * N - 1]

            # Compute smooth max and its gradient
            Z_max = np.max(Z)
            if Z_max > 0:
                weights = np.exp(beta * (Z - Z_max))
                weights /= np.sum(weights)
                smax = Z_max + np.log(np.sum(weights)) / beta
                # Gradient of smooth_max w.r.t. Z: weights / beta
                grad_Z = weights / beta
            else:
                smax = Z_max
                grad_Z = np.zeros_like(Z)
                idx_max = np.argmax(Z)
                grad_Z[idx_max] = 1.0 / beta if beta > 0 else 1.0

            # Value: |2*N*smax/s^2|
            val = abs(2.0 * N * smax / (s * s))

            if val < local_best_val:
                local_best_val = val
                local_best_c = c.copy()
                if val < best_val:
                    best_val = val
                    best_c = c.copy()

            # Compute gradient of value w.r.t. c
            # val = 2*N*smax/s^2 (assuming positive)
            # d(val)/d(c) = 2*N * [dsmax/dc * s^2 - smax * 2*s * ds/dc] / s^4
            #              = 2*N * [dsmax/dc * s - 2*smax * ds/dc] / s^3

            # ds/dc = sum(dh/dc) = sum(basis)
            ds_dc = np.sum(basis, axis=1)

            # dsmax/dc: Z = conv(h,h), so dZ/dh = 2*conv(h, delta) = 2*h
            # Actually Z[k] = sum_i h[i]*h[i+k], so dZ/dh[j] = sum_k (dZ[k]/dh[j]) * grad_Z[k]
            # dZ[k]/dh[j] = h[j+k] if j+k in range + h[j-k] if j-k in range
            # This is equivalent to: dsmax/dc[m] = sum_j basis[m,j] * d(smax)/dh[j]
            # d(smax)/dh[j] = sum_k grad_Z[k] * dZ[k]/dh[j]
            #                = sum_k grad_Z[k] * (h[j+k] + h[j-k]) (with boundary handling)

            # Use FFT for efficiency: dZ/dh is the convolution of h with itself, but we need
            # the gradient of the smooth max w.r.t. h, which is grad_Z convolved with h
            # Actually, d(smax)/dh = grad_Z @ dZ/dh, and dZ/dh[j] = 2*corr(h, delta_j)
            # Simpler: use numerical gradient for the smooth value
            eps = 1e-4
            grad_c = np.zeros(n_basis)
            for m in range(n_basis):
                c_plus = c.copy()
                c_plus[m] += eps
                h_plus = compute_h(c_plus)
                s_plus = np.sum(h_plus)
                if abs(s_plus) < 1e-10:
                    continue
                H_plus = np.fft.rfft(h_plus, fft_size)
                Z_plus = np.fft.irfft(H_plus * H_plus, fft_size)[:2 * N - 1]
                val_plus = abs(2.0 * N * smooth_max(Z_plus, beta) / (s_plus * s_plus))

                c_minus = c.copy()
                c_minus[m] -= eps
                h_minus = compute_h(c_minus)
                s_minus = np.sum(h_minus)
                if abs(s_minus) < 1e-10:
                    continue
                H_minus = np.fft.rfft(h_minus, fft_size)
                Z_minus = np.fft.irfft(H_minus * H_minus, fft_size)[:2 * N - 1]
                val_minus = abs(2.0 * N * smooth_max(Z_minus, beta) / (s_minus * s_minus))

                grad_c[m] = (val_plus - val_minus) / (2 * eps)

            # Update
            v = momentum * v - lr * grad_c
            c = c + v

            # Project: keep c[0] significant
            if abs(c[0]) < 0.05:
                c[0] = 0.05 * max(1.0, np.max(np.abs(c)))

    # Final verification with exact np.convolve
    h_best = compute_h(best_c)
    s = np.sum(h_best)
    if abs(s) > 1e-10:
        z = np.convolve(h_best, h_best)
        exact_val = abs(2.0 * N * np.max(z) / (s * s))
        if exact_val < best_val:
            best_val = exact_val
        return h_best
    else:
        return np.ones(N)
