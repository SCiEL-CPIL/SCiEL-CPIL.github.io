import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def compute_value(h):
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def project(h):
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(200):
        s = h.sum()
        if abs(s - target) < 1e-14:
            break
        h = h + (target - s) / N
        h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if abs(s - target) > 1e-12:
        diff = target - s
        free = (h > 1e-12) & (h < 1.0 - 1e-12)
        if np.any(free):
            h[free] += diff / np.sum(free)
            h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    i = np.arange(N)
    best_h = np.full(N, 0.5)
    best_val = compute_value(best_h)  # 0.5

    def objective_and_grad(x):
        h = x + 0.5
        c = np.correlate(h, 1.0 - h, mode="full")
        c_max = np.max(c)
        c_shifted = c - c_max
        exp_c = np.exp(20.0 * c_shifted)
        sum_exp = np.sum(exp_c)
        obj = c_max + np.log(sum_exp) / 20.0
        weights = exp_c / sum_exp

        # Efficient gradient via convolution
        # grad_h[k] = sum_m weights[m] * [(1 - h[k-m]) - h[k+m]]
        # = 1 - sum_m weights[m]*h[k-m] - sum_m weights[m]*h[k+m]
        # sum_m weights[m]*h[k-m] = conv(h, weights)[k] (with proper alignment)
        # sum_m weights[m]*h[k+m] = conv(h, weights_reversed)[k] (with proper alignment)

        # weights has length 2N-1, indexed 0..2N-2
        # h has length N, indexed 0..N-1

        # For sum_m weights[m] * h[k-m]:
        # This is the correlation of h with weights, but let's use convolve
        # np.convolve(h, weights, 'full')[k] = sum_i h[i] * weights[k-i]
        # We want sum_m weights[m] * h[k-m] = sum_i h[i] * weights[k-i] where i=k-m
        # So this IS np.convolve(h, weights, 'full')[k] for k in 0..N-1
        # But we need to be careful about the range.

        # Actually, let's just compute it directly for correctness since N=100
        grad_h = np.zeros(N)
        for m in range(2 * N - 1):
            w = weights[m]
            if w < 1e-10:
                continue

            # Part A: (1 - h[k-m])
            # h[k-m] is non-zero when 0 <= k-m < N, i.e., m <= k < N+m
            # For m < N: h_right[m:N] = h[0:N-m], rest 0
            # For m >= N: all 0 (since k < N and k >= m >= N is impossible)

            # Part B: -h[k+m]
            # h[k+m] is non-zero when 0 <= k+m < N, i.e., -m <= k < N-m
            # For m < N: h_left[0:N-m] = h[m:N], rest 0
            # For m >= N: all 0 (since k < N and k < N-m <= 0 is impossible)

            if m < N:
                # Part A: 1 - h[k-m]
                # h[k-m] for k in [m, N-1] is h[0, N-1-m]
                h_shifted_right = np.zeros(N)
                h_shifted_right[m:] = h[:N - m]
                grad_h += w * (1.0 - h_shifted_right)

                # Part B: -h[k+m]
                # h[k+m] for k in [0, N-1-m] is h[m, N-1]
                h_shifted_left = np.zeros(N)
                h_shifted_left[:N - m] = h[m:]
                grad_h += w * (-h_shifted_left)

        return obj, grad_h

    bounds = [(-0.5, 0.5)] * N

    # Generate diverse starting points
    starts = [np.zeros(N)]

    # Fourier modes
    for k in range(1, 15):
        for A in [0.1, 0.2, 0.3, 0.4, 0.5]:
            starts.append(A * np.sin(2 * np.pi * k * i / N))
            starts.append(A * np.cos(2 * np.pi * k * i / N))

    # Random starts
    np.random.seed(42)
    for _ in range(20):
        x_rand = np.random.randn(N)
        x_rand -= np.mean(x_rand)
        x_rand /= max(np.max(np.abs(x_rand)), 1e-10)
        x_rand *= 0.4
        starts.append(x_rand)

    # Sawtooth and other structured starts
    for A in [0.2, 0.3, 0.4]:
        saw = np.linspace(-A, A, N)
        starts.append(saw)
        starts.append(-saw)
        tri = np.abs(np.linspace(-A, A, N)) - A/2
        starts.append(tri)
        starts.append(-tri)

    # Square wave
    for A in [0.3, 0.4]:
        sq = np.where(i % 2 == 0, A, -A)
        starts.append(sq)
        sq2 = np.where(i % 4 < 2, A, -A)
        starts.append(sq2)

    for x0 in starts:
        if remaining() < 8:
            break

        # Project initial guess to satisfy constraints
        x0 = np.clip(x0, -0.5, 0.5)
        s = np.sum(x0)
        if abs(s) > 1e-10:
            x0 = x0 - s / N
            x0 = np.clip(x0, -0.5, 0.5)
            for _ in range(20):
                s = np.sum(x0)
                if abs(s) < 1e-12:
                    break
                x0 = x0 - s / N
                x0 = np.clip(x0, -0.5, 0.5)

        try:
            res = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )

            x_opt = res.x
            h_opt = x_opt + 0.5
            h_opt = project(h_opt)
            val = compute_value(h_opt)

            if val < best_val:
                best_val = val
                best_h = h_opt.copy()

        except Exception:
            pass

    # Additional local refinement with higher beta
    if remaining() > 15:
        def objective_refine(x):
            h = x + 0.5
            c = np.correlate(h, 1.0 - h, mode="full")
            c_max = np.max(c)
            c_shifted = c - c_max
            exp_c = np.exp(50.0 * c_shifted)
            sum_exp = np.sum(exp_c)
            obj = c_max + np.log(sum_exp) / 50.0
            weights = exp_c / sum_exp

            grad_h = np.zeros(N)
            for m in range(2 * N - 1):
                w = weights[m]
                if w < 1e-10:
                    continue
                if m < N:
                    h_shifted_right = np.zeros(N)
                    h_shifted_right[m:] = h[:N - m]
                    grad_h += w * (1.0 - h_shifted_right)
                    h_shifted_left = np.zeros(N)
                    h_shifted_left[:N - m] = h[m:]
                    grad_h += w * (-h_shifted_left)
            return obj, grad_h

        try:
            res = minimize(
                objective_refine,
                best_h - 0.5,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_ref = res.x + 0.5
            h_ref = project(h_ref)
            val_ref = compute_value(h_ref)
            if val_ref < best_val:
                best_val = val_ref
                best_h = h_ref.copy()
        except Exception:
            pass

    return best_h
