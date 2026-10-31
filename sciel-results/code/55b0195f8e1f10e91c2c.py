import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def compute_overlap(h):
        """Compute the maximum overlap: max_k sum_i h[i] * (1 - h[(i+k)%N]) = N/2 - min_k corr[k]"""
        # Circular autocorrelation via FFT
        fft_h = np.fft.fft(h, n=2 * N)
        corr = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
        min_corr = np.min(corr)
        return N / 2.0 - min_corr

    def objective(x):
        """Objective: max overlap + penalty for constraint violations."""
        h = np.clip(x, 0, 1)
        # Soft penalty for sum constraint
        sum_val = h.sum()
        sum_penalty = 100.0 * (sum_val - N / 2.0) ** 2
        # Compute overlap
        fft_h = np.fft.fft(h, n=2 * N)
        corr = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
        min_corr = np.min(corr)
        overlap = N / 2.0 - min_corr
        return overlap + sum_penalty

    def gradient(x):
        """Gradient of objective."""
        h = np.clip(x, 0, 1)
        sum_val = h.sum()
        sum_grad = 200.0 * (sum_val - N / 2.0)
        
        fft_h = np.fft.fft(h, n=2 * N)
        corr = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
        k_min = np.argmin(corr)
        
        # Gradient of -min_corr w.r.t. h[i] is -(h[(i+k_min)%N] + h[(i-k_min)%N])
        idx = np.arange(N)
        corr_grad = -(h[(idx + k_min) % N] + h[(idx - k_min) % N])
        
        return corr_grad + sum_grad

    def normalize(h):
        """Project h onto the feasible set: 0 <= h <= 1, sum(h) = N/2."""
        h = np.clip(h, 0, 1)
        s = h.sum()
        if s > 0 and abs(s - N / 2.0) > 1e-10:
            # Iteratively adjust to meet sum constraint while staying in [0,1]
            for _ in range(100):
                diff = N / 2.0 - h.sum()
                if abs(diff) < 1e-12:
                    break
                if diff > 0:
                    # Need to increase sum; add to elements that can be increased
                    mask = h < 1.0 - 1e-12
                    if not mask.any():
                        break
                    add = diff / mask.sum()
                    h[mask] = np.clip(h[mask] + add, 0, 1)
                else:
                    # Need to decrease sum; subtract from elements that can be decreased
                    mask = h > 1e-12
                    if not mask.any():
                        break
                    sub = -diff / mask.sum()
                    h[mask] = np.clip(h[mask] - sub, 0, 1)
        return h

    # Multiple restarts with different initializations
    rng = np.random.default_rng(123)
    best_h = np.full(N, 0.5)
    best_score = compute_overlap(best_h)

    # Try several structured initializations
    inits = [np.full(N, 0.5)]
    # Sinusoidal patterns
    for freq in [1, 2, 3, 5, 7, 11]:
        for amp in [0.15, 0.25, 0.35, 0.45]:
            h = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
            h = normalize(h)
            inits.append(h)
    # Random perturbations
    for _ in range(10):
        h = 0.5 + 0.1 * rng.standard_normal(N)
        h = normalize(h)
        inits.append(h)

    # Optimize from each initialization
    for h0 in inits:
        if time.time() > deadline - 10.0:
            break

        h0 = h0.copy()
        res = minimize(
            objective,
            h0,
            jac=gradient,
            method='L-BFGS-B',
            bounds=[(0, 1)] * N,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
        )

        h_opt = normalize(res.x)
        score = compute_overlap(h_opt)

        if score < best_score:
            best_score = score
            best_h = h_opt.copy()

    # Final refinement with tighter tolerance
    if time.time() < deadline - 5.0:
        res = minimize(
            objective,
            best_h,
            jac=gradient,
            method='L-BFGS-B',
            bounds=[(0, 1)] * N,
            options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12}
        )
        h_opt = normalize(res.x)
        score = compute_overlap(h_opt)
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()

    # Ensure exact constraints
    best_h = normalize(best_h)

    # Final safety check
    if best_h.min() < -1e-9 or best_h.max() > 1 + 1e-9:
        return np.full(N, 0.5)
    if abs(best_h.sum() - N / 2.0) > 1e-6:
        return np.full(N, 0.5)

    return best_h
