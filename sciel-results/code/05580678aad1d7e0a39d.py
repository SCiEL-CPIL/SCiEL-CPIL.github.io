import numpy as np
import time

N = 600
W = 1.0 / 1200.0

def compute_ratio(h, n_grid=2000):
    """Compute max(autoconv) / l2_sq for step function h.

    h: array of length N, step heights on equal intervals of [-1/4, 1/4].
    Returns the ratio of the maximum of the autoconvolution to the L2 norm squared.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    if np.sum(h) < 1e-15:
        return float('inf')

    # Upsample to a fine grid for accurate convolution peak detection
    x_fine = np.linspace(-0.25, 0.25, n_grid, endpoint=False)
    # Map each fine point to its step interval index
    cell_idx = np.clip(((x_fine + 0.25) / W).astype(np.int32), 0, N - 1)
    f_fine = h[cell_idx]

    # Circular autoconvolution via FFT
    n_fft = 1 << (2 * n_grid - 1).bit_length()
    F = np.fft.rfft(f_fine, n_fft)
    conv = np.fft.irfft(F * F, n_fft)[:2 * n_grid - 1]
    dx = 0.5 / n_grid
    conv *= dx
    max_conv = np.max(conv)

    l2_sq = np.sum(f_fine ** 2) * dx
    if l2_sq < 1e-15:
        return float('inf')
    return max_conv / l2_sq


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill

    i = np.arange(N)
    t = i / N  # 0 to ~1

    candidates = []

    # 1. Uniform (baseline, theoretical ratio = 1.0)
    candidates.append(np.ones(N))

    # 2. Slight variations around uniform
    for amp in [0.01, 0.02, 0.05, 0.1]:
        candidates.append(1.0 + amp * np.cos(2 * np.pi * t))

    # 3. Cosine bell shapes
    candidates.append(0.5 * (1 + np.cos(2 * np.pi * t)))
    candidates.append(np.maximum(0, np.cos(np.pi * (t - 0.5))))

    # 4. Triangular
    candidates.append(np.maximum(0, 1 - 2 * np.abs(t - 0.5)))

    # 5. Power law shapes
    for p in [0.5, 2.0, 4.0]:
        base = 1 - np.abs(2 * t - 1)
        candidates.append(np.maximum(0, base ** p))

    # 6. Gaussian-like
    for sigma in [0.1, 0.2, 0.3, 0.5]:
        candidates.append(np.exp(-0.5 * ((t - 0.5) / sigma) ** 2))

    # Evaluate all candidates
    best_h = np.ones(N)
    best_ratio = compute_ratio(best_h)

    for h in candidates:
        h = np.clip(h, 0.0, None)
        if np.sum(h) < 1e-15:
            continue
        r = compute_ratio(h)
        if r < best_ratio:
            best_ratio = r
            best_h = h.copy()

    # Local refinement using coordinate descent (faster than full gradient)
    h = best_h.copy()
    lr = 0.01
    step_size = max(0.01, np.max(h) * 0.01)

    for _ in range(200):
        if time.time() > deadline:
            break

        improved = False
        # Try perturbing a subset of coordinates
        n_trials = 30
        idxs = np.random.choice(N, size=n_trials, replace=False)

        for idx in idxs:
            h_old = h[idx]
            h[idx] = h_old + step_size
            r_plus = compute_ratio(h)
            h[idx] = h_old - step_size
            r_minus = compute_ratio(h)
            h[idx] = h_old

            if r_plus < best_ratio:
                h[idx] = h_old + step_size
                best_ratio = r_plus
                improved = True
            elif r_minus < best_ratio:
                h[idx] = h_old - step_size
                best_ratio = r_minus
                improved = True

        if not improved:
            # Reduce step size if no improvement
            step_size *= 0.8
            if step_size < 1e-6:
                break

    # Final evaluation
    final_r = compute_ratio(best_h)
    if final_r < best_ratio:
        best_ratio = final_r

    return best_h.tolist()
