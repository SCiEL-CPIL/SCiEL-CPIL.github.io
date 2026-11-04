"""Optimized program for the second autocorrelation inequality task using DFT parameterization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
W = 1.0 / (2.0 * N)  # interval width = 0.01


def _autoconvolution_peak(h, n_grid=2048):
    """Compute the peak of the autoconvolution of the step function defined by h."""
    h = np.asarray(h, dtype=float)
    n = len(h)
    w = W
    
    t = np.linspace(-0.5, 0.5, n_grid, endpoint=False)
    dt = t[1] - t[0]
    
    f_grid = np.zeros(n_grid)
    centers = -0.25 + (np.arange(n) + 0.5) * w
    for i in range(n):
        lo = centers[i] - w / 2
        hi = centers[i] + w / 2
        idx_lo = int(round((lo + 0.5) / dt))
        idx_hi = int(round((hi + 0.5) / dt))
        idx_lo = max(0, min(idx_lo, n_grid))
        idx_hi = max(0, min(idx_hi, n_grid))
        if idx_hi > idx_lo:
            f_grid[idx_lo:idx_hi] += h[i]
    
    f_fft = np.fft.fft(f_grid, n=2 * n_grid)
    conv_fft = f_fft * f_fft
    conv = np.real(np.fft.ifft(conv_fft))[:n_grid] * dt
    
    return np.max(conv)


def _l2_norm_sq(h):
    """Compute squared L2 norm."""
    return np.sum(np.asarray(h, dtype=float)**2) * W


def _l1_norm(h):
    """Compute L1 norm."""
    return np.sum(np.asarray(h, dtype=float)) * W


def _ratio(h):
    """Compute ||f||_2^2 / (||f||_1 * ||f*f||_inf)."""
    h = np.asarray(h, dtype=float)
    l1 = _l1_norm(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = _l2_norm_sq(h)
    peak = _autoconvolution_peak(h)
    if peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * peak)


def _smooth_max(conv, temp=0.002):
    """Soft-max approximation of the maximum using log-sum-exp."""
    shifted = conv - np.max(conv)
    return np.max(shifted) + np.log(np.sum(np.exp(shifted / temp))) * temp


def _ratio_smooth(h, temp=0.002):
    """Smooth version of the ratio using soft-max for the autoconvolution peak."""
    h = np.asarray(h, dtype=float)
    l1 = _l1_norm(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = _l2_norm_sq(h)
    
    n_grid = 1024
    t = np.linspace(-0.5, 0.5, n_grid, endpoint=False)
    dt = t[1] - t[0]
    
    f_grid = np.zeros(n_grid)
    centers = -0.25 + (np.arange(len(h)) + 0.5) * W
    for i in range(len(h)):
        lo = centers[i] - W / 2
        hi = centers[i] + W / 2
        idx_lo = int(round((lo + 0.5) / dt))
        idx_hi = int(round((hi + 0.5) / dt))
        idx_lo = max(0, min(idx_lo, n_grid))
        idx_hi = max(0, min(idx_hi, n_grid))
        if idx_hi > idx_lo:
            f_grid[idx_lo:idx_hi] += h[i]
    
    f_fft = np.fft.fft(f_grid, n=2 * n_grid)
    conv_fft = f_fft * f_fft
    conv = np.real(np.fft.ifft(conv_fft))[:n_grid] * dt
    
    peak_smooth = _smooth_max(conv, temp)
    if peak_smooth < 1e-15:
        return 0.0
    return l2_sq / (l1 * peak_smooth)


def _h_from_complex(z):
    """Compute h_k = |sum_j z_j * omega^(jk)|^2 for the DFT parameterization.
    
    z is a complex vector of length N.
    Returns h of length N where h_k = |P(omega^k)|^2 / N.
    """
    z = np.asarray(z, dtype=complex)
    # P(omega^k) = sum_j z_j * omega^(j*k) where omega = exp(-2*pi*i/N)
    # This is essentially the DFT of z (with appropriate sign convention)
    P = np.fft.fft(z)  # P[k] = sum_j z_j * exp(-2*pi*i*j*k/N)
    h = np.abs(P)**2 / N
    return h


def _objective_complex(z_flat):
    """Objective function for complex parameterization.
    
    z_flat is a real vector of length 2*N representing z = z_flat[::2] + 1j*z_flat[1::2].
    Returns negative log ratio for minimization.
    """
    z = z_flat[::2] + 1j * z_flat[1::2]
    h = _h_from_complex(z)
    
    l1 = _l1_norm(h)
    if l1 < 1e-15:
        return 1e10
    
    l2_sq = _l2_norm_sq(h)
    peak = _autoconvolution_peak(h)
    if peak < 1e-15:
        return 1e10
    
    ratio = l2_sq / (l1 * peak)
    if ratio <= 0:
        return 1e10
    
    return -np.log(ratio)


def _objective_complex_smooth(z_flat, temp=0.002):
    """Smooth objective function for complex parameterization."""
    z = z_flat[::2] + 1j * z_flat[1::2]
    h = _h_from_complex(z)
    
    l1 = _l1_norm(h)
    if l1 < 1e-15:
        return 1e10
    
    l2_sq = _l2_norm_sq(h)
    
    n_grid = 1024
    t = np.linspace(-0.5, 0.5, n_grid, endpoint=False)
    dt = t[1] - t[0]
    
    f_grid = np.zeros(n_grid)
    centers = -0.25 + (np.arange(N) + 0.5) * W
    for i in range(N):
        lo = centers[i] - W / 2
        hi = centers[i] + W / 2
        idx_lo = int(round((lo + 0.5) / dt))
        idx_hi = int(round((hi + 0.5) / dt))
        idx_lo = max(0, min(idx_lo, n_grid))
        idx_hi = max(0, min(idx_hi, n_grid))
        if idx_hi > idx_lo:
            f_grid[idx_lo:idx_hi] += h[i]
    
    f_fft = np.fft.fft(f_grid, n=2 * n_grid)
    conv_fft = f_fft * f_fft
    conv = np.real(np.fft.ifft(conv_fft))[:n_grid] * dt
    
    peak_smooth = _smooth_max(conv, temp)
    if peak_smooth < 1e-15:
        return 1e10
    
    ratio = l2_sq / (l1 * peak_smooth)
    if ratio <= 0:
        return 1e10
    
    return -np.log(ratio)


def _optimize_complex(z0_flat, maxiter=3000, ftol=1e-10, gtol=1e-8):
    """Optimize in complex space using L-BFGS-B."""
    result = minimize(_objective_complex_smooth, z0_flat, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50})
    return result.x


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    best_h = np.zeros(N)
    best_h[N // 2] = 1.0
    best_score = _ratio(best_h)

    # Generate diverse starting points in complex space
    z_starts = []

    # Single spike: z = [1, 0, 0, ...]
    z0 = np.zeros(N, dtype=complex)
    z0[0] = 1.0
    z_starts.append(z0)

    # Uniform: z = [1, 1, 1, ...]
    z1 = np.ones(N, dtype=complex)
    z_starts.append(z1)

    # Sidon-like set: select indices to minimize autocorrelation
    # Use a greedy approach
    sidon_indices = [0]
    for i in range(1, N):
        can_add = True
        for j in range(len(sidon_indices)):
            for m in range(len(sidon_indices)):
                if j == m:
                    continue
                diff1 = i - sidon_indices[j]
                diff2 = sidon_indices[m] - sidon_indices[j]
                if diff1 == diff2 and diff1 != 0:
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            sidon_indices.append(i)
        if len(sidon_indices) >= 10:
            break

    z_sidon = np.zeros(N, dtype=complex)
    for idx in sidon_indices:
        z_sidon[idx] = 1.0
    z_starts.append(z_sidon)

    # Gaussian-like in frequency domain
    for sigma in [1.5, 2.5, 4.0]:
        k = np.arange(N)
        z_gauss = np.exp(-0.5 * ((k - N // 2) / sigma)**2)
        z_starts.append(z_gauss)

    # Random starts
    np.random.seed(42)
    for _ in range(5):
        z_rand = np.random.randn(N) + 1j * np.random.randn(N)
        z_starts.append(z_rand)

    # Evaluate and optimize from each starting point
    for z0 in z_starts:
        if time.time() > deadline - 25:
            break

        # Quick evaluation
        h_eval = _h_from_complex(z0)
        s_eval = _ratio(h_eval)
        if s_eval > best_score:
            best_score = s_eval
            best_h = h_eval.copy()

        # Optimize using smooth surrogate
        if time.time() < deadline - 20:
            try:
                z0_flat = np.concatenate([z0.real, z0.imag])
                z_opt_flat = _optimize_complex(z0_flat, maxiter=2000, ftol=1e-10, gtol=1e-8)
                z_opt = z_opt_flat[::2] + 1j * z_opt_flat[1::2]
                h_opt = _h_from_complex(z_opt)
                s_opt = _ratio(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
            except Exception:
                pass

    # Strategy 2: Direct optimization from best found so far
    if time.time() < deadline - 20:
        # Convert best_h back to complex space via inverse DFT
        # h_k = |P(omega^k)|^2 / N, so P(omega^k) = sqrt(N*h_k) * phase
        # We can use the inverse DFT to get z
        P = np.sqrt(N * best_h)  # Assume positive real phase for simplicity
        z_best = np.fft.ifft(P)
        z_best_flat = np.concatenate([z_best.real, z_best.imag])

        try:
            z_opt_flat = _optimize_complex(z_best_flat, maxiter=5000, ftol=1e-12, gtol=1e-10)
            z_opt = z_opt_flat[::2] + 1j * z_opt_flat[1::2]
            h_opt = _h_from_complex(z_opt)
            s_opt = _ratio(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Strategy 3: Final high-precision polish with exact scoring
    if time.time() < deadline - 10:
        try:
            P = np.sqrt(N * best_h)
            z_best = np.fft.ifft(P)
            z_best_flat = np.concatenate([z_best.real, z_best.imag])

            result = minimize(_objective_complex, z_best_flat, method='L-BFGS-B',
                              options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            z_opt = result.x[::2] + 1j * result.x[1::2]
            h_opt = _h_from_complex(z_opt)
            s_opt = _ratio(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Strategy 4: Try more random restarts if time permits
    if time.time() < deadline - 15:
        np.random.seed(123)
        for _ in range(5):
            if time.time() > deadline - 10:
                break

            z_rand = np.random.randn(N) + 1j * np.random.randn(N)
            h_rand = _h_from_complex(z_rand)
            s_rand = _ratio(h_rand)
            if s_rand > best_score:
                best_score = s_rand
                best_h = h_rand.copy()

            # Quick optimization from random start
            if time.time() < deadline - 8:
                try:
                    z_rand_flat = np.concatenate([z_rand.real, z_rand.imag])
                    z_opt_flat = _optimize_complex(z_rand_flat, maxiter=1500, ftol=1e-10, gtol=1e-8)
                    z_opt = z_opt_flat[::2] + 1j * z_opt_flat[1::2]
                    h_opt = _h_from_complex(z_opt)
                    s_opt = _ratio(h_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_h = h_opt.copy()
                except Exception:
                    pass

    # Final polish
    if time.time() < deadline - 3:
        try:
            P = np.sqrt(N * best_h)
            z_best = np.fft.ifft(P)
            z_best_flat = np.concatenate([z_best.real, z_best.imag])

            result = minimize(_objective_complex, z_best_flat, method='L-BFGS-B',
                              options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            z_opt = result.x[::2] + 1j * result.x[1::2]
            h_opt = _h_from_complex(z_opt)
            s_opt = _ratio(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Ensure nonnegativity and return
    best_h = np.abs(best_h)
    return list(best_h)
# EVOLVE-BLOCK-END
