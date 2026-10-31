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

    # Initial guess: all ones (this gives score = 1.0)
    best_h = np.ones(N)
    best_score = compute_autoconv_max(best_h)

    # The key insight: we want to minimize the peak of the autoconvolution.
    # In the frequency domain, the autoconvolution's Fourier transform is |H(k)|^2.
    # The peak of the autoconvolution is related to the peak of |H(k)|^2.
    # 
    # Strategy: Represent h as a mixture of Gaussians in the spatial domain.
    # This gives a smooth, nonnegative function with controllable frequency content.
    # Optimize the Gaussian parameters (centers, widths, amplitudes) to flatten
    # the power spectrum as much as possible.

    def gaussian_mixture_to_h(params, n_gaussians=10):
        """
        Convert Gaussian mixture parameters to step function values.
        params: array of length 3*n_gaussians: [center_1, width_1, amp_1, center_2, ...]
        Returns: array of length N with nonnegative values.
        """
        h = np.zeros(N)
        x = np.linspace(-0.25, 0.25, N, endpoint=False)
        for i in range(n_gaussians):
            center = params[3*i]
            width = params[3*i+1]
            amp = params[3*i+2]
            # Ensure width is positive
            width = max(abs(width), 0.001)
            h += amp * np.exp(-0.5 * ((x - center) / width) ** 2)
        # Ensure nonnegative
        h = np.maximum(h, 0)
        return h

    def soft_max_objective(h, T=0.01):
        """
        Soft approximation of the peak of the autoconvolution.
        Uses frequency domain for efficiency.
        """
        h = np.asarray(h, dtype=float)
        total = np.sum(h)
        if total < 1e-10:
            return 1e10
        h_norm = h / total
        H = np.fft.fft(h_norm)
        power = np.abs(H) ** 2
        pmax = np.max(power)
        if pmax < 1e-15:
            return 1e10
        shifted = (power - pmax) / T
        shifted = np.clip(shifted, -50, 50)
        soft_max = pmax + T * np.log(np.sum(np.exp(shifted)))
        return soft_max

    def exact_objective(h):
        """Exact objective: peak of autoconvolution normalized by (sum h)^2."""
        return compute_autoconv_max(h)

    # Try different numbers of Gaussians
    for n_gauss in [5, 8, 10, 12, 15]:
        if time.time() > deadline - 20:
            break

        # Initialize with diverse starting points
        n_params = 3 * n_gauss
        np.random.seed(42 + n_gauss)

        for trial in range(5):
            if time.time() > deadline - 15:
                break

            # Random initialization
            params = np.zeros(n_params)
            for i in range(n_gauss):
                params[3*i] = np.random.uniform(-0.2, 0.2)  # center
                params[3*i+1] = np.random.uniform(0.01, 0.15)  # width
                params[3*i+2] = np.random.uniform(0.5, 2.0)  # amplitude

            # Optimize using L-BFGS-B
            def objective(params):
                h = gaussian_mixture_to_h(params, n_gauss)
                return soft_max_objective(h, T=0.005)

            def gradient(params):
                # Numerical gradient
                eps = 1e-7
                grad = np.zeros(n_params)
                f0 = objective(params)
                for i in range(n_params):
                    params_plus = params.copy()
                    params_plus[i] += eps
                    f_plus = objective(params_plus)
                    grad[i] = (f_plus - f0) / eps
                return grad

            try:
                result = optimize.minimize(
                    objective,
                    params,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )

                h_opt = gaussian_mixture_to_h(result.x, n_gauss)
                score_opt = exact_objective(h_opt)

                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()

            except Exception:
                pass

    # Now refine the best solution found using direct optimization on h
    if time.time() < deadline - 10:
        # Use L-BFGS-B directly on h with soft-max objective
        def soft_obj_direct(h):
            return soft_max_objective(h, T=0.002)

        try:
            result = optimize.minimize(
                soft_obj_direct,
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_polished = np.maximum(result.x, 0)
            score_polished = exact_objective(h_polished)
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished.copy()
        except Exception:
            pass

    # Final polish with even lower temperature
    if time.time() < deadline - 5:
        try:
            result = optimize.minimize(
                lambda h: soft_max_objective(h, T=0.0005),
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            h_final = np.maximum(result.x, 0)
            score_final = exact_objective(h_final)
            if score_final < best_score:
                best_score = score_final
                best_h = h_final.copy()
        except Exception:
            pass

    return list(best_h)
