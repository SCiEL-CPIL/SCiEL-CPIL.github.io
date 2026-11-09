import numpy as np
import time
from scipy.optimize import minimize

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def compute_normalized_peak(f):
        """Compute max(f*f) / (integral(f))^2 using FFT-based linear autoconvolution."""
        f_pad = np.zeros(2 * N)
        f_pad[:N] = f
        F = np.fft.fft(f_pad)
        autoconv = np.fft.ifft(F * F).real
        peak = np.max(autoconv)
        integral = np.sum(f)
        if integral < 1e-12:
            return np.inf
        return peak / (integral * integral)

    # Baseline: uniform function
    f_uniform = np.ones(N)
    score_uniform = compute_normalized_peak(f_uniform)

    # Structural change: parameterize f[i] = |z[i]|^2 for complex z
    # This ensures non-negativity and allows smooth optimization.
    # We optimize over 2N real variables (real and imaginary parts of z).
    
    def objective_from_z(z_flat):
        """Objective: normalized peak of autoconvolution, given z as flat real array."""
        z = z_flat[:N] + 1j * z_flat[N:]
        f = np.abs(z) ** 2
        return compute_normalized_peak(f)

    # Initial point: z = 1 (real) gives f = 1 (uniform)
    z0 = np.zeros(2 * N)
    z0[:N] = 1.0

    best_f = f_uniform.copy()
    best_score = score_uniform

    # Phase 1: L-BFGS-B optimization from uniform initialization
    try:
        res = minimize(objective_from_z, z0, method='L-BFGS-B',
                       options={'maxiter': 200, 'maxfun': 4000, 'disp': False})
        if res.fun < best_score:
            z_opt = res.x[:N] + 1j * res.x[N:]
            best_f = np.abs(z_opt) ** 2
            best_score = res.fun
    except Exception:
        pass

    if time.time() > deadline - 30:
        return list(best_f)

    # Phase 2: Try several random restarts with different initializations
    rng = np.random.default_rng(123)
    n_restarts = 8
    for r in range(n_restarts):
        if time.time() > deadline - 15:
            break

        # Random complex initialization with varying structure
        if r % 3 == 0:
            # Smooth random: low-frequency components
            freqs = np.fft.rfftfreq(N)
            z_real = np.fft.irfft(np.exp(-10 * freqs) * rng.standard_normal(N // 2 + 1))
            z_imag = np.fft.irfft(np.exp(-10 * freqs) * rng.standard_normal(N // 2 + 1))
        elif r % 3 == 1:
            # Cosine-like with random phase
            x = np.linspace(0, 2 * np.pi, N)
            z_real = np.cos(x + rng.uniform(0, 2 * np.pi)) * 0.5 + 0.5
            z_imag = 0.3 * np.sin(2 * x + rng.uniform(0, 2 * np.pi))
        else:
            # Random with structure: piecewise constant
            n_pieces = rng.integers(2, 8)
            z_real = np.repeat(rng.uniform(0.3, 1.5, n_pieces), N // n_pieces + 1)[:N]
            z_imag = np.repeat(rng.uniform(-0.5, 0.5, n_pieces), N // n_pieces + 1)[:N]

        z0_r = np.concatenate([z_real, z_imag])

        try:
            res_r = minimize(objective_from_z, z0_r, method='L-BFGS-B',
                             options={'maxiter': 150, 'maxfun': 3000, 'disp': False})
            if res_r.fun < best_score:
                z_opt = res_r.x[:N] + 1j * res_r.x[N:]
                best_f = np.abs(z_opt) ** 2
                best_score = res_r.fun
        except Exception:
            pass

    # Phase 3: Local refinement with Nelder-Mead on a reduced set of parameters
    if time.time() < deadline - 10 and best_score > score_uniform * 0.99:
        # If we haven't improved much, try a different approach:
        # optimize a smooth parametric family
        def parametric_objective(params):
            a, b, c = params
            x = np.linspace(-0.25, 0.25, N, endpoint=False)
            f = a + b * np.cos(2 * np.pi * c * x / 0.5)
            f = np.maximum(f, 0)
            if np.sum(f) < 1e-12:
                return np.inf
            return compute_normalized_peak(f)

        best_param_score = np.inf
        best_params = None
        for init in [(1.0, 0.0, 0.0), (1.0, 0.3, 1.0), (1.0, 0.3, 2.0),
                     (0.8, 0.4, 1.0), (1.0, 0.5, 3.0)]:
            if time.time() > deadline - 5:
                break
            try:
                res_p = minimize(parametric_objective, init, method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
                if res_p.fun < best_param_score:
                    best_param_score = res_p.fun
                    best_params = res_p.x
            except Exception:
                pass

        if best_params is not None and best_param_score < best_score:
            a, b, c = best_params
            x = np.linspace(-0.25, 0.25, N, endpoint=False)
            f_p = a + b * np.cos(2 * np.pi * c * x / 0.5)
            f_p = np.maximum(f_p, 0)
            if np.sum(f_p) > 1e-12:
                best_f = f_p
                best_score = best_param_score

    return list(best_f)
