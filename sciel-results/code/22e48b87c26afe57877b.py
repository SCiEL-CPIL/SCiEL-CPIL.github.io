"""
Second autocorrelation inequality construction.

Strategy:
  The ratio R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) is scale-invariant.
  We seek nonnegative h of length 50 that maximizes R.

  Key structural change: parameterize h as the squared magnitudes of a sparse
  complex vector g (h = |g|^2).  We fix a random sparse support (k non-zero
  entries out of 50) and optimize only the 2k real parameters (real and imag
  parts of the non-zero g entries) to minimize the autocorrelation peak,
  equivalently to maximize R.

  This exploits the fact that sparse polynomials (few non-zero Fourier
  coefficients) tend to produce flatter autocorrelation profiles, reducing
  ||h*h||_inf relative to ||h||_2^2.

  We try multiple random sparse supports and multiple random phase initializations,
  using L-BFGS-B on the reduced parameter space.  A final polish pass refines
  the best candidate.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12):
        h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm * h_norm)
    if l2_sq <= 1e-15:
        return 0.0
    conv = np.correlate(h_norm, h_norm, mode='full')
    conv_max = np.max(np.abs(conv))
    if conv_max <= 1e-15:
        return 0.0
    return l2_sq / conv_max

def _ratio_from_g(g):
    """Ratio computed from complex vector g via h = |g|^2."""
    h = np.abs(g) ** 2
    return _ratio(h)

def _ratio_neg_from_params(params, support):
    """Negative ratio for minimization. params is 2*k real numbers.
    support is an array of indices where g is non-zero."""
    k = len(support)
    g = np.zeros(N, dtype=np.complex128)
    g[support] = params[0::2] + 1j * params[1::2]
    return -_ratio_from_g(g)

def _gaussian_profile(x, center, width):
    return np.exp(-0.5 * ((x - center) / width) ** 2)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    def _time_left():
        return deadline - time.time()

    best_score = 0.0
    best_h = np.ones(N) / N
    best_score = _ratio(best_h)

    # Also try a few simple profiles as initial guesses
    x = np.linspace(0, 1, N)
    profiles = [
        np.ones(N) / N,
        _gaussian_profile(x, 0.5, 0.15),
        _gaussian_profile(x, 0.5, 0.25),
        _gaussian_profile(x, 0.3, 0.2) + _gaussian_profile(x, 0.7, 0.2),
        np.maximum(0, 1 - np.abs(x - 0.5) * 2),  # triangle
        np.cos(np.pi * (x - 0.5)) ** 2 + 0.1,    # cosine bump
    ]
    for p in profiles:
        p = np.maximum(p, 0)
        s = _ratio(p)
        if s > best_score:
            best_score = s
            best_h = p.copy()

    # Sparse support optimization
    np.random.seed(12345)

    # Try various sparsity levels: k non-zero entries out of N=50
    k_values = [5, 6, 7, 8, 10, 12, 15]

    for k in k_values:
        if _time_left() < 5.0:
            break
        # Number of random supports to try for this k
        n_supports = max(3, int(20 * _time_left() / (140.0 - (time.time() - start_time))))
        n_supports = min(n_supports, 15)

        for trial in range(n_supports):
            if _time_left() < 3.0:
                break

            # Random support
            support = np.random.choice(N, size=k, replace=False)
            support.sort()

            # Multiple random phase initializations
            n_init = 3
            for init_idx in range(n_init):
                if _time_left() < 2.0:
                    break

                # Random complex values for non-zero entries
                params_init = np.random.randn(2 * k) * 0.5
                # Ensure non-trivial
                if np.allclose(params_init, 0):
                    params_init[0] = 1.0

                def objective(params):
                    return _ratio_neg_from_params(params, support)

                try:
                    res = minimize(
                        objective,
                        params_init,
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'maxfun': 5000, 'ftol': 1e-12}
                    )
                    params_opt = res.x
                    g = np.zeros(N, dtype=np.complex128)
                    g[support] = params_opt[0::2] + 1j * params_opt[1::2]
                    h_opt = np.abs(g) ** 2

                    s = _ratio(h_opt)
                    if s > best_score:
                        best_score = s
                        best_h = h_opt.copy()
                except Exception:
                    pass

    # Polish the best solution using full parameterization (all 50 entries)
    if _time_left() > 15.0:
        try:
            # Use best_h as starting point for a full optimization
            # Parameterize as |g|^2 where g = sqrt(h) * exp(i*theta)
            # Optimize both magnitudes and phases
            g_mag = np.sqrt(np.maximum(best_h, 0))
            g_phase = np.random.randn(N) * 0.1  # small random phase perturbation

            def full_objective(params):
                mag = np.abs(params[0:N])
                phase = params[N:2*N]
                g = mag * np.exp(1j * phase)
                return -_ratio_from_g(g)

            params_init = np.concatenate([g_mag, g_phase])

            res = minimize(
                full_objective,
                params_init,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-14}
            )
            params_opt = res.x
            g_mag_opt = np.abs(params_opt[0:N])
            g_phase_opt = params_opt[N:2*N]
            g_opt = g_mag_opt * np.exp(1j * g_phase_opt)
            h_opt = np.abs(g_opt) ** 2

            s = _ratio(h_opt)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
        except Exception:
            pass

    # Final local refinement with smaller steps
    if _time_left() > 5.0:
        try:
            g_best = np.sqrt(np.maximum(best_h, 0)) + 0j

            def local_obj(params):
                g = params[0::2] + 1j * params[1::2]
                return -_ratio_from_g(g)

            params_init = np.zeros(2 * N)
            params_init[0::2] = g_best.real
            params_init[1::2] = g_best.imag

            res = minimize(
                local_obj,
                params_init,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfun': 20000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            params_opt = res.x
            g_opt = params_opt[0::2] + 1j * params_opt[1::2]
            h_opt = np.abs(g_opt) ** 2

            s = _ratio(h_opt)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
        except Exception:
            pass

    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
