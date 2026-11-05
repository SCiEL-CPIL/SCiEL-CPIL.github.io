"""Combined program: non-harmonic band-limited signal representation with
continuous (arbitrary) frequencies, optimized via gradient-based methods.

The key structural idea: represent h as a sum of complex sinusoids
    h[i] = a0 + sum_k 2*Re( c_k * exp(2*pi*i*f_k*i/N) )
where the frequencies f_k are continuous (not restricted to integers),
amplitudes c_k are complex, and a0 is a real DC term. This gives more
flexible autocorrelation sidelobe shaping than the discrete DFT/DCT bases.

The refinement machinery (smooth-max log-sum-exp objective, L-BFGS-B,
progressively tighter tau) and FFT-based autoconvolution are retained.
"""
import numpy as np
import time
import sys

N = 400
_FFT_SIZE = 1024  # next power of 2 >= 2*400-1 = 799

# EVOLVE-BLOCK-START

def _autoconvolve_fft(h):
    """Compute full autoconvolution h*h via FFT. O(N log N)."""
    size = _FFT_SIZE
    H = np.fft.rfft(h, n=size)
    conv = np.fft.irfft(H * H, n=size)
    return conv[:2 * len(h) - 1]

def construct():
    t0 = time.time()
    DEADLINE = t0 + 145.0  # safety margin before 150s kill

    n_idx = np.arange(N, dtype=np.float64)

    def compute_value(h):
        """Compute the exact objective value (lower is better)."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = _autoconvolve_fft(h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    def smooth_max(x, tau=0.05):
        """Smooth approximation of max via log-sum-exp."""
        x_max = np.max(x)
        return x_max + tau * np.log(np.sum(np.exp((x - x_max) / tau)))

    def nonharmonic_heights(params, K):
        """Build h from non-harmonic sinusoid parameters.
        params layout: [a0, f1, r1, p1, f2, r2, p2, ..., fK, rK, pK]
        h[i] = a0 + sum_k r_k * cos(2*pi*f_k*i/N + p_k)
        f_k are continuous (non-integer) frequencies.
        """
        a0 = params[0]
        h = np.full(N, a0, dtype=np.float64)
        for k in range(K):
            f = params[1 + 3 * k]
            r = params[2 + 3 * k]
            p = params[3 + 3 * k]
            h += r * np.cos(2.0 * np.pi * f * n_idx / N + p)
        return h

    def nonharmonic_objective(params, K, tau=0.05):
        """Differentiable objective for non-harmonic parameterization."""
        h = nonharmonic_heights(params, K)
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = _autoconvolve_fft(h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)

    def direct_objective(h, tau=0.05):
        """Differentiable objective in direct h-space."""
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = _autoconvolve_fft(h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)

    # DFT-basis objective (kept from current program as a fallback / cross-check)
    def dft_heights(params, K):
        dc = params[0]
        h = np.full(N, dc, dtype=np.float64)
        for k in range(1, K + 1):
            r = params[1 + 2 * (k - 1)]
            p = params[2 + 2 * (k - 1)]
            h += r * np.cos(2.0 * np.pi * k * n_idx / N + p)
        return h

    def dft_objective(params, K, tau=0.05):
        h = dft_heights(params, K)
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = _autoconvolve_fft(h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)

    # Initialize with constant ones (value = 2.0, score = 1.0)
    best_h = np.ones(N, dtype=np.float64)
    best_val = 2.0

    # Phase 1: Differential evolution in non-harmonic parameter space (global search)
    if time.time() < DEADLINE - 50.0:
        try:
            from scipy.optimize import differential_evolution

            K = 4  # number of non-harmonic sinusoids
            # params: [a0, f1, r1, p1, f2, r2, p2, f3, r3, p3, f4, r4, p4]
            bounds = [(-2.0, 2.0)]  # a0 (DC)
            for k in range(K):
                bounds.append((0.5, 8.0))   # frequency f_k (continuous, >0)
                bounds.append((-1.5, 1.5))  # amplitude r_k
                bounds.append((0.0, 2.0 * np.pi))  # phase p_k

            def _de_obj_nh(params):
                if time.time() > DEADLINE - 10.0:
                    return 1e10
                return nonharmonic_objective(params, K, tau=0.05)

            de_result = differential_evolution(
                _de_obj_nh,
                bounds,
                maxiter=60,
                popsize=15,
                seed=42,
                tol=1e-6,
                polish=False,
                init='sobol',
                mutation=(0.4, 0.8),
                recombination=0.9,
            )

            if de_result.fun < best_val:
                h_cand = nonharmonic_heights(de_result.x, K)
                val_cand = compute_value(h_cand)
                if val_cand < best_val and np.all(np.isfinite(h_cand)) and abs(np.sum(h_cand)) > 1e-6:
                    best_h = h_cand.copy()
                    best_val = val_cand
                    print(f"DE non-harmonic: {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"DE non-harmonic error: {e}", file=sys.stderr)

    # Phase 2: L-BFGS-B refinement in non-harmonic parameter space
    if time.time() < DEADLINE - 35.0:
        try:
            from scipy.optimize import minimize

            K = 4
            # Start from DE result if available, else from a reasonable init
            if 'de_result' in dir() and de_result.fun < best_val + 0.2:
                x0 = de_result.x.copy()
            else:
                x0 = np.zeros(1 + 3 * K)
                x0[0] = 1.0  # a0 = 1
                rng = np.random.RandomState(7)
                for k in range(K):
                    x0[1 + 3 * k] = rng.uniform(0.5, 6.0)   # frequency
                    x0[2 + 3 * k] = rng.uniform(-0.5, 0.5)  # amplitude
                    x0[3 + 3 * k] = rng.uniform(0, 2 * np.pi)  # phase

            def _nh_obj_local(params):
                if time.time() > DEADLINE - 5.0:
                    return 1e10
                return nonharmonic_objective(params, K, tau=0.02)

            result = minimize(
                _nh_obj_local,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-11}
            )

            if result.fun < best_val:
                h_cand = nonharmonic_heights(result.x, K)
                val_cand = compute_value(h_cand)
                if val_cand < best_val and np.all(np.isfinite(h_cand)) and abs(np.sum(h_cand)) > 1e-6:
                    best_h = h_cand.copy()
                    best_val = val_cand
                    print(f"Non-harmonic L-BFGS: {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Non-harmonic L-BFGS error: {e}", file=sys.stderr)

    # Phase 3: Multiple restarts in non-harmonic space with different initializations
    if time.time() < DEADLINE - 20.0:
        try:
            from scipy.optimize import minimize

            K = 4
            rng = np.random.RandomState(123)

            for restart in range(10):
                if time.time() > DEADLINE - 10.0:
                    break

                x0 = np.zeros(1 + 3 * K)
                x0[0] = rng.uniform(0.5, 2.0)
                for k in range(K):
                    x0[1 + 3 * k] = rng.uniform(0.3, 10.0)   # frequency
                    x0[2 + 3 * k] = rng.uniform(-1.0, 1.0)   # amplitude
                    x0[3 + 3 * k] = rng.uniform(0, 2 * np.pi)  # phase

                def _obj_r(params):
                    if time.time() > DEADLINE - 5.0:
                        return 1e10
                    return nonharmonic_objective(params, K, tau=0.03)

                result = minimize(
                    _obj_r,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-11}
                )

                if result.fun < best_val:
                    h_cand = nonharmonic_heights(result.x, K)
                    val_cand = compute_value(h_cand)
                    if val_cand < best_val and np.all(np.isfinite(h_cand)) and abs(np.sum(h_cand)) > 1e-6:
                        best_h = h_cand.copy()
                        best_val = val_cand
                        print(f"NH restart {restart}: {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"NH restarts error: {e}", file=sys.stderr)

    # Phase 4: Cross-check with DFT-basis refinement (kept from current program)
    if time.time() < DEADLINE - 12.0:
        try:
            from scipy.optimize import minimize

            K_dft = 5
            rng = np.random.RandomState(200)
            x0 = np.zeros(1 + 2 * K_dft)
            x0[0] = 1.0
            for k in range(1, K_dft + 1):
                x0[1 + 2 * (k - 1)] = rng.uniform(-0.5, 0.5)
                x0[2 + 2 * (k - 1)] = rng.uniform(0, 2 * np.pi)

            def _dft_obj_local(params):
                if time.time() > DEADLINE - 5.0:
                    return 1e10
                return dft_objective(params, K_dft, tau=0.02)

            result = minimize(
                _dft_obj_local,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-11}
            )

            if result.fun < best_val:
                h_cand = dft_heights(result.x, K_dft)
                val_cand = compute_value(h_cand)
                if val_cand < best_val and np.all(np.isfinite(h_cand)) and abs(np.sum(h_cand)) > 1e-6:
                    best_h = h_cand.copy()
                    best_val = val_cand
                    print(f"DFT L-BFGS: {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"DFT L-BFGS error: {e}", file=sys.stderr)

    # Phase 5: Direct h-space refinement with L-BFGS-B
    if time.time() < DEADLINE - 8.0:
        try:
            from scipy.optimize import minimize

            def _direct_obj(h):
                if time.time() > DEADLINE - 5.0:
                    return 1e10
                return direct_objective(h, tau=0.01)

            result = minimize(
                _direct_obj,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-11}
            )

            if result.fun < best_val:
                h_cand = result.x
                val_cand = compute_value(h_cand)
                if val_cand < best_val and np.all(np.isfinite(h_cand)) and abs(np.sum(h_cand)) > 1e-6:
                    best_h = h_cand.copy()
                    best_val = val_cand
                    print(f"Direct L-BFGS: {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Direct L-BFGS error: {e}", file=sys.stderr)

    # Phase 6: Tight polish with smaller tau
    if time.time() < DEADLINE - 3.0 and best_val < 2.0:
        try:
            from scipy.optimize import minimize

            def _tight_obj(h):
                if time.time() > DEADLINE - 2.0:
                    return 1e10
                return direct_objective(h, tau=0.003)

            result = minimize(
                _tight_obj,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12}
            )

            if result.fun < best_val:
                h_cand = result.x
                val_cand = compute_value(h_cand)
                if val_cand < best_val and np.all(np.isfinite(h_cand)) and abs(np.sum(h_cand)) > 1e-6:
                    best_h = h_cand.copy()
                    best_val = val_cand
                    print(f"Tight polish: {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Tight polish error: {e}", file=sys.stderr)

    # Phase 7: Final ultra-tight polish
    if time.time() < DEADLINE - 1.5 and best_val < 1.5:
        try:
            from scipy.optimize import minimize

            def _ultra_obj(h):
                if time.time() > DEADLINE - 1.0:
                    return 1e10
                return direct_objective(h, tau=0.001)

            result = minimize(
                _ultra_obj,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-13}
            )

            if result.fun < best_val:
                h_cand = result.x
                val_cand = compute_value(h_cand)
                if val_cand < best_val and np.all(np.isfinite(h_cand)) and abs(np.sum(h_cand)) > 1e-6:
                    best_h = h_cand.copy()
                    best_val = val_cand
                    print(f"Ultra polish: {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Ultra polish error: {e}", file=sys.stderr)

    # Final safety check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N, dtype=np.float64)

    return best_h

# EVOLVE-BLOCK-END
