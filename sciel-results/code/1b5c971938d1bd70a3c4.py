"""
Refined strategy: focus on smooth surrogate optimization with epigraph formulation,
adaptive coordinate descent, and structured signal initialization.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fft import rfft, irfft

N = 400
_FFT_N = 2 * N - 1

# EVOLVE-BLOCK-START

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT for speed."""
    H = rfft(h, n=_FFT_N)
    c = irfft(H * H, n=_FFT_N)
    return c

def _soft_max(x, beta):
    """Differentiable approximation to max using log-sum-exp."""
    m = np.max(x)
    x_shifted = x - m
    exp_vals = np.exp(beta * x_shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return m
    return m + (1.0 / beta) * np.log(sum_exp)

def _evaluate_exact(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    return abs(2.0 * N * np.max(c) / (s * s))

def _objective_full(params, beta):
    """Soft-max objective for full h vector."""
    h = params
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    sm = _soft_max(c, beta)
    val = 2.0 * N * sm / (s * s)
    return val

def _objective_epigraph(params, beta):
    """
    Epigraph formulation: minimize t subject to soft_max(c) <= t * s^2 / (2N)
    params = [h_0, ..., h_{N-1}, t]
    """
    h = params[:N]
    t = params[N]
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    sm = _soft_max(c, beta)
    constraint_val = sm - t * s * s / (2.0 * N)
    if constraint_val > 0:
        return t + 100.0 * constraint_val
    return t

def _generate_structured_h(frequency, phase, amplitude, baseline, rng):
    """Generate structured signal: baseline + amplitude * sin(2*pi*frequency*x + phase)."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    h = baseline + amplitude * np.sin(2.0 * np.pi * frequency * x + phase)
    # Add small random perturbation
    h += rng.standard_normal(N) * 0.01 * amplitude
    return h

def _generate_gaussian_h(center, width, amplitude, baseline, rng):
    """Generate Gaussian bump signal."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    h = baseline + amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)
    h += rng.standard_normal(N) * 0.01 * amplitude
    return h

def _generate_two_gaussian_h(c1, c2, w1, w2, a1, a2, baseline, rng):
    """Generate two Gaussian bumps."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    h = baseline + a1 * np.exp(-0.5 * ((x - c1) / w1) ** 2) + a2 * np.exp(-0.5 * ((x - c2) / w2) ** 2)
    h += rng.standard_normal(N) * 0.01 * max(abs(a1), abs(a2))
    return h

def _coordinate_descent_polish(h, max_iters=2000, step_sizes=None, deadline=None):
    """
    Adaptive coordinate descent on exact objective.
    """
    if step_sizes is None:
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    best_h = h.copy()
    best_val = _evaluate_exact(best_h)
    rng = np.random.default_rng(123)
    
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for _ in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            idx = rng.integers(0, N)
            for delta in step_sizes:
                found = False
                for sign in [1, -1]:
                    h_new = best_h.copy()
                    h_new[idx] += sign * delta
                    s_new = np.sum(h_new)
                    if abs(s_new) < 1e-10:
                        continue
                    val_new = _evaluate_exact(h_new)
                    if val_new < best_val - 1e-14:
                        best_val = val_new
                        best_h = h_new
                        improved = True
                        found = True
                        break
                if found:
                    break
        if not improved:
            step_sizes = [s * 0.5 for s in step_sizes]
            if step_sizes[0] < 1e-7:
                break
    
    return best_h, best_val

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin: evaluator kills at 150s

    rng = np.random.default_rng(42)

    best_h = np.ones(N)
    best_val = _evaluate_exact(best_h)  # should be 2.0

    def try_h(h):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h)):
            return
        s = np.sum(h)
        if abs(s) < 1e-10:
            return
        val = _evaluate_exact(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        return val

    # === Phase 0: Structured signal generation ===
    if time.time() < deadline - 60.0:
        # Try various sinusoidal structures
        for freq in [1, 2, 3, 4, 5, 8, 10]:
            for phase in [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]:
                for amp in [0.5, 1.0, 2.0, 3.0]:
                    for baseline in [0.5, 1.0, 2.0]:
                        if time.time() > deadline - 60.0:
                            break
                        h_struct = _generate_structured_h(freq, phase, amp, baseline, rng)
                        try_h(h_struct)

        # Try Gaussian bumps
        for center in [-0.2, -0.1, 0.0, 0.1, 0.2]:
            for width in [0.05, 0.1, 0.15, 0.2]:
                for amp in [1.0, 2.0, 3.0, 5.0]:
                    for baseline in [0.5, 1.0, 2.0]:
                        if time.time() > deadline - 60.0:
                            break
                        h_gauss = _generate_gaussian_h(center, width, amp, baseline, rng)
                        try_h(h_gauss)

        # Try two Gaussian bumps
        for c1 in [-0.2, -0.1, 0.0]:
            for c2 in [0.0, 0.1, 0.2]:
                if abs(c1 - c2) < 0.05:
                    continue
                for w in [0.05, 0.1, 0.15]:
                    for a1 in [1.0, 2.0, 3.0]:
                        for a2 in [1.0, 2.0, 3.0]:
                            for baseline in [0.5, 1.0]:
                                if time.time() > deadline - 60.0:
                                    break
                                h_two = _generate_two_gaussian_h(c1, c2, w, w, a1, a2, baseline, rng)
                                try_h(h_two)

    # === Phase 1: Full-vector L-BFGS-B with soft-max ===
    for beta in [50.0, 200.0, 500.0, 1000.0]:
        if time.time() > deadline - 40.0:
            break

        for restart in range(8):
            if time.time() > deadline - 40.0:
                break

            if restart == 0:
                x0 = best_h.copy()
            elif restart == 1:
                noise_scale = 0.01 * np.std(best_h)
                x0 = best_h + rng.standard_normal(N) * noise_scale
            elif restart == 2:
                noise_scale = 0.03 * np.std(best_h)
                x0 = best_h + rng.standard_normal(N) * noise_scale
            elif restart == 3:
                noise_scale = 0.05 * np.std(best_h)
                x0 = best_h + rng.standard_normal(N) * noise_scale
            elif restart == 4:
                # Structured restart
                freq = rng.integers(1, 6)
                phase = rng.uniform(0, 2*np.pi)
                amp = rng.uniform(0.5, 2.0)
                baseline = rng.uniform(0.5, 2.0)
                x0 = _generate_structured_h(freq, phase, amp, baseline, rng)
            elif restart == 5:
                # Gaussian restart
                center = rng.uniform(-0.2, 0.2)
                width = rng.uniform(0.05, 0.2)
                amp = rng.uniform(1.0, 3.0)
                baseline = rng.uniform(0.5, 2.0)
                x0 = _generate_gaussian_h(center, width, amp, baseline, rng)
            elif restart == 6:
                # Random sparse
                x0 = np.ones(N)
                n_spikes = rng.integers(5, 20)
                spike_idx = rng.integers(0, N, size=n_spikes)
                spike_amps = rng.standard_normal(n_spikes) * 2.0
                x0[spike_idx] += spike_amps
            else:
                # Blend of best and random
                x0 = 0.5 * best_h + 0.5 * rng.standard_normal(N) * np.std(best_h) + 1.0

            try:
                res = minimize(
                    lambda x: _objective_full(x, beta),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                h_opt = res.x
                val_opt = _evaluate_exact(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass

    # === Phase 2: Epigraph formulation ===
    for beta in [100.0, 500.0]:
        if time.time() > deadline - 20.0:
            break

        # Initialize with best_h and estimated t
        t_init = best_val / (2.0 * N) * (np.sum(best_h) ** 2) / np.sum(best_h ** 2)
        x0 = np.concatenate([best_h, [t_init]])

        try:
            res = minimize(
                lambda x: _objective_epigraph(x, beta),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            h_opt = res.x[:N]
            val_opt = _evaluate_exact(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # === Phase 3: Adaptive coordinate descent polish ===
    if time.time() < deadline - 5.0:
        best_h, best_val = _coordinate_descent_polish(
            best_h,
            max_iters=3000,
            step_sizes=[0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005],
            deadline=deadline - 5.0
        )

    # === Phase 4: Final high-precision subgradient polish ===
    if time.time() < deadline - 2.0:
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            h = best_h.copy()
            s = np.sum(h)
            if abs(s) < 1e-10:
                break
            c = _autoconv_fft(h)
            max_idx = np.argmax(c)
            
            # Compute subgradient direction
            direction = np.zeros(N)
            if max_idx < N:
                lag = max_idx
                for i in range(N - lag):
                    direction[i] -= h[i + lag]
                    direction[i + lag] -= h[i]
            else:
                lag = max_idx - (2*N - 2)
                for i in range(N + lag):
                    direction[i] -= h[i - lag]
                    direction[i - lag] -= h[i]

            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-10:
                direction = direction / norm_dir
                for scale in [0.01, 0.005, 0.001, 0.0005]:
                    h_new = h + scale * direction
                    if abs(np.sum(h_new)) < 1e-10:
                        continue
                    val_new = _evaluate_exact(h_new)
                    if val_new < best_val - 1e-14:
                        best_val = val_new
                        best_h = h_new.copy()
                        break

    # Safety check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N)

    return best_h

# EVOLVE-BLOCK-END
