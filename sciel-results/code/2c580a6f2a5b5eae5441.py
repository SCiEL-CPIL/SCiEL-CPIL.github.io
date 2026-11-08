import numpy as np
import time

N = 400
M = 2 * N
dx = 0.5 / M
DEADLINE = 140.0  # safety margin below 150s kill

# Precompute fine-grid mapping
_m_indices = np.arange(M)
_i_indices = np.clip((_m_indices * N // M), 0, N - 1)
_FFT_N = 2 * M


def _autoconv_from_fine(f_fine):
    """Autoconvolution on fine grid from fine values."""
    fft_f = np.fft.fft(f_fine, n=_FFT_N)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    return autoconv


def _fine_from_step(signed_step):
    return signed_step[_i_indices]


def _compute_c3(signed_step):
    f_fine = _fine_from_step(signed_step)
    autoconv = _autoconv_from_fine(f_fine)
    return np.max(np.abs(autoconv))


def _soft_c3_and_grad(signed_step, beta=50.0):
    """
    Return (soft_c3, grad_step) where grad_step is d(soft_c3)/d(signed_step).
    Uses analytic chain rule through FFT.
    """
    f_fine = _fine_from_step(signed_step).astype(np.float64)
    fft_f = np.fft.fft(f_fine, n=_FFT_N)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx

    abs_a = np.abs(autoconv)
    scaled = beta * abs_a
    max_val = np.max(scaled)
    exp_scaled = np.exp(scaled - max_val)
    sum_exp = np.sum(exp_exp if False else exp_scaled)
    w = exp_scaled / sum_exp  # softmax weights
    soft_c3 = (max_val + np.log(sum_exp)) / beta

    # d(soft_c3)/d(autoconv) = w * sign(autoconv)
    dS_da = w * np.sign(autoconv)

    # da = IFFT(2*Re(F*conj(dF))) * dx ; grad_f = dx * IFFT(2*conj(F)*FFT(dS_da))
    fft_dS = np.fft.fft(dS_da, n=_FFT_N)
    grad_fine = np.fft.ifft(2.0 * np.conj(fft_f) * fft_dS).real * dx

    # Push gradient back to step indices (each step index maps to 2 fine points)
    grad_step = np.zeros(N, dtype=np.float64)
    np.add.at(grad_step, _i_indices, grad_fine)

    return soft_c3, grad_step


def _compute_c3_soft(signed_step, beta=50.0):
    sc, _ = _soft_c3_and_grad(signed_step, beta)
    return sc


def _anneal(start_step, start_time, deadline, beta=40.0, T0=0.05, Tend=1e-4, n_iter=None):
    """
    Simulated annealing on the binary step. Accepts uphill moves with
    probability exp(-delta_soft/T). Uses soft surrogate for acceptance.
    """
    x = start_step.copy()
    best_x = x.copy()
    best_c3 = _compute_c3(x)

    if n_iter is None:
        n_iter = 300000

    # Precompute random indices
    rng = np.random.default_rng(12345)

    for it in range(n_iter):
        if time.time() > deadline:
            break

        # Cooling schedule
        frac = it / max(1, n_iter - 1)
        T = T0 * (Tend / T0) ** frac

        # Random single flip
        i = rng.integers(N)
        old = x[i]
        x[i] = -old
        new_soft, _ = _soft_c3_and_grad(x, beta)
        # Need old soft; recompute cheaply? We'll track.
        # For acceptance we need delta = new_soft - old_soft.
        # Recompute old soft is one FFT; acceptable.
        x[i] = old
        old_soft, _ = _soft_c3_and_grad(x, beta)
        delta = new_soft - old_soft

        accept = (delta <= 0) or (rng.random() < np.exp(-delta / max(T, 1e-9)))
        if accept:
            x[i] = -x[i]
            # Evaluate exact C3 occasionally to track best
            if it % 50 == 0:
                c3_now = _compute_c3(x)
                if c3_now < best_c3:
                    best_c3 = c3_now
                    best_x = x.copy()

    # Final exact eval of annealed state
    c3_end = _compute_c3(x)
    if c3_end < best_c3:
        best_c3 = c3_end
        best_x = x.copy()

    return best_x, best_c3


def _polish(step, start_time, deadline):
    """Exact max-based single-flip local search to polish the true C3."""
    x = step.copy()
    best_c3 = _compute_c3(x)
    rng = np.random.default_rng(777)
    no_improve = 0
    patience = 500

    while time.time() < deadline:
        i = rng.integers(N)
        x[i] = -x[i]
        c3 = _compute_c3(x)
        if c3 < best_c3:
            best_c3 = c3
            no_improve = 0
        else:
            x[i] = -x[i]
            no_improve += 1
        if no_improve >= patience:
            break

    return x, best_c3


def construct():
    start_time = time.time()

    # Initialize with all ones (the seed)
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)

    rng = np.random.default_rng(42)

    # Try a few random initializations
    for trial in range(20):
        if time.time() - start_time > DEADLINE - 30.0:
            break
        random_step = rng.choice([-1, 1], size=N).astype(np.int8)
        c3_val = _compute_c3(random_step)
        if c3_val < best_c3:
            best_step = random_step.copy()
            best_c3 = c3_val

    # Simulated annealing phases with decreasing temperature
    # Phase 1: broad exploration
    if time.time() - start_time < DEADLINE - 40.0:
        phase_deadline = min(start_time + 40.0, DEADLINE - 30.0)
        best_step, best_c3 = _anneal(
            best_step, start_time, phase_deadline,
            beta=40.0, T0=0.10, Tend=1e-3, n_iter=200000
        )

    # Phase 2: focused exploitation
    if time.time() - start_time < DEADLINE - 20.0:
        phase_deadline = min(start_time + 30.0, DEADLINE - 20.0)
        best_step, best_c3 = _anneal(
            best_step, start_time, phase_deadline,
            beta=60.0, T0=0.03, Tend=1e-5, n_iter=200000
        )

    # Final polish with exact max
    if time.time() - start_time < DEADLINE - 5.0:
        best_step, best_c3 = _polish(best_step, start_time, DEADLINE - 3.0)

    # Safety: ensure we return a valid step before kill
    if time.time() > DEADLINE - 2.0:
        return best_step

    return best_step
