"""Signed autocorrelation maximum absolute-value variant - fresh design."""
import numpy as np
from scipy.optimize import minimize
import time

N = 400
DEADLINE = 120.0  # seconds


def _autoconv_max(x):
    """Compute the maximum absolute value of the circular autoconvolution of x."""
    # Use FFT for efficient convolution
    f = np.fft.fft(x, n=2 * N)
    conv = np.fft.ifft(f * np.conj(f)).real[:N]
    return np.max(np.abs(conv))


def _score(x):
    """Score: maximum absolute autoconvolution divided by squared integral (sum)."""
    autoconv = _autoconv_max(x)
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e10
    return autoconv / (integral * integral)


def _init_flat_cosine():
    """Raised-cosine magnitude spectrum with random phases."""
    freq = np.fft.rfftfreq(N)
    # Raised-cosine shape: low-pass-like
    mask = np.ones(N // 2 + 1)
    # Apply raised cosine taper
    for k in range(N // 2 + 1):
        if k > N // 4:
            mask[k] = 0.5 * (1 + np.cos(np.pi * (k - N // 4) / (N // 4)))
    magnitude = mask
    phase = 2 * np.pi * np.random.rand(N // 2 + 1)
    spectrum = magnitude * np.exp(1j * phase)
    # Construct full spectrum (Hermitian)
    full_spec = np.zeros(N, dtype=complex)
    full_spec[:N // 2 + 1] = spectrum
    full_spec[N // 2 + 1:] = np.conj(spectrum[1:N // 2][::-1])
    x = np.fft.ifft(full_spec).real
    return x


def _init_random_phase_flat():
    """Flat magnitude spectrum with random phases."""
    magnitude = np.ones(N // 2 + 1)
    phase = 2 * np.pi * np.random.rand(N // 2 + 1)
    spectrum = magnitude * np.exp(1j * phase)
    full_spec = np.zeros(N, dtype=complex)
    full_spec[:N // 2 + 1] = spectrum
    full_spec[N // 2 + 1:] = np.conj(spectrum[1:N // 2][::-1])
    x = np.fft.ifft(full_spec).real
    return x


def _lbfgs_optimize(x0, deadline):
    """Run L-BFGS to minimize the score."""
    best_x = x0.copy()
    best_score = _score(x0)

    def objective(x):
        return _score(x)

    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    if result.fun < best_score:
        best_x = result.x
        best_score = result.fun

    return best_x, best_score


def _coordinate_descent(x, deadline):
    """Refine using coordinate descent on individual elements."""
    best_x = x.copy()
    best_score = _score(x)
    n_steps = 0
    while time.time() < deadline:
        improved = False
        for i in range(N):
            old_val = best_x[i]
            # Try a few perturbations
            for delta in [0.1, -0.1, 0.01, -0.01, 0.5, -0.5]:
                test_x = best_x.copy()
                test_x[i] = old_val + delta
                s = _score(test_x)
                if s < best_score - 1e-10:
                    best_x[i] = test_x[i]
                    best_score = s
                    improved = True
                    break
        if not improved:
            break
        n_steps += 1
        if n_steps > 20:  # Safety limit
            break
    return best_x, best_score


def construct():
    start_time = time.time()
    best_x = np.ones(N)
    best_score = _score(best_x)

    # Generate diverse initial points
    inits = []
    inits.append(np.ones(N))
    
    # Spectral initializations
    for _ in range(10):
        inits.append(_init_flat_cosine())
    for _ in range(10):
        inits.append(_init_random_phase_flat())
    
    # Random inits
    for _ in range(5):
        inits.append(np.random.randn(N))

    for x0 in inits:
        if time.time() > DEADLINE - 15:
            break
        # Normalize to have reasonable scale
        x0 = x0 / (np.linalg.norm(x0) + 1e-12)
        x_opt, s_opt = _lbfgs_optimize(x0, DEADLINE - 10)
        if s_opt < best_score:
            best_x = x_opt
            best_score = s_opt

    # Coordinate descent refinement
    if time.time() < DEADLINE - 5:
        best_x, best_score = _coordinate_descent(best_x, DEADLINE - 3)

    return best_x
