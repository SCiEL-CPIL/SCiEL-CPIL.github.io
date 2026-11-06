"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_t0 = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill

def _value(h):
    """Compute the exact objective value."""
    h = np.asarray(h, dtype=float)
    if len(h) != N or not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if s == 0.0:
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _fft_convolve_max(h):
    """Compute max of full autocorrelation using FFT (faster)."""
    n = len(h)
    size = 2 * n
    f = np.fft.rfft(h, size)
    conv = np.fft.irfft(f * np.conj(f), size)[:n]
    return float(np.max(conv))

def _fast_value(h):
    """Fast objective using FFT-based convolution."""
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    m = _fft_convolve_max(h)
    return abs(2.0 * N * m / (s * s))

def construct():
    global _t0
    _t0 = time.time()
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if len(h) != N or not np.all(np.isfinite(h)):
            return
        s = np.sum(h)
        if abs(s) < 1e-15:
            return
        v = _value(h)
        if v < best_v - 1e-12:
            best_v = v
            best_h = h.copy()
    
    # === Strategy 1: Symmetric half-vector optimization ===
    # Enforce h[i] = h[N-1-i] to reduce dimensionality by half
    half_N = N // 2
    
    def _symmetric_reconstruct(x):
        h = np.zeros(N)
        h[:half_N] = x
        h[half_N:] = x[::-1]
        return h
    
    def _symmetric_obj(x):
        h = _symmetric_reconstruct(x)
        return _fast_value(h)
    
    # Generate diverse initial guesses for the half vector
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, half_N)
    
    guesses = []
    # Triangular
    guesses.append(1.0 - t)
    # Parabolic
    guesses.append(1.0 - t**2)
    # Sinc-like
    x_sinc = np.linspace(-1, 1, half_N)
    sinc = np.ones_like(x_sinc)
    mask = x_sinc != 0
    sinc[mask] = np.sin(np.pi * x_sinc[mask]) / (np.pi * x_sinc[mask])
    guesses.append(sinc * np.max(sinc))
    # Box with taper
    taper_frac = 0.15
    n_taper = max(1, int(taper_frac * half_N))
    b = np.ones(half_N)
    b[:n_taper] = np.linspace(0, 1, n_taper)
    guesses.append(b)
    # Cosine bell
    guesses.append(0.5 * (1 + np.cos(np.pi * t)))
    # Random positive
    guesses.append(rng.random(half_N) + 0.1)
    # Perturbed constant
    guesses.append(1.0 + 0.2 * rng.standard_normal(half_N))
    
    for g in guesses:
        if not deadline_ok():
            break
        g = g.astype(float).copy()
        s = np.sum(g)
        if s > 0:
            g = g / s
        
        try:
            res = minimize(
                _symmetric_obj,
                g,
                method='L-BFGS-B',
                bounds=[(-10, 10)] * half_N,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.x is not None:
                h = _symmetric_reconstruct(res.x)
                update_best(h)
        except Exception:
            pass
    
    # === Strategy 2: Sparse atom-based representation ===
    # h = sum of a few shifted, sign-flipped copies of a short window
    # This enforces sparsity and reduces effective dimensionality
    
    atom_len = 12
    atom = np.hamming(atom_len)
    atom = atom / np.sum(atom)
    
    # Try different numbers of atoms and their placements
    for num_atoms in [3, 5, 8]:
        if not deadline_ok():
            break
        
        # Parameterize: amplitudes and offsets for each atom
        # Use a fixed set of candidate offsets spread across the domain
        candidate_offsets = list(range(0, N - atom_len + 1, max(1, (N - atom_len) // (num_atoms * 3))))
        candidate_offsets = candidate_offsets[:num_atoms * 3]
        
        def _atom_obj(params):
            h = np.zeros(N)
            for k in range(num_atoms):
                amp = params[2*k]
                off = int(round(params[2*k + 1]))
                sign = 1.0 if amp >= 0 else -1.0
                abs_amp = abs(amp)
                start = off
                end = start + atom_len
                if 0 <= start and end <= N:
                    h[start:end] += sign * abs_amp * atom
            return _fast_value(h)
        
        # Initial: spread atoms evenly with positive amplitudes
        init = np.zeros(2 * num_atoms)
        for k in range(num_atoms):
            init[2*k] = 1.0 / num_atoms
            init[2*k + 1] = k * (N - atom_len) / max(1, num_atoms - 1)
        
        try:
            bounds = [(-5, 5), (0, N - atom_len)] * num_atoms
            res = minimize(
                _atom_obj,
                init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                h = np.zeros(N)
                for k in range(num_atoms):
                    amp = res.x[2*k]
                    off = int(round(res.x[2*k + 1]))
                    sign = 1.0 if amp >= 0 else -1.0
                    abs_amp = abs(amp)
                    start = off
                    end = start + atom_len
                    if 0 <= start and end <= N:
                        h[start:end] += sign * abs_amp * atom
                update_best(h)
        except Exception:
            pass
    
    # === Strategy 3: Low-dimensional Fourier parameterization ===
    # h[i] = a0 + sum_{k=1}^{K} (ak*cos(2*pi*k*i/N) + bk*sin(2*pi*k*i/N))
    i = np.arange(N)
    
    for K in [2, 3, 4]:
        if not deadline_ok():
            break
        
        def _fourier_obj(params):
            a0 = params[0]
            h = a0 * np.ones(N)
            for k in range(1, K + 1):
                ak = params[2*k - 1]
                bk = params[2*k]
                h += ak * np.cos(2 * np.pi * k * i / N)
                h += bk * np.sin(2 * np.pi * k * i / N)
            return _fast_value(h)
        
        init = np.zeros(2 * K + 1)
        init[0] = 1.0  # Constant term
        
        try:
            res = minimize(
                _fourier_obj,
                init,
                method='L-BFGS-B',
                bounds=[(-5, 5)] * (2 * K + 1),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.x is not None:
                a0 = res.x[0]
                h = a0 * np.ones(N)
                for k in range(1, K + 1):
                    ak = res.x[2*k - 1]
                    bk = res.x[2*k]
                    h += ak * np.cos(2 * np.pi * k * i / N)
                    h += bk * np.sin(2 * np.pi * k * i / N)
                update_best(h)
        except Exception:
            pass
    
    # === Strategy 4: Coordinate descent polish on best solution ===
    if deadline_ok():
        h = best_h.copy()
        v = best_v
        
        # In-place coordinate descent with FFT evaluation
        step_sizes = [0.5, 0.1, 0.01, 0.001]
        for _ in range(50):
            if not deadline_ok():
                break
            improved = False
            for idx in range(N):
                if not deadline_ok():
                    break
                old_val = h[idx]
                for delta in step_sizes:
                    for sign in [1, -1]:
                        h[idx] = old_val + sign * delta
                        v_new = _fast_value(h)
                        if v_new < v - 1e-14:
                            v = v_new
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    h[idx] = old_val
            
            if not improved:
                break
        
        # Final exact evaluation
        if np.all(np.isfinite(h)) and abs(np.sum(h)) > 1e-15:
            v_exact = _value(h)
            if v_exact < best_v - 1e-12:
                best_v = v_exact
                best_h = h.copy()
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    
    v_final = _value(best_h)
    if v_final < 2.0 - 1e-12:
        return best_h
    else:
        return np.ones(N)
# EVOLVE-BLOCK-END
