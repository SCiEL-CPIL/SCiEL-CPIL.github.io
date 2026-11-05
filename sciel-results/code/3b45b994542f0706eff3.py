"""
Flattest plus-minus-one polynomials with 24 coefficients.
Strategy:
  1. Precompute DFT matrix for O(1) cost per flip evaluation.
  2. Spectral synthesis initialization (3 variants) + known candidates.
  3. Vectorized 1-flip local search (evaluate all 24 neighbors at once).
  4. Vectorized 2-flip local search (evaluate all 276 pairs).
  5. Iterated local search with random perturbations when stuck.
  6. Deadline guard: return best found before 140s wall-clock.
"""
import numpy as np
import time
import random
import math

N = 24
M = 2**15  # 32768, dense enough for accurate sup-norm
SQRTN = math.sqrt(N)

# Precompute DFT matrix: D[k, n] = exp(-2j*pi*k*n/M)
_k = np.arange(M)
_n = np.arange(N)
_D = np.exp(-2j * np.pi * (_k[:, None] * _n[None, :] / M))

# Precompute all 2-flip combinations
_flip2_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

def _time_left(deadline):
    return deadline - time.time()

def _eval_F(c):
    """Compute DFT of coefficient vector."""
    return _D @ c

def _supnorm(F):
    """Max |F| over the grid."""
    return float(np.max(np.abs(F)))

def _local_search_1flip(c, F, max_val, deadline):
    """
    Greedy 1-flip local search: repeatedly flip the bit that most reduces max|F|.
    Vectorized: evaluate all 24 single-flip neighbors in one pass.
    """
    improved = True
    while improved:
        if _time_left(deadline) < 2.0:
            break
        improved = False
        best_flip = -1
        best_max_val = max_val
        # Evaluate all 24 single-flip neighbors
        for n in range(N):
            F_new = F - 2.0 * c[n] * _D[:, n]
            mv = np.max(np.abs(F_new))
            if mv < best_max_val - 1e-12:
                best_max_val = mv
                best_flip = n
        if best_flip != -1:
            c[best_flip] *= -1.0
            F = _D @ c
            max_val = best_max_val
            improved = True
    return c, F, max_val

def _local_search_2flip(c, F, max_val, deadline):
    """
    Try all 2-flip moves, accept the best improvement.
    """
    if _time_left(deadline) < 2.0:
        return c, F, max_val
    best_pair = (-1, -1)
    best_max_val = max_val
    for (i, j) in _flip2_pairs:
        F_new = F - 2.0 * (c[i] * _D[:, i] + c[j] * _D[:, j])
        mv = np.max(np.abs(F_new))
        if mv < best_max_val - 1e-12:
            best_max_val = mv
            best_pair = (i, j)
    if best_pair != (-1, -1):
        i, j = best_pair
        c[i] *= -1.0
        c[j] *= -1.0
        F = _D @ c
        max_val = best_max_val
    return c, F, max_val

def _spectral_synthesis_init(seed_idx=0):
    """
    Generate a coherent ±1 sequence via inverse FFT of a flattened target waveform.
    Variant 1: cosine-tapered spectrum with random phases.
    """
    rng = np.random.RandomState(seed_idx * 1000 + 42)
    alpha = 0.3 + 0.5 * (seed_idx % 5) / 5.0
    n_spec = 64
    freqs = np.arange(n_spec)
    taper = 0.5 * (1.0 + np.cos(np.pi * freqs / (n_spec - 1)))
    target = (1 - alpha) * np.ones(n_spec) + alpha * taper
    target = target / np.max(target)
    phases = rng.uniform(0, 2 * np.pi, n_spec)
    target_complex = target * np.exp(1j * phases)
    time_signal = np.fft.ifft(target_complex)
    real_part = np.real(time_signal[:N])
    coeffs = np.sign(real_part)
    zero_mask = (coeffs == 0)
    if np.any(zero_mask):
        coeffs[zero_mask] = rng.choice([1, -1], size=np.sum(zero_mask))
    return [int(c) for c in coeffs]

def _spectral_synthesis_init_v2(seed_idx=0):
    """
    Variant 2: sum of random low-frequency cosines as target.
    """
    rng = np.random.RandomState(seed_idx * 2000 + 123)
    n_spec = 64
    freqs = np.arange(n_spec)
    n_components = 2 + (seed_idx % 3)
    target = np.ones(n_spec)
    for k in range(n_components):
        f = rng.uniform(0.5, 3.0)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.1, 0.5)
        target += amp * np.cos(2 * np.pi * f * freqs / n_spec + phase)
    target = np.abs(target)
    target = target / np.max(target)
    time_signal = np.fft.ifft(target)
    real_part = np.real(time_signal[:N])
    coeffs = np.sign(real_part)
    zero_mask = (coeffs == 0)
    if np.any(zero_mask):
        coeffs[zero_mask] = rng.choice([1, -1], size=np.sum(zero_mask))
    return [int(c) for c in coeffs]

def _spectral_synthesis_init_v3(seed_idx=0):
    """
    Variant 3: smoothed spectrum of random polynomial as target.
    """
    rng = np.random.RandomState(seed_idx * 3000 + 456)
    rand_coeffs = rng.choice([1, -1], size=N)
    n_spec = 128
    padded = np.zeros(n_spec, dtype=float)
    padded[:N] = rand_coeffs
    spectrum = np.abs(np.fft.fft(padded))
    kernel_size = 5 + (seed_idx % 4) * 2
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(spectrum, kernel, mode='same')
    smoothed = smoothed / np.max(smoothed)
    n_half = n_spec // 2 + 1
    target_half = smoothed[:n_half]
    target_full = np.zeros(n_spec, dtype=complex)
    target_full[:n_half] = target_half
    for k in range(1, n_spec // 2):
        target_full[n_spec - k] = np.conj(target_full[k])
    time_signal = np.fft.ifft(target_full)
    real_part = np.real(time_signal[:N])
    coeffs = np.sign(real_part)
    zero_mask = (coeffs == 0)
    if np.any(zero_mask):
        coeffs[zero_mask] = rng.choice([1, -1], size=np.sum(zero_mask))
    return [int(c) for c in coeffs]

def construct():
    """
    Return 24 coefficients in {+1, -1} minimizing max_{|z|=1} |P(z)| / sqrt(24).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Known candidate sequences (from literature / Odlyzko tables)
    candidates = [
        [1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1],
        [1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1],
        [1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1],
        [1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1],
        [1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1],
        [1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1],
        [1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1],
        [-1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1],
    ]

    # Generate spectral synthesis initial points
    spectral_inits = []
    for s in range(15):
        spectral_inits.append(_spectral_synthesis_init(s))
        spectral_inits.append(_spectral_synthesis_init_v2(s))
        spectral_inits.append(_spectral_synthesis_init_v3(s))

    # Add some random initial points
    rng = np.random.RandomState(42)
    for _ in range(20):
        spectral_inits.append([int(x) for x in rng.choice([-1, 1], N)])

    best_score = float('inf')
    best_c = None

    # Phase 1: Local search from each candidate and spectral init
    for c_init in candidates + spectral_inits:
        if _time_left(deadline) < 5.0:
            break

        c = np.array(c_init, dtype=float)
        F = _D @ c
        max_val = _supnorm(F)

        # 1-flip local search
        c, F, max_val = _local_search_1flip(c, F, max_val, deadline)

        # Iterate 2-flip + 1-flip to escape local minima
        for _ in range(3):
            if _time_left(deadline) < 3.0:
                break
            c, F, max_val = _local_search_2flip(c, F, max_val, deadline)
            c, F, max_val = _local_search_1flip(c, F, max_val, deadline)

        score = max_val / SQRTN
        if score < best_score:
            best_score = score
            best_c = c.copy()

    # Phase 2: Iterated local search with random perturbations
    if best_c is not None:
        current_c = best_c.copy()
        current_F = _D @ current_c
        current_max = _supnorm(current_F)

        while _time_left(deadline) > 3.0:
            # Try 1-flip
            best_flip_1 = -1
            best_max_1 = current_max
            for n in range(N):
                F_new = current_F - 2.0 * current_c[n] * _D[:, n]
                mv = np.max(np.abs(F_new))
                if mv < best_max_1 - 1e-12:
                    best_max_1 = mv
                    best_flip_1 = n

            if best_flip_1 != -1:
                current_c[best_flip_1] *= -1.0
                current_F = _D @ current_c
                current_max = best_max_1
                if current_max / SQRTN < best_score:
                    best_score = current_max / SQRTN
                    best_c = current_c.copy()
                continue

            # Try 2-flip
            best_pair = (-1, -1)
            best_max_2 = current_max
            for (i, j) in _flip2_pairs:
                F_new = current_F - 2.0 * (current_c[i] * _D[:, i] + current_c[j] * _D[:, j])
                mv = np.max(np.abs(F_new))
                if mv < best_max_2 - 1e-12:
                    best_max_2 = mv
                    best_pair = (i, j)

            if best_pair != (-1, -1):
                i, j = best_pair
                current_c[i] *= -1.0
                current_c[j] *= -1.0
                current_F = _D @ current_c
                current_max = best_max_2
                if current_max / SQRTN < best_score:
                    best_score = current_max / SQRTN
                    best_c = current_c.copy()
                continue

            # Stuck: random perturbation of 2-3 bits, then re-localize
            n_flips = rng.choice([2, 3], p=[0.6, 0.4])
            indices = rng.choice(N, n_flips, replace=False)
            current_c[indices] *= -1.0
            current_F = _D @ current_c
            current_max = _supnorm(current_F)

            # Quick 1-flip local search after restart
            current_c, current_F, current_max = _local_search_1flip(
                current_c, current_F, current_max, deadline)

            if current_max / SQRTN < best_score:
                best_score = current_max / SQRTN
                best_c = current_c.copy()

    if best_c is None:
        # Fallback: return first candidate
        best_c = np.array(candidates[0], dtype=float)

    return [int(x) for x in best_c]


if __name__ == "__main__":
    result = construct()
    print(result)
