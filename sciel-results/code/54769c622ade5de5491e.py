"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

DEADLINE = 150.0
TIME_MARGIN = 5.0  # safety margin in seconds


def _compute_c3(x):
    """
    Compute the upper bound C3 for a signed step function x of length N.
    C3 = max_{k != 0} |R[k]| / R[0]
    where R is the autocorrelation.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    
    R0 = R[0]
    if R0 == 0:
        return 1.0
    
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    max_sidelobe = np.max(np.abs(sidelobes))
    
    C3 = max_sidelobe / R0
    return C3


def _compute_c3_fast(x):
    """
    Fast computation of C3. Returns (C3, max_sidelobe_index).
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    
    R0 = R[0]
    if R0 == 0:
        return 1.0, 0
    
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    abs_sidelobes = np.abs(sidelobes)
    max_idx = np.argmax(abs_sidelobes)
    max_sidelobe = abs_sidelobes[max_idx]
    
    C3 = max_sidelobe / R0
    return C3, max_idx


def _soft_max_c3(x, temperature=3.0):
    """
    Smooth approximation of max sidelobe using log-sum-exp.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    
    R0 = R[0]
    if R0 == 0:
        return 1.0
    
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    abs_sidelobes = np.abs(sidelobes)
    
    max_val = np.max(abs_sidelobes)
    log_sum_exp = np.log(np.sum(np.exp((abs_sidelobes - max_val) / temperature))) + max_val / temperature
    soft_max = temperature * log_sum_exp
    
    return soft_max / R0


def _spectral_init():
    """
    Initialize from inverse FFT of a target low-peak magnitude spectrum.
    The idea: if we want flat autocorrelation sidelobes, we want the power spectrum
    to be as flat as possible (except at DC). So we start with a flat spectrum,
    take inverse FFT, and binarize.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    
    # Create a target power spectrum that is flat (white noise in frequency domain)
    # This corresponds to a sequence with low autocorrelation sidelobes
    rng = np.random.default_rng(42)
    
    # Generate random phases in frequency domain
    num_freqs = n_fft // 2 + 1
    phases = rng.uniform(0, 2 * np.pi, size=num_freqs)
    
    # Create magnitude spectrum: flat for non-DC, with DC set to N
    magnitudes = np.ones(num_freqs)
    magnitudes[0] = N  # DC component
    
    # Combine magnitude and phase
    spectrum = magnitudes * np.exp(1j * phases)
    
    # Make it Hermitian symmetric for real inverse FFT
    # For rfft, we just need the positive frequencies
    x_complex = np.fft.irfft(spectrum, n=n_fft)
    
    # Take the first N samples
    x_real = x_real = x_complex[:N]
    
    # Binarize
    x_binary = np.sign(x_real)
    x_binary[x_binary == 0] = 1
    
    return x_binary.astype(int)


def _spectral_init_flat(x_init):
    """
    Alternative spectral init: start from a given sequence, compute its spectrum,
    flatten the sidelobes, and inverse FFT.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x_init, n=n_fft)
    
    # Compute current power spectrum
    power = F * np.conj(F)
    magnitude = np.abs(F)
    phase = np.angle(F)
    
    # Target: flat magnitude (except DC)
    # Blend current magnitude with flat target
    target_mag = np.ones_like(magnitude)
    target_mag[0] = N
    
    # Use a blend to maintain some structure
    blend = 0.5
    new_mag = blend * target_mag + (1 - blend) * magnitude
    
    # Reconstruct with original phases
    F_new = new_mag * np.exp(1j * phase)
    
    # Inverse FFT
    x_complex = np.fft.irfft(F_new, n=n_fft)
    x_real = x_complex[:N]
    
    # Binarize
    x_binary = np.sign(x_real)
    x_binary[x_binary == 0] = 1
    
    return x_binary.astype(int)


def _local_search_fast(x, max_iter=20000, time_limit=None):
    """
    Fast local search: try flipping individual positions and accept if it improves C3.
    Uses incremental updates where possible.
    """
    start_time = time.time()
    current_c3, _ = _compute_c3_fast(x)
    best_x = x.copy()
    best_c3 = current_c3
    no_improve_count = 0
    max_no_improve = 500
    
    for iteration in range(max_iter):
        # Check time
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        # Check global deadline
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Try a random flip
        idx = random.randint(0, N - 1)
        x_new = x.copy()
        x_new[idx] = -x_new[idx]
        new_c3, _ = _compute_c3_fast(x_new)
        
        if new_c3 < current_c3 - 1e-12:
            x = x_new
            current_c3 = new_c3
            no_improve_count = 0
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    return best_x, best_c3


def _simulated_annealing_fast(x, max_iter=30000, T_start=0.5, T_end=0.001, time_limit=None):
    """
    Simulated annealing to escape local minima.
    """
    start_time = time.time()
    current_c3, _ = _compute_c3_fast(x)
    best_x = x.copy()
    best_c3 = current_c3
    T = T_start
    
    for iteration in range(max_iter):
        # Check time
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Annealing schedule
        T = T_start * (T_end / T_start) ** (iteration / max_iter)
        
        # Try a random flip
        idx = random.randint(0, N - 1)
        x_new = x.copy()
        x_new[idx] = -x_new[idx]
        new_c3, _ = _compute_c3_fast(x_new)
        
        delta = new_c3 - current_c3
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            x = x_new
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
    
    return best_x, best_c3


def _pair_flip_local_search(x, max_iter=10000, time_limit=None):
    """
    Local search that tries flipping pairs of positions simultaneously.
    This can escape local minima that single flips cannot.
    """
    start_time = time.time()
    current_c3, _ = _compute_c3_fast(x)
    best_x = x.copy()
    best_c3 = current_c3
    no_improve_count = 0
    max_no_improve = 300
    
    for iteration in range(max_iter):
        # Check time
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Try flipping two random positions
        idx1 = random.randint(0, N - 1)
        idx2 = random.randint(0, N - 1)
        if idx1 == idx2:
            idx2 = (idx2 + 1) % N
        
        x_new = x.copy()
        x_new[idx1] = -x_new[idx1]
        x_new[idx2] = -x_new[idx2]
        new_c3, _ = _compute_c3_fast(x_new)
        
        if new_c3 < current_c3 - 1e-12:
            x = x_new
            current_c3 = new_c3
            no_improve_count = 0
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    return best_x, best_c3


def _construct():
    """
    Main construction function that searches for a good signed step function.
    """
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = np.ones(N)
    best_c3 = _compute_c3(best_x)
    
    # Strategy 1: Spectral initialization - start from flat spectrum
    x1 = _spectral_init()
    c3_1 = _compute_c3(x1)
    if c3_1 < best_c3:
        best_x = x1
        best_c3 = c3_1
    
    # Refine with local search
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        x1r, c3_1r = _local_search_fast(best_x, max_iter=5000, time_limit=remaining * 0.3)
        if c3_1r < best_c3:
            best_x = x1r
            best_c3 = c3_1r
    
    # Strategy 2: Multiple random spectral inits
    for trial in range(5):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        rng = np.random.default_rng(trial * 100 + 1)
        
        n_fft = 1 << (2 * N - 1).bit_length()
        num_freqs = n_fft // 2 + 1
        phases = rng.uniform(0, 2 * np.pi, size=num_freqs)
        magnitudes = np.ones(num_freqs)
        magnitudes[0] = N
        spectrum = magnitudes * np.exp(1j * phases)
        x_complex = np.fft.irfft(spectrum, n=n_fft)
        x_real = x_complex[:N]
        x_binary = np.sign(x_real)
        x_binary[x_binary == 0] = 1
        x_trial = x_binary.astype(int)
        
        c3_trial = _compute_c3(x_trial)
        if c3_trial < best_c3:
            best_x = x_trial
            best_c3 = c3_trial
        
        # Quick local search on this candidate
        if time.time() < DEADLINE - TIME_MARGIN:
            remaining = DEADLINE - TIME_MARGIN - time.time()
            xr, cr = _local_search_fast(x_trial, max_iter=2000, time_limit=remaining * 0.1)
            if cr < best_c3:
                best_x = xr
                best_c3 = cr
    
    # Strategy 3: Spectral flattening from current best
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        x_flat = _spectral_init_flat(best_x)
        c3_flat = _compute_c3(x_flat)
        if c3_flat < best_c3:
            best_x = x_flat
            best_c3 = c3_flat
        
        # Refine
        xr, cr = _local_search_fast(best_x, max_iter=3000, time_limit=remaining * 0.2)
        if cr < best_c3:
            best_x = xr
            best_c3 = cr
    
    # Strategy 4: Simulated annealing from best
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        x3, c3_3 = _simulated_annealing_fast(best_x, max_iter=15000, T_start=0.3, T_end=0.001,
                                             time_limit=remaining * 0.3)
        if c3_3 < best_c3:
            best_x = x3
            best_c3 = c3_3
    
    # Strategy 5: Pair flip local search
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        x4, c3_4 = _pair_flip_local_search(best_x, max_iter=5000, time_limit=remaining * 0.2)
        if c3_4 < best_c3:
            best_x = x4
            best_c3 = c3_4
    
    # Final local search polish
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        x5, c3_5 = _local_search_fast(best_x, max_iter=10000, time_limit=remaining * 0.5)
        if c3_5 < best_c3:
            best_x = x5
            best_c3 = c3_5
    
    return best_x


def construct():
    """
    Entry point: return a sequence of signed step heights (±1) of length N=400.
    """
    x = _construct()
    # Ensure output is a numpy array of ±1 values
    return np.sign(x).astype(int)
# EVOLVE-BLOCK-END
