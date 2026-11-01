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
TIME_MARGIN = 15.0  # safety margin in seconds


def _compute_c3(x):
    """
    Compute the upper bound C3 for a signed step function x of length N.
    
    The autoconvolution upper bound C3 is related to the maximum absolute
    value of the autocorrelation sidelobes relative to the main lobe.
    
    For a signed step function (values in {-1, +1}), the autocorrelation is:
      R[k] = sum_{i=0}^{N-1-k} x[i] * x[i+k]  for k = 0, ..., N-1
    
    The "upper bound on C3" is typically defined as:
      C3 = max_{k != 0} |R[k]| / R[0]
    
    But since we're dealing with a step function on [-1/4, 1/4] with 400 equal
    intervals, and the task mentions "autoconvolution upper bound", let's define
    it more carefully.
    
    Actually, for a step function f(x) with heights x[i] on intervals, the
    autoconvolution (f * f)(t) = integral f(s) f(t-s) ds.
    
    For a step function with N equal intervals of width w = (1/4 - (-1/4))/N = 1/(2N),
    the autoconvolution at lag k*w is:
      (f*f)(k*w) = w * sum_{i} x[i] x[i+k]  (with appropriate boundary handling)
    
    The peak of the autoconvolution is at k=0: (f*f)(0) = w * sum x[i]^2 = w * N (for ±1)
    
    The "upper bound on C3" likely refers to the maximum sidelobe level relative
    to the main lobe, or some normalized quantity.
    
    Given the task says "score is the upper bound on C3" and "lower is better",
    I'll compute:
      C3 = max_{k=1,...,N-1} |R[k]| / R[0]
    
    where R[k] is the autocorrelation at lag k.
    """
    # Compute autocorrelation using FFT for efficiency
    n_fft = 1 << (2 * N - 1).bit_length()  # next power of 2 for linear correlation
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    # R[0] is the zero-lag correlation = sum x[i]^2
    # R[1:N] are the positive lags
    # R[N:2N-1] are the negative lags (symmetric)
    
    R0 = R[0]
    if R0 == 0:
        return 1.0  # degenerate case
    
    # Maximum absolute sidelobe
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    max_sidelobe = np.max(np.abs(sidelobes))
    
    C3 = max_sidelobe / R0
    return C3


def _flip_random_positions(x, num_flips):
    """Flip a random subset of positions in x."""
    x = x.copy()
    if num_flips == 0:
        return x
    indices = random.sample(range(N), min(num_flips, N))
    for i in indices:
        x[i] = -x[i]
    return x


def _logistic_sequence(n, r=3.99, x0=0.5):
    """Generate a chaotic logistic map sequence and binarize it."""
    x = np.zeros(n)
    val = x0
    for i in range(n):
        val = r * val * (1.0 - val)
        x[i] = 1.0 if val > 0.5 else -1.0
    return x


def _van_der_corput(n, base=2):
    """Generate a van der Corput low-discrepancy sequence and binarize it."""
    seq = np.zeros(n)
    for i in range(1, n + 1):
        v = 0.0
        f = 1.0 / base
        m = i
        while m > 0:
            m, r = divmod(m, base)
            v += r * f
            f /= base
        # Map [0,1) to {-1, +1}
        seq[i-1] = 1.0 if v > 0.5 else -1.0
    return seq


def _chaos_initializations():
    """Generate multiple chaotic/quasi-random initializations."""
    inits = []
    
    # Logistic map with different parameters
    for r in [3.9, 3.99, 3.999]:
        for x0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            inits.append(_logistic_sequence(N, r=r, x0=x0))
    
    # Van der Corput sequences
    inits.append(_van_der_corput(N, base=2))
    inits.append(_van_der_corput(N, base=3))
    inits.append(_van_der_corput(N, base=5))
    
    # Thue-Morse-like (alternating pattern)
    tm = np.zeros(N)
    for i in range(N):
        # Count set bits
        c = 0
        m = i
        while m:
            c += m & 1
            m >>= 1
        tm[i] = 1.0 if c % 2 == 0 else -1.0
    inits.append(tm)
    
    # Mersenne Twister based (deterministic)
    rng = np.random.RandomState(12345)
    inits.append(rng.choice([-1, 1], size=N))
    
    return inits


def _local_search(x, max_iter=5000, time_limit=None):
    """
    Local search: try flipping individual positions and accept if it improves C3.
    """
    start_time = time.time()
    current_c3 = _compute_c3(x)
    best_x = x.copy()
    best_c3 = current_c3
    no_improve_count = 0
    max_no_improve = 100
    
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
        new_c3 = _compute_c3(x_new)
        
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


def _simulated_annealing(x, max_iter=20000, T_start=1.0, T_end=0.01, time_limit=None):
    """
    Simulated annealing to escape local minima.
    """
    start_time = time.time()
    current_c3 = _compute_c3(x)
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
        new_c3 = _compute_c3(x_new)
        
        delta = new_c3 - current_c3
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            x = x_new
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
    
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
    
    # Strategy 1: Chaotic/quasi-random initializations + local search
    chaos_inits = _chaos_initializations()
    for i, x_init in enumerate(chaos_inits):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        # Give more time to earlier (likely better) initializations
        time_limit = max(10.0, 25.0 - i * 2.0)
        x_trial, c3_trial = _local_search(x_init.copy(), max_iter=3000, time_limit=time_limit)
        if c3_trial < best_c3:
            best_x = x_trial
            best_c3 = c3_trial
    
    # Strategy 2: Standard random initialization + local search
    for trial in range(5):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        x_init = np.random.choice([-1, 1], size=N)
        x2, c3_2 = _local_search(x_init, max_iter=2000, time_limit=15)
        if c3_2 < best_c3:
            best_x = x2
            best_c3 = c3_2
    
    # Check time
    if time.time() > DEADLINE - TIME_MARGIN:
        return best_x
    
    # Strategy 3: Simulated annealing from best solution
    if time.time() < DEADLINE - TIME_MARGIN:
        x3, c3_3 = _simulated_annealing(best_x, max_iter=10000, T_start=0.5, T_end=0.01,
                                         time_limit=DEADLINE - TIME_MARGIN - time.time())
        if c3_3 < best_c3:
            best_x = x3
            best_c3 = c3_3
    
    # Final local search polish
    if time.time() < DEADLINE - TIME_MARGIN:
        x4, c3_4 = _local_search(best_x, max_iter=5000, time_limit=DEADLINE - TIME_MARGIN - time.time())
        if c3_4 < best_c3:
            best_x = x4
            best_c3 = c3_4
    
    return best_x

def construct():
    """
    Entry point: return a sequence of signed step heights (±1) of length N=400.
    """
    x = _construct()
    # Ensure output is a numpy array of ±1 values
    return np.sign(x).astype(int)
# EVOLVE-BLOCK-END
