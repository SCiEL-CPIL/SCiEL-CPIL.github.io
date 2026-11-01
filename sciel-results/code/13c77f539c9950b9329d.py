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
    C3 = max_{k=1,...,N-1} |R[k]| / R[0]
    where R[k] is the autocorrelation at lag k.
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


def _init_autocorr(x):
    """Initialize autocorrelation array from x using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    return R


def _update_autocorr_flip(R, x, idx):
    """
    Update autocorrelation R after flipping x[idx].
    Flipping x[idx] changes x[idx] to -x[idx].
    For each lag k, R[k] changes by:
      -2 * x[idx] * (sum of x[j]*x[j+k] where j or j+k == idx)
    More precisely:
      R[k] = sum_i x[i]*x[i+k]
      After flip at idx: x[idx] -> -x[idx]
      Delta R[k] = -2*x[idx] * (sum over terms involving idx)
      Terms involving idx:
        - When i = idx: x[idx]*x[idx+k]
        - When i+k = idx, i.e., i = idx-k: x[idx-k]*x[idx]
      So Delta R[k] = -2*x[idx]*(x[idx+k] if idx+k < N else 0)
                      -2*x[idx]*(x[idx-k] if idx-k >= 0 else 0)
    For negative lags (R[N+k] corresponds to lag -k), similar logic applies.
    """
    R = R.copy()
    old_val = x[idx]
    x[idx] = -old_val  # flip
    
    # For positive lags k = 1..N-1:
    # R[k] = sum_{i=0}^{N-1-k} x[i]*x[i+k]
    # After flipping x[idx]:
    #   If idx < N-k: term i=idx exists: contribution was old_val*x[idx+k], now -old_val*x[idx+k], delta = -2*old_val*x[idx+k]
    #   If idx-k >= 0 and idx-k < N-k (i.e., 0 <= idx-k <= N-1-k): term i=idx-k exists: contribution was x[idx-k]*old_val, now x[idx-k]*(-old_val), delta = -2*x[idx-k]*old_val
    # Note: if idx < N-k AND idx-k >= 0, both terms exist.
    
    for k in range(1, N):
        delta = 0.0
        # Term where i = idx (requires idx <= N-1-k, i.e., k <= N-1-idx)
        if idx + k < N:
            delta -= 2.0 * old_val * x[idx + k]
        # Term where i+k = idx, i.e., i = idx-k (requires idx-k >= 0 and idx-k <= N-1-k, i.e., idx >= k and k <= N-1-(idx-k) -> always true if idx>=k)
        if idx - k >= 0:
            delta -= 2.0 * x[idx - k] * old_val
        R[k] += delta
    
    # For negative lags: R[N+k] corresponds to correlation at lag -k
    # R[N+k] = sum_{i=k}^{N-1} x[i]*x[i-k]
    # After flipping x[idx]:
    #   If idx >= k: term i=idx: was x[idx]*x[idx-k], now -x[idx]*x[idx-k], delta = -2*old_val*x[idx-k]
    #   If idx-k >= k, i.e., idx >= 2k: term i=idx+k: was x[idx+k]*x[idx], now x[idx+k]*(-old_val), delta = -2*x[idx+k]*old_val
    for k in range(1, N):
        delta = 0.0
        # Term where i = idx (requires idx >= k)
        if idx >= k:
            delta -= 2.0 * old_val * x[idx - k]
        # Term where i-k = idx, i.e., i = idx+k (requires idx+k < N and idx+k >= k, i.e., idx >= 0)
        if idx + k < N:
            delta -= 2.0 * x[idx + k] * old_val
        R[N + k] += delta
    
    return R, x


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


def _local_search(x, max_iter=50000, time_limit=None):
    """
    Local search with incremental autocorrelation update for speed.
    """
    start_time = time.time()
    R = _init_autocorr(x)
    R0 = R[0]
    current_c3 = _compute_c3(x)
    best_x = x.copy()
    best_c3 = current_c3
    no_improve_count = 0
    max_no_improve = 200
    
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
        R_new, x_new = _update_autocorr_flip(R, x_new, idx)
        
        # Compute new C3 from R_new
        if R_new[0] == 0:
            new_c3 = 1.0
        else:
            sidelobes = np.concatenate([R_new[1:N], R_new[N:2*N-1]])
            max_sidelobe = np.max(np.abs(sidelobes))
            new_c3 = max_sidelobe / R_new[0]
        
        if new_c3 < current_c3 - 1e-12:
            x = x_new
            R = R_new
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


def _simulated_annealing(x, max_iter=50000, T_start=1.0, T_end=0.01, time_limit=None):
    """
    Simulated annealing with incremental autocorrelation update.
    """
    start_time = time.time()
    R = _init_autocorr(x)
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
        R_new, x_new = _update_autocorr_flip(R, x_new, idx)
        
        # Compute new C3 from R_new
        if R_new[0] == 0:
            new_c3 = 1.0
        else:
            sidelobes = np.concatenate([R_new[1:N], R_new[N:2*N-1]])
            max_sidelobe = np.max(np.abs(sidelobes))
            new_c3 = max_sidelobe / R_new[0]
        
        delta = new_c3 - current_c3
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            x = x_new
            R = R_new
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
        x_trial, c3_trial = _local_search(x_init.copy(), max_iter=50000, time_limit=time_limit)
        if c3_trial < best_c3:
            best_x = x_trial
            best_c3 = c3_trial
    
    # Strategy 2: Standard random initialization + local search
    for trial in range(8):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        x_init = np.random.choice([-1, 1], size=N)
        x2, c3_2 = _local_search(x_init, max_iter=30000, time_limit=12)
        if c3_2 < best_c3:
            best_x = x2
            best_c3 = c3_2
    
    # Check time
    if time.time() > DEADLINE - TIME_MARGIN:
        return best_x
    
    # Strategy 3: Simulated annealing from best solution
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        x3, c3_3 = _simulated_annealing(best_x, max_iter=50000, T_start=0.5, T_end=0.01,
                                         time_limit=remaining * 0.5)
        if c3_3 < best_c3:
            best_x = x3
            best_c3 = c3_3
    
    # Final local search polish
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        x4, c3_4 = _local_search(best_x, max_iter=50000, time_limit=remaining)
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
