"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

DEADLINE = 150.0
TIME_MARGIN = 15.0


def _compute_c3(x):
    """Compute C3 = max_{k!=0} |R[k]| / R[0] using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    R0 = R[0]
    if R0 == 0:
        return 1.0
    # R[1:N] positive lags, R[N:2N-1] negative lags
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    max_sidelobe = np.max(np.abs(sidelobes))
    return max_sidelobe / R0


def _compute_R(x):
    """Compute full autocorrelation R[0..2N-2] via FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    return np.fft.irfft(power, n=n_fft)[:2 * N - 1]


def _c3_from_R(R):
    """Compute C3 from precomputed R."""
    R0 = R[0]
    if R0 == 0:
        return 1.0
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    return np.max(np.abs(sidelobes)) / R0


def _flip_effect(x, i, R):
    """
    Compute new R after flipping x[i].
    R[k] = sum_{j=0}^{N-1-k} x[j] x[j+k] for k=0..N-1
    After flip at i: x'[i] = -x[i]
    delta_R[k] = -2*x[i]*x[i+k] (if k <= N-1-i) - 2*x[i-k]*x[i] (if k <= i)
    R_full has length 2N-1, symmetric: R_full[2N-2-k] = R_full[k]
    """
    R_new = R.copy()
    xi = x[i]
    # Vectorized computation of delta_R for k=0..N-1
    # Term 1: -2*xi*x[i+k] for k where i+k < N, i.e., k in 0..N-1-i
    # Term 2: -2*x[i-k]*xi for k where i-k >= 0, i.e., k in 0..i
    delta = np.zeros(N)
    if N - 1 - i >= 0:
        delta[0:N-i] += -2.0 * xi * x[i:i+N-i]
    if i >= 0:
        delta[0:i+1] += -2.0 * x[i::-1] * xi
    R_new[0:N] += delta
    # Reimpose symmetry: R_full[2N-2-k] = R_full[k] for k=0..N-1
    for k in range(N):
        R_new[2*N-2-k] = R_new[k]
    return R_new


def _local_search_fast(x, R, max_iter=100000, time_limit=None):
    """Local search with O(N) flip evaluation."""
    start_time = time.time()
    current_c3 = _c3_from_R(R)
    best_x = x.copy()
    best_R = R.copy()
    best_c3 = current_c3
    no_improve = 0
    max_no_improve = 200
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        idx = random.randint(0, N - 1)
        R_new = _flip_effect(x, idx, R)
        new_c3 = _c3_from_R(R_new)
        
        if new_c3 < current_c3 - 1e-12:
            x[idx] = -x[idx]
            R = R_new
            current_c3 = new_c3
            no_improve = 0
            if current_c3 < best_c3:
                best_x = x.copy()
                best_R = R.copy()
                best_c3 = current_c3
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
    
    return best_x, best_R, best_c3


def _simulated_annealing_fast(x, R, max_iter=50000, T_start=0.1, T_end=0.001, time_limit=None):
    """Simulated annealing with O(N) flip evaluation."""
    start_time = time.time()
    current_c3 = _c3_from_R(R)
    best_x = x.copy()
    best_R = R.copy()
    best_c3 = current_c3
    T = T_start
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        T = T_start * (T_end / T_start) ** (it / max_iter)
        
        idx = random.randint(0, N - 1)
        R_new = _flip_effect(x, idx, R)
        new_c3 = _c3_from_R(R_new)
        
        delta = new_c3 - current_c3
        if delta < 0 or (T > 1e-10 and random.random() < np.exp(-delta / T)):
            x[idx] = -x[idx]
            R = R_new
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_x = x.copy()
                best_R = R.copy()
                best_c3 = current_c3
    
    return best_x, best_R, best_c3


def _thue_morse(n):
    """Generate Thue-Morse sequence."""
    seq = np.zeros(n)
    for i in range(n):
        c = bin(i).count('1')
        seq[i] = 1.0 if c % 2 == 0 else -1.0
    return seq


def _m_sequence(n, poly=0x5D, seed=1):
    """Generate m-sequence (maximal length LFSR)."""
    # Use a simple LFSR
    state = seed
    seq = np.zeros(n)
    for i in range(n):
        # Extract bit
        bit = state & 1
        seq[i] = 1.0 if bit == 0 else -1.0
        # Update state (simple Fibonacci LFSR)
        # For a 8-bit LFSR with polynomial x^8 + x^6 + x^5 + x^4 + 1
        # This is a simplified version; use numpy random as fallback
        pass
    # Fallback to random
    rng = np.random.RandomState(seed)
    return rng.choice([-1, 1], size=n)


def _vectorized_best_flip(x, R):
    """
    For each position i, compute the C3 if we flip x[i].
    Return the index i that gives the best (lowest) C3, and the new R.
    This is O(N^2) but vectorized with numpy.
    """
    # Precompute for each position i, the delta_R vector
    # delta_R_i[k] = -2*x[i]*x[i+k] (if i+k < N) - 2*x[i-k]*x[i] (if i-k >= 0)
    
    # We'll compute C3 for each possible flip
    # R_new_i = R + delta_i
    # C3_i = max(|R_new_i[1:N]|, |R_new_i[N:2N-1]|) / R_new_i[0]
    
    # Note: R_new_i[0] = R[0] - 2*x[i]^2 = R[0] - 2 (since x[i] = ±1)
    # So R_new_i[0] is constant for all i!
    R0_new = R[0] - 2.0  # since x[i]^2 = 1
    
    if R0_new == 0:
        return -1, R
    
    best_c3 = float('inf')
    best_i = -1
    best_R_new = R
    
    # For efficiency, we compute delta for each i and track the max sidelobe
    # delta_i[k] for k in 0..N-1
    # The sidelobes are at positions 1..N-1 and N..2N-2
    
    # Let's compute all deltas at once using broadcasting
    # delta[i, k] = -2*x[i]*x[i+k] (if i+k < N) - 2*x[i-k]*x[i] (if i-k >= 0)
    
    # Create index matrix
    i_idx = np.arange(N)
    k_idx = np.arange(N)
    
    # i+k matrix
    ik = i_idx[:, None] + k_idx[None, :]
    # i-k matrix
    ik_minus = i_idx[:, None] - k_idx[None, :]
    
    # Mask for valid indices
    mask_ik = (ik >= 0) & (ik < N)
    mask_ik_minus = (ik_minus >= 0) & (ik_minus < N)
    
    # Get x values with masking
    # For x[i+k]: valid where mask_ik
    x_ik = np.where(mask_ik, x[np.clip(ik, 0, N-1)], 0.0)
    # For x[i-k]: valid where mask_ik_minus
    x_ik_minus = np.where(mask_ik_minus, x[np.clip(ik_minus, 0, N-1)], 0.0)
    
    # delta[i, k] = -2*x[i]*x[i+k] - 2*x[i-k]*x[i]
    delta = -2.0 * x[:, None] * x_ik - 2.0 * x_ik_minus * x[:, None]
    
    # Now for each i, R_new_i = R + delta[i, :]
    # We need to compute C3 for each
    # Sidelobes: positions 1..N-1 and N..2N-2
    # R_new_i[k] for k in 1..N-1
    # R_new_i[2N-2-k] = R_new_i[k] for k in 0..N-1 (symmetry)
    # So negative lags are at positions N..2N-2, which correspond to R_new_i[2N-2-k] for k=0..N-2
    # = R_new_i[k] for k=2..N-1 (by symmetry)
    
    # Actually, let's just compute the max of |R_new_i[1:N]| and |R_new_i[N:2N-1]|
    # R_new_i has length 2N-1
    # Positions 1..N-1: positive lags
    # Positions N..2N-2: negative lags
    
    # R_new_i[k] = R[k] + delta[i, k] for k in 0..N-1
    # R_new_i[2N-2-k] = R_new_i[k] for k in 0..N-1
    
    # So for k in N..2N-2, let k' = 2N-2-k, so k' in 0..N-2
    # R_new_i[k] = R_new_i[k'] = R[k'] + delta[i, k']
    
    # Max sidelobe for flip i = max( max_{k=1..N-1} |R[k] + delta[i,k]|, max_{k=N..2N-2} |R[2N-2-k] + delta[i,2N-2-k]| )
    # = max( max_{k=1..N-1} |R[k] + delta[i,k]|, max_{k'=0..N-2} |R[k'] + delta[i,k']| )
    # = max( max_{k=1..N-1} |R[k] + delta[i,k]|, max_{k'=0..N-2} |R[k'] + delta[i,k']| )
    # = max( max_{k=0..N-1, k!=0} |R[k] + delta[i,k]| )
    # Wait, that's not quite right. Let me be more careful.
    
    # R_new_i has indices 0..2N-2
    # R_new_i[0] = R[0] + delta[i,0]
    # R_new_i[k] for k=1..N-1: = R[k] + delta[i,k]
    # R_new_i[N] = R_new_i[N-1] by symmetry? No.
    # R_new_i[2N-2-k] = R_new_i[k] for k=0..N-1
    # So R_new_i[N] = R_new_i[2N-2-N] = R_new_i[N-2]
    # R_new_i[N+1] = R_new_i[N-3]
    # ...
    # R_new_i[2N-2] = R_new_i[0]
    
    # So the sidelobes are at indices 1..N-1 and N..2N-2
    # Index N..2N-2 correspond to R_new_i[2N-2-k] for k=N-2..0
    # = R_new_i[k] for k=N-2..0 (by symmetry, but these are the same as indices 0..N-2)
    
    # So max sidelobe = max( max_{k=1..N-1} |R_new_i[k]|, max_{k=N..2N-2} |R_new_i[k]| )
    # = max( max_{k=1..N-1} |R[k] + delta[i,k]|, max_{k=N..2N-2} |R[2N-2-k] + delta[i,2N-2-k]| )
    # Let j = 2N-2-k, so when k=N, j=N-2; when k=2N-2, j=0
    # = max( max_{k=1..N-1} |R[k] + delta[i,k]|, max_{j=0..N-2} |R[j] + delta[i,j]| )
    # = max( max_{k=1..N-1} |R[k] + delta[i,k]|, max_{j=0..N-2} |R[j] + delta[i,j]| )
    # = max( max_{k=0..N-1} |R[k] + delta[i,k]| )  [since k=0 is included in second max]
    
    # So for each i, max_sidelobe_i = max_{k=0..N-1} |R[k] + delta[i,k]|
    # And C3_i = max_sidelobe_i / R0_new
    
    # Compute R[k] + delta[i,k] for all i,k
    R_ext = R[0:N]  # R[0..N-1]
    R_plus_delta = R_ext[None, :] + delta  # shape (N, N)
    
    # Max absolute value for each i
    max_sidelobes = np.max(np.abs(R_plus_delta), axis=1)
    
    # C3 for each flip
    c3_values = max_sidelobes / R0_new
    
    # Find best
    best_i = np.argmin(c3_values)
    best_c3 = c3_values[best_i]
    
    if best_c3 < float('inf'):
        best_R_new = _flip_effect(x, best_i, R)
    
    return best_i, best_R_new, best_c3


def _local_search_systematic(x, R, max_iter=100000, time_limit=None):
    """Local search that systematically finds the best flip at each step."""
    start_time = time.time()
    current_c3 = _c3_from_R(R)
    best_x = x.copy()
    best_R = R.copy()
    best_c3 = current_c3
    no_improve = 0
    max_no_improve = 50  # Fewer no-improve iterations since we're doing systematic search
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Find the best flip
        best_i, best_R_new, best_c3_new = _vectorized_best_flip(x, R)
        
        if best_i >= 0 and best_c3_new < current_c3 - 1e-12:
            x[best_i] = -x[best_i]
            R = best_R_new
            current_c3 = best_c3_new
            no_improve = 0
            if current_c3 < best_c3:
                best_x = x.copy()
                best_R = R.copy()
                best_c3 = current_c3
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
    
    return best_x, best_R, best_c3


def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = np.ones(N)
    best_R = _compute_R(best_x)
    best_c3 = _c3_from_R(best_R)
    
    # Initializations
    inits = []
    inits.append(_thue_morse(N))
    rng = np.random.RandomState(12345)
    inits.append(rng.choice([-1, 1], size=N))
    rng2 = np.random.RandomState(67890)
    inits.append(rng2.choice([-1, 1], size=N))
    # Logistic map
    val = 0.5
    log_seq = np.zeros(N)
    for i in range(N):
        val = 3.99 * val * (1.0 - val)
        log_seq[i] = 1.0 if val > 0.5 else -1.0
    inits.append(log_seq)
    
    # Local search from each initialization - use systematic search
    for i, x_init in enumerate(inits):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        R_init = _compute_R(x_init)
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = min(20.0, time_left * 0.25)
        if time_limit < 1.0:
            break
        x_trial, R_trial, c3_trial = _local_search_systematic(x_init.copy(), R_init, max_iter=10000, time_limit=time_limit)
        if c3_trial < best_c3:
            best_x = x_trial
            best_R = R_trial
            best_c3 = c3_trial
    
    # Simulated annealing from best
    if time.time() < DEADLINE - TIME_MARGIN - 5.0:
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = min(30.0, time_left * 0.4)
        if time_limit > 2.0:
            x_sa, R_sa, c3_sa = _simulated_annealing_fast(best_x.copy(), best_R.copy(), max_iter=50000, time_limit=time_limit)
            if c3_sa < best_c3:
                best_x = x_sa
                best_R = R_sa
                best_c3 = c3_sa
    
    # Final systematic local search polish
    if time.time() < DEADLINE - TIME_MARGIN - 2.0:
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = max(1.0, time_left * 0.8)
        x_final, R_final, c3_final = _local_search_systematic(best_x.copy(), best_R.copy(), max_iter=10000, time_limit=time_limit)
        if c3_final < best_c3:
            best_x = x_final
            best_R = R_final
            best_c3 = c3_final
    
    return best_x
# EVOLVE-BLOCK-END
