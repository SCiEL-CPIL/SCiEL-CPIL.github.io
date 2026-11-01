"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

DEADLINE = 120.0
TIME_MARGIN = 15.0  # safety margin in seconds


def _compute_c3(x):
    """
    Compute the upper bound C3 for a signed step function x of length N.
    
    The autoconvolution upper bound C3 is related to the maximum absolute
    value of the autocorrelation sidelobes relative to the main lobe.
    
    For a signed step function (values in {-1, +1}), the autocorrelation is:
      R[k] = sum_{i=0}^{N-1-k} x[i] * x[i+k]  for k = 0, ..., N-1
    
    The "upper bound on C3" is defined as:
      C3 = max_{k != 0} |R[k]| / R[0]
    
    where R[0] = sum x[i]^2 = N for ±1 values.
    """
    # Compute autocorrelation using FFT for efficiency
    n_fft = 1 << (2 * N - 1).bit_length()  # next power of 2 for linear correlation
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    
    R0 = R[0]
    if R0 == 0:
        return 1.0  # degenerate case
    
    # Maximum absolute sidelobe
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    max_sidelobe = np.max(np.abs(sidelobes))
    
    C3 = max_sidelobe / R0
    return C3


def _compute_c3_fast(x):
    """
    Fast computation of C3 using precomputed FFT structure.
    Only the changed element affects the spectrum, so we can update efficiently.
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


def _local_search(x, max_iter=5000, time_limit=None):
    """
    Local search: try flipping individual positions and accept if it improves C3.
    """
    start_time = time.time()
    current_c3 = _compute_c3(x)
    best_x = x.copy()
    best_c3 = current_c3
    no_improve_count = 0
    max_no_improve = 200
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Try a random flip
        idx = random.randint(0, N - 1)
        x_new = x.copy()
        x_new[idx] = -x_new[idx]
        new_c3 = _compute_c3_fast(x_new)
        
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
        new_c3 = _compute_c3_fast(x_new)
        
        delta = new_c3 - current_c3
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            x = x_new
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
    
    return best_x, best_c3


def _parallel_tempering(x, max_iter=15000, n_chains=4, T_start=1.0, T_end=0.01, time_limit=None):
    """
    Parallel tempering (replica exchange) to escape local minima more effectively.
    Maintains multiple chains at different temperatures and swaps configurations
    between them with a Metropolis criterion.
    """
    start_time = time.time()
    
    # Initialize chains
    chains = []
    for i in range(n_chains):
        if i == 0:
            chains.append(x.copy())
        else:
            # Perturb from the initial solution
            cx = x.copy()
            n_flips = random.randint(10, 50)
            for _ in range(n_flips):
                idx = random.randint(0, N - 1)
                cx[idx] = -cx[idx]
            chains.append(cx)
    
    # Compute C3 for all chains
    chain_c3s = [_compute_c3(c) for c in chains]
    
    # Best solution tracking
    best_x = chains[0].copy()
    best_c3 = min(chain_c3s)
    if chain_c3s[0] < best_c3:
        best_c3 = chain_c3s[0]
        best_x = chains[0].copy()
    
    # Temperature schedule for each chain
    # Chain 0 is the coldest, chain n-1 is the hottest
    temps = np.linspace(T_end, T_start, n_chains)
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Update temperatures
        temps = T_end + (T_start - T_end) * (1 - iteration / max_iter) ** 0.5
        temps = np.linspace(T_end, T_start, n_chains) * (1 - 0.3 * iteration / max_iter)
        temps = np.maximum(temps, 0.005)
        
        # Move each chain (single flip)
        for i in range(n_chains):
            idx = random.randint(0, N - 1)
            chains[i][idx] = -chains[i][idx]
            new_c3 = _compute_c3_fast(chains[i])
            
            delta = new_c3 - chain_c3s[i]
            T_i = max(temps[i], 1e-10)
            
            if delta < 0 or random.random() < np.exp(-delta / T_i):
                chain_c3s[i] = new_c3
                if new_c3 < best_c3:
                    best_c3 = new_c3
                    best_x = chains[i].copy()
            else:
                chains[i][idx] = -chains[i][idx]  # revert
        
        # Try swap between adjacent chains
        if n_chains > 1 and iteration % 3 == 0:
            i = random.randint(0, n_chains - 2)
            j = i + 1
            T_i = max(temps[i], 1e-10)
            T_j = max(temps[j], 1e-10)
            
            # Swap acceptance probability
            delta = (1/T_i - 1/T_j) * (chain_c3s[j] - chain_c3s[i])
            if random.random() < np.exp(-delta):
                # Swap configurations
                chains[i], chains[j] = chains[j], chains[i]
                chain_c3s[i], chain_c3s[j] = chain_c3s[j], chain_c3s[i]
    
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
    
    # Strategy 1: Start with all ones, do local search
    x1, c3_1 = _local_search(np.ones(N), max_iter=3000, time_limit=20)
    if c3_1 < best_c3:
        best_x = x1
        best_c3 = c3_1
    
    # Check time
    if time.time() > DEADLINE - TIME_MARGIN:
        return best_x
    
    # Strategy 2: Random initialization + local search
    for trial in range(4):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        x_init = np.random.choice([-1, 1], size=N)
        x2, c3_2 = _local_search(x_init, max_iter=1500, time_limit=12)
        if c3_2 < best_c3:
            best_x = x2
            best_c3 = c3_2
    
    # Check time
    if time.time() > DEADLINE - TIME_MARGIN:
        return best_x
    
    # Strategy 3: Parallel tempering from best solution
    remaining_time = DEADLINE - TIME_MARGIN - time.time()
    if remaining_time > 15:
        x3, c3_3 = _parallel_tempering(best_x, max_iter=15000, n_chains=4, 
                                        T_start=0.5, T_end=0.01,
                                        time_limit=min(remaining_time - 5, 60))
        if c3_3 < best_c3:
            best_x = x3
            best_c3 = c3_3
    
    # Final local search polish
    remaining_time = DEADLINE - TIME_MARGIN - time.time()
    if remaining_time > 5:
        x4, c3_4 = _local_search(best_x, max_iter=5000, time_limit=remaining_time - 2)
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
