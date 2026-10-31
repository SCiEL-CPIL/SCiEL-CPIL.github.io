import numpy as np
import time
import math

N = 400

def _compute_psl(f):
    """
    Computes the peak sidelobe magnitude of the linear autocorrelation.
    f is a 1D array of length N with values +/- 1.
    """
    # Use FFT for efficient autocorrelation
    # Linear autocorrelation can be computed via circular autocorrelation of a zero-padded signal
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    PSD = F * np.conj(F)
    R = np.fft.irfft(PSD, n=n_fft)
    # R[0] is the peak (energy = N for binary)
    # R[1:N] are the positive lags sidelobes
    # Due to symmetry, R[N+1:] are negative lags, same magnitude as R[1:N] reversed
    # We only need to check R[1:N]
    return np.max(np.abs(R[1:N]))

def _local_search_block(f, max_iter=2000, time_limit=5.0):
    """
    Local search using single-bit flips and pair swaps.
    Returns improved f and psl.
    """
    start_time = time.time()
    current_f = f.copy()
    current_psl = _compute_psl(current_f)
    improved = True
    iterations = 0
    
    while improved and iterations < max_iter:
        if time.time() - start_time > time_limit:
            break
        improved = False
        iterations += 1
        
        # 1. Try single bit flips
        # Optimization: We can precompute the change in PSL for each flip?
        # PSL is max of sidelobes. Changing one bit changes all sidelobes.
        # It's O(N) to compute new PSL. O(N^2) total per pass.
        # N=400, N^2=160,000. Fast enough.
        
        best_flip_idx = -1
        best_flip_psl = current_psl
        
        for i in range(N):
            old_val = current_f[i]
            current_f[i] = -old_val
            new_psl = _compute_psl(current_f)
            if new_psl < best_flip_psl:
                best_flip_psl = new_psl
                best_flip_idx = i
            else:
                current_f[i] = old_val
        
        if best_flip_idx != -1:
            current_f[best_flip_idx] = -current_f[best_flip_idx]
            current_psl = best_flip_psl
            improved = True
            continue # Restart scan after a flip
        
        # 2. Try pair swaps (i, j)
        # This preserves the sum of the sequence.
        # Only try if no single flip improved.
        # We limit pair swaps to keep runtime manageable.
        # Random sampling of pairs or structured?
        # Let's try a few random pairs per iteration if no single flip works.
        # Or structured: swap i and i+k for small k.
        
        # For efficiency, we will just do a few random pair swaps in the SA phase.
        # In this local search, we stick to single flips to guarantee convergence to a local min.
        # Pair swaps are better handled in the global SA.
        
    return current_f, current_psl

def _simulated_annealing(f_init, time_limit=20.0):
    """
    Simulated Annealing with block flips.
    """
    start_time = time.time()
    current_f = f_init.copy()
    current_psl = _compute_psl(current_f)
    best_f = current_f.copy()
    best_psl = current_psl
    
    T = 0.5 # Initial temperature
    cooling_rate = 0.999
    min_T = 0.01
    
    # Block size varies
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        # Generate neighbor
        # Choose a random block length L between 2 and N//4
        L = np.random.randint(2, max(3, N // 4))
        start_idx = np.random.randint(0, N - L)
        
        neighbor_f = current_f.copy()
        neighbor_f[start_idx:start_idx+L] *= -1
        
        new_psl = _compute_psl(neighbor_f)
        
        delta = new_psl - current_psl
        
        if delta < 0 or np.random.rand() < math.exp(-delta / T):
            current_f = neighbor_f
            current_psl = new_psl
            
            if current_psl < best_psl:
                best_f = current_f.copy()
                best_psl = current_psl
        
        T *= cooling_rate
        
    return best_f, best_psl

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_f = None
    best_psl = float('inf')
    
    rng = np.random.default_rng(42)
    
    # 1. Initial candidates
    # Random
    f_rand = rng.choice([-1, 1], size=N).astype(float)
    
    # Legendre Sequence (for prime N, but N=400 is not prime. Use Legendre for N=397 or 401 and pad/truncate?)
    # Let's use a simple "Balanced" random sequence.
    
    # M-Sequence (as in seed, but ensure it's valid)
    # The seed's m-seq might have been okay, but let's generate a proper one for length 511 and take middle 400?
    # Or just use random.
    
    initial_sequences = [f_rand]
    
    # Add a few more diverse random starts
    for _ in range(2):
        f_rand2 = rng.choice([-1, 1], size=N).astype(float)
        initial_sequences.append(f_rand2)
        
    # Try a "Chirp-like" binary sequence? 
    # f[i] = 1 if (i^2 mod N) < N/2 else -1 ?
    x = np.arange(N)
    f_chirp = np.where((x**2) % (2*N) < N, 1, -1).astype(float)
    initial_sequences.append(f_chirp)
    
    for f_init in initial_sequences:
        if time.time() > deadline - 10:
            break
            
        # Phase 1: Simulated Annealing
        f_sa, psl_sa = _simulated_annealing(f_init, time_limit=15.0)
        
        # Phase 2: Local Search (Polishing)
        f_ls, psl_ls = _local_search_block(f_sa, max_iter=1000, time_limit=5.0)
        
        if psl_ls < best_psl:
            best_psl = psl_ls
            best_f = f_ls.copy()
            
        # Check time
        if time.time() > deadline - 5:
            break
            
    # Final Polish on the best found
    if best_f is not None and time.time() < deadline - 2:
        f_final, psl_final = _local_search_block(best_f, max_iter=2000, time_limit=5.0)
        if psl_final < best_psl:
            best_psl = psl_final
            best_f = f_final.copy()
            
    if best_f is None:
        best_f = np.ones(N)
        
    return best_f
