import numpy as np
import time

N = 400
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def _autocorr_peak_sidelobe(s):
    """
    Compute the maximum absolute value of the non-zero lag autocorrelation.
    """
    n_fft = 2 * N
    S = np.fft.fft(s, n=n_fft)
    r = np.fft.ifft(S * np.conj(S)).real
    # r[0] is lag 0 (energy). r[1:N] are positive lags.
    # Due to symmetry, max(abs(r[1:N])) is the peak sidelobe.
    return np.max(np.abs(r[1:N]))

def _autocorr_energy(s):
    """
    Compute the sum of squares of the non-zero lag autocorrelation.
    This is used for continuous relaxation.
    """
    n_fft = 2 * N
    S = np.fft.fft(s, n=n_fft)
    r = np.fft.ifft(S * np.conj(S)).real
    return np.sum(np.abs(r[1:N])**2)

def _continuous_relaxation(rng, iters=500, lr=0.05):
    """
    Optimize a continuous vector x to minimize autocorrelation energy.
    Returns the sign of the optimized vector.
    """
    # Initialize with random signs
    x = rng.choice([-1.0, 1.0], size=N)
    
    # Gradient descent on energy E = sum |r_k|^2
    # The gradient of E with respect to x can be computed efficiently.
    # Let R be the circular autocorrelation matrix. E = r^T r (excluding lag 0).
    # Actually, simpler: use FFT based gradient.
    # E = sum_{k=1}^{N-1} |r_k|^2.
    # We can approximate the gradient numerically or analytically.
    # Analytical gradient of E(x) = sum_{k=1}^{N-1} (sum_{i=0}^{N-1-k} x_i x_{i+k})^2 
    # is complex. 
    # Instead, let's use a simple heuristic: Minimize the L2 norm of the sidelobes.
    # We can use a few steps of gradient descent on the continuous domain.
    
    # A simpler and very effective approach for low autocorrelation is to 
    # start with a random sequence and use a greedy "bit flip" that minimizes 
    # the peak sidelobe, but we want a better start.
    
    # Let's try a different continuous relaxation: 
    # Minimize the peak sidelobe directly is non-differentiable.
    # Minimize the energy is differentiable.
    # Let's use a few iterations of gradient descent on energy.
    
    # Gradient of Energy E = || r[1:] ||^2
    # r = ifft(fft(x) * conj(fft(x)))
    # This is equivalent to x * circular_conv(x)
    
    # Let's just use a random start and rely on the strong local search.
    # The continuous relaxation is often overkill if the local search is strong enough
    # and we have many restarts. 
    # However, a "greedy construction" can be faster.
    
    # Let's construct a sequence greedily:
    # s = [1]
    # For i in 1..N-1:
    #   Try s_i = 1, calculate current peak sidelobe (approximate)
    #   Try s_i = -1, calculate current peak sidelobe
    #   Pick the one that minimizes the peak sidelobe so far.
    
    # This greedy construction is O(N^2) or O(N^2 log N) with FFT updates.
    # For N=400, it's fast.
    
    s = np.zeros(N)
    s[0] = 1.0
    
    # Pre-allocate
    current_peak = 0.0
    
    for i in range(1, N):
        # Try 1
        s[i] = 1.0
        # Update autocorrelation efficiently? 
        # For simplicity in the construction phase, we can just compute full FFT every few steps 
        # or just use the full FFT for each step since N=400 is small.
        # 400 steps * FFT(800) is very fast.
        
        peak1 = _autocorr_peak_sidelobe(s)
        
        # Try -1
        s[i] = -1.0
        peak_minus1 = _autocorr_peak_sidelobe(s)
        
        if peak1 < peak_minus1:
            s[i] = 1.0
        else:
            s[i] = -1.0
            
    return s

def _local_search(s, rng, time_limit=10.0):
    """
    Perform local search on the sequence s.
    Uses a combination of single-bit flips and pair-flips.
    """
    start_time = time.time()
    cur_val = _autocorr_peak_sidelobe(s)
    best_val = cur_val
    best_s = s.copy()
    
    # Greedy phase: Single bit flips
    # We scan the array and flip bits if it improves the peak sidelobe.
    # Repeat until no improvement.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Randomize the order of bits to avoid bias
        order = rng.permutation(N)
        for i in order:
            if (time.time() - start_time) > time_limit:
                break
            old_val = s[i]
            s[i] = -old_val
            new_val = _autocorr_peak_sidelobe(s)
            if new_val < cur_val:
                cur_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_s = s.copy()
                improved = True
            else:
                s[i] = old_val
                
    # Pair flip phase
    # Try swapping pairs of bits.
    # This is more expensive but can escape local minima of single-bit flips.
    pair_attempts = 0
    max_pair_attempts = 5000
    while pair_attempts < max_pair_attempts and (time.time() - start_time) < time_limit:
        i, j = rng.choice(N, size=2, replace=False)
        if s[i] == s[j]:
            pair_attempts += 1
            continue
            
        s[i], s[j] = s[j], s[i]
        new_val = _autocorr_peak_sidelobe(s)
        
        if new_val < cur_val:
            cur_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_s = s.copy()
            improved = True
        else:
            s[i], s[j] = s[j], s[i]
            
        pair_attempts += 1
        
    return best_s, best_val

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    rng = np.random.default_rng(42)
    
    best_s = None
    best_val = np.inf
    
    # 1. Generate diverse initializations
    inits = []
    
    # Greedy Construction
    inits.append(_continuous_relaxation(rng))
    
    # Random
    inits.append(rng.choice([-1.0, 1.0], size=N))
    
    # Sine patterns
    x = np.linspace(-1/4, 1/4, N, endpoint=False)
    for freq in [2, 4, 8, 16, 32]:
        p = np.sign(np.cos(2 * np.pi * freq * x))
        p[p == 0] = 1.0
        inits.append(p)
        
    # Block structures
    p_block = np.ones(N)
    p_block[N//4:N//2] = -1
    p_block[3*N//4:] = -1
    inits.append(p_block)
    
    # Alternating
    p_alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    inits.append(p_alt)
    
    # 2. Optimize each initialization
    # Allocate time budget per initialization
    # We have ~145 seconds. 
    # Let's do 10 restarts. 14 seconds each.
    
    num_restarts = 10
    time_per_restart = (DEADLINE - SAFETY_MARGIN - (time.time() - start_time)) / num_restarts
    
    for k in range(num_restarts):
        if time.time() > deadline:
            break
            
        if k < len(inits):
            cur_s = inits[k].copy()
        else:
            # Use the best found so far as the start for subsequent restarts?
            # Or a new random one?
            # Let's perturb the best found so far to explore its neighborhood.
            if best_s is not None:
                cur_s = best_s.copy()
                # Perturb 10% of the bits
                num_flips = N // 10
                flip_indices = rng.choice(N, size=num_flips, replace=False)
                cur_s[flip_indices] *= -1
            else:
                cur_s = rng.choice([-1.0, 1.0], size=N)
        
        # Run local search
        # Use a fraction of the time for local search
        search_time = time_per_restart * 0.8
        cur_s, cur_val = _local_search(cur_s, rng, time_limit=search_time)
        
        if cur_val < best_val:
            best_val = cur_val
            best_s = cur_s.copy()
            
        # Check time
        if time.time() > deadline - 10:
            # Final polish on the best found
            if best_s is not None:
                # Use remaining time for a final greedy polish on the best
                final_time = deadline - time.time() - 1
                if final_time > 1:
                    best_s, best_val = _local_search(best_s, rng, time_limit=final_time)
            break

    if best_s is None:
        best_s = np.ones(N)
        
    return best_s
