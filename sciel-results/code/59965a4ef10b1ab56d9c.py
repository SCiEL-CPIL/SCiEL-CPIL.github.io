"""Rewritten program for third autocorrelation inequality optimization."""
import numpy as np
import time
import scipy.optimize

N = 400
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds before deadline to stop

def _compute_autocorr(h):
    """Compute full autocorrelation array R[0..N-1] for h of length N."""
    n_fft = 1 << (2 * N - 1).bit_length()
    f = np.zeros(n_fft, dtype=np.float64)
    f[:N] = h
    F = np.fft.rfft(f)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    return acf

def _generate_msequence(L, seed=0):
    """Generate a length-L m-sequence using a linear feedback shift register."""
    np.random.seed(seed)
    # Use a primitive polynomial of degree 8 for good properties
    # x^8 + x^6 + x^5 + x^4 + 1 (0x141)
    poly = 0x141
    degree = 8
    # Initialize LFSR with random nonzero state
    state = np.random.randint(1, 2**degree)
    seq = []
    for _ in range(L):
        # Output bit is the LSB
        seq.append(1 if state & 1 else -1)
        # Feedback
        fb = (state ^ (state >> 1) ^ (state >> 2) ^ (state >> 3)) & 1
        state = (state >> 1) | (fb << (degree - 1))
    return np.array(seq, dtype=np.float64)

def _generate_msequence_truncated(N, seed=0):
    """Generate an m-sequence of length >= N, then truncate to N."""
    # Generate a longer m-sequence and take the first N samples
    L = N + 10  # slight over-generation for boundary flexibility
    seq = _generate_msequence(L, seed)
    return seq[:N].copy()

def _max_sidelobe(R):
    """Return max absolute value of R[1:]."""
    return np.max(np.abs(R[1:]))

def _generate_h_from_params(params, N, K):
    """
    Generate a signed sequence from sinusoidal parameters.
    params: array of length 2*K. [A1, phi1, A2, phi2, ...]
    x[n] = sum_{k=1}^K A_k * cos(2*pi*k*n/N + phi_k)
    h[n] = sign(x[n])
    """
    n = np.arange(N)
    x = np.zeros(N)
    for k in range(K):
        A = params[2*k]
        phi = params[2*k+1]
        x += A * np.cos(2 * np.pi * (k+1) * n / N + phi)
    
    # Handle zeros: if x[n] is 0, assign +1 or -1 arbitrarily (or based on previous)
    # sign(0) is 0 in numpy, so we map 0 to 1
    h = np.sign(x)
    h[h == 0] = 1.0
    return h

def _objective_sin_smooth(params, N, K, temp=1.0):
    """
    Objective function for sinusoidal parameter optimization using soft-sign.
    Returns a smooth surrogate for the max sidelobe (log-sum-exp of squared sidelobes).
    """
    n = np.arange(N)
    x = np.zeros(N)
    for k in range(K):
        A = params[2*k]
        phi = params[2*k+1]
        x += A * np.cos(2 * np.pi * (k+1) * n / N + phi)
    
    # Soft sign
    h_soft = np.tanh(x / temp)
    
    R = _compute_autocorr(h_soft)
    
    # Smooth max surrogate: log-sum-exp of squared sidelobes
    sidelobes = R[1:]**2
    max_val = np.max(sidelobes)
    if max_val > 0:
        lse = max_val + np.log(np.sum(np.exp(sidelobes - max_val)))
    else:
        lse = 0.0
    return lse

def _optimize_sinusoidal(N, K, seed, time_limit):
    """
    Optimize sinusoidal parameters using L-BFGS-B.
    """
    np.random.seed(seed)
    # Initial parameters: random amplitudes and phases
    A_init = np.random.uniform(0.5, 1.5, size=K)
    phi_init = np.random.uniform(0, 2*np.pi, size=K)
    params_init = np.concatenate([A_init, phi_init])
    
    start_time = time.time()
    def callback(xk):
        if time.time() - start_time > time_limit:
            return True
        return False

    try:
        res = scipy.optimize.minimize(
            _objective_sin_smooth, 
            params_init, 
            args=(N, K, 1.0), 
            method='L-BFGS-B', 
            options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6},
            callback=callback
        )
        best_params = res.x
        n = np.arange(N)
        x = np.zeros(N)
        for k in range(K):
            A = best_params[2*k]
            phi = best_params[2*k+1]
            x += A * np.cos(2 * np.pi * (k+1) * n / N + phi)
        best_h = np.sign(x)
        best_h[best_h == 0] = 1.0
        return best_h
    except Exception:
        return None

def construct():
    """
    Construct a signed step function with N intervals to minimize the
    upper bound on C3 (maximum autocorrelation sidelobe).
    
    Uses a hybrid approach:
    1. m-sequence initialization (near-ideal periodic autocorrelation).
    2. Sinusoidal parameter optimization for diverse starts.
    3. Aggressive multi-pass greedy local search with batch evaluation.
    """
    start_time = time.time()
    best_h = None
    best_val = np.inf
    
    def _fast_local_search(h, R, current_max, time_limit, max_passes=50):
        """
        Greedy local search with vectorized batch delta evaluation.
        Tries all flips in parallel, picks the best improving move per pass.
        """
        h = h.copy()
        R = R.copy()
        current_max = float(current_max)
        
        for _pass in range(max_passes):
            if time.time() >= time_limit:
                break
            
            # Compute delta for each possible flip position
            # Flipping h[i] changes R[j] for j where the autocorrelation involves h[i]
            # delta[i] = 2 * h[i] * (h[:i][::-1] + h[i+1:]) padded to length N
            # More precisely: R_new = R - 2*h[i]*[h[i-1], h[i-2], ..., h[0], 0, h[i+1], ..., h[N-1-i]]
            
            # Vectorized: for each i, the effect on R[1:] 
            # R[k] = sum_j h[j]*h[j+k], flipping h[i] changes R[k] by -2*h[i]*h[i+k] for k>0 where i+k<N,
            # and -2*h[i]*h[i-k] for k>0 where i-k>=0
            # So delta_R[i, k] = -2*h[i]*(h[i+k] if i+k<N else 0) - 2*h[i]*(h[i-k] if i-k>=0 else 0) for k=1..N-1
            
            # Build the delta matrix efficiently
            # For position i, the change in R[1:] is:
            #   for lag k: delta = -2*h[i] * (h[i+k] if i+k<N else 0) - 2*h[i] * (h[i-k] if i-k>=0 else 0)
            
            # Use FFT-based approach: flipping h[i] is equivalent to adding -2*h[i]*e_i to h
            # The change in autocorrelation is the cross-correlation of -2*h[i]*e_i with h
            # which is -2*h[i] * (shifted h)
            
            # Simpler: just compute new_R for the best candidate positions
            # But that's O(N^2) per pass. Let's use the analytical formula.
            
            # For each i, compute the new max sidelobe after flipping h[i]
            # new_R = R - 2*h[i]*d_i where d_i[k] = (h[i+k] if i+k<N else 0) + (h[i-k] if i-k>=0 else 0)
            
            # Precompute the "neighbor" vectors
            # d_i for k=1..N-1:
            #   right part: h[i+1], h[i+2], ..., h[N-1] (length N-1-i)
            #   left part: h[i-1], h[i-2], ..., h[0] (length i)
            #   d_i = [left_part, 0, right_part] but aligned to lag 1..N-1
            #   Actually: d_i[k] = h[i+k] (if i+k<N) + h[i-k] (if i-k>=0)
            
            # Vectorized computation of all deltas:
            # Create a matrix D of shape (N, N-1) where D[i, k-1] = h[i+k] + h[i-k] (with boundary handling)
            
            # For efficiency, use the fact that we only need to check if any flip improves
            # Let's do a simpler approach: iterate through positions but use numpy for the R update
            
            improved = False
            # Try each position, keep the best improvement
            best_i = -1
            best_new_max = current_max
            best_delta = None
            
            for i in range(N):
                if time.time() >= time_limit:
                    break
                
                # Compute delta vector for R[1:]
                # delta[k] = -2*h[i] * (h[i+k] if i+k<N else 0) - 2*h[i] * (h[i-k] if i-k>=0 else 0)
                
                # Right side: h[i+1], h[i+2], ..., h[N-1] -> contributes to lags 1, 2, ..., N-1-i
                # Left side: h[i-1], h[i-2], ..., h[0] -> contributes to lags 1, 2, ..., i
                # Combined: for lag k (1-indexed), delta_R[k] = -2*h[i] * (h[i+k] if i+k<N else 0) - 2*h[i] * (h[i-k] if i-k>=0 else 0)
                
                # Build the delta for R[1:]
                # R has indices 0..N-1, we care about R[1:] which has N-1 elements
                delta_R = np.zeros(N - 1)
                
                if i > 0:
                    # Left neighbors: h[i-1], h[i-2], ..., h[0]
                    # These contribute to lags 1, 2, ..., i
                    left = h[:i][::-1]  # h[i-1], h[i-2], ..., h[0]
                    delta_R[:i] -= 2.0 * h[i] * left
                
                if i < N - 1:
                    # Right neighbors: h[i+1], h[i+2], ..., h[N-1]
                    # These contribute to lags 1, 2, ..., N-1-i
                    right = h[i+1:]
                    delta_R[:len(right)] -= 2.0 * h[i] * right
                
                new_R_sidelobes = R[1:] - delta_R
                new_max = np.max(np.abs(new_R_sidelobes))
                
                if new_max < best_new_max - 1e-12:
                    best_new_max = new_max
                    best_i = i
                    best_delta = delta_R
            
            if best_i >= 0:
                h[best_i] *= -1.0
                R[1:] -= best_delta
                current_max = best_new_max
                improved = True
            else:
                break
        
        return h, R, current_max
    
    # Phase 1: m-sequence based initializations
    mseq_seeds = list(range(20))
    for seed in mseq_seeds:
        if time.time() >= DEADLINE - SAFETY_MARGIN:
            break
        
        h_candidate = _generate_msequence_truncated(N, seed)
        R = _compute_autocorr(h_candidate)
        current_max = _max_sidelobe(R)
        
        if current_max < best_val:
            best_val = current_max
            best_h = h_candidate.copy()
        
        # Local search polish
        remaining = DEADLINE - SAFETY_MARGIN - time.time()
        if remaining > 3.0:
            h_candidate, R, current_max = _fast_local_search(
                h_candidate, R, current_max, 
                time.time() + min(remaining - 1.0, 30.0)
            )
            if current_max < best_val:
                best_val = current_max
                best_h = h_candidate.copy()
    
    # Phase 2: Sinusoidal parameter optimization for diverse starts
    K_values = [5, 10, 15, 20, 25, 30]
    seed_counter = 100
    
    while time.time() < DEADLINE - SAFETY_MARGIN:
        seed_counter += 1
        K = K_values[seed_counter % len(K_values)]
        
        coarse_time_limit = 2.0
        if time.time() + coarse_time_limit + 5.0 > DEADLINE - SAFETY_MARGIN:
            break
            
        h_candidate = _optimize_sinusoidal(N, K, seed_counter, coarse_time_limit)
        
        if h_candidate is None:
            h_candidate = np.random.choice([-1.0, 1.0], size=N)
        
        R = _compute_autocorr(h_candidate)
        current_max = _max_sidelobe(R)
        
        if current_max < best_val:
            best_val = current_max
            best_h = h_candidate.copy()
            
        remaining_time = DEADLINE - SAFETY_MARGIN - time.time()
        if remaining_time > 3.0:
            h_candidate, R, current_max = _fast_local_search(
                h_candidate, R, current_max,
                time.time() + min(remaining_time - 1.0, 40.0)
            )
            if current_max < best_val:
                best_val = current_max
                best_h = h_candidate.copy()
    
    # Phase 3: Final polish on the best solution found
    if best_h is not None and time.time() < DEADLINE - SAFETY_MARGIN - 2.0:
        R_best = _compute_autocorr(best_h)
        best_val = _max_sidelobe(R_best)
        best_h, R_best, best_val = _fast_local_search(
            best_h, R_best, best_val,
            time.time() + (DEADLINE - SAFETY_MARGIN - 2.0 - time.time())
        )
    
    return best_h if best_h is not None else np.ones(N)
