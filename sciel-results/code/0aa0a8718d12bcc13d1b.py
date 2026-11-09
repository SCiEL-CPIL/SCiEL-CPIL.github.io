"""Rewritten program for third autocorrelation inequality optimization."""
import numpy as np
import time

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

def _max_sidelobe(R):
    """Return max absolute value of R[1:]."""
    return np.max(np.abs(R[1:]))

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    t = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        t[i] = t[i >> 1] ^ (i & 1)
    return t

def construct():
    """
    Construct a signed step function with N intervals to minimize the
    upper bound on C3 (maximum autocorrelation sidelobe).
    
    Uses a fast incremental greedy search with random restarts and structured initialization.
    Returns an array of signed step heights.
    """
    start_time = time.time()
    best_h = None
    best_val = np.inf
    
    def optimize_local(h, R, current_max, time_limit):
        """
        Perform greedy local search starting from h, R, current_max.
        Returns improved h, R, current_max.
        """
        improved = True
        # Use a fixed number of passes to avoid infinite loops if stuck, but check time
        max_passes = 50
        pass_count = 0
        
        while improved and time.time() < time_limit and pass_count < max_passes:
            improved = False
            pass_count += 1
            
            # Shuffle indices to avoid bias towards early indices
            indices = np.random.permutation(N)
            
            for i in indices:
                if time.time() > time_limit:
                    return h, R, current_max
                
                # Calculate change in autocorrelation if h[i] is flipped
                # Delta R(k) = -2 * h[i] * (h[i-k] + h[i+k]) for valid indices
                # We only care about k=1..N-1
                
                delta = np.zeros(N)
                
                # Left contribution: k=1..i
                # delta[k] += -2 * h[i] * h[i-k]
                if i > 0:
                    # k=1 corresponds to h[i-1], k=i corresponds to h[0]
                    # Vector: h[i-1], h[i-2], ..., h[0]
                    # Slice: h[:i][::-1]
                    delta[1:i+1] -= 2.0 * h[i] * h[:i][::-1]
                
                # Right contribution: k=1..N-1-i
                # delta[k] += -2 * h[i] * h[i+k]
                if i < N - 1:
                    # k=1 corresponds to h[i+1], k=N-1-i corresponds to h[N-1]
                    # Vector: h[i+1], h[i+2], ..., h[N-1]
                    # Slice: h[i+1:]
                    delta[1:N-i] -= 2.0 * h[i] * h[i+1:]
                
                new_R = R - delta
                new_max = _max_sidelobe(new_R)
                
                if new_max < current_max - 1e-12:
                    h[i] *= -1.0
                    R = new_R
                    current_max = new_max
                    improved = True
                    
        return h, R, current_max

    def generate_initial_h(seed):
        """Generate diverse initial configurations."""
        rng = np.random.RandomState(seed)
        
        # 1. Random
        if seed % 3 == 0:
            return rng.choice([-1.0, 1.0], size=N)
        
        # 2. Thue-Morse (structured)
        elif seed % 3 == 1:
            t = _thue_morse(N)
            h = 1.0 - 2.0 * t.astype(np.float64)
            # Add slight randomness to break perfect symmetry?
            # Or just use it as is
            return h
        
        # 3. Block structure (alternating blocks of random length)
        else:
            h = np.zeros(N)
            pos = 0
            while pos < N:
                # Random block length 1-10
                block_len = rng.randint(1, 11)
                sign = 1.0 if rng.rand() > 0.5 else -1.0
                h[pos:pos+block_len] = sign
                pos += block_len
            return h

    # Main loop: Multiple restarts
    n_restarts = 0
    while time.time() < DEADLINE - SAFETY_MARGIN:
        n_restarts += 1
        seed = n_restarts
        
        # Generate initial h
        h = generate_initial_h(seed)
        
        # Initial R
        R = _compute_autocorr(h)
        current_max = _max_sidelobe(R)
        
        if current_max < best_val:
            best_val = current_max
            best_h = h.copy()
        
        # Optimize
        h, R, current_max = optimize_local(h, R, current_max, DEADLINE - SAFETY_MARGIN)
        
        if current_max < best_val:
            best_val = current_max
            best_h = h.copy()
            
        # Check time
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
            
    # Final polish: Run local search on the best solution found with more time budget if available
    if best_h is not None:
        # Ensure we have a bit of time left
        remaining = DEADLINE - SAFETY_MARGIN - time.time()
        if remaining > 5.0:
            R = _compute_autocorr(best_h)
            current_max = _max_sidelobe(R)
            best_h, R, current_max = optimize_local(best_h, R, current_max, DEADLINE - 1.0)
            if current_max < best_val:
                best_val = current_max
        
    return best_h if best_h is not None else np.ones(N)
