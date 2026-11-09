import numpy as np
import time

N = 400

def construct():
    """
    Constructs a signed step function (sequence of +/- 1) of length N
    that minimizes the peak sidelobe level of its autocorrelation.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_psl = np.inf
    best_h = None
    
    # Use a fixed number of restarts, but also stop if time runs out
    # Since each restart is fast, we can do many.
    max_restarts = 1000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize with random +/- 1 sequence
        h = np.random.choice([-1, 1], size=N)
        
        # Local Search: Greedy Flipping
        # We will keep flipping the index that most reduces the PSL
        # until no single flip improves the PSL.
        
        # To speed up, we can use a heuristic: 
        # Instead of checking all N flips from scratch each time (O(N^2 log N) per pass),
        # we can just do a few passes. 
        # For N=400, a full pass is very fast.
        
        current_psl = _compute_psl(h)
        
        # If this random start is already good, great.
        # Otherwise, optimize.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_flip_idx = -1
            best_new_psl = current_psl
            
            # Check all possible single flips
            # Optimization: We can break early if we find a flip that reduces PSL significantly?
            # No, we need the best one for greedy.
            # But N=400 is small.
            
            # To make it faster, we can sample a subset of indices if time is tight,
            # but with 140s budget, full passes are fine.
            
            for i in range(N):
                # Flip h[i]
                h[i] *= -1
                new_psl = _compute_psl(h)
                h[i] *= -1  # Flip back
                
                if new_psl < best_new_psl:
                    best_new_psl = new_psl
                    best_flip_idx = i
            
            if best_flip_idx != -1 and best_new_psl < current_psl:
                h[best_flip_idx] *= -1
                current_psl = best_new_psl
                improved = True
        
        # After local search, update global best
        if current_psl < best_psl:
            best_psl = current_psl
            best_h = h.copy()
            
            # If we get a very low PSL, we might stop early?
            # Theoretical minimum for random-like sequences is around sqrt(N) or lower.
            # For N=400, sqrt(400)=20. If we get below 10, it's very good.
            # Let's not stop early to be safe, but the loop will break on time.

    if best_h is None:
        # Fallback: return a random sequence or all ones?
        # All ones is bad. Random is better.
        best_h = np.random.choice([-1, 1], size=N)
        
    return best_h

def _compute_psl(h):
    """
    Computes the Peak Sidelobe Level (PSL) of the aperiodic autocorrelation of h.
    PSL = max_{tau != 0} |R(tau)|
    """
    # Use FFT for efficient autocorrelation
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.fft(h, n=n_fft)
    R = np.real(np.fft.ifft(H * np.conj(H)))
    # The autocorrelation is circular, but for zero-padded input, 
    # the first N values (excluding the wrap-around if any) correspond to aperiodic.
    # Actually, np.fft.ifft(H * conj(H)) gives the circular autocorrelation.
    # With zero padding to >= 2N-1, the circular autocorrelation matches the aperiodic one for lags 0..N-1.
    # The aperiodic autocorrelation R(tau) = sum_{i=0}^{N-1-tau} h[i] h[i+tau] for tau >= 0.
    # The FFT method gives R[0] = sum h^2 = N.
    # R[1] corresponds to tau=1, etc.
    # We need to check lags 1 to N-1.
    
    # The output of ifft is length n_fft.
    # The first N elements are the autocorrelation for lags 0, 1, ..., N-1.
    # Note: The correlation is symmetric, so we only need to look at positive lags.
    
    # Extract lags 1 to N-1
    sidelobes = R[1:N]
    
    # Find the maximum absolute value
    max_sidelobe = np.max(np.abs(sidelobes))
    
    return max_sidelobe
