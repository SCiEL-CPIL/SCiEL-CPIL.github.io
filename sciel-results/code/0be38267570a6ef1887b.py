import numpy as np
import time

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    def greedy_descent(s, A, current_val, time_limit):
        """Perform greedy coordinate descent until no improvement or time limit reached."""
        improved = True
        max_no_improve = 500
        no_improve = 0
        
        while improved and no_improve < max_no_improve and time.time() < time_limit:
            improved = False
            
            # Find indices where |A[k]| is large - these are the "problem" lags
            # Focus on the top few lags to guide flip selection
            abs_A = np.abs(A[1:])
            if len(abs_A) == 0:
                break
            top_k = np.argsort(abs_A)[-5:] + 1  # Top 5 lags (1-indexed in A)
            
            # Generate candidate flip positions
            # Prioritize positions that affect the top lags
            candidates = set()
            for k in top_k:
                # Flipping s[i] affects A[k] via s[i+k] and s[i-k]
                # So positions i where i+k or i-k is in [0, N) are relevant
                # More precisely, flipping s[i] changes A[k] by -2*s[i]*(s[i+k]+s[i-k])
                # We want to flip i where this change is large in magnitude
                # For a given k, the affected i values are those where the term is nonzero
                # Let's just pick random positions and evaluate
                pass
            
            # Strategy: try flipping each position in a smart order
            # Start with positions near the top lag indices
            # For lag k, flipping s[i] for i in [0, N-k) and [k, N) can affect A[k]
            
            # Simple approach: try all positions in random order, but prioritize
            # positions that are "central" (more likely to affect multiple lags)
            
            # Even simpler: just do random flips but with more iterations
            # Since each flip is O(N log N) ~ 0.1ms, we can do ~1000 flips in 0.1s
            # So in 1s we can do ~10000 flips, which is plenty
            
            # Let's do a full pass over all positions, trying each flip
            # If any flip improves, accept it and restart the pass
            # This is standard greedy coordinate descent
            
            for i in np.random.permutation(N):
                if time.time() > time_limit:
                    break
                
                old_si = s[i]
                s[i] = -old_si
                
                A_new = _calc_acf(s)
                val_new = _max_abs_acf_from_array(A_new)
                
                if val_new < current_val - 1e-15:
                    current_val = val_new
                    A = A_new
                    improved = True
                    no_improve = 0
                else:
                    s[i] = old_si
                    no_improve += 1
                    
                if no_improve > 200:
                    break
            
        return s, A, current_val
    
    # Multiple restarts
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        # Random initialization
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Greedy descent
        s, A, current_val = greedy_descent(s, A, current_val, deadline)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
        
        # Check if we're making progress
        if restart_count % 10 == 0:
            # Log progress (no-op, just for clarity)
            pass
    
    return best_seq
