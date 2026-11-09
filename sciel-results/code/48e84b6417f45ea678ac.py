"""Rewritten program for third autocorrelation inequality optimization."""
import numpy as np
import time

N = 400
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds before deadline to stop

def _compute_autocorr(h):
    """Compute full autocorrelation array R[0..N-1] for h of length N."""
    # Using FFT is standard, but for N=400, direct computation is also fast and avoids FFT overhead for single calls.
    # However, we will use FFT for initialization as it's robust.
    n_fft = 1 << (2 * N - 1).bit_length()
    f = np.zeros(n_fft, dtype=np.float64)
    f[:N] = h
    F = np.fft.rfft(f)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    return acf

def _max_sidelobe(R):
    """Return max absolute value of R[1:]."""
    return np.max(np.abs(R[1:]))

def construct():
    """
    Construct a signed step function with N intervals to minimize the
    upper bound on C3 (maximum autocorrelation sidelobe).
    
    Uses a fast incremental greedy search with random restarts.
    Returns an array of signed step heights.
    """
    start_time = time.time()
    best_h = None
    best_val = np.inf
    
    # Precompute indices for delta calculation
    # For a flip at i, delta_R[k] = -2 * h[i] * (h[i+k] if i+k < N else 0 + h[i-k] if i-k >= 0 else 0)
    # We can vectorize this.
    
    def optimize_local(h, R, current_max, time_limit):
        """
        Perform greedy local search starting from h, R, current_max.
        Returns improved h, R, current_max.
        """
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try flipping each bit. To avoid bias, we can shuffle indices or just iterate.
            # Iterating in order is fine for a first pass.
            for i in range(N):
                if time.time() > time_limit:
                    return h, R, current_max
                
                # Calculate change in autocorrelation if h[i] is flipped
                # delta_R[k] = -2 * h[i] * ( h[i+k] * (i+k < N) + h[i-k] * (i-k >= 0) )
                
                # Vectorized calculation of the term in parentheses
                # Let term[k] = h[i+k] if i+k < N else 0 + h[i-k] if i-k >= 0 else 0
                # We can create an array of zeros and fill the valid parts.
                
                # Actually, constructing the vector every time might be slow.
                # Let's precompute the structure of the update.
                # The update depends on i.
                
                # Efficient way:
                # Left part: h[i-k] for k=0..i -> indices i-0, i-1, ..., i-i=0. Reversed?
                # R[k] = sum h[j]h[j+k].
                # Term 1: h[i] is the second factor. j+k=i => j=i-k. Valid for k=0..i.
                #   Values: h[i], h[i-1], ..., h[0].
                # Term 2: h[i] is the first factor. j=i. h[i]h[i+k]. Valid for k=0..N-1-i.
                #   Values: h[i], h[i+1], ..., h[N-1].
                
                # So for a given i:
                # term[0..i] gets contribution from h[i-k] (which is h[i], h[i-1]...)
                # term[0..N-1-i] gets contribution from h[i+k] (which is h[i], h[i+1]...)
                
                # Note: at k=0, both terms contribute h[i]*h[i] = 1? 
                # Wait, R(0) = sum h[j]^2 = N.
                # If we flip h[i], R(0) changes by -2 h[i] h[i] = -2?
                # Let's check the formula: Delta R(k) = -2 h[i] (h[i-k] + h[i+k])
                # At k=0: Delta R(0) = -2 h[i] (h[i] + h[i]) = -4 h[i]^2 = -4.
                # Original R(0) = N. New R(0) = N - 4?
                # Let's verify.
                # R(0) = sum h[j]^2. If we flip h[i], h[i]^2 is still 1. So R(0) should be unchanged!
                # Where is the error?
                # The sum is over j.
                # R(k) = sum_{j} h[j] h[j+k].
                # If we flip h[i], the terms involving h[i] are:
                # 1. j=i: term is h[i] h[i+k]. Change: -2 h[i] h[i+k].
                # 2. j=i-k: term is h[i-k] h[i]. Change: -2 h[i-k] h[i].
                # For k=0:
                # 1. j=i: h[i]h[i]. Change -2 h[i]^2 = -2.
                # 2. j=i: h[i]h[i]. Change -2 h[i]^2 = -2.
                # Total change -4.
                # But h[i]^2 is 1. Sum of squares is invariant under sign flip.
                # Ah, the sum for k=0 is sum_{j=0}^{N-1} h[j]h[j].
                # The term j=i is h[i]h[i].
                # In the decomposition:
                # "j=i" corresponds to the first factor being h[i].
                # "j=i-k" corresponds to the second factor being h[i].
                # For k=0, these are the SAME term in the sum.
                # So we are double counting the change for k=0?
                # No, the sum is unique for each j.
                # For k=0, j runs 0 to N-1.
                # The term is h[j]h[j].
                # If j=i, the term is h[i]h[i].
                # My decomposition:
                # Term where h[i] is the "first" element (index j): j=i. Contributes h[i]h[i+k].
                # Term where h[i] is the "second" element (index j+k): j+k=i => j=i-k. Contributes h[i-k]h[i].
                # For k=0, j=i and j=i-k=i are the same index.
                # So the total change is indeed -2 h[i]h[i] - 2 h[i]h[i] = -4?
                # Let's check: R_new(0) = R_old(0) - 2 h[i] h[i] (from j=i) - 2 h[i] h[i] (from j=i-k)?
                # No.
                # Let R = sum h[j]h[j+k].
                # R' = R - 2 h[i] h[i+k] (removing old, adding new -h[i]h[i+k] => diff -2 h[i]h[i+k])
                # AND R' = R - 2 h[i-k] h[i] (removing old, adding new h[i-k](-h[i]) => diff -2 h[i-k]h[i]).
                # For k=0, these are the same term h[i]h[i].
                # So the total change is -2 h[i]^2 - 2 h[i]^2 = -4.
                # But physically, R(0) = sum h[j]^2 = N. It should not change.
                # Why?
                # Because for k=0, the term h[i]h[i] appears ONLY ONCE in the sum.
                # My decomposition splits the sum into "terms where i is the first index" and "terms where i is the second index".
                # For k=0, the set of indices j is 0..N-1.
                # The term for j=i is h[i]h[i].
                # It is counted in "i is first index" (j=i).
                # It is ALSO counted in "i is second index" (j=i-k=i).
                # So yes, it is counted twice in my delta formula?
                # No, the sum R(k) is a single sum.
                # The change in the sum is the sum of changes of individual terms.
                # The term at j=i changes from h[i]h[i+k] to -h[i]h[i+k]. Delta: -2 h[i]h[i+k].
                # The term at j=i-k changes from h[i-k]h[i] to h[i-k](-h[i]). Delta: -2 h[i-k]h[i].
                # If k=0, j=i and j=i-k are the SAME term.
                # So the total change is -2 h[i]h[i] - 2 h[i]h[i] = -4?
                # Let's test with N=1, h=[1]. R(0)=1. Flip to -1. R(0)=1. Change 0.
                # Formula: -2(1)(1) - 2(1)(1) = -4. Incorrect.
                
                # The issue is that for k=0, the "j=i" and "j=i-k" refer to the same element in the summation index j.
                # We must not double count.
                # For k > 0, j=i and j=i-k are distinct.
                # For k = 0, they are the same.
                
                # So, Delta R(0) = 0.
                # For k > 0:
                # If i+k < N and i-k >= 0: Delta R(k) = -2 h[i] (h[i+k] + h[i-k]).
                # If i+k >= N and i-k >= 0: Delta R(k) = -2 h[i] h[i-k].
                # If i+k < N and i-k < 0: Delta R(k) = -2 h[i] h[i+k].
                
                # We only care about k=1..N-1 for the metric.
                # So we can ignore k=0 entirely for the max calculation, as long as we don't use R(0) in the max.
                
                # Let's construct the delta vector for k=1..N-1.
                # We will create an array `delta` of length N.
                # delta[0] = 0.
                
                # Left part (i-k >= 0 => k <= i):
                # k ranges 1 to i.
                # delta[k] += -2 * h[i] * h[i-k]
                
                # Right part (i+k < N => k < N-i):
                # k ranges 1 to N-1-i.
                # delta[k] += -2 * h[i] * h[i+k]
                
                # Implementation:
                delta = np.zeros(N)
                
                # Left contribution
                if i > 0:
                    # k=1..i
                    # h[i-1], h[i-2], ..., h[0]
                    # We need h[i-k] for k=1..i.
                    # This is h[i-1], h[i-2], ..., h[0].
                    # Slicing h[:i][::-1] gives h[i-1]...h[0].
                    # We want to add -2 h[i] * this vector to delta[1:i+1].
                    delta[1:i+1] -= 2.0 * h[i] * h[:i][::-1]
                
                # Right contribution
                if i < N - 1:
                    # k=1..N-1-i
                    # h[i+1], h[i+2], ..., h[N-1]
                    # Slicing h[i+1:] gives h[i+1]...h[N-1].
                    # We want to add -2 h[i] * this vector to delta[1:N-i].
                    delta[1:N-i] -= 2.0 * h[i] * h[i+1:]
                
                new_R = R - delta
                new_max = _max_sidelobe(new_R)
                
                if new_max < current_max - 1e-12:
                    h[i] *= -1.0
                    R = new_R
                    current_max = new_max
                    improved = True
                    # If we find an improvement, we might want to restart the loop or continue?
                    # Standard greedy: continue scanning.
                    
        return h, R, current_max

    # Main loop: Multiple restarts
    n_restarts = 0
    while time.time() < DEADLINE - SAFETY_MARGIN:
        n_restarts += 1
        np.random.seed(n_restarts)
        h = np.random.choice([-1.0, 1.0], size=N)
        
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
            
        # If we are stuck at a high value, we might try a "kick" before the next restart?
        # The restart itself is a kick.
        
        # Check time
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
            
    return best_h if best_h is not None else np.ones(N)
