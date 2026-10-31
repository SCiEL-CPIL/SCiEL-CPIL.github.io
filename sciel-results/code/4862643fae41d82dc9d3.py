import numpy as np
import time
import math

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    rng = np.random.default_rng(42)
    
    # Best solution found so far
    best_f = None
    best_psl = np.inf
    
    # We will run multiple restarts
    # Since N=400 is small, we can do many restarts or one long SA.
    # Let's do a few restarts with SA.
    
    max_restarts = 10
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize random sequence
        f = rng.choice([-1, 1], size=N).astype(np.int32)
        
        # Compute initial autocorrelation R
        # R[k] = sum f[j] f[j+k]
        # Using FFT for initialization is fast enough
        nfft = 1
        while nfft < 2 * N:
            nfft *= 2
        F = np.fft.rfft(f, nfft)
        acf_full = np.fft.irfft(F * np.conj(F), nfft)
        # acf_full[0:N] are the lags 0..N-1
        R = acf_full[:N].astype(np.int64)
        
        current_psl = np.max(np.abs(R[1:]))
        
        if current_psl < best_psl:
            best_psl = current_psl
            best_f = f.copy()
            
        # Simulated Annealing
        T = 1.0  # Start temperature high enough to accept bad moves
        # Scale T relative to typical PSL changes? 
        # PSL is integer. Changes are small.
        # Let's use a cooling schedule.
        cooling = 0.9999
        min_T = 1e-6
        
        # Pre-compute indices for potential flips? No, random is fine.
        
        while T > min_T and time.time() < deadline - 5.0:
            # Pick random index
            i = rng.integers(0, N)
            
            # Calculate delta R
            # Delta R[k] = -2 * f[i] * ( f[i+k] + f[i-k] )
            # We need to update R for all k > 0
            
            # To avoid allocating new array every time, we can compute the change
            # and apply it to a copy or in-place with revert.
            
            # Efficient way:
            # 1. Save old R values? No, R is large.
            # 2. Compute new R = R + Delta
            # 3. Check max
            # 4. Revert if rejected
            
            # Let's compute the vector of changes
            # k ranges from 1 to N-1
            # f[i+k] is valid if i+k < N
            # f[i-k] is valid if i-k >= 0
            
            # Create indices
            # This might be slow if done in Python loops.
            # Vectorized approach:
            
            # Right side: k such that i+k < N => k < N-i
            # Left side: k such that i-k >= 0 => k <= i
            
            # Let's just create a zero array and fill it
            delta_R = np.zeros(N, dtype=np.int64)
            
            # Right part: k in [1, N-i-1]
            # delta_R[k] += -2 * f[i] * f[i+k]
            if i < N - 1:
                # k=1..N-i-1
                # f[i+1 : N]
                delta_R[1:N-i] += -2 * f[i] * f[i+1:N]
                
            # Left part: k in [1, i]
            # delta_R[k] += -2 * f[i] * f[i-k]
            if i > 0:
                # k=1..i
                # f[i-1 : 0 : -1] is f[i-1], f[i-2]... f[0]
                # We need to map k=1 -> f[i-1], k=2 -> f[i-2]
                # So delta_R[1:i+1] += -2 * f[i] * f[i-1::-1]
                delta_R[1:i+1] += -2 * f[i] * f[i-1::-1]
            
            # Apply delta
            new_R = R + delta_R
            
            # Calculate new PSL
            new_psl = np.max(np.abs(new_R[1:]))
            
            # Acceptance criterion
            if new_psl < current_psl:
                # Always accept improvement
                R = new_R
                f[i] *= -1
                current_psl = new_psl
                
                if current_psl < best_psl:
                    best_psl = current_psl
                    best_f = f.copy()
            else:
                # Simulated Annealing: accept with probability exp(-delta/T)
                delta = new_psl - current_psl
                if delta < 0:
                    # Should not happen if we checked <
                    pass
                else:
                    # delta >= 0
                    if rng.random() < math.exp(-delta / T):
                        R = new_R
                        f[i] *= -1
                        current_psl = new_psl
                        if current_psl < best_psl:
                            best_psl = current_psl
                            best_f = f.copy()
                    # else reject, R and f unchanged
            
            T *= cooling
            
            # Optional: if we hit a very low PSL, reduce T faster?
            # Or just keep cooling.
            
    return best_f

# EVOLVE-BLOCK-END
