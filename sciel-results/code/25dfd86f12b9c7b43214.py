import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_s = None
    best_score = float('inf')
    
    # Try a few random restarts or structured starts
    for attempt in range(10):
        if time.time() > deadline:
            break
            
        # Initialize
        # Use a mix of random and structured
        if attempt == 0:
            # Greedy-like: start with all 1s, then randomize a bit?
            # Or just random
            s = np.random.choice([-1, 1], N)
        else:
            s = np.random.choice([-1, 1], N)
            
        # Compute initial autocorrelation
        # np.correlate is fast for N=400
        R = np.correlate(s, s, mode='full')
        # R has length 2*N - 1. Peak is at N-1.
        # Off-peak is R[:N-1] and R[N:]
        # We care about max absolute value of off-peak
        # Symmetric, so just look at R[N:] (lags 1 to N-1)
        
        def get_score(R):
            # R is full correlation
            # Peak is R[N-1]
            # Sidelobes are R[0:N-1] and R[N:2*N-1]
            # Due to symmetry, R[k] == R[2N-2-k]
            # We can just take R[N:] which corresponds to lags 1..N-1
            # Wait, np.correlate(s, s, 'full')
            # Index 0: lag -(N-1)
            # Index N-1: lag 0
            # Index 2N-2: lag N-1
            # So lags 1..N-1 are at indices N..2N-2
            # Lags -1..-(N-1) are at indices 0..N-2
            # We want max abs of these.
            # Since R is real and symmetric, max(abs(R[0:N-1]), abs(R[N:]))
            # R[N:] has length N-1.
            return np.max(np.abs(R[N:]))
            
        current_score = get_score(R)
        
        # Local Search Loop
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. L2 Minimization Phase (Flat Spectrum)
            # Objective: minimize sum(R[k]^2) for k != 0
            # We can approximate by just checking if a flip reduces the L2 norm
            # Incremental update for L2 is complex, so we might skip this or do a simple version
            # Let's stick to L-inf (Peak) minimization as it directly targets the score.
            # But L-inf is non-smooth. 
            # Strategy: Flip bit if it reduces the current max sidelobe.
            
            # To escape local minima, we can also consider reducing the 2nd largest, etc.
            # For now, simple greedy on Peak.
            
            # We need to track the current max and its location to update efficiently?
            # Or just recompute max after each potential flip?
            # Recomputing max is O(N).
            # Updating R for a flip is O(N).
            # So O(N) per flip check.
            # N flips per pass -> O(N^2) per pass.
            
            for i in range(N):
                if time.time() > deadline:
                    return best_s if best_s is not None else s
                
                # Calculate delta R if s[i] is flipped
                # s[i] -> -s[i]
                # Change in R[k]:
                # R[k] = sum_{j} s[j] s[j+k]
                # Terms involving s[i]:
                # 1. j=i, term s[i] s[i+k]. Valid if i+k < N.
                # 2. j+k=i => j=i-k, term s[i-k] s[i]. Valid if i-k >= 0.
                
                # We update R array in place
                # Save old R values? No, we can revert if worse.
                
                # Determine affected k indices
                # k ranges from 1 to N-1
                # For a specific i:
                # Affected k are those where i is in [0, N-1-k] OR i-k is in [0, N-1-k]?
                # Actually, simpler:
                # The term s[i] s[i+k] exists for k in 1..N-1-i
                # The term s[i-k] s[i] exists for k in 1..i
                # So affected k are 1..max(i, N-1-i)? No, union of [1, N-1-i] and [1, i].
                # So k in 1..max(i, N-1-i).
                
                max_k = max(i, N - 1 - i)
                
                # We need to update R for k in 1..max_k
                # Note: R array from np.correlate is indexed by lag offset.
                # Index N-1 is lag 0.
                # Index N-1+k is lag k.
                # Index N-1-k is lag -k.
                
                # We store R_full.
                # Let's just update the relevant parts.
                
                # To avoid complex bookkeeping, we can just compute the new R
                # for the affected k's, compare the new max to old max.
                
                # Optimization: 
                # 1. Find current max M and index k_max.
                # 2. For each i, check if flipping i reduces M.
                
                # Let's do a simpler O(N^2) pass:
                # For each i, flip s[i], compute new score, if better keep, else revert.
                # Computing new score from scratch is O(N) if we update R incrementally, 
                # but finding max is O(N).
                # So O(N) per i. Total O(N^2) per pass.
                
                # Let's implement incremental R update.
                
                # Store original s[i]
                old_s_i = s[i]
                
                # Update R
                # We need to modify R for k=1..max_k
                # R[N-1+k] and R[N-1-k]
                
                # Let's create a list of (index, delta) to apply
                # Or just apply to R, check, revert.
                
                # Apply flip to s
                s[i] = -old_s_i
                
                # Update R
                # For k in 1..max_k:
                #   delta_R_k = -2 * old_s_i * (s[i+k] if i+k<N else 0 + s[i-k] if i-k>=0 else 0)
                #   R[N-1+k] += delta_R_k
                #   R[N-1-k] += delta_R_k
                
                # This is a Python loop, might be slow.
                # N=400. Inner loop up to 400.
                # 400 * 400 = 160,000 iterations.
                # This is fine for 1 pass.
                
                deltas = np.zeros(2*N-1)
                # Vectorized update?
                # k indices: 1..max_k
                ks = np.arange(1, max_k + 1)
                
                # Term 1: s[i] s[i+k] -> -s[i] s[i+k] => delta -2 s[i] s[i+k]
                # Valid for i+k < N => k < N-i
                mask1 = ks < (N - i)
                if np.any(mask1):
                    k1 = ks[mask1]
                    # s[i+k1]
                    vals1 = s[i + k1]
                    delta1 = -2 * old_s_i * vals1
                    # Apply to R[N-1+k1]
                    R[N-1+k1] += delta1
                    # Apply to R[N-1-k1]
                    R[N-1-k1] += delta1
                    
                # Term 2: s[i-k] s[i] -> s[i-k] (-s[i]) => delta -2 s[i-k] s[i]
                # Valid for i-k >= 0 => k <= i
                mask2 = ks <= i
                if np.any(mask2):
                    k2 = ks[mask2]
                    # s[i-k2]
                    vals2 = s[i - k2]
                    delta2 = -2 * vals2 * old_s_i
                    # Apply to R[N-1+k2]
                    R[N-1+k2] += delta2
                    # Apply to R[N-1-k2]
                    R[N-1-k2] += delta2
                    
                new_score = get_score(R)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    # Keep the flip
                else:
                    # Revert
                    s[i] = old_s_i
                    # Revert R
                    # Subtract the same deltas
                    if np.any(mask1):
                        R[N-1+k1] -= delta1
                        R[N-1-k1] -= delta1
                    if np.any(mask2):
                        R[N-1+k2] -= delta2
                        R[N-1-k2] -= delta2
                    
        # End local search
        if current_score < best_score:
            best_score = current_score
            best_s = s.copy()
            
    # Return the best found
    if best_s is not None:
        return best_s
    else:
        return s
