import time
import random
import numpy as np
from scipy.fft import fft, ifft

def construct():
    start_time = time.time()
    deadline = start_time + 115
    N = 1000
    
    # Precompute FFT sizes for convolution
    # We need to compute counts of sums a+b.
    # S is a boolean vector of length N+1 (indices 0..N).
    # We want C = S * S (convolution).
    # C[k] = sum_{i+j=k} S[i]S[j].
    # We need C[s] <= 2 for all s.
    # Note: C[s] counts pairs (i,j) with i+j=s. 
    # The problem asks for number of representations a+b=s with a<=b.
    # Let R(s) be the count of pairs a<=b.
    # C[s] = sum_{a+b=s} S[a]S[b].
    # If s is even, C[s] includes the term a=b=s/2.
    # R(s) = (C[s] + (S[s/2] if s even else 0)) / 2.
    # We require R(s) <= 2.
    # This is equivalent to C[s] <= 4 generally, but stricter for even s if S[s/2]=1.
    # Specifically:
    # If s even and S[s/2]=1: R(s) = (C[s]+1)/2 <= 2 => C[s] <= 3.
    # If s even and S[s/2]=0: R(s) = C[s]/2 <= 2 => C[s] <= 4.
    # If s odd: R(s) = C[s]/2 <= 2 => C[s] <= 4.
    
    # To make it simple and robust, we can just enforce C[s] <= 4 and handle the even case carefully 
    # or just enforce the exact condition R(s) <= 2 in the validation step.
    # However, for the annealing objective, we need a fast penalty.
    # Let's define Penalty = sum_{s} max(0, R(s) - 2)^2.
    # Calculating R(s) from C(s) is cheap.
    
    # FFT size
    # Max sum is 2N. Length of convolution is 2N+1.
    # Next power of 2 >= 2N+1.
    L = 1
    while L < 2 * N + 1:
        L *= 2
    
    def get_counts_from_mask(mask):
        """Compute representation counts R(s) for all s using FFT."""
        # Pad mask to length L
        x = np.zeros(L)
        x[1:N+1] = mask[1:N+1]
        
        # FFT
        X = fft(x)
        C = ifft(X * X)
        C = np.real(C).astype(int)
        
        # Extract relevant sums 2..2N
        # C[k] corresponds to sum k.
        # C[0] is 0 (since mask[0]=0).
        # C[1] is 0.
        # Sums range from 2 to 2N.
        
        R = np.zeros(2*N + 1, dtype=np.int32)
        
        for s in range(2, 2*N + 1):
            c_s = C[s]
            if s % 2 == 0:
                # a = b = s/2
                if 1 <= s//2 <= N:
                    if mask[s//2]:
                        r_s = (c_s + 1) // 2
                    else:
                        r_s = c_s // 2
                else:
                    r_s = 0 # Should not happen for valid s range
            else:
                r_s = c_s // 2
            R[s] = r_s
            
        return R

    def calculate_penalty(R):
        """Calculate penalty based on violations of R[s] <= 2."""
        # We want to minimize sum of (R[s] - 2)^2 for R[s] > 2
        # And maybe reward for being close to 2? No, just minimize violations.
        # A quadratic penalty helps gradient-like moves.
        viol = np.maximum(0, R - 2)
        return np.sum(viol * viol)

    def is_valid(R):
        return np.all(R <= 2)

    def size_of_mask(mask):
        return np.sum(mask)

    # Initialize with a greedy solution to have a good starting point
    def greedy_init():
        mask = np.zeros(N + 1, dtype=bool)
        counts = np.zeros(2 * N + 1, dtype=np.int32)
        for x in range(1, N + 1):
            present = np.where(mask)[0]
            present = present[present >= 1]
            would_exceed = False
            if len(present) > 0:
                sums = x + present
                if np.any(counts[sums] >= 2): # Wait, counts here are R(s)?
                    # In greedy, we maintain R(s) directly?
                    # Let's stick to the previous greedy logic which was correct for R(s) <= 2
                    # The previous code used 'counts' as R(s).
                    # Let's re-verify: 
                    # If we add x, new pairs (x,y) for y in S.
                    # Sum s = x+y.
                    # If y < x, pair is (y,x). If y >= x, pair is (x,y).
                    # In both cases, it's one new representation for sum s.
                    # So R[x+y] += 1.
                    # We need R[s] <= 2.
                    # So if any R[x+y] >= 2, adding x would make it 3.
                    pass
            # Let's just use the simple check:
            # Compute R for current mask? Too slow for greedy.
            # Use incremental R update.
            
            # Re-implementing incremental R update for greedy
            # R[s] = count of pairs a<=b in S with a+b=s.
            # Adding x:
            # For each y in S:
            #   If y < x: pair (y,x) added to sum x+y.
            #   If y >= x: pair (x,y) added to sum x+y.
            #   So R[x+y] += 1 for all y in S.
            #   Also, if x in S? No, x is new.
            #   Wait, what about pair (x,x)?
            #   If we add x, is (x,x) a new pair?
            #   Yes, if x is in the set.
            #   But the loop "for y in S" doesn't include x if x is not in S yet.
            #   So we must manually add R[2x] += 1 for the pair (x,x).
            #   And for y in S, R[x+y] += 1.
            
            # Let's check if adding x violates constraint.
            # Check R[2x] + 1 <= 2 => R[2x] <= 1.
            # Check R[x+y] + 1 <= 2 => R[x+y] <= 1 for all y in S.
            
            if counts[2*x] > 1:
                continue
            if len(present) > 0:
                sums = x + present
                if np.any(counts[sums] > 1):
                    continue
            
            # Commit
            if len(present) > 0:
                counts[x + present] += 1
            counts[2*x] += 1
            mask[x] = True
            
        return mask, counts

    mask_greedy, counts_greedy = greedy_init()
    best_mask = mask_greedy.copy()
    best_size = size_of(best_mask)
    best_R = get_counts_from_mask(best_mask)
    best_penalty = calculate_penalty(best_R)
    
    # Simulated Annealing
    # Objective: Maximize Size - Lambda * Penalty
    # Or just minimize Penalty, and track Size.
    # Better: Minimize (Penalty * 1000 - Size) or similar.
    # Let's try: Cost = Penalty * 10 - Size.
    # We want to minimize Cost.
    
    def cost(mask):
        R = get_counts_from_mask(mask)
        p = calculate_penalty(R)
        s = size_of(mask)
        return p * 10 - s, R, p, s

    # Initial state
    current_mask = mask_greedy.copy()
    current_R = get_counts_from_mask(current_mask)
    current_penalty = calculate_penalty(current_R)
    current_size = size_of(current_mask)
    current_cost = current_penalty * 10 - current_size
    
    T = 50.0
    T_min = 0.1
    alpha = 0.995
    rng = random.Random(42)
    
    while time.time() < deadline - 5:
        if T < T_min:
            break
            
        # Generate neighbor
        # Random flip
        x = rng.randint(1, N)
        
        # Calculate new state
        # Instead of full FFT every step, can we update?
        # FFT update for a single bit flip is complex (requires updating spectrum).
        # Given N=1000, L=2048. FFT is ~2048*log(2048) ~ 20k ops.
        # Python overhead might be high.
        # Let's measure if full FFT is fast enough.
        # 1000 iterations/sec? 115s -> 100k iterations.
        # If FFT takes 1ms, that's 100ms per 100 iters.
        # Let's try.
        
        new_mask = current_mask.copy()
        new_mask[x] = not new_mask[x]
        
        new_R = get_counts_from_mask(new_mask)
        new_penalty = calculate_penalty(new_R)
        new_size = size_of(new_mask)
        new_cost = new_penalty * 10 - new_size
        
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or (T > 0 and rng.random() < np.exp(-delta_cost / T)):
            current_mask = new_mask
            current_R = new_R
            current_penalty = new_penalty
            current_size = new_size
            current_cost = new_cost
            
            if current_penalty == 0 and current_size > best_size:
                best_mask = current_mask.copy()
                best_size = current_size
                best_R = current_R
                best_penalty = 0
        
        T *= alpha
        
    # Final polish: Local search on best_mask to fix any small issues or add more
    # Since SA might end with penalty > 0, we should ensure we return a valid set.
    # If best_mask has penalty > 0, we need to repair it.
    # The greedy init was valid. If SA improved size but broke validity, best_size might be larger but invalid.
    # We only update best_mask if current_penalty == 0.
    # So best_mask is always valid.
    
    # Try to improve best_mask with a quick local search (add/swap)
    # Using the valid best_mask as start
    mask_ls = best_mask.copy()
    R_ls = get_counts_from_mask(mask_ls)
    size_ls = size_of(mask_ls)
    
    # Simple local search: try to add any number
    for _ in range(50): # Limited iterations
        if time.time() > deadline - 2: break
        added = False
        for x in range(1, N + 1):
            if mask_ls[x]: continue
            # Check if adding x is valid
            # R[2x] + 1 <= 2 => R[2x] <= 1
            if R_ls[2*x] > 1: continue
            # R[x+y] + 1 <= 2 => R[x+y] <= 1 for all y in S
            present = np.where(mask_ls)[0]
            present = present[present >= 1]
            if len(present) > 0:
                sums = x + present
                if np.any(R_ls[sums] > 1): continue
            
            # Add x
            if len(present) > 0:
                R_ls[x + present] += 1
            R_ls[2*x] += 1
            mask_ls[x] = True
            size_ls += 1
            added = True
            break
        
        if not added:
            # Try swaps?
            # Skip for simplicity, just return best valid found
            break
        else:
            if size_ls > best_size:
                best_size = size_ls
                best_mask = mask_ls.copy()
                best_R = R_ls.copy()
    
    result = [int(x) for x in np.where(best_mask)[0] if x >= 1]
    return result

