import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = 128
    # Target sums are [128, 256]
    # We represent A as a boolean array of length n+1 (0 to 128)
    # Initial solution: A = [64, 128] -> Sums: 128, 192, 256. 
    # This is invalid as it doesn't cover [128, 256].
    # We need a valid starting point or handle invalid states in SA.
    # Let's start with a dense random set to ensure validity, then optimize.
    # A dense set like [0, 128] has high multiplicity but is valid.
    # A = range(0, 129) -> Sums cover [0, 256].
    # Multiplicity for sum S in [128, 256] with A=[0..128]:
    # S=128: pairs (0,128)..(128,0) -> unordered: (0,128)..(64,64). Count = 65.
    # S=256: (128,128). Count = 1.
    # Peak is 65.
    
    # Better initial: A = [64, 128] is sparse but invalid.
    # Let's try A = [0, 128] and let SA reduce the set.
    # Or start with a structured set like difference set if we knew one, 
    # but random dense is safer for validity.
    
    # Actually, let's start with A = [0, 128] (all elements).
    # We will try to remove elements to reduce multiplicity while maintaining coverage.
    # Or start with a subset and add/remove.
    
    # Let's use a bitmask or boolean array.
    A = np.ones(n + 1, dtype=bool)
    
    def get_cost(A):
        """
        Returns (peak_multiplicity, invalid_count)
        Lower is better. We primarily minimize peak, but must maintain validity (invalid_count == 0).
        """
        # Compute all pair sums
        # a_i + a_j where a_i, a_j in A, i <= j (unordered)
        # Efficient way: Convolution of A with A?
        # Convolution gives ordered sums. 
        # Let C = conv(A, A).
        # For unordered pairs (i,j) with i<=j:
        # If i != j, the sum s = a_i + a_j appears twice in convolution (once as i,j once as j,i).
        # If i == j, the sum s = 2*a_i appears once.
        # We want the count of unordered pairs.
        # Let N_unordered(s) be the count.
        # Conv(s) = 2 * (number of unordered pairs with i != j summing to s) + (number of unordered pairs with i == j summing to s)
        # Let N_neq(s) be count of i<j such that a_i+a_j=s.
        # Let N_eq(s) be count of i such that 2*a_i=s.
        # Conv(s) = 2*N_neq(s) + N_eq(s).
        # N_unordered(s) = N_neq(s) + N_eq(s).
        # We can compute N_eq(s) easily: N_eq(s) = 1 if s is even and s/2 is in A, else 0.
        # Then N_neq(s) = (Conv(s) - N_eq(s)) / 2.
        # N_unordered(s) = (Conv(s) - N_eq(s)) / 2 + N_eq(s) = (Conv(s) + N_eq(s)) / 2.
        
        # Convolution of boolean array with itself.
        # A is length 129. Conv length 257.
        # We only care about sums in [128, 256].
        
        conv = np.convolve(A, A, mode='full')
        
        # Calculate N_eq for sums 128..256
        # s is in [128, 256]
        # s is even? s/2 is in [64, 128].
        # A has indices 0..128.
        
        peak = 0
        invalid_count = 0
        
        for s in range(128, 257):
            n_eq = 1 if (s % 2 == 0 and A[s//2]) else 0
            n_unordered = (conv[s] + n_eq) // 2
            
            if n_unordered == 0:
                invalid_count += 1
            else:
                if n_unordered > peak:
                    peak = n_unordered
        
        # Combine into a single score for SA
        # If invalid, penalty is huge.
        # score = peak * 1000 + invalid_count
        # We want to minimize this.
        return peak, invalid_count

    current_peak, current_invalid = get_cost(A)
    current_score = current_peak * 1000 + current_invalid
    
    best_A = A.copy()
    best_peak, best_invalid = current_peak, current_invalid
    best_score = best_peak * 1000 + best_invalid
    
    temperature = 10.0
    cooling = 0.995
    
    while time.time() < deadline:
        # Pick a random element to toggle
        idx = random.randint(0, n)
        
        # Flip
        A[idx] = not A[idx]
        
        new_peak, new_invalid = get_cost(A)
        new_score = new_peak * 1000 + new_invalid
        
        delta = new_score - current_score
        
        if delta < 0 or (temperature > 0 and random.random() < np.exp(-delta / temperature)):
            current_peak, current_invalid = new_peak, new_invalid
            current_score = new_score
            
            if new_score < best_score:
                best_score = new_score
                best_peak, best_invalid = new_peak, new_invalid
                best_A = A.copy()
        else:
            # Revert
            A[idx] = not A[idx]
        
        temperature *= cooling
        if temperature < 0.01:
            break
            
    # Return the best valid solution found
    # If best_invalid > 0, we have a problem. 
    # The initial solution A=[0..128] is valid.
    # SA might drift to invalid if not careful, but we track best.
    # Ensure we return a valid set.
    if best_invalid > 0:
        # Fallback to a known valid set if something went wrong
        # A = [0, 128] is valid.
        return list(range(n + 1))
    
    return [i for i in range(n + 1) if best_A[i]]
