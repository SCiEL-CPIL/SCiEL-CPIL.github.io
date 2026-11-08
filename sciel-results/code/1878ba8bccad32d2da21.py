"""
Strategy: Simulated Annealing over binary vectors representing set membership.
Instead of greedily adding integers, we treat the set U as a binary vector x of length M+1.
We perform random flips (add/remove elements) and accept them based on the Metropolis criterion.
This allows escaping local optima that trap greedy methods.
"""
import time
import math
import numpy as np

def compute_score_fast(U_list, max_val_hint=None):
    """
    Computes the lower bound exponent using FFT for sumset and bitsets for diffset.
    U_list: list of nonnegative integers
    Returns: score
    """
    if len(U_list) < 2:
        return 0.0
    
    # Ensure 0 is in the set for the problem definition, though score formula works generally
    # The problem states U contains 0.
    # If 0 is missing, we should probably add it or return 0? 
    # The prompt says "Construct a finite set U ... containing 0".
    # The evaluator likely checks if 0 is in the result.
    # For scoring purposes, if 0 is not in U, |U+U| and |U-U| are still defined.
    # But the score formula uses max U.
    # Let's assume the input is valid (contains 0). If not, we handle it gracefully.
    
    U = sorted(set(U_list))
    if 0 not in U:
        # If 0 is not in U, the problem statement is violated.
        # However, the score formula is mathematical. 
        # We'll just compute it. But construct() must return a set with 0.
        pass
        
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    # 1. Compute Sumset |U+U| using FFT
    max_sum = 2 * maxU
    n_fft = 1
    while n_fft < max_sum + 1:
        n_fft *= 2
        
    A = np.zeros(n_fft, dtype=np.float64)
    for u in U:
        A[u] = 1.0
        
    A_fft = np.fft.rfft(A)
    C_fft = A_fft * A_fft
    C = np.fft.irfft(C_fft, n=n_fft)
    
    # Count non-zero entries in C[0..2*maxU]
    sum_set_size = np.sum(C[:max_sum + 1] > 0.5)
    
    # 2. Compute Difference Set |U-U|
    # Using set comprehension for speed in Python for small N
    # N is bounded by M+1. If M is ~100, N is ~100. N^2 = 10k ops. Fast.
    # If M is larger, say 500, N^2 = 250k. Still fast.
    diff_set = set()
    for u1 in U:
        for u2 in U:
            diff_set.add(u1 - u2)
            
    size_diff = len(diff_set)
    size_sum = sum_set_size
    
    if size_sum == 0:
        return 0.0
        
    log_ratio = math.log(size_diff / size_sum)
    log_range = math.log(2 * maxU + 1)
    
    return 1.0 + log_ratio / log_range

def construct():
    """
    Entry point. Returns a list of nonnegative integers containing 0.
    Uses Simulated Annealing over a binary vector representation.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Define the search space boundary
    # The paper mentions sizes 2003 and 54265, but those are likely |U| or related.
    # The score depends on max(U). Larger max(U) reduces the log term in denominator,
    # but increases the numerator if the ratio improves.
    # Let's bound the maximum element M to keep computation tractable and score competitive.
    # A geometric progression with base 2 up to 2^10 is 1024.
    # Let's try M = 200 initially, then maybe expand if we find good structures.
    # Actually, the score is 1 + log(ratio)/log(2M+1).
    # If ratio is ~2, log(2)~0.69.
    # If M=10, log(21)~3.0, score ~ 1.23.
    # If M=100, log(201)~5.3, score ~ 1.13.
    # So smaller M is better IF ratio stays high.
    # However, small M limits the set size and structure.
    # Let's try M=50, 100, 200.
    
    # We will run SA for a few different M values and pick the best.
    best_overall_score = 0.0
    best_overall_U = [0, 1]
    
    M_values = [20, 50, 100, 200]
    
    for M in M_values:
        if time.time() > deadline - 10:
            break
            
        # Initialize binary vector
        # x[i] = 1 if i is in U
        # 0 must be in U.
        x = np.zeros(M + 1, dtype=np.int8)
        x[0] = 1
        
        # Initialize with a random small set to give it some structure
        # e.g., geometric progression
        val = 1
        while val <= M:
            x[val] = 1
            val *= 2
        
        # Convert to list for initial score
        current_U = [i for i in range(M + 1) if x[i] == 1]
        current_score = compute_score_fast(current_U)
        
        best_U_in_sa = list(current_U)
        best_score_in_sa = current_score
        
        # Simulated Annealing Parameters
        T_start = 1.0
        T_end = 0.01
        alpha = 0.995  # Cooling rate
        
        T = T_start
        
        while T > T_end:
            if time.time() > deadline - 5:
                break
                
            # Generate a neighbor
            # Flip a random bit
            # Bias towards flipping 1->0 or 0->1?
            # Let's just pick a random index.
            idx = np.random.randint(1, M + 1)  # Don't flip 0
            
            if x[idx] == 1:
                # Remove idx
                new_U = [u for u in current_U if u != idx]
                if len(new_U) < 2:
                    T *= alpha
                    continue
            else:
                # Add idx
                new_U = current_U + [idx]
                new_U.sort()
            
            new_score = compute_score_fast(new_U)
            delta = new_score - current_score
            
            if delta > 0:
                # Accept
                current_U = new_U
                current_score = new_score
                x[idx] = 1 - x[idx]
                
                if current_score > best_score_in_sa:
                    best_score_in_sa = current_score
                    best_U_in_sa = list(current_U)
            else:
                # Probabilistic accept
                # Accept with probability exp(delta / T)
                # delta is negative
                prob = math.exp(delta / T)
                if np.random.random() < prob:
                    current_U = new_U
                    current_score = new_score
                    x[idx] = 1 - x[idx]
            
            T *= alpha
        
        # End of SA for this M
        if best_score_in_sa > best_overall_score:
            best_overall_score = best_score_in_sa
            best_overall_U = best_U_in_sa
            
    # Ensure 0 is in the final result
    if 0 not in best_overall_U:
        best_overall_U.append(0)
        
    return sorted(best_overall_U)

# EVOLVE-BLOCK-END
