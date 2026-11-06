import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n = 128
    target_lo = 128
    target_hi = 256
    target_range = np.arange(target_lo, target_hi + 1)
    num_targets = len(target_range)
    
    # Precompute pair structure for vectorized count calculation
    # We need to compute counts[s] = sum_{x,y in A, x<=y} [x+y == s]
    # Using NumPy, we can compute this efficiently for a given A.
    
    # Let A be a boolean vector of length 129.
    # We can compute the convolution of A with A (reversed?) 
    # Actually, standard convolution (A * A)[k] = sum_i A[i] A[k-i].
    # This counts ordered pairs (i, j) such that i+j=k.
    # We want unordered pairs {i, j} with i <= j.
    # Let C[k] be the number of ordered pairs.
    # If i != j, the pair (i,j) and (j,i) are both counted in C[k].
    # If i == j, the pair (i,i) is counted once in C[k].
    # We want U[k] = (C[k] + D[k]) / 2, where D[k] is the number of i such that 2i=k and A[i]=1.
    # Wait, let's verify.
    # C[k] = sum_{i+j=k} A[i]A[j].
    # Let S1 = sum_{i<j, i+j=k} A[i]A[j].
    # Let S2 = sum_{i, 2i=k} A[i].
    # Then C[k] = 2*S1 + S2.
    # We want U[k] = S1 + S2.
    # So 2*U[k] = 2*S1 + 2*S2 = C[k] + S2.
    # Thus U[k] = (C[k] + S2[k]) / 2.
    
    # This is very fast to compute using np.convolve.
    
    def compute_counts(A):
        # A is a boolean array of length 129
        # Convolution of A with A gives ordered pair sums
        conv = np.convolve(A, A, mode='full')
        # conv[k] corresponds to sum i+j=k.
        # Length of conv is 257 (indices 0..256).
        # We need sums in [128, 256].
        
        # Calculate S2: number of i in A such that 2i = s.
        # For a given sum s, i = s/2.
        # So S2[s] is 1 if s is even and A[s//2] is 1, else 0.
        # We only care about s in [128, 256].
        
        # Let's compute S2 for the target range.
        s2 = np.zeros(257, dtype=np.int64)
        # s is even, i = s/2.
        # i must be in [0, 128].
        # s in [128, 256] => i in [64, 128].
        # So for s in 128..256 step 2:
        # s2[s] = A[s//2]
        
        # Vectorized way:
        # Indices of even numbers in 0..256
        even_indices = np.arange(0, 257, 2)
        # Corresponding i values
        i_vals = even_indices // 2
        # Filter i_vals to be in [0, 128]
        valid_mask = (i_vals >= 0) & (i_vals <= 128)
        valid_evens = even_indices[valid_mask]
        valid_is = i_vals[valid_mask]
        
        # s2[valid_evens] = A[valid_is]
        # A is boolean, so this works.
        s2[valid_evens] = A[valid_is].astype(np.int64)
        
        # Now compute U[s] for s in target_range
        # U[s] = (conv[s] + s2[s]) // 2
        counts = (conv[target_range] + s2[target_range]) // 2
        return counts

    def evaluate(A):
        # A is boolean array
        counts = compute_counts(A)
        
        # Coverage penalty: sum of 1 for s where counts[s] == 0
        # We want to minimize max(counts).
        # Objective: 
        # 1. If not all covered, return a large penalty.
        # 2. Else, return max(counts) + small regularization (e.g., sum of squares to smooth landscape).
        
        uncovered = np.sum(counts == 0)
        if uncovered > 0:
            # Penalty proportional to number of uncovered sums
            # Also add max count to encourage balancing even if not fully covered?
            # Let's just return a large number.
            # But for annealing, we need a continuous landscape.
            # Let's use: penalty = 1000 * uncovered + max(counts)
            # This ensures coverage is prioritized, but allows exploration.
            return 1000 * uncovered + np.max(counts)
        
        # If covered, minimize max(counts)
        # To make the landscape smoother for SA, we can use max(counts) + 0.01 * np.sum(counts**2)
        # The sum of squares penalizes uneven distribution.
        return np.max(counts) + 0.01 * np.sum(counts**2)

    # Initialize A
    # Start with a random set that has reasonable density
    # Or start with a structured set?
    # Let's start with a random set of size ~40-50, which is likely to cover or be close.
    # Actually, starting with a "greedy" solution might be a good initial state.
    # But the directive asks for a different strategy. SA from a random start is fine.
    
    # Let's try a few different initializations and pick the best one for SA.
    # Or just run SA with multiple restarts?
    
    best_A = None
    best_score = float('inf')
    
    # We have ~85 seconds.
    # Let's do 5 restarts of SA, each with a time budget.
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts
    
    for r in range(num_restarts):
        # Initialize A
        # Random initialization: pick k elements
        # k around 40 seems reasonable based on previous analysis.
        k = random.randint(35, 50)
        indices = np.random.choice(129, size=k, replace=False)
        A = np.zeros(129, dtype=bool)
        A[indices] = True
        
        current_score = evaluate(A)
        
        # Simulated Annealing
        T = 10.0  # Initial temperature
        T_min = 0.1
        cooling_rate = 0.995  # Adjust based on steps
        
        # Estimate steps:
        # If time_per_restart is ~17s, and each step is ~0.001s (vectorized eval is fast),
        # we can do ~10000-20000 steps.
        # Let's use a time-based loop instead of fixed steps.
        
        sa_start = time.time()
        sa_deadline = sa_start + time_per_restart * 0.9  # Leave margin
        
        while time.time() < sa_deadline:
            # Flip a random bit
            idx = random.randint(0, 128)
            A[idx] = not A[idx]
            
            new_score = evaluate(A)
            
            delta = new_score - current_score
            
            # Accept if better, or with probability exp(-delta/T)
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = A.copy()
            else:
                # Revert
                A[idx] = not A[idx]
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
                # Maybe reheat? Or just stop cooling.
                # For simplicity, keep T at T_min.
        
        # Check if we improved
        if best_A is not None:
            # Verify best_A is valid
            final_counts = compute_counts(best_A)
            if np.all(final_counts > 0):
                # It's a valid solution.
                # We can break if we find a very good one?
                # Theoretical lower bound for peak is around 3-4?
                # If best_score < 5, we might be close to optimal.
                if best_score < 5.0:
                    break

    # Fallback if best_A is None or invalid
    if best_A is None:
        # Return a valid set, e.g., all even + 1?
        # Just return range(129) to be safe, though peak will be high.
        A_fallback = np.zeros(129, dtype=bool)
        A_fallback[:] = True
        best_A = A_fallback
        best_score = evaluate(best_A)

    # Convert numpy boolean array to list of integers
    result_list = [i for i in range(129) if best_A[i]]
    
    return result_list

if __name__ == "__main__":
    A = construct()
    print(A)
