"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
TARGET_SUM = N / 2.0

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_overlaps_binary(b):
    """
    Compute C_k = sum_i b_i * (1 - b_{i+k}) for all k.
    b is a binary array of length N.
    """
    # C_k is the count of 1s in b shifted by k that overlap with 0s in b.
    # This is equivalent to:
    # Let B be the set of indices where b_i = 1.
    # C_k = |{i in B : i+k not in B}|
    # We can compute this efficiently using FFT or just direct summation for N=100.
    # Direct summation O(N^2) is 10,000 ops, which is trivial.
    # Let's do it directly for clarity and speed in Python if vectorized.
    
    # b is 0/1. 
    # 1-b is 0/1.
    # corr[k] = sum_i b[i] * (1-b[(i+k)%N])
    
    # Using FFT for speed, though N=100 is small.
    # Actually, direct loop in numpy might be faster for small N than FFT overhead.
    # But let's stick to FFT or matrix multiplication.
    
    # Matrix approach:
    # M is circulant matrix? No.
    # Let's just use a loop over k, vectorized over i.
    max_c = 0.0
    for k in range(1, N):
        # shift b by k
        shifted = np.roll(b, -k)
        # count where b is 1 and shifted is 0
        c_k = np.sum(b * (1 - shifted))
        if c_k > max_c:
            max_c = c_k
    return max_c

def construct():
    start = time.time()
    deadline = start + 120.0
    
    # We search in the space of binary vectors with exactly N/2 ones.
    # This ensures sum(h) = N/2 exactly.
    # h = b.
    
    n_ones = N // 2
    
    # Initial guess: Random binary vector with n_ones ones
    indices = np.arange(N)
    ones_indices = np.random.choice(indices, n_ones, replace=False)
    b = np.zeros(N)
    b[ones_indices] = 1.0
    
    current_cost = _compute_overlaps_binary(b)
    best_b = b.copy()
    best_cost = current_cost
    
    # Simulated Annealing
    T = 5.0
    T_min = 0.01
    alpha = 0.99
    
    # Precompute neighbors? No, just random swaps.
    # A move: pick a 1 and a 0, swap them.
    
    while time.time() < deadline - 5.0:
        # Pick random 1 index and random 0 index
        ones = np.where(b == 1.0)[0]
        zeros = np.where(b == 0.0)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            break
            
        i = np.random.choice(ones)
        j = np.random.choice(zeros)
        
        # Perform swap
        b[i] = 0.0
        b[j] = 1.0
        
        new_cost = _compute_overlaps_binary(b)
        
        if new_cost < current_cost or np.random.random() < np.exp(-(new_cost - current_cost) / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_b = b.copy()
        else:
            # Revert
            b[i] = 1.0
            b[j] = 0.0
            
        T *= alpha
        if T < T_min:
            T = T_min

    # Convert best_b to h
    h = best_b.copy()
    
    # Optional: Try to improve with small continuous perturbations?
    # The problem allows h in [0,1]. Binary is a subset.
    # If we allow h to be non-binary, we might lower the max overlap further?
    # The overlap is linear in h. The max of linear functions is convex.
    # Minimizing a convex function over a polytope usually happens at vertices.
    # The vertices of the polytope {0<=h<=1, sum(h)=N/2} are not just binary.
    # Wait, the polytope is defined by box constraints and one equality.
    # The vertices of this polytope are points where N-1 variables are at bounds (0 or 1) and one is determined by the sum.
    # So vertices are mostly binary, with at most one fractional component.
    # Since the objective is convex (max of linear functions), the minimum MUST be at a vertex.
    # Therefore, the optimal solution is either binary (if N/2 is integer, which it is) or has one fractional value.
    # Since N=100, N/2=50 is integer. So binary solutions are valid vertices.
    # Thus, searching the binary space is theoretically sufficient to find the global minimum!
    
    # So the binary Simulated Annealing is actually a complete search strategy for the global optimum
    # (modulo local minima traps, but SA helps escape them).
    
    # Final check and return
    return h

# EVOLVE-BLOCK-END
