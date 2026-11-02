import numpy as np
import time

N = 400
DEADLINE = 115.0

def _compute_c3_direct(h):
    """
    Compute C3[k] = sum_i h[i] h[i+k] h[i+2k] for k=0..N//2.
    This is O(N^2) but fast enough for initialization.
    """
    c3 = np.zeros(N // 2 + 1)
    h = h.astype(np.float64)
    for k in range(N // 2 + 1):
        # i goes from 0 to N - 1 - 2k
        # sum h[i] h[i+k] h[i+2k]
        # Vectorized:
        max_i = N - 2 * k
        if max_i > 0:
            c3[k] = np.sum(h[:max_i] * h[k:k+max_i] * h[2*k:2*k+max_i])
    return c3

def construct():
    start_time = time.time()
    
    # Initialize with random signs
    h = np.random.choice([-1.0, 1.0], size=N)
    
    # Initial C3 calculation
    c3 = _compute_c3_direct(h)
    
    # We track the max absolute value
    best_score = np.max(np.abs(c3))
    best_h = h.copy()
    current_score = best_score
    
    # Simulated Annealing parameters
    T = 1.0
    T_min = 1e-8
    cooling_rate = 0.999
    
    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE

    # Precompute indices for neighbors? 
    # For a flip at j, we need to update c3[k] for all k.
    # The change in c3[k] when flipping h[j] is:
    # -2 * h[j] * (sum of products of the other two elements in triples containing j at lag k)
    
    while elapsed_ok():
        # Pick a random index to flip
        j = np.random.randint(0, N)
        old_h_j = h[j]
        new_h_j = -old_h_j
        
        # Calculate the change in c3 for all k
        # We only need to check k such that j is part of the triple (i, i+k, i+2k)
        # 1. j = i => i=j. Valid if j+2k < N => k < (N-j)/2.
        # 2. j = i+k => i=j-k. Valid if j-k >= 0 and j+k < N.
        # 3. j = i+2k => i=j-2k. Valid if j-2k >= 0.
        
        # Instead of iterating k and checking conditions, we can iterate over the possible roles.
        # But it's easier to just iterate k=0..N//2 and compute the delta.
        # Delta_c3[k] = -2 * old_h_j * (Product1 + Product2 + Product3)
        
        # Let's optimize the delta calculation.
        # We can create an array of deltas, but we only need to update c3 in place.
        
        # To avoid creating a new array every time, we can just compute the new c3 values
        # for the affected k's? No, ALL k's are potentially affected?
        # Wait. If I flip h[j], does it affect c3[k] for ALL k?
        # Yes, if there is ANY i such that j is in {i, i+k, i+2k}.
        # For a given k, is there always such an i?
        # If k is small, yes. If k is large, maybe not.
        # But we can just compute the delta for all k=0..N//2. It's only 200 iterations.
        
        deltas = np.zeros(N // 2 + 1)
        
        # Role 1: j = i. i=j.
        # Term: h[j] h[j+k] h[j+2k]. Other product: h[j+k] h[j+2k].
        # Valid for k where j+2k < N.
        max_k1 = (N - 1 - j) // 2
        if max_k1 >= 0:
            # k ranges from 0 to max_k1
            # We need h[j+k] * h[j+2k] for k=0..max_k1
            # This is a convolution-like operation but small.
            # We can just loop or use slicing.
            # Slicing is faster.
            # k=0: h[j]h[j]h[j] -> other: h[j]h[j] = 1
            # Let's just do a loop for clarity and speed (200 iterations is tiny)
            pass # We will use a unified loop below for all roles to avoid complex slicing logic errors
            
        # Unified Loop for Delta Calculation
        # For each k in 0..N//2:
        #   prod_sum = 0
        #   if j + 2*k < N: prod_sum += h[j+k] * h[j+2*k]  # j is first
        #   if j - k >= 0 and j + k < N: prod_sum += h[j-k] * h[j+k] # j is middle
        #   if j - 2*k >= 0: prod_sum += h[j-2*k] * h[j-k] # j is last
        #   delta_c3[k] = -2 * old_h_j * prod_sum
        
        # This is 200 iterations with ~3 condition checks and multiplications.
        # Very fast.
        
        for k in range(N // 2 + 1):
            prod_sum = 0.0
            # j is i
            if j + 2 * k < N:
                prod_sum += h[j + k] * h[j + 2 * k]
            # j is i+k
            if j - k >= 0 and j + k < N:
                prod_sum += h[j - k] * h[j + k]
            # j is i+2k
            if j - 2 * k >= 0:
                prod_sum += h[j - 2 * k] * h[j - k]
                
            deltas[k] = -2.0 * old_h_j * prod_sum
            
        # Update c3
        c3 += deltas
        
        # Calculate new score
        new_score = np.max(np.abs(c3))
        
        # Acceptance criterion
        delta_score = new_score - current_score
        if delta_score < 0 or (T > T_min and np.random.random() < np.exp(-delta_score / T)):
            # Accept
            h[j] = new_h_j
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            # Reject
            c3 -= deltas
            # h[j] remains old_h_j
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_h

if __name__ == "__main__":
    h = construct()
    print("Constructed:", h[:20])
    c3 = _compute_c3_direct(h)
    print("Score:", np.max(np.abs(c3)))
