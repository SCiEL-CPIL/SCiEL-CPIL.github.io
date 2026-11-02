"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # We use a binary sequence of 0s and 1s.
    # We want to minimize the maximum overlap with the shifted complement.
    # Overlap for shift k is sum_{i} h[i] * (1 - h[(i+k) % N]) for circular,
    # or sum_{i=0}^{N-k-1} h[i] * (1 - h[i+k]) for linear.
    # Given the ambiguity, circular is standard for "autocorrelation" type problems.
    # However, linear allows O(0)=0.
    # Let's assume circular first, but if we can't beat 0.5, we might need to rethink.
    # Actually, let's try to minimize the max circular overlap.
    
    # Initial random balanced sequence
    h = np.zeros(N)
    indices = np.random.choice(N, N//2, replace=False)
    h[indices] = 1.0
    
    def get_overlap(h):
        # Returns max overlap
        max_ov = 0
        for k in range(1, N):
            # Circular shift
            shifted = np.roll(h, -k)
            ov = np.sum(h * (1 - shifted))
            if ov > max_ov:
                max_ov = ov
        return max_ov
    
    current_h = h.copy()
    current_max_ov = get_overlap(current_h)
    
    # Local search
    import random
    for _ in range(5000):
        # Pick two indices to swap (keep sum constant)
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        if i == j:
            continue
        if current_h[i] == current_h[j]:
            continue
            
        # Swap
        current_h[i], current_h[j] = current_h[j], current_h[i]
        
        new_max_ov = get_overlap(current_h)
        
        if new_max_ov < current_max_ov:
            current_max_ov = new_max_ov
        else:
            # Revert
            current_h[i], current_h[j] = current_h[j], current_h[i]
            
    # Convert to float array for return
    return current_h

# EVOLVE-BLOCK-END
