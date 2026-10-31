import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 140  # 150s limit, keep margin
    
    # Objective: Minimize max overlap for shifts 1 and 4 (C5 neighbors)
    # Overlap A_s = N/2 - C_s
    # Minimize max(A_1, A_4) <=> Maximize min(C_1, C_4)
    
    def calc_cost(h):
        # h is a numpy array of 0s and 1s
        # C_s = sum(h[i] * h[(i+s)%N])
        c1 = np.sum(h * np.roll(h, -1))
        c4 = np.sum(h * np.roll(h, -4))
        # We want to maximize min(c1, c4)
        # So cost is -min(c1, c4)
        return -min(c1, c4)
    
    # Initialize with a random permutation of N/2 ones and N/2 zeros
    h = np.zeros(N)
    h[:N//2] = 1.0
    np.random.shuffle(h)
    
    best_h = h.copy()
    best_cost = calc_cost(best_h)
    
    T = 1.0
    cooling = 0.99
    
    while time.time() - start_time < timeout:
        # Pick two random indices to swap
        i = np.random.randint(N)
        j = np.random.randint(N)
        if i == j:
            continue
        if h[i] == h[j]:
            continue
            
        # Swap
        h[i], h[j] = h[j], h[i]
        new_cost = calc_cost(h)
        delta = new_cost - best_cost
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            best_h = h.copy()
            best_cost = new_cost
        else:
            # Revert
            h[i], h[j] = h[j], h[i]
            
        T *= cooling
        
    # Convert to float array in [0,1]
    return best_h.astype(float)
# EVOLVE-BLOCK-END
