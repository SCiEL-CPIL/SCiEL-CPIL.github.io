import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Generate Thue-Morse sequence
    # t_n = popcount(n) % 2
    # We want sum to be N/2.
    # Thue-Morse is balanced for powers of 2.
    # For N=100, we can take the first 100 terms.
    tm = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=float)
    
    # Adjust to ensure exactly N/2 ones
    # If sum != N/2, we can swap some 0s and 1s
    num_ones = np.sum(tm)
    target_ones = N // 2
    
    h = tm.copy()
    
    # If num_ones > target, change some 1s to 0s
    # If num_ones < target, change some 0s to 1s
    # To minimize disturbance, we can just pick random ones.
    # But let's start with a random binary sequence if Thue-Morse is not balanced enough?
    # Actually, Thue-Morse is very good.
    
    # Let's just use a random binary sequence as the base for hill climbing,
    # as it explores the space well.
    # But Thue-Morse is a good "structured" start.
    
    # Let's initialize with Thue-Morse, then fix the sum.
    if num_ones > target_ones:
        excess = num_ones - target_ones
        ones_indices = np.where(h == 1)[0]
        to_flip = np.random.choice(ones_indices, excess, replace=False)
        h[to_flip] = 0
    elif num_ones < target_ones:
        deficit = target_ones - num_ones
        zeros_indices = np.where(h == 0)[0]
        to_flip = np.random.choice(zeros_indices, deficit, replace=False)
        h[to_flip] = 1

    def calc_score(h):
        max_overlap = 0.0
        # Linear shifts
        for k in range(1, N):
            prod = h[:N-k] * (1 - h[k:])
            s = np.sum(prod)
            if s > max_overlap:
                max_overlap = s
        return max_overlap

    current_score = calc_score(h)
    best_h = h.copy()
    best_score = current_score

    # Hill Climbing with Simulated Annealing
    T = 1.0
    # Number of steps
    max_steps = 50000
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Pick two random indices to swap
        i, j = np.random.randint(0, N, 2)
        if h[i] == h[j]:
            continue
        
        # Swap
        h[i], h[j] = h[j], h[i]
        
        new_score = calc_score(h)
        
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            # Revert
            h[i], h[j] = h[j], h[i]
            # Check acceptance
            if T > 1e-6:
                if np.random.rand() < np.exp(-delta / T):
                    h[i], h[j] = h[j], h[i]
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_h = h.copy()
        
        # Cool down
        T *= 0.999
        if T < 1e-6:
            T = 1e-6

    return best_h

# EVOLVE-BLOCK-END
