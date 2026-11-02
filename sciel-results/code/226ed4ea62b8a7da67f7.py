import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds safety margin for 120s limit? No, 150s kill. 120s is task limit. Use 100s.
    
    # Initialize with random binary sequence with exactly N/2 ones
    h = np.zeros(N)
    indices = np.random.permutation(N)
    h[indices[:N//2]] = 1.0
    
    best_h = h.copy()
    best_score = np.inf
    
    def calc_score(h):
        # Calculate max overlap for shifts 1 to N-1
        # O(k) = sum h_i (1 - h_{i-k})
        # Using FFT for speed? N=100 is small, direct is fine.
        max_overlap = 0
        for k in range(1, N):
            # Shifted h
            h_shifted = np.roll(h, k)
            overlap = np.sum(h * (1 - h_shifted))
            if overlap > max_overlap:
                max_overlap = overlap
        return max_overlap

    current_score = calc_score(h)
    if current_score < best_score:
        best_score = current_score
        best_h = h.copy()
        
    # Simple Hill Climbing
    # We want to minimize max overlap.
    # Flipping a bit changes the overlap for all shifts.
    
    no_improvement_count = 0
    while time.time() < deadline:
        if no_improvement_count > 500:
            break
            
        # Pick a random index to flip
        idx = np.random.randint(N)
        
        # Ensure sum constraint: if we flip 1->0, we must flip 0->1
        # Or just swap two indices.
        if h[idx] == 1:
            # Find a 0 to swap with
            zeros = np.where(h == 0)[0]
            if len(zeros) == 0: continue
            swap_idx = zeros[np.random.randint(len(zeros))]
        else:
            ones = np.where(h == 1)[0]
            if len(ones) == 0: continue
            swap_idx = ones[np.random.randint(len(ones))]
            
        # Perform swap
        h[idx], h[swap_idx] = h[swap_idx], h[idx]
        
        new_score = calc_score(h)
        
        if new_score < best_score:
            best_score = new_score
            best_h = h.copy()
            no_improvement_count = 0
        else:
            # Revert
            h[idx], h[swap_idx] = h[swap_idx], h[idx]
            no_improvement_count += 1
            
    return best_h
# EVOLVE-BLOCK-END
