import numpy as np
import time

N = 100
W = 2.0 / N

def _overlap_max(h):
    """Compute the maximum overlap of h with shifted 1-h."""
    # Overlap at shift k is sum_{i} h[i] * (1-h[i-k])
    # Sum h[i] - h[i]h[i-k]
    # Sum h[i] is N/2.
    # So we minimize max_k (N/2 - autocorr[k])
    # Which is equivalent to minimizing max_k (N/2 - autocorr[k])
    # Or maximizing min_k autocorr[k]
    
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    # autocorr[k] corresponds to circular cross-correlation
    # Note: ifft returns unnormalized correlation. 
    # np.fft.ifft(np.fft.fft(x)*np.fft.fft(y).conj()) gives circular cross-correlation.
    # The sum of x is N/2.
    # overlap[k] = sum_i h[i] * (1-h[i-k]) = sum_i h[i] - sum_i h[i]h[i-k]
    #            = N/2 - autocorr[k]
    
    overlaps = (N / 2.0 - autocorr) * W
    # We want to minimize the max overlap.
    # Due to floating point, overlaps might be slightly negative or > 1.
    # But max is what matters.
    return np.max(overlaps)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Binary representation: h_i is 0 or 1.
    # Sum must be N/2 = 50.
    # So we have exactly 50 ones and 50 zeros.
    
    # Initial guess: alternating pattern or random
    # Alternating 0101... has autocorr 0 for odd shifts, 50 for even shifts (roughly).
    # Let's start with a random permutation.
    h = np.zeros(N)
    h[:N//2] = 1.0
    
    # Shuffle to break symmetry
    np.random.shuffle(h)
    
    best_h = h.copy()
    best_val = _overlap_max(h)
    current_val = best_val
    
    # Simulated Annealing
    T_start = 0.5
    T_end = 0.01
    alpha = 0.999 # Slow cooling
    
    T = T_start
    step = 0
    max_steps = 50000
    
    # Precompute indices of ones and zeros for efficient swap
    # We will maintain a list of indices where h==1 and h==0
    ones_indices = np.where(h == 1.0)[0]
    zeros_indices = np.where(h == 0.0)[0]
    
    while step < max_steps and time.time() < deadline:
        # Pick a random 1 and a random 0 to swap
        # This maintains the sum constraint automatically.
        if len(ones_indices) == 0 or len(zeros_indices) == 0:
            break
            
        i_idx = np.random.randint(len(ones_indices))
        j_idx = np.random.randint(len(zeros_indices))
        
        i_pos = ones_indices[i_idx]
        j_pos = zeros_indices[j_idx]
        
        # Perform swap
        h[i_pos] = 0.0
        h[j_pos] = 1.0
        
        # Update index lists
        ones_indices[i_idx] = j_pos
        zeros_indices[j_idx] = i_pos
        
        val_new = _overlap_max(h)
        
        # Acceptance
        if val_new < current_val:
            current_val = val_new
            if current_val < best_val:
                best_val = current_val
                best_h = h.copy()
        else:
            delta = val_new - current_val
            if delta < 0:
                accept = True
            else:
                # Delta is positive (bad move)
                # T is in units of overlap (roughly 0 to 0.5)
                # exp(-delta/T)
                if T > 0:
                    accept = np.random.random() < np.exp(-delta / T)
                else:
                    accept = False
            
            if accept:
                current_val = val_new
            else:
                # Revert swap
                h[i_pos] = 1.0
                h[j_pos] = 0.0
                ones_indices[i_idx] = i_pos
                zeros_indices[j_idx] = j_pos
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end
            
        step += 1
        
        # Periodic restart if stuck? 
        # For now, keep it simple.
        
    # The result h is binary. 
    # The task requires heights in [0,1] with sum N/2.
    # Binary 0/1 satisfies this.
    
    # Ensure exact sum (it is exactly 50.0)
    # Return the binary vector
    
    return best_h
