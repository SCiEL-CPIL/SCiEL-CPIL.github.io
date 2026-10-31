import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize with 0.5
    h = np.full(N, 0.5)
    
    def compute_overlaps(h):
        # h is length N
        # overlap_k = sum_i h_i * (1 - h_{i-k})
        # = sum_i h_i - sum_i h_i h_{i-k}
        # sum_i h_i is constant N/2
        # We need C_k = sum_i h_i h_{i-k}
        # This is circular cross-correlation
        # Using FFT for efficiency
        H = np.fft.fft(h)
        C = np.fft.ifft(H * np.conj(H)).real
        # C[k] corresponds to shift k
        overlaps = N * 0.5 - C
        return overlaps

    # Initial evaluation
    overlaps = compute_overlaps(h)
    current_max_overlap = np.max(overlaps)
    
    # Local search
    # We will try to reduce the max overlap by adjusting h
    # Use a simple gradient-like step or random perturbations
    # Since the problem is symmetric, let's try a few random starts and local search
    
    best_h = h.copy()
    best_score = current_max_overlap
    
    # Try a few random initializations
    rng = np.random.default_rng(42)
    
    for _ in range(5):
        # Random initialization
        h = rng.uniform(0, 1, N)
        # Project to sum = N/2
        h = h - np.mean(h) + 0.5
        h = np.clip(h, 0, 1)
        # Re-project to sum exactly N/2
        diff = N/2 - np.sum(h)
        h += diff / N
        h = np.clip(h, 0, 1)
        
        # Local search steps
        step_size = 0.05
        for i in range(1000):
            if time.time() > deadline:
                return best_h
            
            overlaps = compute_overlaps(h)
            max_overlap = np.max(overlaps)
            
            if max_overlap < best_score:
                best_score = max_overlap
                best_h = h.copy()
                
            # Find the shift with max overlap
            k_max = np.argmax(overlaps)
            
            # We want to reduce overlap at k_max
            # overlap_k = sum_i h_i (1 - h_{i-k})
            # Derivative wrt h_j: (1 - h_{j-k}) - h_{j+k}
            # We want to move h in direction -grad
            # grad_j = (1 - h_{j-k}) - h_{j+k}
            # Let's compute gradient
            h_shifted_down = np.roll(h, k_max) # h_{i-k}
            h_shifted_up = np.roll(h, -k_max) # h_{i+k}
            grad = (1 - h_shifted_down) - h_shifted_up
            
            # Update h
            h = h - step_size * grad
            h = np.clip(h, 0, 1)
            
            # Project to sum N/2
            diff = N/2 - np.sum(h)
            h += diff / N
            h = np.clip(h, 0, 1)
            
            # Adaptive step size
            new_overlaps = compute_overlaps(h)
            new_max_overlap = np.max(new_overlaps)
            
            if new_max_overlap < max_overlap:
                step_size *= 1.1
            else:
                step_size *= 0.9
                # Revert if worse
                h = h + step_size * grad
                h = np.clip(h, 0, 1)
                diff = N/2 - np.sum(h)
                h += diff / N
                h = np.clip(h, 0, 1)
                
    return best_h
# EVOLVE-BLOCK-END
