import numpy as np
import time

N = 100
DEADLINE = 145.0 # 5s margin

def construct():
    start_time = time.time()
    
    # Initial best: Constant function
    best_h = np.full(N, 0.5)
    best_val = 0.5
    
    # Helper to calculate max overlap
    def calc_max_overlap(h):
        # h is array of N floats
        # C(k) = (2/N) * sum(h[i] * h[(i-k)%N])
        # O(k) = 1 - C(k)
        # We want min C(k) to be max.
        # Using FFT for speed
        h_pad = np.zeros(2*N)
        h_pad[:N] = h
        # Autocorrelation
        # c = np.correlate(h, h, mode='full')? No, FFT is faster.
        H = np.fft.fft(h, 2*N)
        C = np.fft.ifft(H * np.conj(H)).real[:N]
        C *= (2.0 / N)
        min_C = np.min(C)
        return 1.0 - min_C
        
    current_val = calc_max_overlap(best_h)
    
    # Random search with local optimization
    # We will try to perturb the constant function
    # But since constant is likely optimal, we might just verify.
    
    # Try a few random initializations
    for _ in range(20):
        if time.time() > DEADLINE:
            break
        
        # Random h in [0,1] with sum N/2
        # Simple method: random permutation of 0s and 1s? No, must be continuous.
        # Let's try random Gaussian perturbations
        h = np.full(N, 0.5)
        # Add random noise
        noise = np.random.normal(0, 0.1, N)
        h += noise
        # Project to [0,1]
        h = np.clip(h, 0, 1)
        # Adjust sum to N/2
        current_sum = np.sum(h)
        if current_sum > 0:
            h *= (N/2.0) / current_sum
            h = np.clip(h, 0, 1)
            # Re-adjust sum if clipping broke it
            # Simple iterative adjustment
            for _ in range(10):
                diff = N/2.0 - np.sum(h)
                if abs(diff) < 1e-9: break
                # Add diff/N to all
                h += diff / N
                h = np.clip(h, 0, 1)
        
        val = calc_max_overlap(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
        # Local search: flip small bits or adjust neighbors
        # Since it's continuous, do gradient descent?
        # For simplicity, just random perturbations around best_h
        for _ in range(100):
            if time.time() > DEADLINE:
                break
            h_new = best_h.copy()
            idx = np.random.randint(N)
            # Random adjustment
            h_new[idx] += np.random.uniform(-0.1, 0.1)
            h_new[idx] = np.clip(h_new[idx], 0, 1)
            # Fix sum
            diff = N/2.0 - np.sum(h_new)
            h_new += diff / N
            h_new = np.clip(h_new, 0, 1)
            
            val_new = calc_max_overlap(h_new)
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
                
    return best_h
