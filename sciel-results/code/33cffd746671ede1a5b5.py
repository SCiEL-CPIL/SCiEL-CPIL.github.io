import numpy as np
import time

N = 400

def construct():
        # Start with seed
        h_seed = np.ones(N)
        val_seed = abs(2 * N * np.max(np.convolve(h_seed, h_seed)) / np.sum(h_seed) ** 2)
        best_h = h_seed
        best_val = val_seed
        
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 100  # 100s budget for search
        
        # Simulated Annealing on binary sequences
        # Initialize with a random binary sequence or seed
        # Let's try a few random initializations
        for _ in range(5):
            h = np.random.choice([1, -1], size=N)
            # Ensure sum is not zero
            if np.sum(h) == 0:
                h[0] = 1 # Perturb
                
            current_val = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
            
            T = 1.0
            T_min = 1e-4
            cooling = 0.99
            
            while T > T_min and time.time() < deadline:
                # Flip a random element
                idx = np.random.randint(N)
                old_val = h[idx]
                h[idx] = -old_val
                
                new_sum = np.sum(h)
                if new_sum == 0:
                    # Revert if invalid
                    h[idx] = old_val
                    T *= cooling
                    continue
                    
                new_conv = np.convolve(h, h)
                new_max = np.max(new_conv)
                new_val = abs(2 * N * new_max / (new_sum ** 2))
                
                delta = new_val - current_val
                
                if delta < 0 or np.random.random() < np.exp(-delta / T):
                    current_val = new_val
                    if current_val < best_val:
                        best_val = current_val
                        best_h = h.copy()
                else:
                    h[idx] = old_val
                    
                T *= cooling
        
        # Return the best found
        return best_h
