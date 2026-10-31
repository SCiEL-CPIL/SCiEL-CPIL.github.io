import numpy as np
import random

def construct():
        def calc_score(U):
            U = sorted(set(U))
            if not U or 0 not in U:
                return -np.inf
            max_u = max(U)
            if max_u == 0:
                return 1.0
            
            # Pad to power of 2 > 2*max_u
            size_needed = 2 * max_u + 1
            n_fft = 1
            while n_fft < size_needed:
                n_fft *= 2
            
            vec = np.zeros(n_fft, dtype=np.float64)
            for x in U:
                vec[x] = 1.0
            
            fft_vec = np.fft.fft(vec)
            
            # Sumset
            sum_poly = np.fft.ifft(fft_vec * fft_vec).real
            sum_counts = np.round(sum_poly).astype(int)
            # U+U is in [0, 2*max_u]
            size_sum = np.sum(sum_counts[:2*max_u+1] > 0)
            
            # Diffset
            # Autocorrelation
            corr = np.fft.ifft(fft_vec * np.conj(fft_vec)).real
            diff_counts = np.round(corr).astype(int)
            
            # Differences are in [-max_u, max_u]
            # In the circular convolution of length n_fft:
            # Index k corresponds to diff k if k <= n_fft/2
            # Index n_fft-k corresponds to diff -k
            size_diff = 0
            for d in range(-max_u, max_u+1):
                if d >= 0:
                    idx = d
                else:
                    idx = n_fft + d
                if diff_counts[idx] > 0:
                    size_diff += 1
            
            if size_sum == 0 or size_diff == 0:
                return -np.inf
                
            ratio = size_diff / size_sum
            if ratio <= 0:
                return -np.inf
                
            score = 1 + np.log(ratio) / np.log(2 * max_u + 1)
            return score

        # Initial set: Random sparse set
        # Let's start with a size around 50-100
        # Range should be large enough to allow distinct differences
        # But not too large to make the log denominator huge?
        # Actually, if max_u is large, log(2 max_u + 1) is large.
        # If ratio is close to 1, log(ratio) is small.
        # We want log(ratio) to be as large (positive) as possible relative to log(2 max_u + 1).
        
        # Heuristic: Sets that are "geometric" or have large gaps tend to have |U-U| > |U+U|.
        # E.g., powers of 2.
        # Let's try initializing with a mix of small and large numbers.
        
        random.seed(42)
        
        # Start with a set of size 50
        n_init = 50
        max_val_init = 500
        best_set = [0]
        # Add random distinct integers
        candidates = list(range(1, max_val_init + 1))
        random.shuffle(candidates)
        best_set.extend(candidates[:n_init-1])
        best_set = sorted(best_set)
        
        best_score = calc_score(best_set)
        
        # Local Search
        # We will iterate many times.
        # Each iteration: pick a random move.
        # Move types:
        # 1. Add a new element (if size < max_size)
        # 2. Remove an element (if size > min_size)
        # 3. Move an element (remove one, add another)
        
        max_size = 200
        min_size = 10
        max_val_limit = 2000 # Keep max_u bounded to keep FFT fast and denominator controlled?
                             # Actually, larger max_u increases denominator. 
                             # If we keep size fixed and increase max_u, ratio might drop?
                             # Let's allow max_u to grow but monitor score.
                             
        iterations = 5000
        current_set = best_set.copy()
        current_score = best_score
        
        for i in range(iterations):
            # Pick move
            move_type = random.random()
            new_set = current_set.copy()
            
            if move_type < 0.33:
                # Add
                if len(new_set) < max_size:
                    # Choose value
                    # Bias towards smaller values? Or random?
                    # Let's pick from a range slightly larger than current max
                    current_max = max(new_set)
                    # Try values up to current_max * 1.5 or so
                    upper_bound = max(100, int(current_max * 1.2))
                    new_val = random.randint(0, upper_bound)
                    if new_val not in new_set:
                        new_set.append(new_val)
                    else:
                        continue # Skip if no change
            elif move_type < 0.66:
                # Remove
                if len(new_set) > min_size:
                    # Don't remove 0
                    if len(new_set) > 1:
                        idx = random.randint(1, len(new_set)-1)
                        new_set.pop(idx)
                    else:
                        continue
            else:
                # Move (Remove + Add)
                if len(new_set) > min_size:
                    idx = random.randint(1, len(new_set)-1)
                    old_val = new_set.pop(idx)
                    current_max = max(new_set) if new_set else 0
                    upper_bound = max(100, int(current_max * 1.2) + 10)
                    new_val = random.randint(0, upper_bound)
                    if new_val not in new_set:
                        new_set.append(new_val)
                    else:
                        # Revert
                        new_set.append(old_val)
                        continue
                else:
                    continue
            
            if 0 not in new_set:
                continue
                
            new_score = calc_score(new_set)
            
            if new_score > current_score:
                current_score = new_score
                current_set = new_set
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set.copy()
                    
        return sorted(best_set)
