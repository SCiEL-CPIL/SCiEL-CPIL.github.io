<<<<<<< SEARCH
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    N = 1000
    best_set = []
    best_size = 0
    
    def get_tight_count(x, chosen, counts):
        """
        Returns the number of sums that will become 'tight' (count 1 -> 2)
        if x is added to chosen.
        Assumes x is valid (i.e., no sum becomes count 3).
        """
        tight = 0
        # Sum 2*x
        if counts.get(2 * x, 0) == 1:
            tight += 1
        
        # Sums x + y for y in chosen
        for y in chosen:
            s = x + y
            if counts.get(s, 0) == 1:
                tight += 1
        return tight

    def is_valid(x, chosen, counts):
        if counts.get(2 * x, 0) >= 2:
            return False
        for y in chosen:
            if counts.get(x + y, 0) >= 2:
                return False
        return True

    def add_to_set(x, chosen, counts):
        counts[2 * x] = counts.get(2 * x, 0) + 1
        for y in chosen:
            s = x + y
            counts[s] = counts.get(s, 0) + 1
        chosen.append(x)

    def run_greedy(noise_scale=0.0):
        chosen = []
        counts = defaultdict(int)
        candidates = list(range(1, N + 1))
        
        while candidates:
            best_x = None
            best_score = float('inf')
            
            # Optional: Early exit if time is tight
            if time.time() > deadline:
                break
                
            for x in candidates:
                if is_valid(x, chosen, counts):
                    tight = get_tight_count(x, chosen, counts)
                    # Add noise to break ties and explore
                    score = tight + (random.random() * noise_scale if noise_scale > 0 else 0)
                    
                    if score < best_score:
                        best_score = score
                        best_x = x
                        # If score is very low (0), we might want to check if there are others?
                        # No, we want the min. If we add noise, 0 + noise is just noise.
                        # If noise_scale is 0, we pick the first min (smallest x due to list order? 
                        # Wait, candidates is not sorted in the loop? 
                        # candidates is a list. We iterate through it.
                        # If we want smallest x on tie, we need to sort or handle carefully.
                        # With noise, ties are broken randomly.
            
            if best_x is None:
                break
                
            add_to_set(best_x, chosen, counts)
            candidates.remove(best_x) # O(N) removal
            
        return chosen

    # Phase 1: Deterministic Min-Cost Greedy (Noise = 0)
    # This gives a strong baseline
    initial_set = run_greedy(noise_scale=0.0)
    best_set = initial_set
    best_size = len(initial_set)
    
    # Phase 2: Randomized Noisy Min-Cost Greedy Restarts
    # We use noise to explore the space of "good" greedy choices
    while time.time() < deadline:
        # Increase noise over time? Or keep it fixed?
        # Fixed small noise is good for diversity.
        current_set = run_greedy(noise_scale=0.5)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return best_set
=======
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    N = 1000
    best_set = []
    best_size = 0
    fft_size = 2 * N + 1
    
    def compute_counts(S_bool):
        """Compute representation counts via FFT convolution."""
        S_padded = np.zeros(fft_size, dtype=np.float64)
        S_padded[1:N+1] = S_bool[1:N+1]
        S_fft = np.fft.fft(S_padded)
        corr = np.fft.ifft(S_fft * np.conj(S_fft)).real
        counts = np.zeros(fft_size, dtype=np.int32)
        for s in range(2, 2*N+1):
            diag = 1.0 if (s % 2 == 0 and s//2 <= N and S_bool[s//2] == 1.0) else 0.0
            counts[s] = int(round((corr[s] + diag) / 2.0))
        return counts
    
    def run_greedy(noise_scale=0.0):
        chosen = set()
        S_bool = np.zeros(N + 1, dtype=np.float64)
        candidates = list(range(1, N + 1))
        
        while candidates:
            if time.time() > deadline:
                break
            
            counts = compute_counts(S_bool)
            
            best_x = None
            best_score = float('inf')
            
            for x in candidates:
                if counts[2*x] >= 2:
                    continue
                valid = True
                for y in chosen:
                    if counts[x+y] >= 2:
                        valid = False
                        break
                if not valid:
                    continue
                
                tight = 0
                if counts[2*x] == 1:
                    tight += 1
                for y in chosen:
                    if counts[x+y] == 1:
                        tight += 1
                
                score = tight + (random.random() * noise_scale if noise_scale > 0 else 0)
                if score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
            
            chosen.add(best_x)
            S_bool[best_x] = 1.0
            candidates.remove(best_x)
        
        return list(chosen)
    
    # Phase 1: Deterministic greedy
    initial_set = run_greedy(noise_scale=0.0)
    best_set = initial_set
    best_size = len(initial_set)
    
    # Phase 2: Randomized restarts
    while time.time() < deadline:
        noise = random.uniform(0.0, 0.5)
        current_set = run_greedy(noise_scale=noise)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    return best_set
>>>>>>> REPLACE
