import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initialize with a smooth wave or random
    # Try a cosine wave to break symmetry
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    h = 0.5 + 0.1 * np.cos(x)
    
    # Ensure constraints: 0 <= h <= 1, sum = 50
    # Normalize
    def normalize(h):
        h = np.clip(h, 0, 1)
        current_sum = np.sum(h)
        if current_sum > 0:
            h = h * (50.0 / current_sum)
        h = np.clip(h, 0, 1)
        # Fine tune to get sum exactly 50?
        # The task says "sum is exactly len/2".
        # We might need to adjust one element.
        diff = 50.0 - np.sum(h)
        idx = np.argmax(h) if diff > 0 else np.argmin(h)
        h[idx] += diff
        h = np.clip(h, 0, 1)
        return h

    h = normalize(h)
    
    best_h = h.copy()
    best_score = float('inf')
    
    # Calculate score
    def score(h):
        # Linear overlap
        max_ov = 0
        for k in range(1, N):
            # sum h[i] * (1 - h[i+k]) for i=0 to N-1-k
            prod = h[:-k] * (1 - h[k:])
            ov = np.sum(prod)
            if ov > max_ov:
                max_ov = ov
        return max_ov
    
    current_score = score(h)
    if current_score < best_score:
        best_score = current_score
        best_h = h.copy()
        
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    step_size = 0.1
    
    while time.time() < deadline:
        # Perturb
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        if i == j:
            continue
        delta = np.random.uniform(-step_size, step_size)
        
        h_new = h.copy()
        h_new[i] += delta
        h_new[j] -= delta # Keep sum roughly constant
        
        h_new = normalize(h_new)
        
        new_score = score(h_new)
        
        if new_score < current_score or np.random.random() < np.exp(-(new_score - current_score) / T):
            h = h_new
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
                
        T *= cooling
        
    return best_h
