import numpy as np
import time

N = 100
TARGET_SUM = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Initial best
    best_h = np.full(N, 0.5)
    best_val = 0.5 # seed value
    
    def evaluate(h):
        c = np.correlate(h, 1 - h, mode="full")
        return np.max(c) / N * 2

    # 1. Random Search
    h = np.array([1]*TARGET_SUM + [0]*(N-TARGET_SUM))
    
    # Run random search for first 20 seconds
    rand_deadline = start_time + 20
    while time.time() < rand_deadline:
        np.random.shuffle(h)
        val = evaluate(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    # 2. Local Search
    # Use the best found so far
    h = best_h.copy()
    
    # If best_h is still 0.5, we might want to start local search from a random binary seq
    if np.allclose(h, 0.5):
         np.random.shuffle(h)
         val = evaluate(h)
         if val < best_val:
             best_val = val
             best_h = h.copy()
         else:
             # Keep 0.5? Or use this random one for search?
             # If random is worse, search might improve it.
             # But 0.5 is a strong baseline.
             # Let's try to improve the random one.
             pass 

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try swaps
        # To make it faster, we can pre-compute indices?
        # But h changes.
        
        # Batch of swaps
        for _ in range(5000):
            if time.time() > deadline:
                break
                
            ones = np.where(h == 1)[0]
            zeros = np.where(h == 0)[0]
            if len(ones) == 0 or len(zeros) == 0:
                break
            
            i = np.random.choice(ones)
            j = np.random.choice(zeros)
            
            # Swap
            h[i], h[j] = h[j], h[i]
            
            new_val = evaluate(h)
            
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
                improved = True
            else:
                h[i], h[j] = h[j], h[i]
                
    return best_h
