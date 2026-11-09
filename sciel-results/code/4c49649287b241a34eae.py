import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial best: constant 0.5
    best_h = np.full(N, 0.5)
    best_score = 25.0  # Theoretical lower bound for circular
    
    def get_overlap(h):
        # Compute max overlap of h with shifted 1-h (circular)
        g = 1.0 - h
        max_ov = 0.0
        for k in range(N):
            # Circular shift of g by k
            shifted_g = np.roll(g, k)
            ov = np.sum(h * shifted_g)
            if ov > max_ov:
                max_ov = ov
        return max_ov
    
    # Try a few random binary sequences
    rng = np.random.default_rng(42)
    
    for _ in range(50):
        if time.time() > deadline:
            break
        h = (rng.random(N) < 0.5).astype(float)
        # Ensure sum is exactly 50
        # If sum is not 50, adjust randomly
        while np.sum(h) != 50:
            if np.sum(h) < 50:
                idx = np.where(h == 0)[0]
                if len(idx) == 0: break
                h[rng.choice(idx)] = 1.0
            else:
                idx = np.where(h == 1)[0]
                if len(idx) == 0: break
                h[rng.choice(idx)] = 0.0
        
        score = get_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
        # Local search on this binary sequence
        for _ in range(100):
            if time.time() > deadline:
                break
            i = rng.integers(N)
            h[i] = 1.0 - h[i]
            # Re-normalize if necessary? No, flipping keeps sum 50 if we flip one?
            # Wait, flipping one bit changes sum by 1 or -1.
            # We need sum exactly 50.
            # So we must flip two bits? Or accept the drift?
            # The task says "sum is exactly len/2".
            # So we must maintain sum 50.
            # Strategy: Flip i and j.
            j = rng.integers(N)
            if i == j: continue
            if h[i] == h[j]:
                # Flipping both doesn't change sum
                h[i] = 1.0 - h[i]
                h[j] = 1.0 - h[j]
            else:
                # Flipping both changes sum by 0?
                # 1->0 and 0->1. Sum change -1 + 1 = 0.
                h[i] = 1.0 - h[i]
                h[j] = 1.0 - h[j]
                
            score = get_overlap(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
            else:
                # Revert
                h[i] = 1.0 - h[i]
                h[j] = 1.0 - h[j]

    return best_h
