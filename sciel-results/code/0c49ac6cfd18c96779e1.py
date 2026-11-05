import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    def evaluate(h):
        s = np.sum(h)
        if s == 0:
            return np.inf
        c = np.convolve(h, h)
        max_c = np.max(c)
        return abs(2 * N * max_c / (s ** 2))
    
    # Initial best
    best_h = np.ones(N, dtype=np.int8)
    best_val = evaluate(best_h)
    
    # Helper for local search
    def local_search(h, max_iters=1000):
        current_h = h.copy()
        current_val = evaluate(current_h)
        s = np.sum(current_h)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try flipping each sign
            for i in range(N):
                # Flip h[i]
                old_val = current_h[i]
                new_val = -old_val
                current_h[i] = new_val
                
                new_s = s - 2 * old_val
                if new_s == 0:
                    current_h[i] = old_val
                    continue
                
                # Calculate new objective
                # We can optimize the convolution update, but N=400 is small
                c = np.convolve(current_h, current_h)
                max_c = np.max(c)
                new_obj = abs(2 * N * max_c / (new_s ** 2))
                
                if new_obj < current_val:
                    current_val = new_obj
                    s = new_s
                    improved = True
                else:
                    current_h[i] = old_val
        
        return current_h, current_val

    # Strategy: Multiple random restarts with local search
    # We can also mix in some structured starts
    
    # 1. Try the all-ones vector (already done)
    
    # 2. Try random binary vectors
    rng = np.random.default_rng(42)
    
    # We will run restarts until deadline
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        
        # Generate a random starting point
        # Mix of pure random and "chirp-like" or "block" structures
        if restarts % 3 == 0:
            # Random signs
            h_init = rng.choice([-1, 1], size=N).astype(np.int8)
        elif restarts % 3 == 1:
            # Block structure: first half 1, second half -1? Or random blocks
            h_init = np.ones(N, dtype=np.int8)
            # Flip a random contiguous block
            start = rng.integers(0, N)
            length = rng.integers(1, N // 2)
            h_init[start:(start + length) % N] = -1
            # Ensure it's not all 1s or all -1s (sum 0 check in evaluate handles it, but sum=0 is bad)
            # If sum is 0, flip one bit
            if np.sum(h_init) == 0:
                h_init[0] = 1
        else:
            # Chirp-like: sign of cos(k^2)
            k = np.arange(N)
            h_init = np.sign(np.cos(2 * np.pi * k**2 / N)).astype(np.int8)
            # Ensure not all zero (sign(0)=0)
            h_init[h_init == 0] = 1
            if np.sum(h_init) == 0:
                h_init[0] = 1

        # Run local search
        h_local, val_local = local_search(h_init, max_iters=200)
        
        if val_local < best_val:
            best_val = val_local
            best_h = h_local.copy()
            # Optional: Print progress if it was a significant improvement
            # print(f"Restart {restarts}: Improved to {best_val:.4f}")

    return best_h

