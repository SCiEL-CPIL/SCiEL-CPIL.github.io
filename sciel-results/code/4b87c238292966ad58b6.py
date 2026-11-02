import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    best_f = None
    best_score = float('inf')
    
    # We will try multiple random restarts and local search
    # Objective: Minimize L1 norm of linear convolution f * f
    # Note: For f in {+1, -1}, ||f||_1 is constant (N).
    
    def compute_conv(f):
        return np.convolve(f, f, 'full')
    
    def compute_score(f):
        c = compute_conv(f)
        return np.sum(np.abs(c))
    
    # Initial random seed
    # Try a few different initializations
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Initialize f with random signs
        f = np.random.choice([-1, 1], size=N)
        
        c = compute_conv(f)
        score = np.sum(np.abs(c))
        
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # To speed up, we can iterate indices in random order or sequential
            # Sequential is fine, but might get stuck in patterns. Random order helps.
            indices = np.random.permutation(N)
            for i in indices:
                if time.time() > deadline:
                    return best_f if best_f is not None else f
                
                # Try flipping f[i]
                # c_new[k] = c[k] - 2*f[i]*f[k-i] for valid k
                # The convolution index k ranges from 0 to 2N-2.
                # f[k-i] is valid if 0 <= k-i < N.
                # So k ranges from i to i+N-1.
                
                # Extract the affected part of c
                # c[i : i+N] corresponds to terms where the second factor index is 0..N-1
                # Wait, c[k] = sum_{j} f[j] f[k-j].
                # If we flip f[i] (first factor), then for a fixed k,
                # the term f[i] f[k-i] is part of the sum if 0 <= k-i < N.
                # So the affected k are i <= k <= i+N-1.
                
                # Let's compute the delta for these k.
                # current_c_segment = c[i : i+N]
                # The term being removed is f[i] * f[0 : N] (shifted)
                # Specifically, for k in i..i+N-1, the term is f[i] * f[k-i].
                # So the change is -2 * f[i] * f[k-i].
                
                # f[k-i] for k=i..i+N-1 is f[0..N-1] = f.
                # So the update is:
                # c_new_segment = c[i:i+N] - 2 * f[i] * f
                
                # We need to calculate the change in sum of absolute values.
                old_segment = c[i:i+N]
                new_segment = old_segment - 2 * f[i] * f
                
                old_sum = np.sum(np.abs(old_segment))
                new_sum = np.sum(np.abs(new_segment))
                
                delta = new_sum - old_sum
                
                if delta < -1e-9: # Strict improvement
                    # Apply flip
                    f[i] = -f[i]
                    c[i:i+N] = new_segment
                    score += delta
                    improved = True
        
        # After local search, check if this is the best
        if score < best_score:
            best_score = score
            best_f = f.copy()
            
    if best_f is None:
        # Fallback to random
        best_f = np.random.choice([-1, 1], size=N)
        
    return best_f
