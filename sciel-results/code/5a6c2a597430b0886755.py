import numpy as np

def construct():
    N = 50
    DELTA = 0.5 / N
    
    # We want to maximize R = sum(h^2) / (sum(h) * max(h*h))
    # Assuming sum(h) = 1 is the effective constraint or normalization.
    # If the evaluator normalizes, we should return a shape.
    # If the evaluator does not normalize, we should return a shape with sum=1 
    # to be consistent with the seed's behavior, or check if scaling helps.
    # Given the seed normalizes to sum=1, I will assume the optimal shape is 
    # independent of scale if the evaluator normalizes, or we stick to sum=1.
    
    def score_h(h):
        # h is a numpy array of non-negative numbers
        s = np.sum(h)
        if s < 1e-15:
            return 0.0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv < 1e-15:
            return 0.0
        # The seed score formula: l2_sq / (sum * DELTA * max_conv)
        # We want to maximize this.
        # Since DELTA is constant, we maximize l2_sq / (sum * max_conv)
        return l2_sq / (s * max_conv)

    # Initialize with uniform
    best_h = np.ones(N) / N
    best_score = score_h(best_h)
    
    # Random search / Hill climbing
    # We will perturb the vector, renormalize to sum=1, and check score.
    # We will also try to break symmetry.
    
    for i in range(2000):
        # Random perturbation
        # Add noise, clip to 0, normalize
        noise = np.random.randn(N) * 0.1
        h_new = best_h + noise
        h_new[h_new < 0] = 0
        s_new = np.sum(h_new)
        if s_new < 1e-15:
            continue
        h_new = h_new / s_new
        
        # Also try a "spike" move: pick a random index, increase it, decrease others
        # Or just stick to Gaussian noise which is simpler
        
        s_new_score = score_h(h_new)
        
        if s_new_score > best_score:
            best_score = s_new_score
            best_h = h_new
            
            # If we find a significant improvement, try to refine with smaller noise
            # For now, just continue
    
    # Optional: Try to find a shape with R > 1
    # One specific candidate: alternating signs? No, non-negative.
    # What about a "comb" pattern?
    
    # Let's try a few specific patterns as initial seeds for the hill climb
    patterns = [
        np.ones(N), # Uniform
        np.eye(N)[0], # Spike
        np.array([1 if i % 2 == 0 else 0 for i in range(N)]), # Comb
        np.array([1 if i < N//2 else 0 for i in range(N)]), # Half block
        np.array([1 if i == N//2 else 0 for i in range(N)]), # Center spike
    ]
    
    for p in patterns:
        p = p.astype(float)
        p = p / np.sum(p)
        s_p = score_h(p)
        if s_p > best_score:
            best_score = s_p
            best_h = p.copy()
            
            # Refine this pattern
            for j in range(500):
                noise = np.random.randn(N) * 0.05
                h_new = best_h + noise
                h_new[h_new < 0] = 0
                s_new = np.sum(h_new)
                if s_new < 1e-15: continue
                h_new = h_new / s_new
                s_new_score = score_h(h_new)
                if s_new_score > best_score:
                    best_score = s_new_score
                    best_h = h_new

    return list(best_h)
