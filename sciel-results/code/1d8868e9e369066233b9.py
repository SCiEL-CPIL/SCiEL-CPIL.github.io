import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = 120.0
    margin = 10.0
    
    # Initial best: constant 0.5
    h_best = np.full(N, 0.5)
    score_best = 25.0 # Max overlap for constant is 25
    
    # Function to calculate max overlap
    def get_max_overlap(h):
        max_ov = 0
        # Linear shifts
        for k in range(N):
            # h and 1-h shifted by k
            # Overlap sum over valid indices
            # h[0:N-k] * (1 - h[k:N])
            ov = np.sum(h[:N-k] * (1 - h[k:]))
            if ov > max_ov:
                max_ov = ov
        return max_ov

    # Verify initial
    score_best = get_max_overlap(h_best)
    
    # Simple local search
    # Try to flip small sections or adjust values
    # Since N=100, full evaluation is O(N^2) = 10000 ops.
    # 150s allows for many iterations.
    
    # We'll use a simple hill climbing with random restarts
    current_h = h_best.copy()
    current_score = score_best
    
    np.random.seed(42)
    
    while time.time() < deadline - margin:
        # Perturb
        # Try a few random moves
        for _ in range(100):
            if time.time() > deadline - margin:
                break
            # Pick a random index and add noise
            idx = np.random.randint(0, N)
            # Or swap two elements?
            # Or add small noise to a block?
            
            # Let's try adjusting a single element
            old_val = current_h[idx]
            new_val = np.clip(old_val + np.random.uniform(-0.1, 0.1), 0, 1)
            
            # Adjust sum to maintain integral 1 (sum = N/2)
            # This is tricky for single element.
            # Instead, let's use a projection method or just restart with random binary and polish.
            
            pass
        
        # Better approach: Start with random binary, then project to [0,1] and sum=N/2, then polish.
        h_rand = np.random.randint(0, 2, N).astype(float)
        # Scale to sum N/2
        s = np.sum(h_rand)
        if s > 0:
            h_rand = h_rand * (N/2) / s
        h_rand = np.clip(h_rand, 0, 1)
        
        # Local polish: gradient descent step
        # Objective: max_k sum h_i (1-h_{i+k})
        # This is non-differentiable due to max.
        # Use subgradient or just random search on the continuous space.
        
        # Given time constraints, I'll stick to the constant solution if optimization is complex.
        # But let's try one specific pattern: Alternating 0.2, 0.8?
        # Sum = 50.
        # h = [0.2, 0.8, 0.2, 0.8...]
        # O_0 = 50 * (0.2*0.8 + 0.8*0.2) / 2? No.
        # O_0 = sum h(1-h) = 50 * (0.16 + 0.16) / 2? 
        # 50 pairs of (0.2, 0.8).
        # 0.2*0.8 = 0.16. 0.8*0.2 = 0.16.
        # Sum = 100 * 0.16 / 2? No, N=100.
        # 50 zeros and 50 ones in binary.
        # Here 50 of 0.2 and 50 of 0.8.
        # O_0 = 50 * (0.2 * 0.8) + 50 * (0.8 * 0.2) = 8 + 8 = 16.
        # This is < 25!
        
        # Check O_1 for alternating 0.2, 0.8
        # h = 0.2, 0.8, 0.2, 0.8...
        # k=1:
        # pairs: (0.2, 0.8) -> 0.2 * (1-0.8) = 0.04
        # (0.8, 0.2) -> 0.8 * (1-0.2) = 0.64
        # Sum approx 50 * (0.04 + 0.64) / 2 = 50 * 0.34 = 17.
        # Max so far 17.
        
        # Check k=2:
        # (0.2, 0.2) -> 0.2 * 0.8 = 0.16
        # (0.8, 0.8) -> 0.8 * 0.2 = 0.16
        # Sum = 100 * 0.16 / 2 = 8.
        
        # It seems alternating 0.2, 0.8 gives max overlap around 17.
        # This is better than 25!
        
        # Let's verify this hypothesis.
        h_alt = np.array([0.2, 0.8] * (N//2))
        score_alt = get_max_overlap(h_alt)
        
        if score_alt < score_best:
            h_best = h_alt
            score_best = score_alt
            
    return h_best
