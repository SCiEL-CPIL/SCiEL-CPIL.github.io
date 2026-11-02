import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Initialize with a random sequence of -1 and 1
    # Using a fixed seed for reproducibility in the function context is not required 
    # but helps debugging. The evaluator runs construct() fresh.
    x = np.random.choice([-1, 1], size=N)
    
    def score(x):
        # Calculate autocorrelation
        # np.correlate is O(N^2) which is 160,000 ops, very fast.
        r = np.correlate(x, x, mode='full')
        # The full correlation has length 2N-1.
        # Lag 0 is at index N-1.
        # We want max absolute value.
        max_abs_r = np.max(np.abs(r))
        
        # Calculate sum
        s = np.sum(x)
        s_sq = s * s
        
        if s_sq == 0:
            return float('inf')
            
        return max_abs_r / s_sq

    best_x = x
    best_score = score(x)
    
    # Local Search
    # We try to flip bits to improve the score
    # To keep it fast, we do a limited number of iterations
    num_iterations = 5000
    np.random.seed(42) # Fixed seed for the search process to be deterministic if needed, 
                        # but usually random is fine. Let's use default random.
    
    for _ in range(num_iterations):
        # Pick a random index to flip
        idx = np.random.randint(N)
        
        # Flip
        x[idx] *= -1
        
        # Calculate new score
        # Optimization: We can update R and S incrementally, but N=400 is small.
        # np.correlate is fast enough.
        new_score = score(x)
        
        if new_score < best_score:
            best_score = new_score
            best_x = x.copy()
        else:
            # Revert
            x[idx] *= -1
            
    return best_x
# EVOLVE-BLOCK-END
