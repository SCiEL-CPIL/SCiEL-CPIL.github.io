import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a signed step function with 400 intervals.
    Uses a Chirp sequence initialization followed by local search (hill climbing)
    to minimize the sum of squared autocorrelation sidelobes.
    This approach improves the upper bound on the third autocorrelation coefficient
    by minimizing the overall sidelobe energy.
    """
    # Initialize with a Chirp sequence: s_n = (-1)^(n(n-1)/2)
    n = np.arange(N)
    signs = np.ones(N)
    # Calculate quadratic phase
    # n(n-1)/2 mod 2 determines the sign
    # Using integer arithmetic to avoid floating point issues
    phase = (n * (n - 1)) // 2
    signs[phase % 2 == 1] = -1.0
    
    # Define cost function: Sum of squared autocorrelation sidelobes
    def cost(s):
        # Compute full autocorrelation
        # np.correlate(s, s, mode='full')
        # Index N-1 is lag 0.
        # Indices N to 2N-2 are lags 1 to N-1.
        ac = np.correlate(s, s, mode='full')
        sidelobes = ac[N:]
        return np.sum(sidelobes**2)
        
    current_cost = cost(signs)
    
    # Hill Climbing Optimization
    # Perform multiple passes, flipping bits if it reduces cost
    max_passes = 200
    for _ in range(max_passes):
        improved = False
        for i in range(N):
            # Try flipping the sign at index i
            old_val = signs[i]
            signs[i] = -old_val
            
            new_cost = cost(signs)
            
            if new_cost < current_cost:
                current_cost = new_cost
                improved = True
            else:
                # Revert if cost did not improve
                signs[i] = old_val
                
        # If no improvement in a full pass, stop
        if not improved:
            break
            
    return signs
# EVOLVE-BLOCK-END
