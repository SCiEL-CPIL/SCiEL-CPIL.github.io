import numpy as np
import time

N = 17

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Leave 5 seconds margin for overhead and finalization
    deadline = start_time + 85.0
    
    # Initial Network: Odd-Even Transposition
    # This is a valid sorting network with N^2/2 comparators for even N, 
    # and similar for odd. For N=17, it is 136 comparators.
    net = [(i, i + 1) for phase in range(N)
           for i in range(phase % 2, N - 1, 2)]
           
    current_net = net
    
    # Pre-compute powers of 2 for input generation
    # We generate inputs as binary representations of 0 to 2^N - 1
    powers = 2 ** np.arange(N, 0, -1, dtype=np.uint32)
    
    # Pre-generate all 2^N inputs
    # Shape: (2^N, N)
    # dtype uint32 to allow bitwise operations
    inputs = np.arange(2**N, dtype=np.uint32).reshape(-1, 1)
    inputs = (inputs // powers) % 2
    
    def check_validity(network):
        # Copy inputs to state
        state = inputs.copy()
        
        # Apply comparators
        for i, j in network:
            # min is AND, max is OR for bits
            min_val = state[:, i] & state[:, j]
            max_val = state[:, i] | state[:, j]
            state[:, i] = min_val
            state[:, j] = max_val
            
        # Check if all rows are sorted
        # A row is sorted if state[k, i] <= state[k, i+1] for all i
        # We check if any state[k, i] > state[k, i+1]
        # state[:, 1:] > state[:, :-1]
        # If any is True, it's not sorted.
        return not np.any(state[:, 1:] > state[:, :-1])
        
    # Verify initial network (should be true)
    # if not check_validity(current_net):
    #     raise Exception("Initial network is invalid")
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each comparator
        # Shuffle indices to avoid bias and find different local minima
        indices = list(range(len(current_net)))
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
                
            # Create new network without comparator at idx
            new_net = current_net[:idx] + current_net[idx+1:]
            
            # Check validity
            if check_validity(new_net):
                current_net = new_net
                improved = True
                # Break to restart the scan with the new network
                break
                
    return current_net
# EVOLVE-BLOCK-END
